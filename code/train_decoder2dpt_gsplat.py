"""
Pipeline 2 — Étape 4 (2D+gsplat) : Entraînement du décodeur 2D(+t)GS via gsplat.

Utilise GaussianSplatDecoder2pt_gsplat : même Schur complement que le décodeur 2D
original, mais rasterisation CUDA via gsplat (~100× plus rapide que le rendu
Python sur grille pixel).

Différences vs le décodeur 2D+t de référence :
  - Rasterisation via gsplat.rasterization (compositing alpha séquentiel)
  - Pruning + réinit des Gaussiennes mortes (alpha < seuil)
  - Encodeur du pipeline 3D (avec ENC_NORMALIZE/whitening)
  - Loss : L1 + SSIM (pas de MSE)
  - Pas de viz.py requis

Prérequis : encoder_finetuned.pt (ou encoder.pt) dans SAVE_DIR.
Produit   : <SAVE_DIR>/decoder2dpt.pt

Usage :
    py train_decoder2dpt_gsplat.py --config ../cases/krauss2026_2seg_npz/config.py
"""

import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_fn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path

from _bootstrap import load_config
config = load_config()

# Overrides rapides en ligne de commande, consommés avant main(). Ils écrivent dans
# le module config déjà chargé, donc le reste du script les voit comme des réglages
# ordinaires (aucun config_*.py à modifier pour un run de ré-entraînement).
_ap_ep = argparse.ArgumentParser(add_help=False)
_ap_ep.add_argument('--epochs', type=int, default=None)
_ap_ep.add_argument('--n-gaussians', type=int, default=None,
                    help='surcharge DEC2PT_GSPLAT_N_GAUSSIANS')
_ap_ep.add_argument('--alpha-w', type=float, default=None,
                    help='surcharge DEC2PT_ALPHA_W (pression douce sur les opacités)')
_ap_ep.add_argument('--n-plots', type=int, default=None,
                    help='nombre TOTAL de plots de debug (fixe DEC_PLOT_EVERY)')
_ap_ep.add_argument('--out', type=str, default='decoder2dpt.pt',
                    help='nom du checkpoint écrit dans SAVE_DIR (et repris s\'il existe)')
_ep_args, _ = _ap_ep.parse_known_args()
if _ep_args.epochs is not None:
    config.DEC_EPOCHS = _ep_args.epochs
if _ep_args.n_gaussians is not None:
    config.DEC2PT_GSPLAT_N_GAUSSIANS = _ep_args.n_gaussians
if _ep_args.alpha_w is not None:
    config.DEC2PT_ALPHA_W = _ep_args.alpha_w
if _ep_args.n_plots is not None:
    config.DEC_PLOT_EVERY = max(config.DEC_EPOCHS // max(_ep_args.n_plots, 1), 1)

from dataset import VideoFrameDataset
from models import Encoder, build_encoder
from models_2pt import build_decoder2pt


# ─────────────────────────────────────────────────────────────────────────────
# Chargement de l'encodeur
# ─────────────────────────────────────────────────────────────────────────────

def load_encoder(device):
    n_ch = 3 if config.ENC_COLOR else 1
    encoder = build_encoder(
        img_size    = config.IMG_SIZE,
        latent_dim  = config.LATENT_DIM,
        hidden_dims = config.ENC_HIDDEN,
        n_channels  = n_ch,
        normalize   = getattr(config, 'ENC_NORMALIZE', False),
    ).to(device)
    # encoder_ae.pt (encodeur GELÉ de l'AE conjoint) EN PREMIER : c'est l'espace
    # latent sur lequel compute_latent_whiten ajuste le blanchiment et que l'éval /
    # le rollout décodent (whiten.inverse → z brut). Le décodeur doit donc être
    # entraîné dans CET espace pour rester cohérent avec l'aval.
    for candidate in [
        config.SAVE_DIR / 'encoder_ae.pt',
        config.SAVE_DIR / 'encoder_finetuned.pt',
        config.SAVE_DIR / 'encoder.pt',
    ]:
        if candidate.exists():
            encoder.load_state_dict(torch.load(candidate, map_location=device))
            print(f'Encodeur : {candidate}')
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad_(False)
            return encoder
    raise FileNotFoundError(
        f'Aucun checkpoint encodeur dans {config.SAVE_DIR} — '
        'lance train_ae.py d\'abord'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pruning / réinit des Gaussiennes mortes
# ─────────────────────────────────────────────────────────────────────────────

def prune_and_reinit_2d(
    decoder,
    optimizer,
    frames_dec_t,   # (N, C, H, W) tensor CPU — frames de supervision
    z_all,          # (N, d) tensor CPU
    device,
    sigma_xy,       # σ_xy cible pour les Gaussiennes réinitialisées
    alpha_thresh,
    epoch=0,
):
    """
    Réinitialise les Gaussiennes dont l'opacité < alpha_thresh.
    Les nouveaux centres sont échantillonnés dans la carte d'erreur L1.
    """
    with torch.no_grad():
        alpha = torch.sigmoid(decoder.log_alpha)   # (K,)
        dead  = alpha < alpha_thresh
        n_dead = dead.sum().item()
        if n_dead == 0:
            return 0

        # Frame de référence aléatoire
        N       = frames_dec_t.shape[0]
        ref_idx = int(torch.randint(0, N, (1,)).item())
        z_ref   = z_all[ref_idx:ref_idx + 1].to(device)
        rgb_ref = frames_dec_t[ref_idx].to(device)
        # frames_dec_t peut être uint8 [0,255] (train_ae store_uint8) → float [0,1].
        if rgb_ref.dtype == torch.uint8:
            rgb_ref = rgb_ref.float() / 255.0

        # Carte d'erreur L1
        decoder.eval()
        rgb_pred = decoder(z_ref)[0].detach()   # (C, H, W)
        decoder.train()
        err_map  = (rgb_pred - rgb_ref).abs().mean(0)           # (H, W)
        err_np   = err_map.cpu().numpy().astype(np.float64).clip(0, None)
        err_flat = err_np.flatten()
        err_sum  = err_flat.sum()
        probs    = err_flat / err_sum if err_sum > 1e-10 else np.ones_like(err_flat) / len(err_flat)

        H_out, W_out = decoder.img_size
        rng      = np.random.default_rng()
        flat_idx = rng.choice(len(probs), size=n_dead, replace=True, p=probs)
        py_i     = flat_idx // W_out
        px_i     = flat_idx  % W_out

        # Nouvelles positions normalisées → inverse de sigmoid(x)*1.2-0.1
        mu_x_new = px_i.astype(np.float32) / max(W_out - 1, 1)
        mu_y_new = py_i.astype(np.float32) / max(H_out - 1, 1)
        def _inv_ext_sig(u):
            v = ((u + 0.1) / 1.2).clip(0.01, 0.99)
            return np.log(v / (1 - v))
        logit_xy = _inv_ext_sig(np.stack([mu_x_new, mu_y_new], axis=1))  # (n_dead, 2)

        # mu_z : distribué autour du z de la frame de référence
        z_ref_np = z_all[ref_idx].cpu().numpy()
        mu_z_new = np.tile(z_ref_np, (n_dead, 1))                         # (n_dead, d)
        mu_raw_new = np.concatenate([logit_xy, mu_z_new], axis=1)         # (n_dead, 2+d)

        # Couleurs depuis la frame de référence
        C    = decoder.n_channels
        img  = rgb_ref.cpu().numpy()
        cols = img[0, py_i, px_i][:, None] if C == 1 else img[:, py_i, px_i].T
        cols = cols.clip(0.01, 0.99)

        # L_raw : diagonale initialisée aux σ cibles
        z_std        = z_all.cpu().numpy().std(axis=0).clip(1e-3)
        sigma_target = np.concatenate([np.full(2, sigma_xy), z_std])
        def _sp_inv(x):
            return np.log(np.exp(np.clip(x, 1e-4, 30)) - 1 + 1e-6)
        tril_r = decoder.tril_rows.cpu().numpy()
        tril_c = decoder.tril_cols.cpu().numpy()
        L_raw_new = np.zeros((n_dead, decoder.n_tril), dtype=np.float32)
        for i in range(decoder.full_dim):
            idx_d = np.where((tril_r == i) & (tril_c == i))[0][0]
            L_raw_new[:, idx_d] = _sp_inv(sigma_target[i])

        # Écriture
        dev_d    = decoder.mu_raw.device
        dead_idx = dead.nonzero(as_tuple=False).squeeze(1)
        decoder.mu_raw.data[dead_idx]    = torch.tensor(mu_raw_new, dtype=torch.float32).to(dev_d)
        decoder.L_raw.data[dead_idx]     = torch.tensor(L_raw_new,  dtype=torch.float32).to(dev_d)
        decoder.color_raw.data[dead_idx] = torch.tensor(
            np.log(cols / (1 - cols)), dtype=torch.float32
        ).to(dev_d)
        decoder.log_alpha.data[dead_idx] = -3.0   # sigmoid(-3) ≈ 0.047

        # Reset moments Adam pour les slots modifiés
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state.get(p)
                if state is None or 'exp_avg' not in state:
                    continue
                idx = dead_idx.to(p.device)
                if state['exp_avg'].shape[0] == decoder.n_gaussians:
                    state['exp_avg'][idx]    = 0.0
                    state['exp_avg_sq'][idx] = 0.0

        print(f'  [prune2d ep{epoch+1}] {n_dead}/{decoder.n_gaussians} '
              f'Gaussiennes réinitialisées (frame {ref_idx})')
        return n_dead


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _to_img(t):
    img = t.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    return img[:, :, 0] if img.shape[2] == 1 else img


def _f01(x):
    """uint8 [0,255] (store_uint8) → float [0,1] ; passe-plat si déjà float."""
    return x.float().div(255.0) if x.dtype == torch.uint8 else x


def plot_epoch(decoder, z_samples, samples, device, epoch, save_dir):
    debug_dir = Path(save_dir) / 'debug_plots_2dpt_gsplat'
    debug_dir.mkdir(exist_ok=True)

    decoder.eval()
    with torch.no_grad():
        recons = decoder(z_samples.to(device)).cpu()   # (n, C, H, W)

        # Ellipses 2D conditionnelles (Schur complement, même logique que le parent)
        mu_xy, mu_z, Sigma, alpha_p, _ = decoder._decode_params()
        Sigma_xx = Sigma[:, :2, :2]
        Sigma_xz = Sigma[:, :2, 2:]
        Sigma_zz = Sigma[:, 2:, 2:]
        d = decoder.latent_dim
        eye_d = torch.eye(d, device=device, dtype=Sigma_zz.dtype).unsqueeze(0)
        Szz_inv = torch.linalg.inv(Sigma_zz + 1e-4 * eye_d)
        Sigma_cond = Sigma_xx - (Sigma_xz @ Szz_inv) @ Sigma_xz.transpose(-1, -2)
        dz = z_samples[0:1].to(device).unsqueeze(1) - mu_z.unsqueeze(0)
        shift = (Sigma_xz @ Szz_inv).unsqueeze(0) @ dz.unsqueeze(-1)
        mu_cond = mu_xy.unsqueeze(0) + shift.squeeze(-1)   # (1, K, 2)
        # Forme close 2x2 — remplace eigh qui diverge sur matrices mal conditionnées
        S_c = (Sigma_cond + Sigma_cond.transpose(-1, -2)) * 0.5
        S_c = S_c + 1e-5 * torch.eye(2, device=device, dtype=S_c.dtype).unsqueeze(0)
        _a = S_c[:, 0, 0]; _b = S_c[:, 0, 1]; _cc = S_c[:, 1, 1]
        _m = (_a + _cc) * 0.5
        _p = (((_a - _cc) * 0.5) ** 2 + _b ** 2).clamp(min=1e-10).sqrt()
        eigvals  = torch.stack([_m - _p, _m + _p], dim=1).clamp(1e-8)
        sigma_px = eigvals.sqrt()                          # (K, 2)
        # Vecteur propre de λ₁ : [-b, (a-c)/2 + p]
        angles   = torch.atan2((_a - _cc) * 0.5 + _p, -_b)  # (K,)

    mu_cond_np = mu_cond[0].cpu().numpy()
    sigma_np   = sigma_px.cpu().numpy()
    angles_np  = angles.cpu().numpy()
    alpha_np   = torch.sigmoid(decoder.log_alpha).detach().cpu().numpy().clip(0, 1)

    H, W = decoder.img_size
    n    = samples.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    fig.suptitle(f'Epoch {epoch}  [2D+t GS / gsplat]')

    for i in range(n):
        axes[0, i].imshow(_to_img(samples[i]))
        axes[0, i].axis('off')
        axes[1, i].imshow(_to_img(recons[i]))
        axes[1, i].axis('off')
        if i == 0:
            for k in range(decoder.n_gaussians):
                ell = Ellipse(
                    xy=(mu_cond_np[k, 0] * W, mu_cond_np[k, 1] * H),
                    width=2 * sigma_np[k, 0] * W,
                    height=2 * sigma_np[k, 1] * H,
                    angle=float(np.degrees(angles_np[k])),
                    edgecolor='red', facecolor='none',
                    linewidth=0.8, alpha=float(alpha_np[k]),
                )
                axes[1, 0].add_patch(ell)

    axes[0, 0].set_ylabel('Original',    fontsize=8)
    axes[1, 0].set_ylabel('Reconstruit', fontsize=8)
    plt.tight_layout()
    fig.savefig(debug_dir / f'epoch_{epoch:04d}.png', dpi=100)
    plt.close(fig)
    decoder.train()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    config.SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    n_ch_enc = 3 if config.ENC_COLOR  else 1
    n_ch_dec = 3 if getattr(config, 'DEC_COLOR', True) else 1

    # ── Dataset ───────────────────────────────────────────────────────────────
    print('Chargement des frames…')
    enc_dataset = VideoFrameDataset(
        video_dir           = config.VIDEO_DIR,
        img_size            = config.IMG_SIZE,
        n_channels          = n_ch_enc,
        crop                = getattr(config, 'CROP', None),
        rest_video          = config.REST_VIDEO,
        rest_n_frames       = config.REST_N_FRAMES,
        rest_first_n_frames = getattr(config, 'REST_FIRST_N_FRAMES', 0),
        exclude_videos      = [config.VAL_VIDEO] if config.VAL_VIDEO else None,
        store_uint8         = True,   # ~4× moins de RAM (vidéo 15 min → float32 = OOM)
    )
    if n_ch_dec != n_ch_enc:
        dec_dataset = VideoFrameDataset(
            video_dir           = config.VIDEO_DIR,
            img_size            = config.IMG_SIZE,
            n_channels          = n_ch_dec,
            crop                = getattr(config, 'CROP', None),
            rest_video          = config.REST_VIDEO,
            rest_n_frames       = config.REST_N_FRAMES,
            rest_first_n_frames = getattr(config, 'REST_FIRST_N_FRAMES', 0),
            exclude_videos      = [config.VAL_VIDEO] if config.VAL_VIDEO else None,
            store_uint8         = True,
        )
    else:
        dec_dataset = enc_dataset
    N = len(dec_dataset)
    print(f'  {N} frames  (enc: {"RGB" if n_ch_enc==3 else "gris"}, '
          f'dec: {"RGB" if n_ch_dec==3 else "gris"})')

    # ── Encodeur figé + précalcul z_all ───────────────────────────────────────
    encoder = load_encoder(device)

    print('Précalcul z_all…')
    all_enc = torch.from_numpy(enc_dataset.frames)   # (N, C_enc, H, W)
    batch   = getattr(config, 'DEC_BATCH', 1)
    with torch.no_grad():
        parts = []
        for i in range(0, N, max(batch, 32)):
            parts.append(encoder(_f01(all_enc[i:i + max(batch, 32)].to(device))).cpu())
        z_all = torch.cat(parts, dim=0)   # (N, d)
    print(f'  z_all : {z_all.shape}  '
          f'mean={z_all.mean(0).numpy().round(3)}  '
          f'std={z_all.std(0).numpy().round(3)}')

    # ── Décodeur 2D+t gsplat ──────────────────────────────────────────────────
    # DEC2PT_GSPLAT_N_GAUSSIANS prend priorité sur DEC_N_GAUSSIANS (qui vaut 150 par défaut)
    n_gaussians = getattr(config, 'DEC2PT_GSPLAT_N_GAUSSIANS',
                          getattr(config, 'DEC_N_GAUSSIANS', 2048))
    decoder = build_decoder2pt(
        latent_dim  = config.LATENT_DIM,
        n_gaussians = n_gaussians,
        img_size    = config.IMG_SIZE,
        n_channels  = n_ch_dec,
    ).to(device)

    # Reprise depuis checkpoint si disponible
    ckpt_path = config.SAVE_DIR / _ep_args.out
    if ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location=device)
        if sd['mu_raw'].shape[0] == n_gaussians:
            decoder.load_state_dict(sd)
            print(f'  Checkpoint chargé : {ckpt_path}')
        else:
            print(f'  Checkpoint ignoré (K={sd["mu_raw"].shape[0]} ≠ {n_gaussians})'
                  f' — init smart_init')
            ref_frame = _f01(torch.from_numpy(dec_dataset.frames[0]))
            decoder.smart_init(
                ref_image      = ref_frame,
                z_all          = z_all,
                sigma_xy_scale = getattr(config, 'DEC2PT_SIGMA_XY', 0.05),
                sigma_z_scale  = getattr(config, 'DEC2PT_SIGMA_Z',  1.0),
                sigma_z_floor  = getattr(config, 'DEC2PT_SIGMA_Z_FLOOR', 0.5),
            )
    else:
        ref_frame = _f01(torch.from_numpy(dec_dataset.frames[0]))
        decoder.smart_init(
            ref_image      = ref_frame,
            z_all          = z_all,
            sigma_xy_scale = getattr(config, 'DEC2PT_SIGMA_XY', 0.05),
            sigma_z_scale  = getattr(config, 'DEC2PT_SIGMA_Z',  1.0),
            sigma_z_floor  = getattr(config, 'DEC2PT_SIGMA_Z_FLOOR', 0.5),
        )

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f'Décodeur 2D+t gsplat : {n_gaussians} gaussiennes, {n_params:,} paramètres')

    # ── Hyperparamètres d'entraînement ────────────────────────────────────────
    lr          = getattr(config, 'DEC_LR',          1e-3)
    lr_factor   = getattr(config, 'DEC_LR_FACTOR',   0.5)
    lr_patience = getattr(config, 'DEC_LR_PATIENCE', 200)
    lr_min      = getattr(config, 'DEC_LR_MIN',      1e-6)
    n_epochs    = getattr(config, 'DEC_EPOCHS',       5000)
    l1_w        = getattr(config, 'DEC_L1_W',         0.08)
    ssim_w      = getattr(config, 'DEC_SSIM_W',       0.02)
    aniso_w     = getattr(config, 'DEC_ANISO_W',      0.01)
    plot_every  = getattr(config, 'DEC_PLOT_EVERY',   10)

    # Pruning
    alpha_thresh  = getattr(config, 'DEC2PT_ALPHA_THRESH',  0.02)
    prune_every   = getattr(config, 'DEC2PT_PRUNE_EVERY',   200)
    prune_warmup  = getattr(config, 'DEC2PT_PRUNE_WARMUP',  500)

    # Pression douce vers le bas sur les opacités (pendant exact de AE_ALPHA_W dans
    # train_ae.py et de DEC3PT_ALPHA_W dans train_decoder3pt.py) : pénalise la MOYENNE
    # des α = sigmoid(log_alpha). Une gaussienne utile voit sa L1/SSIM compenser ce
    # coût et garde son opacité ; une gaussienne inutile n'a rien pour s'y opposer et
    # descend sous DEC2PT_ALPHA_THRESH, où le prune/réinit la recycle vers une zone de
    # forte erreur. Sans ce terme, rien ne pousse α vers le bas et le pruning ne se
    # déclenche jamais. Poids à garder FAIBLE : c'est un départage entre primitives
    # équivalentes, pas un objectif. 0 (défaut) ⟹ loss inchangée bit à bit.
    alpha_w       = getattr(config, 'DEC2PT_ALPHA_W', 0.0)

    print(f'  Loss : L1×{l1_w} + SSIM×{ssim_w} + Aniso×{aniso_w} + Alpha×{alpha_w}')
    print(f'  Pruning : alpha_thresh={alpha_thresh}, every={prune_every}, '
          f'warmup={prune_warmup}')

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience, min_lr=lr_min,
    )

    # Samples fixes pour debug
    n_show  = min(8, N)
    dbg_idx = np.linspace(0, N - 1, n_show, dtype=int)
    dbg_z   = z_all[dbg_idx]
    dbg_rgb = _f01(torch.from_numpy(dec_dataset.frames[dbg_idx]))

    # Tenseur frames CPU (pour pruning + lookup direct)
    frames_t  = torch.from_numpy(dec_dataset.frames)   # (N, C_dec, H, W)
    z_all_dev = z_all.to(device)

    losses, losses_l1, losses_ssim, losses_aniso, losses_alpha = [], [], [], [], []

    decoder.train()
    print(f'\nEntraînement : {n_epochs} epochs, {N} frames')

    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        ep_loss = ep_l1 = ep_ssim = ep_aniso = ep_alpha = 0.0

        for start in range(0, N, batch):
            idx      = perm[start:start + batch]
            z        = z_all_dev[idx]
            x_target = _f01(frames_t[idx].to(device))

            x_recon = decoder(z)

            loss_l1   = F.l1_loss(x_recon, x_target) if l1_w   > 0 else x_recon.new_zeros(1)
            loss_ssim = (1.0 - ssim_fn(x_recon, x_target,
                                       data_range=1.0, size_average=True)) \
                        if ssim_w > 0 else x_recon.new_zeros(1)

            if aniso_w > 0:
                Sigma_xx  = decoder._build_Sigma()[:, :2, :2]
                # Forme close 2x2 : évite eigvalsh qui diverge sur matrices mal conditionnées
                _a = Sigma_xx[:, 0, 0]; _b = Sigma_xx[:, 0, 1]; _cc = Sigma_xx[:, 1, 1]
                _m = (_a + _cc) * 0.5
                _p = (((_a - _cc) * 0.5) ** 2 + _b ** 2).clamp(min=1e-10).sqrt()
                eigvals   = torch.stack([_m - _p, _m + _p], dim=1).clamp(1e-8)
                sigma_eig = eigvals.sqrt()
                ratio     = sigma_eig[:, 1] / sigma_eig[:, 0].clamp(1e-8)
                loss_aniso = (
                    F.relu(ratio.clamp(min=1).log() - math.log(5)) +
                    F.relu((1 / ratio).clamp(min=1).log() - math.log(5))
                ).mean()
            else:
                loss_aniso = x_recon.new_zeros(1)

            loss_alpha = (torch.sigmoid(decoder.log_alpha).mean() if alpha_w > 0
                          else x_recon.new_zeros(1))

            loss = (l1_w * loss_l1 + ssim_w * loss_ssim + aniso_w * loss_aniso
                    + alpha_w * loss_alpha)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()

            B = len(idx)
            ep_loss  += loss.item()       * B
            ep_l1    += loss_l1.item()    * B
            ep_ssim  += loss_ssim.item()  * B
            ep_aniso += loss_aniso.item() * B
            ep_alpha += loss_alpha.item() * B

        ep_loss  /= N; ep_l1 /= N; ep_ssim /= N; ep_aniso /= N; ep_alpha /= N
        losses.append(ep_loss); losses_l1.append(ep_l1)
        losses_ssim.append(ep_ssim); losses_aniso.append(ep_aniso)
        losses_alpha.append(ep_alpha)
        scheduler.step(ep_loss)

        # ── Pruning ────────────────────────────────────────────────────────
        # ⚠️ JAMAIS à la dernière époque : le prune/réinit repositionne des gaussiennes à
        # l'aveugle (α = 0.047, zone de forte erreur) et il leur faut quelques centaines
        # d'époques pour redevenir utiles. Comme la sauvegarde suit immédiatement, pruner au
        # dernier tour écrit un checkpoint DÉGRADÉ. Mesuré sur `sac` : 7196/15000 gaussiennes
        # (48 %) réinitialisées à l'époque 1000, juste avant l'écriture du fichier.
        if (prune_every > 0
                and epoch >= prune_warmup
                and epoch < n_epochs - 1
                and (epoch + 1) % prune_every == 0):
            prune_and_reinit_2d(
                decoder, optimizer, frames_t, z_all, device,
                sigma_xy     = getattr(config, 'DEC2PT_SIGMA_XY', 0.05),
                alpha_thresh = alpha_thresh,
                epoch        = epoch,
            )

        # ── Sauvegarde ────────────────────────────────────────────────────
        has_nan = any(p.isnan().any().item() for p in decoder.parameters())
        if not has_nan:
            torch.save(decoder.state_dict(), ckpt_path)

        if (epoch + 1) % max(n_epochs // 20, 1) == 0:
            lr_cur = optimizer.param_groups[0]['lr']
            n_alive = int((torch.sigmoid(decoder.log_alpha) >= alpha_thresh).sum().item())
            print(f'  [{epoch+1:5d}/{n_epochs}]  '
                  f'total={ep_loss:.5f}  L1={ep_l1:.5f}  '
                  f'SSIM={ep_ssim:.5f}  Aniso={ep_aniso:.5f}  '
                  f'Alpha={ep_alpha:.5f}  alive={n_alive}/{decoder.n_gaussians}  '
                  f'lr={lr_cur:.2e}')

        if plot_every > 0 and (epoch + 1) % plot_every == 0:
            plot_epoch(decoder, dbg_z, dbg_rgb, device, epoch + 1, config.SAVE_DIR)

    # ── Courbe de loss ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(losses,       label='total')
    ax.semilogy(losses_l1,    label='L1',   linestyle='--')
    ax.semilogy(losses_ssim,  label='SSIM', linestyle='-.')
    if aniso_w > 0:
        ax.semilogy(losses_aniso, label='Aniso', linestyle=':')
    if alpha_w > 0:
        ax.semilogy(losses_alpha, label='Alpha', linestyle=':')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Décodeur 2D+t gsplat')
    ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(config.SAVE_DIR / 'decoder2dpt_gsplat_loss.png', dpi=150)
    plt.close(fig)

    plot_epoch(decoder, dbg_z, dbg_rgb, device, n_epochs, config.SAVE_DIR)
    print(f'\nDécodeur sauvegardé : {ckpt_path}')
    print(decoder.get_params_summary())


if __name__ == '__main__':
    # --config est consommé en tête de module par _bootstrap.load_config (avant l'import
    # de models.py), de sorte que config est déjà le bon module ici.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='Config propre au cas test (cf. _bootstrap.load_config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='override DEC_EPOCHS (consommé en tête de module)')
    # Les quatre suivants sont eux aussi consommés en tête de module (_ep_args) ;
    # ils sont redéclarés ici pour l'aide et pour que --help reste exhaustif.
    parser.add_argument('--n-gaussians', type=int, default=None,
                        help='override DEC2PT_GSPLAT_N_GAUSSIANS')
    parser.add_argument('--alpha-w', type=float, default=None,
                        help='override DEC2PT_ALPHA_W')
    parser.add_argument('--n-plots', type=int, default=None,
                        help='nombre TOTAL de plots de debug')
    parser.add_argument('--out', type=str, default='decoder2dpt.pt',
                        help='nom du checkpoint écrit dans SAVE_DIR')
    parser.parse_args()
    main()
