"""
Pipeline 2 — Autoencodeur conjoint : encodeur + décodeur 2D+t GS (gsplat).

Entraîne l'encodeur ET le décodeur EN MÊME TEMPS par reconstruction pure
(L1 + SSIM + aniso), sans LNN. À la différence de train_decoder2dpt_gsplat.py
(encodeur figé), ici l'encodeur n'est PAS détaché : le gradient de la loss de
reconstruction remonte dans l'encodeur, qui apprend donc un latent z « rendu-
able » en même temps que le décodeur. C'est la phase « warmup » de train_all
extraite en script autonome (pas de loss physique).

Cohérence d'initialisation :
  - Encodeur : build_encoder(IMG_SIZE, ENC_HIDDEN, LATENT_DIM, n_channels,
               normalize=ENC_NORMALIZE) — identique à train_encoder.py.
               Reprise depuis encoder_ae.pt puis encoder.pt si présents.
               Régularisation : nonlocal_penalty (CpAE) ou gradient penalty (MLP).
  - Décodeur : GaussianSplatDecoder2pt_gsplat + smart_init(ref_frame, z_all)
               — identique à train_decoder2dpt_gsplat.py. Reprise depuis
               decoder2dpt_ae.pt si la taille correspond.
  - Pruning / réinit des gaussiennes mortes + plot debug : réutilisés tels quels
    depuis train_decoder2dpt_gsplat.py.

Normalisation z :
  - ENC_NORMALIZE=True (défaut) : encoder.forward() blanchit z DANS le graphe
    (WhiteningLayer, anti-collapse actif) ; z est passé tel quel au décodeur.
  - ENC_NORMALIZE=False : z bruts transmis directement au décodeur.

Prérequis : une vidéo (config.VIDEO_DIR). train_tsne/encoder NE sont PAS requis
            (l'encodeur peut partir de zéro), mais encoder.pt est repris s'il existe.
Lance     : py -3.10 train_ae.py [--config ../<cas>/config.py]
Produit   : <SAVE_DIR>/encoder_ae.pt
            <SAVE_DIR>/decoder2dpt_ae.pt
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
from pathlib import Path

from _bootstrap import load_config
config = load_config()
from dataset import VideoFrameDataset
from models import build_encoder
from models_2pt import build_decoder2pt
# Réutilisation directe du pruning et du plot debug du décodeur gsplat figé.
from train_decoder2dpt_gsplat import prune_and_reinit_2d, plot_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Régularisation Lipschitz de l'encodeur MLP (repris de train_encoder.py)
# ─────────────────────────────────────────────────────────────────────────────

def gradient_penalty(encoder, x_batch: torch.Tensor) -> torch.Tensor:
    """Pénalise (‖J_{x̂} enc(x̂)‖_F - 1)² sur des interpolations aléatoires."""
    B = x_batch.shape[0]
    idx1  = torch.randperm(B, device=x_batch.device)
    idx2  = torch.randperm(B, device=x_batch.device)
    alpha = torch.rand(B, 1, 1, 1, device=x_batch.device)
    x_hat = (alpha * x_batch[idx1] + (1 - alpha) * x_batch[idx2]).requires_grad_(True)

    z_hat = encoder(x_hat)
    D = z_hat.shape[1]
    jac_norm_sq = torch.zeros(B, device=x_batch.device)
    for d in range(D):
        grad = torch.autograd.grad(
            z_hat[:, d].sum(), x_hat, create_graph=True, retain_graph=True
        )[0]
        jac_norm_sq = jac_norm_sq + grad.flatten(1).pow(2).sum(dim=1)
    return (jac_norm_sq.sqrt() - 1).pow(2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Encodage complet sans gradient (smart_init, pruning, stats)
# ─────────────────────────────────────────────────────────────────────────────

def compute_z_all(encoder, enc_frames_np, device, batch_size):
    """Encode toutes les frames en eval (whitening via running stats si actif)."""
    encoder.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(enc_frames_np), batch_size):
            # frames stockées en uint8 [0,255] → float [0,1] par batch (cf. train_ae main)
            x = torch.from_numpy(enc_frames_np[i:i + batch_size]).to(device).float().div_(255.0)
            parts.append(encoder(x).cpu())
    encoder.train()
    return torch.cat(parts, dim=0)   # (N, D) CPU


def refresh_whitening_global(encoder, enc_frames_np, device, batch_size):
    """
    Repose les buffers de la WhiteningLayer sur les stats GLOBALES (tout le
    dataset) et renvoie z_all BLANCHI par cette transformée (N, D) sur CPU.
    Renvoie None si l'encodeur n'a pas de whitening.

    POURQUOI (cause des reconstructions noires à d élevé, ex. Krauss 2-seg d=4) :
    en entraînement, WhiteningLayer.forward recalcule μ, W depuis le BATCH courant
    (AE_BATCH frames) et ÉCRASE running_mean/running_W (copy_, pas d'EMA). En eval
    (plot, checkpoint, scripts aval) tout le dataset est donc blanchi par la
    transformée du DERNIER batch. Or W = eigvecs · clamp(λ, eps)^(-1/2) : dans les
    directions à petite valeur propre (latent partiellement dégénéré quand les DDL
    réels < d, fréquent à d=4), rsqrt amplifie ×milliers le bruit d'estimation
    d'une covariance sur 32 échantillons. La transformée eval s'éloigne alors des
    transformées vues à l'entraînement → z eval loin des μ_z appris → poids
    conditionnels de Schur w_z = exp(-½·d²_Mahalanobis) saturés à ~exp(-10) ≈ 0
    → rendu NOIR, alors que la loss (calculée en train, blanchiment par batch)
    descend normalement. À d=2 la cov 2×2 sur 32 échantillons est stable → pas de
    symptôme. On repose donc la transformée GLOBALE (déterministe, sur les N
    frames) avant chaque sauvegarde/plot. N'affecte PAS l'entraînement (toujours
    en blanchiment par batch dans le graphe) — uniquement l'eval.
    """
    if not getattr(encoder, 'normalize', False):
        return None
    wl = encoder.whitening
    # Force compute_z_all (eval) à renvoyer le z BRUT (branche non-initialisée).
    wl.initialized.fill_(False)
    z = compute_z_all(encoder, enc_frames_np, device, batch_size).to(device)  # brut
    mu = z.mean(0)
    M2 = z.T @ z / max(len(z), 1)                     # E[zzᵀ] global
    # set_global pose running_mean/running_M2/running_W de façon cohérente : la
    # moyenne mobile par batch reprend depuis ces stats globales (pas de réamorçage
    # instable au step suivant). _global_W reproduit la même transformée.
    wl.set_global(mu, M2)
    return ((z - mu) @ wl.running_W).detach().cpu()


def prime_whitening(encoder, enc_frames_np, device, batch_size):
    """
    Initialise les buffers de la WhiteningLayer depuis les stats GLOBALES du
    dataset, AVANT smart_init (n'agit que si pas déjà initialisée, ex. checkpoint).

    Sans ça, un encodeur neuf (pas de checkpoint) a une WhiteningLayer non
    initialisée : en eval elle renvoie z BRUT (variance minuscule) → smart_init
    fixe mu_z/sigma_z à cette échelle, alors qu'en entraînement l'encodeur
    renvoie z BLANCHI (échelle ~unité). Le mismatch fait s'effondrer les poids
    conditionnels de Schur (w_z → 0) → rendu vide → gradient nul → loss figée.
    """
    if not getattr(encoder, 'normalize', False):
        return
    if bool(encoder.whitening.initialized.item()):
        return   # déjà initialisée (ex. checkpoint chargé)
    refresh_whitening_global(encoder, enc_frames_np, device, batch_size)
    print('  WhiteningLayer initialisée sur les stats globales du dataset')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    config.SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    n_ch_enc = 3 if config.ENC_COLOR else 1
    n_ch_dec = 3 if getattr(config, 'DEC_COLOR', True) else 1

    # ── Datasets ───────────────────────────────────────────────────────────────
    print('Chargement des frames…')
    # store_uint8=True : self.frames en uint8 [0,255] (~4× moins de RAM qu'en float32 —
    # indispensable en 256×256 où le dataset RGB float32 ≈ 41 GB faisait crasher la machine).
    # La conversion en float [0,1] se fait PAR BATCH ci-dessous (boucle + compute_z_all).
    enc_dataset = VideoFrameDataset(
        video_dir           = config.VIDEO_DIR,
        img_size            = config.IMG_SIZE,
        n_channels          = n_ch_enc,
        crop                = getattr(config, 'CROP', None),
        rest_video          = config.REST_VIDEO,
        rest_n_frames       = config.REST_N_FRAMES,
        rest_first_n_frames = getattr(config, 'REST_FIRST_N_FRAMES', 0),
        exclude_videos      = [config.VAL_VIDEO] if config.VAL_VIDEO else None,
        store_uint8         = True,
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

    enc_frames_np = enc_dataset.frames   # (N, C_enc, H, W) uint8 [0,255]
    frames_t      = torch.from_numpy(dec_dataset.frames)   # (N, C_dec, H, W) uint8 CPU (partagé)

    # ── Encodeur (entraînable) ─────────────────────────────────────────────────
    _enc_normalize = getattr(config, 'ENC_NORMALIZE', False)
    encoder = build_encoder(
        img_size    = config.IMG_SIZE,
        hidden_dims = config.ENC_HIDDEN,
        latent_dim  = config.LATENT_DIM,
        n_channels  = n_ch_enc,
        normalize   = _enc_normalize,
    ).to(device)
    _resumed_enc = False
    for cand in [config.SAVE_DIR / 'encoder_ae.pt', config.SAVE_DIR / 'encoder.pt']:
        if cand.exists():
            try:
                encoder.load_state_dict(torch.load(cand, map_location=device))
                print(f'Encodeur repris : {cand}')
                _resumed_enc = True
                break
            except RuntimeError as e:
                # Checkpoint incompatible (ex. canaux gris→couleur, hidden dims, latent…) :
                # on n'interrompt pas, on repart d'un encodeur neuf. Typique après un
                # changement de config (ENC_COLOR, IMG_SIZE, LATENT_DIM…).
                print(f'Encodeur : checkpoint "{cand.name}" incompatible — ignoré '
                      f'(init aléatoire). Détail : {str(e).splitlines()[0]}')
    if not _resumed_enc:
        print('Encodeur : initialisation aléatoire (aucun checkpoint compatible)')
    encoder.train()
    print(f'Encodeur : {sum(p.numel() for p in encoder.parameters()):,} paramètres '
          f'({"CpAE/CNN" if getattr(config, "ENC_CPAE", False) else "MLP"}, '
          f'normalize={_enc_normalize})')

    # ── z_all initial (pour smart_init) ────────────────────────────────────────
    # Primer le whitening AVANT d'encoder : garantit que z_all (et donc mu_z du
    # décodeur) est à la même échelle blanchie que le z vu en entraînement.
    prime_whitening(encoder, enc_frames_np, device, config.AE_BATCH)
    z_all = compute_z_all(encoder, enc_frames_np, device, config.AE_BATCH)
    print(f'  z_all init : {z_all.shape}  '
          f'mean={z_all.mean(0).numpy().round(3)}  std={z_all.std(0).numpy().round(3)}')

    # ── Décodeur 2D+t gsplat ───────────────────────────────────────────────────
    n_gaussians = getattr(config, 'AE_N_GAUSSIANS',
                          getattr(config, 'DEC2PT_GSPLAT_N_GAUSSIANS',
                                  getattr(config, 'DEC_N_GAUSSIANS', 2048)))
    decoder = build_decoder2pt(
        latent_dim  = config.LATENT_DIM,
        n_gaussians = n_gaussians,
        img_size    = config.IMG_SIZE,
        n_channels  = n_ch_dec,
    ).to(device)

    # uint8 [0,255] → float [0,1] (1 frame) pour smart_init.
    ref_frame = torch.from_numpy(dec_dataset.frames[0]).float().div_(255.0)   # (C_dec, H, W)
    # Priorité au décodeur de l'autoencodeur (decoder2dpt_ae.pt) s'il existe,
    # sinon repli sur le décodeur séparé (decoder2dpt.pt). Repris seulement si
    # le nombre de gaussiennes correspond.
    # Le décodeur de l'AE est TOUJOURS sauvegardé sous decoder2dpt_ae.pt (sortie
    # documentée, chargée en priorité par precompute_metric_geom). On ne dérive plus
    # le chemin de save de la variable de boucle (qui, sans reprise, pointait par
    # erreur sur decoder2dpt.pt et écrasait le « décodeur seul »).
    ckpt_path = config.SAVE_DIR / 'decoder2dpt_ae.pt'
    _resumed = False
    for _cand in [config.SAVE_DIR / 'decoder2dpt_ae.pt',
                  config.SAVE_DIR / 'decoder2dpt.pt']:
        if _cand.exists() and \
           torch.load(_cand, map_location='cpu')['mu_raw'].shape[0] == n_gaussians:
            decoder.load_state_dict(torch.load(_cand, map_location=device))
            print(f'  Décodeur repris : {_cand}')
            _resumed = True
            break
    if not _resumed:
        decoder.smart_init(
            ref_image      = ref_frame,
            z_all          = z_all,
            sigma_xy_scale = getattr(config, 'DEC2PT_SIGMA_XY', 0.05),
            sigma_z_scale  = getattr(config, 'DEC2PT_SIGMA_Z',  1.0),
            sigma_z_floor  = getattr(config, 'DEC2PT_SIGMA_Z_FLOOR', 0.5),
        )
        print('  Décodeur : smart_init depuis frame 0 + z_all')
    decoder.train()
    print(f'Décodeur 2D+t gsplat : {n_gaussians} gaussiennes, '
          f'{sum(p.numel() for p in decoder.parameters()):,} paramètres')

    # ── Optimiseur conjoint (deux groupes : encodeur + décodeur) ───────────────
    enc_lr = getattr(config, 'AE_ENC_LR', config.ENC_LR)
    dec_lr = getattr(config, 'AE_DEC_LR', config.DEC_LR)
    optimizer = torch.optim.Adam([
        {'params': list(encoder.parameters()), 'lr': enc_lr},
        {'params': list(decoder.parameters()), 'lr': dec_lr},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor   = getattr(config, 'AE_LR_FACTOR',   config.DEC_LR_FACTOR),
        patience = getattr(config, 'AE_LR_PATIENCE', config.DEC_LR_PATIENCE),
        min_lr   = getattr(config, 'AE_LR_MIN',      config.DEC_LR_MIN),
    )

    # ── Hyperparamètres ────────────────────────────────────────────────────────
    n_epochs    = getattr(config, 'AE_EPOCHS', config.DEC_EPOCHS)
    batch       = getattr(config, 'AE_BATCH',  config.DEC_BATCH)
    l1_w        = config.DEC_L1_W
    ssim_w      = config.DEC_SSIM_W
    aniso_w     = config.DEC_ANISO_W
    plot_every  = getattr(config, 'AE_PLOT_EVERY',  config.DEC_PLOT_EVERY)
    print_every = getattr(config, 'AE_PRINT_EVERY', 50)

    # Poids de la pression sur les opacités (0 = désactivé, comportement historique).
    # Pendant du DEC3PT_ALPHA_W de train_decoder3pt.py ; sans lui le prune/réinit ne
    # se déclenche jamais, faute de force qui pousse les α inutiles sous le seuil.
    alpha_w      = getattr(config, 'AE_ALPHA_W', 0.0)
    alpha_thresh = getattr(config, 'AE_ALPHA_THRESH', config.DEC2PT_ALPHA_THRESH)
    prune_every  = getattr(config, 'AE_PRUNE_EVERY',  config.DEC2PT_PRUNE_EVERY)
    prune_warmup = getattr(config, 'AE_PRUNE_WARMUP', config.DEC2PT_PRUNE_WARMUP)
    sigma_xy     = getattr(config, 'DEC2PT_SIGMA_XY', 0.05)

    # Régularisation encodeur, mise à l'échelle par AE_ENC_REG_W. En AE pur, la
    # supervision est faible (L1×0.08 + SSIM×0.02) ; appliquer la pénalité
    # nonlocale à λ_J=1.0 la rend dominante et aplatit les filtres CpAE en une
    # époque (encodeur effondré). Comme la phase warmup de train_all, on la coupe
    # par défaut (AE_ENC_REG_W=0) ; l'anti-collapse est assuré par le whitening.
    _reg_scale = getattr(config, 'AE_ENC_REG_W', 0.0)
    _use_cpae = getattr(config, 'ENC_CPAE', False)
    _lambda_j = getattr(config, 'ENC_CPAE_LAMBDA_J', 1.0) * _reg_scale
    _gp_lambda = getattr(config, 'ENC_GP_LAMBDA', 0) * _reg_scale
    if _use_cpae and _lambda_j > 0:
        print(f'  Régul. encodeur : nonlocal_penalty (CpAE)  poids={_lambda_j:g}')
    elif _gp_lambda > 0:
        print(f'  Régul. encodeur : gradient penalty (MLP)  poids={_gp_lambda:g}')
    else:
        print('  Régul. encodeur : aucune (AE_ENC_REG_W=0 ; anti-collapse via whitening)')
    print(f'  Loss : L1×{l1_w} + SSIM×{ssim_w} + Aniso×{aniso_w}'
          + (f' + Alpha×{alpha_w}' if alpha_w > 0 else ''))
    print(f'  Pruning : alpha_thresh={alpha_thresh}, every={prune_every}, '
          f'warmup={prune_warmup}')
    print(f'  LR : enc={enc_lr}  dec={dec_lr}')

    # ── Samples fixes pour debug ───────────────────────────────────────────────
    n_show  = min(8, N)
    dbg_idx = np.linspace(0, N - 1, n_show, dtype=int)
    # uint8 [0,255] → float [0,1] (8 frames) → plot_epoch reçoit du float, inchangé.
    dbg_rgb = torch.from_numpy(dec_dataset.frames[dbg_idx]).float().div_(255.0)

    losses, losses_l1, losses_ssim, losses_aniso, losses_reg = [], [], [], [], []

    print(f'\nEntraînement : {n_epochs} epochs, {N} frames')
    for epoch in range(n_epochs):
        decoder.train(); encoder.train()
        perm = torch.randperm(N)
        ep_loss = ep_l1 = ep_ssim = ep_aniso = ep_reg = 0.0

        for start in range(0, N, batch):
            idx      = perm[start:start + batch]
            # frames uint8 [0,255] (store_uint8) → float [0,1] par batch sur le GPU.
            x_enc    = torch.from_numpy(enc_frames_np[idx.numpy()]).to(device).float().div_(255.0)
            x_target = frames_t[idx].to(device).float().div_(255.0)

            # Encode SANS detach → le gradient recon remonte dans l'encodeur.
            # encoder.forward() blanchit z dans le graphe si ENC_NORMALIZE.
            z       = encoder(x_enc)
            x_recon = decoder(z)

            loss_l1   = F.l1_loss(x_recon, x_target) if l1_w   > 0 else x_recon.new_zeros(1)
            loss_ssim = (1.0 - ssim_fn(x_recon, x_target,
                                       data_range=1.0, size_average=True)) \
                        if ssim_w > 0 else x_recon.new_zeros(1)

            if aniso_w > 0:
                Sigma_xx = decoder._build_Sigma()[:, :2, :2]
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

            # Régularisation encodeur (cohérente avec train_encoder.py)
            if _use_cpae and _lambda_j > 0:
                loss_reg = encoder.nonlocal_penalty(); w_reg = _lambda_j
            elif _gp_lambda > 0:
                loss_reg = gradient_penalty(encoder, x_enc); w_reg = _gp_lambda
            else:
                loss_reg = x_recon.new_zeros(1); w_reg = 0.0

            # Pression douce vers le bas sur les opacités (cf. DEC3PT_ALPHA_W dans
            # train_decoder3pt.py) : pénalise la MOYENNE des α = sigmoid(log_alpha).
            # Une gaussienne réellement utile voit sa L1/SSIM compenser largement ce
            # coût et garde son opacité ; une gaussienne inutile n'a rien pour s'y
            # opposer et descend jusqu'à passer sous AE_ALPHA_THRESH, où le
            # prune/réinit la recycle vers une zone de forte erreur. Sans ce terme
            # rien ne pousse α vers le bas : sur `dp`, aucune des 2048 gaussiennes ne
            # descendait sous 0.25, donc le pruning ne se déclenchait jamais.
            # Poids à garder FAIBLE (1e-3) : c'est un départage entre primitives
            # équivalentes, pas un objectif. 0 (défaut) ⟹ loss inchangée bit à bit.
            loss_alpha = (torch.sigmoid(decoder.log_alpha).mean() if alpha_w > 0
                          else x_recon.new_zeros(1))

            loss = (l1_w * loss_l1 + ssim_w * loss_ssim +
                    aniso_w * loss_aniso + w_reg * loss_reg +
                    alpha_w * loss_alpha)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            B = len(idx)
            ep_loss  += loss.item()       * B
            ep_l1    += loss_l1.item()    * B
            ep_ssim  += loss_ssim.item()  * B
            ep_aniso += loss_aniso.item() * B
            ep_reg   += loss_reg.item()   * B

        ep_loss /= N; ep_l1 /= N; ep_ssim /= N; ep_aniso /= N; ep_reg /= N
        losses.append(ep_loss); losses_l1.append(ep_l1)
        losses_ssim.append(ep_ssim); losses_aniso.append(ep_aniso)
        losses_reg.append(ep_reg)
        scheduler.step(ep_loss)

        # ── Whitening : repose la transformée GLOBALE (stable) avant
        #    pruning/sauvegarde/plot. Sinon running_W = stats du DERNIER batch
        #    (bruité à d élevé → z eval loin des μ_z → rendu noir, cf.
        #    refresh_whitening_global). z_all_white est réutilisé ci-dessous
        #    (1 seul encodage complet/époque). None si pas de whitening.
        z_all_white = refresh_whitening_global(encoder, enc_frames_np, device, batch)

        # ── Pruning / réinit (z_all recalculé depuis l'encodeur courant) ───────
        if (prune_every > 0 and epoch >= prune_warmup
                and (epoch + 1) % prune_every == 0):
            z_all = z_all_white if z_all_white is not None \
                    else compute_z_all(encoder, enc_frames_np, device, batch)
            prune_and_reinit_2d(
                decoder, optimizer, frames_t, z_all, device,
                sigma_xy     = sigma_xy,
                alpha_thresh = alpha_thresh,
                epoch        = epoch,
            )

        # ── Sauvegarde ─────────────────────────────────────────────────────────
        nan_enc = any(p.isnan().any().item() for p in encoder.parameters())
        nan_dec = any(p.isnan().any().item() for p in decoder.parameters())
        if not (nan_enc or nan_dec):
            torch.save(encoder.state_dict(), config.SAVE_DIR / 'encoder_ae.pt')
            torch.save(decoder.state_dict(), ckpt_path)
        else:
            print(f'  [epoch {epoch+1}] NaN détecté — sauvegarde ignorée')

        if (epoch + 1) % print_every == 0:
            lr_dec = optimizer.param_groups[1]['lr']
            print(f'  [{epoch+1:5d}/{n_epochs}]  total={ep_loss:.5f}  '
                  f'L1={ep_l1:.5f}  SSIM={ep_ssim:.5f}  Aniso={ep_aniso:.5f}  '
                  f'reg={ep_reg:.5f}  lr_dec={lr_dec:.2e}')

        if plot_every > 0 and (epoch + 1) % plot_every == 0:
            dbg_z = (z_all_white[dbg_idx] if z_all_white is not None
                     else compute_z_all(encoder, enc_frames_np, device, batch)[dbg_idx])
            plot_epoch(decoder, dbg_z, dbg_rgb, device, epoch + 1, config.SAVE_DIR)

    # ── Courbe de loss ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(losses,       label='total')
    ax.semilogy(losses_l1,    label='L1',   linestyle='--')
    ax.semilogy(losses_ssim,  label='SSIM', linestyle='-.')
    if aniso_w > 0:
        ax.semilogy(losses_aniso, label='Aniso', linestyle=':')
    if (_use_cpae and _lambda_j > 0) or _gp_lambda > 0:
        ax.semilogy(losses_reg, label='reg enc', linestyle=(0, (3, 1, 1, 1)))
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Autoencodeur conjoint (encodeur + décodeur 2D+t gsplat)')
    ax.legend(); ax.grid(True); fig.tight_layout()
    fig.savefig(config.SAVE_DIR / 'ae_loss.png', dpi=150)
    plt.close(fig)

    z_all_white = refresh_whitening_global(encoder, enc_frames_np, device, batch)
    dbg_z = (z_all_white[dbg_idx] if z_all_white is not None
             else compute_z_all(encoder, enc_frames_np, device, batch)[dbg_idx])
    plot_epoch(decoder, dbg_z, dbg_rgb, device, n_epochs, config.SAVE_DIR)

    print(f'\nEncodeur sauvegardé : {config.SAVE_DIR / "encoder_ae.pt"}')
    print(f'Décodeur sauvegardé : {ckpt_path}')
    print(decoder.get_params_summary())


if __name__ == '__main__':
    # --config est consommé en tête de module par _bootstrap.load_config (avant
    # l'import de models.py), donc config est déjà le bon module ici.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='Config propre au cas test (cf. _bootstrap.load_config)')
    parser.parse_args()
    main()
