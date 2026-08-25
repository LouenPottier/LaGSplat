"""
train_lnn_krauss.py — entraîne le LNN (AUTOENCODEUR GELÉ) avec la LOSS DE KRAUSS
et al. 2026 (VON, RA-L) : rollout 1-pas, cible de POSITION DÉCODÉE (image rendue
par le décodeur GS gsplat) + cohérence dynamique latente (position et vitesse).

Motivation
----------
`train_lnn_fixedae.py` minimise le résidu Euler-Lagrange en différences finies
(teacher-forcing single-step, cible = q̈ le long de la trajectoire vraie lissée).
Krauss, lui, entraîne sa dynamique sur une loss de RECONSTRUCTION DÉCODÉE à un pas :
Éq. (5), termes dynamiques (AE figé ⟹ les termes statiques / KL ne concernent pas
le LNN) :
  - reconstruction dynamique : MSE(φ⁻¹(ẑ_{i+1}), o_{i+1})   ← DÉCODAGE gsplat de l'état
      prédit à un pas, comparé à la VRAIE frame suivante (position décodée) ;
  - cohérence latente : MSE(ẑ_{i+1}, z_{i+1}) + MSE(Δt·ż̂_{i+1}, Δt·ż_{i+1})
      (position et vitesse latentes prédites vs encodées).

On reproduit ces termes ici : depuis (z_i, ż_i) teacher-forcé (latent encodé, figé),
UN pas de velocity-Verlet différentiable (mêmes forces/signe que
`viz.simulate_rk4` et le résidu FD, via `finetune_lnn_fixedae._verlet_step`) donne
(ẑ_{i+1}, ż̂_{i+1}) ; on DÉCODE ẑ_{i+1} par le décodeur GS gsplat (whiten⁻¹ →
GaussianSplatDecoder2pt_gsplat, décodeur GELÉ) et on compare à la frame o_{i+1}.

Transparence
------------
AE (encodeur + décodeur + blanchiment latent) 100 % GELÉ : seul le LNN est
entraîné. On N'ÉCRASE PAS `lnn.pt` / `lnn_rollout.pt` : sauvegarde DÉDIÉE
`lnn_krauss.pt` + plots `debug_plots_lnn_krauss/`. Clés `LNN_KRAUSS_*` lues
uniquement ici (aucun impact sur les autres entraînements).

Lancer :  py train_lnn_krauss.py --config ../cases/krauss2026_2seg_npz/config.py
Prérequis : encoder_ae.pt + decoder2dpt_ae.pt + latent_whiten.pt dans SAVE_DIR
            (AE déjà entraîné).
"""
import argparse
import time

from _bootstrap import load_config
config = load_config()

# ── Knobs spécifiques Krauss (CLI, priorité sur config.LNN_KRAUSS_*) ─────────
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--epochs', type=int, default=None, help='override config.LNN_EPOCHS')
_ap.add_argument('--dec-res', type=int, default=None,
                 help='résolution de rendu gsplat pour la loss (déf. 64)')
_ap.add_argument('--windows-per-epoch', type=int, default=None,
                 help='nb de fenêtres tirées par époque (déf. 3000 ; <=0 = toutes)')
_ap.add_argument('--batch', type=int, default=None, help='taille de batch (déf. 32)')
_ap.add_argument('--w-dec', type=float, default=None,
                 help='poids position décodée = lam_o de Krauss (déf. 10.0)')
_ap.add_argument('--w-z', type=float, default=None,
                 help='poids position latente = lam_z de Krauss (déf. 0.1)')
_ap.add_argument('--w-v', type=float, default=None,
                 help='poids vitesse latente = lam_z de Krauss (déf. 0.1)')
_ap.add_argument('--lr', type=float, default=None,
                 help='LR (déf. 1e-3 = référence Krauss ; PAS config.LNN_LR=1e-2, instable)')
_ap.add_argument('--init-lnn', type=str, default=None,
                 help='checkpoint LNN de départ dans SAVE_DIR (ex. lnn.pt) : warm-start / '
                      'fine-tuning au lieu d\'un LNN neuf (déf. None = neuf)')
_ap.add_argument('--metric-latent', action='store_true',
                 help='pondère les termes latents (z, v) par la métrique de VISIBILITÉ Ḡ '
                      '(visibility_metric.pt) au lieu d\'une MSE isotrope : substitut au '
                      'premier ordre du terme décodé de Krauss, SANS le bruit du plancher AE '
                      '(Δuᵀ Ḡ Δu ≈ ‖ΔI‖²). Recommandé avec --w-dec 0.')
_ap.add_argument('--plot-every', type=int, default=None,
                 help='plot debug tous les N époques (déf. 25 ; final toujours tracé)')
_ap.add_argument('--seq-stride', type=int, default=None,
                 help='override config.SEQ_STRIDE : espacement (en frames) des échantillons '
                      'de la fenêtre. 2 = décimation ×2 ≈ 60 fps (dt de Krauss). L\'ODE reste '
                      'PAR FRAME (v en pos/frame, 1 pas Verlet de h=stride frames) ⟹ les poids '
                      'de lnn.pt gardent leur sens, warm-start direct.')
_ap.add_argument('--out', type=str, default=None,
                 help='nom du checkpoint de sortie dans SAVE_DIR (déf. lnn_krauss.pt). '
                      'Les artefacts dérivés (courbe de perte, plots debug) en sont '
                      'nommés ⟹ deux runs de réglages différents ne s\'écrasent pas.')
_ap.add_argument('--sigma', type=float, default=None,
                 help='override config.SMOOTH_LATENT_SIGMA (lissage gaussien de q(t), en '
                      'FRAMES). Le checkpoint produit devra être ÉVALUÉ au même σ '
                      '(eval_multistep_mse --sigma), sinon les conditions initiales des '
                      "fenêtres ne sont pas celles vues à l'entraînement.")
_ap.add_argument('--sigma-pressure', type=float, default=None,
                 help='override config.SMOOTH_PRESSURE_SIGMA (lissage gaussien de P(t), en '
                      'FRAMES). INDÉPENDANT de --sigma (le repli « pression = σ de q » ne '
                      'joue que si SMOOTH_PRESSURE_SIGMA vaut None dans la config).')
_ap.add_argument('--cq-eps', type=float, default=None,
                 help="override config.LNN_RAYLEIGH_CQ_EPS : plancher d’amortissement "
                      "ISOTROPE de C(q)=L(q)L(q)ᵀ+εI. Ce n’est PAS qu’un garde-fou SPD : "
                      "trop bas, la plus petite valeur propre de C vaut exactement ε sur "
                      "toutes les données (dissipation de rang d−1, une direction non "
                      "amortie, rollout long instable) ; trop haut, l’isotrope remplace "
                      "la dissipation apprise.")
_ap.add_argument('--metric-ridge', type=float, default=None,
                 help='override config.LNN_ROLLOUT_METRIC_RIDGE : ρ de la métrique de '
                      'visibilité, en fraction de λmax (A ∝ Ḡ + ρ·λmax·I). Borne le rapport '
                      'de pondération entre modes à ~1/ρ, donc ρ GRAND = repondération plus '
                      'DOUCE (les modes très visibles écrasent moins les autres). Défaut '
                      '0.01 ⟹ rapport ~87 ; ρ=0.36 est la valeur transposée des λ officiels '
                      'de Krauss (lam_o=10 / lam_z=0.1) ⟹ rapport ~3.8. Sans effet sans '
                      '--metric-latent.')
_ap.add_argument('--seed', type=int, default=0)
_ap.add_argument('--smoke', action='store_true',
                 help='smoke test : 2 époques, 128 fenêtres/époque (mesure de timing)')
_args, _ = _ap.parse_known_args()

if _args.sigma is not None:
    config.SMOOTH_LATENT_SIGMA = float(_args.sigma)
if _args.sigma_pressure is not None:
    config.SMOOTH_PRESSURE_SIGMA = float(_args.sigma_pressure)
if _args.cq_eps is not None:
    config.LNN_RAYLEIGH_CQ_EPS = float(_args.cq_eps)
if _args.metric_ridge is not None:
    config.LNN_ROLLOUT_METRIC_RIDGE = float(_args.metric_ridge)

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import VideoFrameDataset
from models import LNN, build_encoder, WhitenedEncoder, load_latent_whiten
from models_2pt import build_decoder2pt
from viz import simulate_rk4, initial_velocity, get_sim_pressure
from train_lnn_fixedae import (encode_all, smooth_latents, smooth_pressures,
                               plot_latent_rollout)
from finetune_lnn_fixedae import _verlet_step, load_rollout_metric, rollout_loss


def _cfg(name, default):
    return getattr(config, name, default)


def build_step_windows(z_all, video_lengths, pressures=None, stride=1):
    """Fenêtres de 4 échantillons espacés de `stride` frames (indices s, s+k, s+2k,
    s+3k), NE chevauchant PAS deux vidéos. Retourne :
      Z      : (Nw, 4, d)  échantillons latents encodés
      P      : (Nw, 4, n_c) pression ZOH aux mêmes échantillons (ou None)
      starts : (Nw,) index de frame du 1ᵉʳ échantillon (s) dans z_all/frames

    Le pas Verlet part de l'échantillon 1 (s+k, v0 = diff. centrée sur s,s+2k) et
    prédit l'échantillon 2 (s+2k) ; la cible image décodée est la frame s+2k.
    """
    k = max(int(stride), 1)
    span = 3 * k
    starts, off = [], 0
    for vlen in video_lengths:
        if vlen > span:
            starts += [off + s for s in range(vlen - span)]
        off += vlen
    assert starts, f'Aucune vidéo > 3·stride={span} frames. Longueurs={list(video_lengths)}'
    starts_arr = np.asarray(starts, dtype=np.int64)
    Z = torch.stack([z_all[s:s + span + 1:k] for s in starts])          # (Nw,4,d)
    P = None
    if pressures is not None:
        P = torch.stack([torch.from_numpy(pressures[s:s + span + 1:k]) for s in starts])
    return Z, P, starts_arr


def main():
    config.SAVE_DIR.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(_args.seed)
    np.random.seed(_args.seed)
    print(f'Device : {device}')
    assert _cfg('LNN_FREEZE_ENCODER', False), \
        'train_lnn_krauss suppose LNN_FREEZE_ENCODER=True (AE figé).'
    _use_pressure = _cfg('LNN_PRESSURE', False)

    # ── Hyperparamètres (CLI > config.LNN_KRAUSS_* > défaut) ─────────────────
    epochs   = _args.epochs if _args.epochs is not None else int(_cfg('LNN_EPOCHS', 500))
    dec_res  = _args.dec_res if _args.dec_res is not None else int(_cfg('LNN_KRAUSS_DEC_RES', 64))
    wpe      = (_args.windows_per_epoch if _args.windows_per_epoch is not None
               else int(_cfg('LNN_KRAUSS_WINDOWS_PER_EPOCH', 3000)))
    bs       = _args.batch if _args.batch is not None else int(_cfg('LNN_KRAUSS_BATCH', 32))
    # Pondérations alignées sur le VRAI article Krauss (VON, RA-L 2026), configs/base.yaml :
    #   lam_o = 10.0  (reconstruction DÉCODÉE au pas suivant)  → w_dec
    #   lam_z = 0.1   (cohérence latente position + vitesse)   → w_z ET w_v
    # (rapport décodé:latent = 100:1). L'ancien 1:1:1 sous-pondérait le terme décodé ~200×.
    w_dec    = _args.w_dec if _args.w_dec is not None else float(_cfg('LNN_KRAUSS_W_DEC', 10.0))
    w_z      = _args.w_z if _args.w_z is not None else float(_cfg('LNN_KRAUSS_W_Z', 0.1))
    w_v      = _args.w_v if _args.w_v is not None else float(_cfg('LNN_KRAUSS_W_V', 0.1))
    # LR par défaut = 1e-3 (référence Krauss), PAS config.LNN_LR (1e-2) : ce dernier rend le
    # fine-tuning instable sur ce warm-start (constaté).
    lr       = _args.lr if _args.lr is not None else float(_cfg('LNN_KRAUSS_LR', 1e-3))
    plot_every = (_args.plot_every if _args.plot_every is not None
                  else int(_cfg('LNN_KRAUSS_PLOT_EVERY', 25)))
    if _args.smoke:
        epochs, wpe, plot_every = 2, 128, 1
    # Métrique de visibilité Ḡ pour les termes latents (opt-in --metric-latent) :
    # réutilise load_rollout_metric (ridge + normalisation trace(A)=d). A=None sinon.
    metric_A = None
    if _args.metric_latent:
        config.LNN_ROLLOUT_METRIC = True                    # active le chargement
        metric_A = load_rollout_metric(device)
        assert metric_A is not None, 'visibility_metric.pt introuvable (lancer ' \
                                     'compute_visibility_metric.py --source data d\'abord).'
    n_ch     = 3 if _cfg('ENC_COLOR', True) else 1
    n_ch_dec = 3 if _cfg('DEC_COLOR', True) else 1
    assert n_ch == n_ch_dec, ('train_lnn_krauss suppose ENC_COLOR==DEC_COLOR '
                              '(frames encodeur = cibles décodeur).')

    # ── Données (frames + frontières vidéos + pressions) ─────────────────────
    fds = VideoFrameDataset(
        video_dir=config.VIDEO_DIR, img_size=config.IMG_SIZE, n_channels=n_ch,
        rest_video=config.REST_VIDEO, rest_n_frames=config.REST_N_FRAMES,
        rest_first_n_frames=_cfg('REST_FIRST_N_FRAMES', 0),
        crop=_cfg('CROP', None),
        exclude_videos=[config.VAL_VIDEO] if config.VAL_VIDEO else None,
        load_pressure=_use_pressure, pressure_dir=_cfg('PRESSURE_DIR', None),
        pressure_cols=_cfg('PRESSURE_COLS', None),
        pressure_norm=_cfg('PRESSURE_NORM', 101325.0), pressure_dt=config.DT,
        pressure_sync_offsets=_cfg('PRESSURE_SYNC_OFFSETS', None),
        store_uint8=True)

    # ── Encodeur gelé (+ blanchiment latent figé) ────────────────────────────
    enc = build_encoder(config.IMG_SIZE, config.ENC_HIDDEN, config.LATENT_DIM,
                        n_channels=n_ch, normalize=_cfg('ENC_NORMALIZE', False)).to(device)
    enc_path = config.SAVE_DIR / 'encoder_ae.pt'
    if not enc_path.exists():
        enc_path = config.SAVE_DIR / 'encoder.pt'
    assert enc_path.exists(), f'encodeur introuvable dans {config.SAVE_DIR}'
    enc.load_state_dict(torch.load(enc_path, map_location=device))
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    print(f'Encodeur gelé : {enc_path.name}')

    whiten = load_latent_whiten(config.SAVE_DIR, device, config.LATENT_DIM)
    if whiten is not None:
        enc = WhitenedEncoder(enc, whiten).to(device).eval()
        print('Encodeur enveloppé : sortie en espace latent blanchi (LatentWhiten).')

    # ── Décodeur GS gsplat GELÉ, rendu à dec_res (params indép. de la résolution) ─
    dec_path = config.SAVE_DIR / 'decoder2dpt_ae.pt'
    if not dec_path.exists():
        dec_path = config.SAVE_DIR / 'decoder2dpt.pt'
    assert dec_path.exists(), f'décodeur introuvable dans {config.SAVE_DIR}'
    dec_state = torch.load(dec_path, map_location=device)
    n_gaussians = dec_state['mu_raw'].shape[0]
    # Construit à la résolution native (le buffer `grid` du state_dict en dépend, mais
    # il n'est PAS utilisé par le rendu gsplat, qui dérive la caméra de self.img_size)
    # puis on rabaisse img_size à dec_res : les paramètres GS sont en coords normalisées
    # [0,1] ⟹ indépendants de la résolution, seul K_cam/width/height changent.
    decoder = build_decoder2pt(
        latent_dim=config.LATENT_DIM, n_gaussians=n_gaussians,
        img_size=config.IMG_SIZE, n_channels=n_ch_dec).to(device)
    decoder.load_state_dict(dec_state)
    decoder.img_size = (dec_res, dec_res)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    print(f'Décodeur GS gsplat GELÉ : {dec_path.name} ({n_gaussians} gaussiennes), '
          f'rendu {dec_res}×{dec_res}')

    # ── Précalcul de tous les q + lissages (identiques à train_lnn_fixedae) ───
    print('Encodage de toutes les frames (une fois)…')
    z_all = encode_all(enc, fds.frames, device)                        # (N,d) CPU
    if _cfg('SMOOTH_LATENT', False):
        z_all = smooth_latents(
            z_all, fds.video_lengths,
            _cfg('SMOOTH_LATENT_WINDOW', 13), _cfg('SMOOTH_LATENT_POLY', 3),
            _cfg('SMOOTH_LATENT_MODE', 'savgol'), _cfg('SMOOTH_LATENT_SIGMA', 10.0))
        print(f'Lissage latent ({_cfg("SMOOTH_LATENT_MODE", "savgol")}, '
              f'σ={_cfg("SMOOTH_LATENT_SIGMA", None)} frames).')
    if _use_pressure and _cfg('SMOOTH_PRESSURE', False):
        fds.pressures = smooth_pressures(fds.pressures, fds.video_lengths)
        print(f'Pression lissée (σ={_cfg("SMOOTH_PRESSURE_SIGMA", None)} frames).')

    _seq_stride = (_args.seq_stride if _args.seq_stride is not None
                   else int(_cfg('SEQ_STRIDE', 1)))
    Z, P, starts = build_step_windows(z_all, fds.video_lengths,
                                      fds.pressures if _use_pressure else None,
                                      stride=_seq_stride)
    Z = Z.to(device)
    P = P.to(device) if P is not None else None
    starts_t = torch.from_numpy(starts).to(device)
    k = max(_seq_stride, 1)
    tgt_frame_idx = starts + 2 * k          # frame décodée cible (échantillon 2), numpy
    N = Z.shape[0]
    print(f'  {len(fds)} frames → {N} fenêtres (4 échantillons, d={config.LATENT_DIM}'
          + (f', stride={_seq_stride}' if _seq_stride > 1 else '')
          + (f', pression {P.shape[2]} chambre(s)' if P is not None else '') + ')')

    # frames CPU (uint8) : gather par batch pour la cible décodée.
    frames_np = fds.frames                                              # (N,C,H,W) uint8

    # ── LNN : neuf, ou WARM-START depuis un checkpoint (--init-lnn) ───────────
    lnn = LNN(config.LATENT_DIM, config.LNN_HIDDEN).to(device)
    mode = _cfg('Z_REST_MODE', 'barycenter')
    if mode == 'rest_frame':
        rf = torch.from_numpy(fds.rest_frame).unsqueeze(0).to(device)
        with torch.no_grad():
            zr = enc(rf).squeeze(0)
        lnn.energy.set_z_rest(zr, learnable=False)
        print(f'z_rest depuis rest_frame : {zr.cpu().numpy().round(3)}')
    else:
        lnn.energy.set_z_rest(torch.zeros(config.LATENT_DIM, device=device), learnable=True)
        print('z_rest = 0 (barycentre, apprenable)')

    if _args.init_lnn is not None:
        init_path = config.SAVE_DIR / _args.init_lnn
        assert init_path.exists(), f'checkpoint de départ introuvable : {init_path}'
        state = torch.load(init_path, map_location=device)
        missing, unexpected = lnn.load_state_dict(state, strict=False)
        print(f'WARM-START depuis {init_path.name} (fine-tuning)'
              + (f' — {len(missing)} clés manquantes' if missing else '')
              + (f', {len(unexpected)} clés en trop' if unexpected else ''))
        if missing:
            print(f'    manquantes : {missing}')
        if unexpected:
            print(f'    en trop    : {unexpected}')
    else:
        print('LNN NEUF (pas de warm-start).')

    out_path = config.SAVE_DIR / (_args.out or 'lnn_krauss.pt')
    opt = torch.optim.Adam(lnn.parameters(), lr=lr)
    print(f'LNN : {sum(p.numel() for p in lnn.parameters() if p.requires_grad):,} '
          f'paramètres, LR {lr:g}, {epochs} époques, batch {bs}, '
          f'{wpe if wpe > 0 else N} fenêtres/époque')
    _lat = 'Ḡ-métrique(ẑ₁,z₁)' if metric_A is not None else 'MSE(ẑ₁,z₁)'
    _latv = 'Ḡ-métrique(h·ż̂₁,h·ż₁)' if metric_A is not None else 'MSE(h·ż̂₁,h·ż₁)'
    print(f'Loss = {w_dec:g}·MSE(decode(ẑ₁), o₁) + {w_z:g}·{_lat} + {w_v:g}·{_latv}')
    print(f'Sortie : {out_path.name} (lnn.pt / lnn_rollout.pt CONSERVÉS intacts)')

    # ── Debug plot (rollout d'inférence latent, comme train_lnn_fixedae) ─────
    debug_dir = config.SAVE_DIR / f'debug_plots_{out_path.stem}'
    debug_dir.mkdir(exist_ok=True)
    splits = np.cumsum([0] + list(fds.video_lengths))
    s0, e0 = int(splits[0]), int(splits[1])

    def _plot_intermediate(ep):
        lnn.eval()
        _max = _cfg('VIZ_MAX_FRAMES', None)
        n = e0 - s0
        z_e = z_all[s0:e0].to(device)
        if _max is not None and n > _max:
            n, z_e = _max, z_e[:_max]
        v0 = initial_velocity(z_e)
        p_s = get_sim_pressure(lnn, fds, s0, n, device) if _use_pressure else None
        with torch.no_grad():
            z_s = simulate_rk4(lnn, z_e[0], v0, n_steps=n, dt=1.0, pressure=p_s)
        mse = plot_latent_rollout(z_e.cpu().numpy(), z_s.cpu().numpy(), config.DT,
                                  f'epoch {ep + 1} — vidéo 0',
                                  debug_dir / f'epoch_{ep + 1:04d}_rollout_train0.png')
        lnn.train()
        return mse

    def _decode(z_bd):
        """(B,d) latent (blanchi) → (B,C,dec_res,dec_res) [0,1], DIFFÉRENTIABLE en z."""
        zb = whiten.inverse(z_bd) if whiten is not None else z_bd
        return decoder(zb)                       # déjà clampé [0,1] dans forward

    def _target_images(idx_cpu):
        """Frames cibles o_{i+1} (échantillon 2) → (B,C,dec_res,dec_res) [0,1] device."""
        fi = tgt_frame_idx[idx_cpu]                          # (B,) numpy
        x = torch.from_numpy(np.ascontiguousarray(frames_np[fi])).to(device)
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        if x.shape[-1] != dec_res:
            x = F.interpolate(x, size=(dec_res, dec_res), mode='area')
        return x

    def _batch_loss(idx, idx_cpu):
        """Loss de Krauss sur un lot (pas de Verlet + décodage). Retourne
        (loss, loss_dec) ou (None, None) si le pas a divergé / n'est pas fini."""
        Zb = Z[idx]                                          # (b,4,d)
        z0 = Zb[:, 1]
        v0 = (Zb[:, 2] - Zb[:, 0]) / (2.0 * k)               # diff centrée sur s+k
        z1_true = Zb[:, 2]
        v1_true = (Zb[:, 3] - Zb[:, 1]) / (2.0 * k)          # diff centrée sur s+2k
        p_step = P[idx][:, 1] if P is not None else None     # pression ZOH à l'échantillon 1
        try:
            z1_pred, v1_pred = _verlet_step(lnn, z0, v0, p_step, h=float(k))
        except torch.linalg.LinAlgError:
            return None, None
        if not torch.isfinite(z1_pred).all():
            return None, None
        if metric_A is not None:
            # Δuᵀ Ḡ Δu / d (rollout_loss réduit à la MSE isotrope si A=I, cf. trace(A)=d)
            loss_z = rollout_loss(z1_pred, z1_true, metric_A)
            loss_v = rollout_loss(k * v1_pred, k * v1_true, metric_A)
        else:
            loss_z = F.mse_loss(z1_pred, z1_true)
            loss_v = F.mse_loss(k * v1_pred, k * v1_true)    # vitesse en unités de position/pas
        # w_dec = 0 (recette de référence) : le terme décodé ne
        # pèse ni dans la loss ni dans le gradient, mais le décoder quand même coûtait un
        # rendu GS par pas — prohibitif sous le repli torch (~100× plus lent que gsplat).
        # On le saute et on journalise 0.
        if w_dec != 0.0:
            img_pred = _decode(z1_pred)                      # (b,C,dec_res,dec_res)
            o_next = _target_images(idx_cpu)
            loss_dec = F.mse_loss(img_pred, o_next)
        else:
            loss_dec = torch.zeros((), device=z1_pred.device)
        loss = w_dec * loss_dec + w_z * loss_z + w_v * loss_v
        if not torch.isfinite(loss):
            return None, None
        return loss, loss_dec

    def _eval_loss(n_windows=3000):
        """Loss de Krauss moyenne SANS mise à jour (état courant du LNN), sur un
        échantillon déterministe de fenêtres. Sert de mesure avant/après.
        NB : pas de `torch.no_grad()` — `lnn.accel` construit son propre graphe
        autograd (dérivée de l'énergie cinétique) ; on ne fait juste pas de backward."""
        lnn.eval()
        g = torch.Generator(device=device).manual_seed(12345)
        perm = torch.randperm(N, generator=g, device=device)[:min(n_windows, N)]
        tot, tdec, nb = 0.0, 0.0, 0
        for i in range(0, perm.shape[0], bs):
            idx = perm[i:i + bs]
            loss, loss_dec = _batch_loss(idx, idx.cpu().numpy())
            if loss is None:
                continue
            tot += loss.item() * len(idx)
            tdec += loss_dec.item() * len(idx)
            nb += len(idx)
        lnn.train()
        return (tot / nb, tdec / nb) if nb else (float('nan'), float('nan'))

    # ── Loss AVANT entraînement (état de départ = warm-start ou LNN neuf) ─────
    init_loss, init_dec = _eval_loss()
    print(f'Loss Krauss AVANT (no-grad, {_args.init_lnn or "LNN neuf"}) : '
          f'{init_loss:.4e}  (dec={init_dec:.4e})')

    # ── Boucle d'entraînement ────────────────────────────────────────────────
    losses, dec_losses = [], []
    for ep in range(epochs):
        lnn.train()
        if wpe > 0 and wpe < N:
            perm = torch.randperm(N, device=device)[:wpe]
        else:
            perm = torch.randperm(N, device=device)
        n_used = perm.shape[0]
        ep_loss, ep_dec, nb, nskip = 0.0, 0.0, 0, 0
        t0 = time.time()
        for i in range(0, n_used, bs):
            idx = perm[i:i + bs]
            idx_cpu = idx.cpu().numpy()
            loss, loss_dec = _batch_loss(idx, idx_cpu)
            if loss is None:
                nskip += 1
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lnn.parameters(), config.LNN_GRAD_CLIP)
            opt.step()
            ep_loss += loss.item() * len(idx)
            ep_dec += loss_dec.item() * len(idx)
            nb += len(idx)
        ep_loss = ep_loss / nb if nb else float('nan')
        ep_dec = ep_dec / nb if nb else float('nan')
        losses.append(ep_loss)
        dec_losses.append(ep_dec)
        dt_fit = time.time() - t0
        # plot debug (final toujours ; sinon tous les plot_every)
        if (ep + 1) % max(1, plot_every) == 0 or ep == epochs - 1:
            try:
                _plot_intermediate(ep)
            except Exception as e:
                print(f'  [epoch {ep + 1}] plot échoué ({type(e).__name__}: {e})')
        # Chrono APRÈS le plot : `_plot_intermediate` déroule VIZ_MAX_FRAMES pas
        # séquentiels de simulate_rk4 (autograd interne, une seule trajectoire, donc
        # aucun parallélisme) et coûte, mesuré sur Krauss 2-seg à VIZ_MAX_FRAMES=5000,
        # ~83 s contre ~11 s pour l'entraînement — soit 8× l'époque. Chronométrer avant
        # le plot donnait une estimation de durée fausse d'un facteur ~9 sous
        # --plot-every 1. On reporte donc les deux temps séparément.
        dt_ep = time.time() - t0
        if ep % max(1, epochs // 50) == 0 or ep == epochs - 1:
            _plt = f'+{dt_ep - dt_fit:.0f}s plot' if dt_ep - dt_fit > 1.0 else ''
            print(f'  epoch {ep:4d}  loss={ep_loss:.4e}  (dec={ep_dec:.4e})  '
                  f'{dt_ep:.1f}s/ep ({dt_fit:.1f}s fit{" " + _plt if _plt else ""})'
                  + (f'  ({nskip} sautés)' if nskip else ''))
        # checkpoint atomique à chaque époque
        _tmp = out_path.with_suffix('.pt.tmp')
        torch.save(lnn.state_dict(), _tmp)
        _tmp.replace(out_path)

    print(f'LNN Krauss sauvegardé : {out_path}')

    # ── Loss APRÈS entraînement (même échantillon déterministe qu'AVANT) ──────
    final_loss, final_dec = _eval_loss()
    print('──────────── Loss Krauss (no-grad, même échantillon) ────────────')
    print(f'  AVANT : {init_loss:.4e}  (dec={init_dec:.4e})')
    print(f'  APRÈS : {final_loss:.4e}  (dec={final_dec:.4e})')
    print(f'  ratio : {final_loss / init_loss:.3f}× (total), '
          f'{final_dec / init_dec:.3f}× (terme décodé)')
    print('─────────────────────────────────────────────────────────────────')

    # ── Courbe de perte ──────────────────────────────────────────────────────
    lnn.eval()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, color='tab:purple', label='loss totale')
    ax.plot(dec_losses, color='tab:orange', alpha=0.7, label='terme décodé (MSE image)')
    ax.set_yscale('log'); ax.set_xlabel('époque'); ax.set_ylabel('loss')
    ax.set_title(f'LNN Krauss ({config.SAVE_DIR.parent.name})')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    loss_png = config.SAVE_DIR / f'{out_path.stem}_loss.png'
    fig.savefig(loss_png, dpi=130)
    plt.close(fig)
    print(f'Courbe de perte : {loss_png}')


if __name__ == '__main__':
    main()
