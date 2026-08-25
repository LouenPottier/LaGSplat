"""
eval_multistep_mse.py — reproduit la métrique quantitative de Krauss et al. 2026
(VON, RA-L) : **MSE image de reconstruction multi-step** sur un horizon de 0.5 s.

Protocole Krauss (IV-D) répliqué au plus près :
  - on tire `--n-traj` fenêtres de départ aléatoires dans la vidéo ;
  - depuis chaque départ, on initialise (q0, q̇0) = (enc(x0), vitesse latente par
    diff. avant d'ordre 2) — comme eux (observation + vitesse latente) ;
  - on déroule le LNN sur `round(0.5 / DT)` pas (0.5 s), forcé par la pression
    mesurée alignée (get_sim_pressure) ;
  - on décode chaque état latent prédit en image (whiten⁻¹ → décodeur GS) ;
  - MSE image [0,1] (F.mse_loss, moyenne sur C×H×W) contre la vérité terrain,
    par pas puis moyennée sur les fenêtres.

Krauss calcule sa MSE à **32×32**. Notre décodeur rend à IMG_SIZE (256²) ; on
reporte donc la MSE à la résolution native ET après downsample area 32×32 (le
chiffre directement comparable à Krauss). On reporte aussi le **plancher AE**
(MSE de reconstruction statique decode(enc(x)) vs x, sans dynamique) pour séparer
l'erreur de dynamique de l'erreur d'autoencodeur.

Chiffres Krauss à confronter (MSE image multi-step, 0.5 s, 32×32) :
    1 segment : VON 5.74e-4  | osc. std 2.46e-4
    2 segments: VON 6.56e-3  | Koopman+ABCD 9.84e-4  | Koopman std 5.66e-3

⚠️ Held-out : Krauss mesure sur 50 trajectoires de VALIDATION. Nos cas ont
VAL_VIDEO=None (on s'entraîne sur toute la 1ʳᵉ moitié) → par défaut ce script
évalue sur VIDEO_DIR (IN-SAMPLE, optimiste). Passer `--video <fichier>` pour
pointer la part held-out, ici les 20 % finaux du split officiel.

Lancer (interpréteur `py`) :
    py eval_multistep_mse.py --config ../cases/krauss2026_2seg_npz/config.py \
        --video scr_2seg_32x32_val20.npz          # held-out (20 % finaux)

Sortie : impression des chiffres + PNG `multistep_mse[_<tag>].png` dans SAVE_DIR.
"""
import argparse

from _bootstrap import load_config
config = load_config()

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--frames', type=int, default=6000,
                 help='nb de frames chargées/encodées (borne mémoire ; défaut 6000 ≈ 50 s)')
_ap.add_argument('--n-traj', type=int, default=50, help='nb de fenêtres/trajectoires (Krauss : 50)')
_ap.add_argument('--stride', type=int, default=0,
                 help='0 (défaut) = tirage ALÉATOIRE de --n-traj fenêtres (protocole Krauss, '
                      'chiffres historiques reproduits) ; ≥1 = énumération EXHAUSTIVE de toutes '
                      'les fenêtres de [skip, n−horizon] avec ce pas (1 = tout le held-out, '
                      'une fenêtre par frame de départ ; --n-traj est alors ignoré)')
_ap.add_argument('--skip-sec', type=float, default=8.0,
                 help='ignore les N premières s (tête statique avant le 1ᵉʳ échelon de pression ; défaut 8)')
_ap.add_argument('--horizon-sec', type=float, default=0.5, help='horizon multi-step en s (Krauss : 0.5)')
_ap.add_argument('--checkpoints-sec', type=str, default=None,
                 help="liste de temps (s) où reporter l'erreur, ex '0.5,1,2,4,8' ; horizon = max")
_ap.add_argument('--eval-res', type=int, default=32, help='résolution de comparaison Krauss (défaut 32)')
_ap.add_argument('--lnn', type=str, default='auto',
                 help="checkpoint LNN : 'auto' (lnn_rollout.pt sinon lnn.pt) | 'lnn.pt' | 'lnn_rollout.pt'")
_ap.add_argument('--decoder', type=str, default='auto',
                 help="checkpoint décodeur : 'auto' (decoder2dpt_ae.pt sinon decoder2dpt.pt) | nom explicite")
_ap.add_argument('--video', type=str, default=None,
                 help='nom de fichier vidéo (dans le dossier de VIDEO_DIR) pour évaluer en held-out')
_ap.add_argument('--seed', type=int, default=0, help='graine du tirage des fenêtres')
_ap.add_argument('--step-frames', type=int, default=1,
                 help="pas d'intégration en frames (1 = natif ; 2 = 30 pas sur 0.5 s, "
                      'comme les 59.94 fps de Krauss). Horizon PHYSIQUE inchangé.')
_ap.add_argument('--integrator', type=str, default=None,
                 choices=['verlet', 'rk4', 'gen_alpha'], help='override config.LNN_INTEGRATOR')
_ap.add_argument('--no-smooth-latent', action='store_true', help='ne PAS lisser q(t) (init)')
_ap.add_argument('--sigma', type=float, default=None,
                 help='override config.SMOOTH_LATENT_SIGMA (lissage gaussien de q(t), en '
                      "FRAMES). ⚠️ Doit valoir le σ d'ENTRAÎNEMENT du checkpoint : l'éval "
                      "lisse les conditions initiales avec cette clé, donc un σ différent "
                      'change toutes les fenêtres. Ex. lnn_krauss_sig5_100ep.pt → --sigma 5.')
_ap.add_argument('--sigma-pressure', type=float, default=None,
                 help='override config.SMOOTH_PRESSURE_SIGMA (lissage gaussien de P(t), en '
                      "FRAMES). ⚠️ INDÉPENDANT de --sigma : ce dernier n'écrase que le σ de "
                      'q(t), et le repli « pression = σ de q » ne joue que si '
                      'SMOOTH_PRESSURE_SIGMA vaut None dans la config. Doit valoir le σ de '
                      "PRESSION d'entraînement du checkpoint, sinon le forçage b(q)ᵀP de "
                      "l'évaluation ne correspond pas à celui vu à l'entraînement.")
_ap.add_argument('--cq-eps', type=float, default=None,
                 help='override config.LNN_RAYLEIGH_CQ_EPS (plancher isotrope de C(q)). '
                      "⚠️ Comme --sigma, doit valoir le c₀ d'ENTRAÎNEMENT du checkpoint : "
                      "c'est une composante de la dynamique simulée, pas un réglage "
                      "d’affichage. Mêmes sémantique et nom que dans les autres "
                      "scripts d’évaluation.")
_ap.add_argument('--out', type=str, default=None, help='nom du PNG de sortie')
_args, _ = _ap.parse_known_args()

if _args.integrator is not None:
    config.LNN_INTEGRATOR = _args.integrator
if _args.cq_eps is not None:
    config.LNN_RAYLEIGH_CQ_EPS = float(_args.cq_eps)
if _args.sigma is not None:
    config.SMOOTH_LATENT_SIGMA = float(_args.sigma)
if _args.sigma_pressure is not None:
    config.SMOOTH_PRESSURE_SIGMA = float(_args.sigma_pressure)

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
from train_lnn_fixedae import encode_all, smooth_latents, smooth_pressures


def _to_float01(frames_np):
    """(T,C,H,W) uint8[0,255] ou float[0,1] → tensor float [0,1] CPU."""
    t = torch.from_numpy(np.ascontiguousarray(frames_np))
    if t.dtype == torch.uint8:
        t = t.float().div_(255.0)
    return t.float()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _use_pressure = getattr(config, 'LNN_PRESSURE', False)
    n_ch = 3 if config.ENC_COLOR else 1
    n_ch_dec = 3 if getattr(config, 'DEC_COLOR', True) else 1
    checkpoints = ([float(x) for x in _args.checkpoints_sec.split(',')]
                   if _args.checkpoints_sec else None)
    horizon_sec = max(checkpoints) if checkpoints else _args.horizon_sec
    horizon_frames = int(round(horizon_sec / config.DT))
    # ── Pas d'intégration en FRAMES (`--step-frames k`) ──────────────────────
    # Le LNN est une ODE lagrangienne CONTINUE : le même horizon physique peut
    # s'intégrer en `horizon_frames` pas de dt=1 frame (défaut) ou en
    # `horizon_frames/k` pas de dt=k. Sert à vérifier que notre chiffre à 0.5 s
    # ne dépend pas du NOMBRE de pas — Krauss, dont les modèles sont des cartes
    # discrètes à 59.94 fps, n'en fait que 30 là où nous en faisons 60.
    k = max(1, int(_args.step_frames))
    horizon = max(1, horizon_frames // k)
    R = _args.eval_res

    video_dir = config.VIDEO_DIR
    if _args.video is not None:
        video_dir = config.VIDEO_DIR.parent / _args.video
    print(f'Device : {device}')
    print(f'Vidéo  : {video_dir.name}'
          + ('  (VIDEO_DIR = in-sample)' if _args.video is None else '  (--video)'))
    print(f'Horizon: {horizon} pas × {k} frame(s) = {horizon*k*config.DT:.3f} s  |  '
          f'{_args.n_traj} fenêtres  |  éval @ {R}×{R} + natif {config.IMG_SIZE[0]}²')

    # ── Données ──────────────────────────────────────────────────────────────
    ds = VideoFrameDataset(
        video_dir=video_dir, img_size=config.IMG_SIZE, n_channels=n_ch,
        rest_video=config.REST_VIDEO, rest_n_frames=config.REST_N_FRAMES,
        rest_first_n_frames=getattr(config, 'REST_FIRST_N_FRAMES', 0),
        crop=getattr(config, 'CROP', None),
        load_pressure=_use_pressure, pressure_dir=getattr(config, 'PRESSURE_DIR', None),
        pressure_cols=getattr(config, 'PRESSURE_COLS', None),
        pressure_norm=getattr(config, 'PRESSURE_NORM', 101325.0), pressure_dt=config.DT,
        pressure_sync_offsets=getattr(config, 'PRESSURE_SYNC_OFFSETS', None),
        max_frames=_args.frames, store_uint8=True)
    n = len(ds.frames)
    if n_ch_dec != n_ch:
        ds_dec = VideoFrameDataset(
            video_dir=video_dir, img_size=config.IMG_SIZE, n_channels=n_ch_dec,
            rest_video=config.REST_VIDEO, rest_n_frames=config.REST_N_FRAMES,
            rest_first_n_frames=getattr(config, 'REST_FIRST_N_FRAMES', 0),
            crop=getattr(config, 'CROP', None), max_frames=_args.frames, store_uint8=True)
        gt_frames = ds_dec.frames
    else:
        gt_frames = ds.frames
    print(f'{n} frames chargées.')
    assert n > horizon * k + 3, f'pas assez de frames ({n}) pour {horizon} pas de {k} frames'

    # ── Encodeur gelé + blanchiment latent figé ──────────────────────────────
    enc = build_encoder(config.IMG_SIZE, config.ENC_HIDDEN, config.LATENT_DIM,
                        n_channels=n_ch, normalize=getattr(config, 'ENC_NORMALIZE', False)).to(device)
    enc_path = config.SAVE_DIR / 'encoder_ae.pt'
    if not enc_path.exists():
        enc_path = config.SAVE_DIR / 'encoder.pt'
    enc.load_state_dict(torch.load(enc_path, map_location=device))
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    whiten = load_latent_whiten(config.SAVE_DIR, device, config.LATENT_DIM)
    if whiten is not None:
        enc = WhitenedEncoder(enc, whiten).to(device).eval()
    print(f'Encodeur : {enc_path.name}' + ('  + LatentWhiten' if whiten is not None else ''))

    # ── Décodeur GS (AE) ─────────────────────────────────────────────────────
    if _args.decoder == 'auto':
        dec_path = config.SAVE_DIR / 'decoder2dpt_ae.pt'
        if not dec_path.exists():
            dec_path = config.SAVE_DIR / 'decoder2dpt.pt'
    else:
        dec_path = config.SAVE_DIR / _args.decoder
    dec_state = torch.load(dec_path, map_location=device)
    n_gaussians = dec_state['mu_raw'].shape[0]
    decoder = build_decoder2pt(
        latent_dim=config.LATENT_DIM, n_gaussians=n_gaussians,
        img_size=config.IMG_SIZE, n_channels=n_ch_dec).to(device)
    decoder.load_state_dict(dec_state)
    decoder.eval()
    print(f'Décodeur : {dec_path.name}  ({n_gaussians} gaussiennes)')

    # ── LNN ──────────────────────────────────────────────────────────────────
    if _args.lnn == 'auto':
        lnn_path = config.SAVE_DIR / 'lnn_rollout.pt'
        if not lnn_path.exists():
            lnn_path = config.SAVE_DIR / 'lnn.pt'
    else:
        lnn_path = config.SAVE_DIR / _args.lnn
    lnn = LNN(config.LATENT_DIM, config.LNN_HIDDEN).to(device)
    state = torch.load(lnn_path, map_location=device)
    try:
        lnn.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f'⚠ strict load échoué ({e}); repli strict=False.')
        lnn.load_state_dict(state, strict=False)
    lnn.eval()
    print(f'LNN      : {lnn_path.name}  (use_pressure={getattr(lnn, "use_pressure", False)}, '
          f'integrator={getattr(config, "LNN_INTEGRATOR", "verlet")})')
    print(f'Lissage q: σ={getattr(config, "SMOOTH_LATENT_SIGMA", None)} frames'
          f' | P: σ={getattr(config, "SMOOTH_PRESSURE_SIGMA", None)} frames'
          f'  (doivent être les σ d\'entraînement du checkpoint)')

    # ── Encodage + lissage (init seulement ; la GT image reste brute) ─────────
    print('Encodage…')
    z_raw = encode_all(enc, ds.frames, device)                   # (n, d) CPU, NON lissé
    z_init = z_raw
    if getattr(config, 'SMOOTH_LATENT', False) and not _args.no_smooth_latent:
        z_init = smooth_latents(
            z_raw.clone(), [n],
            getattr(config, 'SMOOTH_LATENT_WINDOW', 13),
            getattr(config, 'SMOOTH_LATENT_POLY', 3),
            getattr(config, 'SMOOTH_LATENT_MODE', 'savgol'),
            getattr(config, 'SMOOTH_LATENT_SIGMA', 10.0))
    if _use_pressure and getattr(config, 'SMOOTH_PRESSURE', False):
        ds.pressures = smooth_pressures(ds.pressures, ds.video_lengths)

    def decode(z_bd):
        """(B,d) espace latent (blanchi) → (B,C,H,W) [0,1] sur device."""
        with torch.no_grad():
            zb = z_bd.to(device)
            if whiten is not None:
                zb = whiten.inverse(zb)
            return decoder(zb).clamp(0.0, 1.0)

    def down(x):
        return F.interpolate(x, size=(R, R), mode='area') if R < config.IMG_SIZE[0] else x

    # ── Tirage des fenêtres (on saute la tête statique) ──────────────────────
    skip = int(round(_args.skip_sec / config.DT))
    rng = np.random.default_rng(_args.seed)
    # hi exclusif ⟹ la dernière frame comparée (s + (horizon−1)·k) reste dans
    # la vidéo.
    lo, hi = skip, n - horizon * k
    assert hi > lo, f'skip {_args.skip_sec}s trop grand pour {n} frames'
    if _args.stride > 0:
        starts = list(range(lo, hi, _args.stride))
        print(f'Fenêtres ÉNUMÉRÉES dans [{skip}, {hi}] au pas {_args.stride} : '
              f'{len(starts)} fenêtres (couverture exhaustive du held-out chargé ; '
              f'tête statique {_args.skip_sec}s = {skip} frames ignorée)')
    else:
        starts = rng.integers(lo, hi, size=min(_args.n_traj, hi - lo)).tolist()
        print(f'Fenêtres tirées dans [{skip}, {hi}]  (tête statique {_args.skip_sec}s = {skip} frames ignorée)')

    mse_nat = np.zeros(horizon)   # Σ MSE par pas, résolution native
    mse_sml = np.zeros(horizon)   # Σ MSE par pas, résolution Krauss (R×R)
    # Convention d'alignement : `simulate_rk4` renvoie traj[0] = CONDITION
    # INITIALE (état à t₀), donc l'état d'indice i se compare à la frame s+i·k.
    # C'est la convention retenue pour toutes les mesures publiées.
    per_win_sml = []              # MSE moyennée sur l'horizon, PAR FENÊTRE
    floor_nat = 0.0               # Σ MSE plancher AE (statique) natif
    floor_sml = 0.0
    n_finite = 0

    z_init_d = z_init.to(device)
    for s in starts:
        z0 = z_init_d[s]
        v0 = initial_velocity(z_init_d[s:s + 3])
        if _use_pressure:
            # Pression au DÉBUT de chaque pas : frames s, s+k, s+2k… (ZOH sur le pas).
            p_full = get_sim_pressure(lnn, ds, s, horizon * k, device)
            p = p_full[::k][:horizon] if p_full is not None else None
        else:
            p = None
        with torch.no_grad():
            z_s = simulate_rk4(lnn, z0, v0, n_steps=horizon, dt=float(k), pressure=p)  # (H,d)
        if not torch.isfinite(z_s).all():
            continue
        n_finite += 1
        # z_s[i] est l'état à la frame s+(i+1)·k
        idx_p0 = np.arange(horizon) * k + s
        gt = _to_float01(gt_frames[idx_p0]).to(device)            # (H,C,H,W) [0,1]
        # décodage par blocs pour borner la VRAM
        preds = []
        for i in range(0, horizon, 64):
            preds.append(decode(z_s[i:i + 64]))
        pred = torch.cat(preds, 0)                                # (H,C,H,W)
        gt_s, pred_s = down(gt), down(pred)
        mse_nat += ((pred - gt).pow(2).mean(dim=(1, 2, 3))).cpu().numpy()
        step_sml = ((pred_s - gt_s).pow(2).mean(dim=(1, 2, 3))).cpu().numpy()
        mse_sml += step_sml
        per_win_sml.append(step_sml.mean())

    assert n_finite > 0, 'toutes les trajectoires ont divergé (non-finies).'
    mse_nat /= n_finite
    mse_sml /= n_finite
    per_win_sml = np.asarray(per_win_sml)
    se_sml = per_win_sml.std(ddof=1) / np.sqrt(n_finite) if n_finite > 1 else float('nan')

    # ── Plancher AE (statique) sur un sous-échantillon régulier de frames ─────
    # Même région que les fenêtres (tête statique exclue) pour un plancher comparable.
    idx_floor = np.linspace(skip, n - 1, num=min(1000, n - skip), dtype=int)
    with torch.no_grad():
        for i in range(0, len(idx_floor), 64):
            sl = idx_floor[i:i + 64]
            gt = _to_float01(gt_frames[sl]).to(device)
            rec = decode(z_raw[torch.from_numpy(sl)])
            floor_nat += (rec - gt).pow(2).mean(dim=(1, 2, 3)).sum().item()
            floor_sml += (down(rec) - down(gt)).pow(2).mean(dim=(1, 2, 3)).sum().item()
    floor_nat /= len(idx_floor)
    floor_sml /= len(idx_floor)

    # ── Rapport ──────────────────────────────────────────────────────────────
    print(f'\n{n_finite}/{len(starts)} trajectoires finies.')
    print('──────────── MSE image multi-step (0.5 s) ────────────')
    print(f'{"":16s}{"natif "+str(config.IMG_SIZE[0]):>16s}{"Krauss "+str(R)+"²":>16s}')
    print(f'{"moyenne horizon":16s}{mse_nat.mean():>16.3e}{mse_sml.mean():>16.3e}')
    print(f'{"dernier pas":16s}{mse_nat[-1]:>16.3e}{mse_sml[-1]:>16.3e}')
    print(f'{"1er pas":16s}{mse_nat[0]:>16.3e}{mse_sml[0]:>16.3e}')
    print(f'{"plancher AE":16s}{floor_nat:>16.3e}{floor_sml:>16.3e}')
    print('───────────────────────────────────────────────────────')
    print(f'{"± erreur-type (sur les fenêtres)":32s}{se_sml:>16.2e}')
    seg = '2 seg' if config.LATENT_DIM >= 4 else '1 seg'
    print(f'Krauss ({seg}) @32² : VON '
          + ('6.56e-3, Koopman+ABCD 9.84e-4' if seg == '2 seg' else '5.74e-4, osc.std 2.46e-4'))

    # ── Erreur à plusieurs horizons (cumul [0,T] et instantané à T) @32² ──────
    if checkpoints:
        print('\n──────── MSE @32² par horizon (cumul [0,T] | instant T) ────────')
        print(f'{"T (s)":>8s}{"cumul [0,T]":>16s}{"instant T":>16s}')
        for T in checkpoints:
            kk = min(int(round(T / (config.DT * k))), horizon)   # pas (de k frames)
            cum = mse_sml[:kk].mean()
            inst = mse_sml[kk - 1]
            print(f'{T:>8.2f}{cum:>16.3e}{inst:>16.3e}')
        print(f'{"plancher":>8s}{floor_sml:>16.3e}{"":>16s}')
        print('────────────────────────────────────────────────────────────────')

    # ── Plot MSE vs temps (comme Fig. 5 de Krauss) ───────────────────────────
    t_s = np.arange(horizon) * config.DT * k
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(t_s, mse_sml, label=f'LaGSplat rollout @ {R}²', color='tab:blue')
    ax.plot(t_s, mse_nat, label=f'LaGSplat rollout @ {config.IMG_SIZE[0]}²', color='tab:cyan', alpha=0.6)
    ax.axhline(floor_sml, ls='--', color='tab:blue', alpha=0.5, label=f'plancher AE @ {R}²')
    kv = 6.56e-3 if seg == '2 seg' else 5.74e-4
    ax.axhline(kv, ls=':', color='tab:red', label=f'Krauss VON @32² ({kv:.2e})')
    ax.set_xlabel('temps (s)'); ax.set_ylabel('MSE image [0,1]')
    ax.set_yscale('log'); ax.legend(fontsize=8)
    ax.set_title(f'MSE multi-step — {video_dir.name}\n'
                 f'{n_finite} fenêtres, {lnn_path.name}, {config.LNN_INTEGRATOR}')
    fig.tight_layout()
    tag = _args.out or f'multistep_mse{"_"+_args.video.split(".")[0] if _args.video else ""}.png'
    out = config.SAVE_DIR / tag
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f'Plot : {out}')

    # Dump de la courbe par-pas (pour figures externes : overlay Krauss Fig. 5).
    npz = out.with_suffix('.npz')
    np.savez(npz, t_s=t_s, mse_sml=mse_sml, mse_nat=mse_nat,
             floor_sml=floor_sml, floor_nat=floor_nat, dt=config.DT,
             n_windows=n_finite, seg=config.LATENT_DIM,
             se_sml=se_sml, per_win_sml=per_win_sml)
    print(f'Data : {npz}')


if __name__ == '__main__':
    main()
