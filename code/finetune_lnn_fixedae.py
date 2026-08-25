"""
finetune_lnn_fixedae.py — RAFFINE un LNN déjà entraîné (loss ODE / résidu FD) par
une loss de ROLLOUT différentiable, autoencodeur GELÉ et q(t) PRÉCALCULÉ.

Motivation
----------
`train_lnn_fixedae.py` minimise le résidu Euler-Lagrange en différences finies : à
chaque point on INJECTE le q(t) vrai (teacher-forcing) et on demande q̈ ≈ accel(q,q̇).
Ce critère single-step ne contraint la dynamique QUE le long de la trajectoire vraie
lissée — jamais la stabilité hors-variété ni l'enveloppe d'amortissement long-horizon
(c'est précisément ce qui a laissé passer le bug 'invex' : résidu FD faible mais rollout
divergent ; et ce qui manque à Gamma sur Krauss). Ici on déroule le LNN par un
intégrateur DIFFÉRENTIABLE (velocity-Verlet, mêmes forces que `viz.simulate_rk4`) et on
back-propage MSE(rollout, q_vrai) à travers l'horizon. Recette « pré-entraînement
derivative-matching → fine-tuning rollout court » (SymODEN / Symplectic RNN).

Fenêtre temporelle
------------------
Chaque fenêtre de batch fait W = SEQ_LEN + LNN_ROLLOUT_STEPS frames : le rollout part de
la 1re frame (v0 par différence avant d'ordre 2, cohérent avec le résidu) et prédit toute
la fenêtre ; la loss compare aux W frames encodées. LNN_ROLLOUT_STEPS s'AJOUTE donc à la
taille de fenêtre du résidu FD (SEQ_LEN).

Transparence
------------
On CHARGE `lnn.pt` (le LNN ODE) et on sauvegarde le LNN raffiné dans un fichier DÉDIÉ
`lnn_rollout.pt` : `lnn.pt` n'est JAMAIS écrasé (baseline ODE préservée). Pour utiliser
le LNN raffiné en aval, pointer explicitement dessus (copier sur lnn.pt, ou charger
lnn_rollout.pt). Toutes les clés `LNN_ROLLOUT_* / LNN_FINETUNE_*` de la config ne sont
lues QUE par ce script (aucun impact sur les autres entraînements).

Métrique de la loss (Ḡ)
-----------------------
Par défaut la loss de rollout est isotrope, ce qui donne le même poids à un mode très
visible et à un mode qui ne déplace presque aucun pixel. `--metric` la remplace par la
métrique de VISIBILITÉ `Ḡ` (précalculée par `compute_visibility_metric.py`), c'est-à-dire
le terme décodé de Krauss au premier ordre sans rasteriser dans le graphe BPTT. Voir
`load_rollout_metric` pour la ridge `ρ` et la normalisation `trace(A)=d`.

Lancer :  py finetune_lnn_fixedae.py --config ../cases/krauss2026_2seg_npz/config.py --metric
Prérequis : encoder_ae.pt/encoder.pt + lnn.pt (+ metric_geom/latent_whiten si utilisés)
            dans SAVE_DIR (i.e. train_lnn_fixedae déjà passé) ; avec `--metric`,
            visibility_metric.pt (recalculé avec le décodeur COURANT).

CLI (tout est optionnel, la config fait foi sans les flags) :
  --metric / --no-metric / --metric-ridge ρ   pondération par Ḡ
  --init-lnn <f.pt>  (déf. lnn.pt)   --out <f.pt>  (déf. lnn_rollout.pt)
  --lr / --epochs / --batch / --seq-stride / --sigma / --plot-every
Les artefacts dérivés (courbe de perte, held-out, plots debug) sont nommés d'après
`--out`, donc deux runs de réglages différents ne s'écrasent pas.
"""
import argparse
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _bootstrap import load_config
config = load_config()

# ── Knobs CLI (priorité sur la config ; appliqués UNIQUEMENT dans main()) ─────
# Le parseur est défini au niveau module mais n'écrit RIEN sur `config` ici :
# `train_lnn_krauss.py` importe ce module (load_rollout_metric / rollout_loss /
# _verlet_step), et un import ne doit pas modifier sa config. parse_known_args
# ignore les options des autres scripts (dont --config, lu par _bootstrap).
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--metric', dest='metric', action='store_true', default=None,
                 help='pondère la loss de rollout par la métrique de VISIBILITÉ Ḡ '
                      '(visibility_metric.pt) au lieu de la MSE isotrope : '
                      'Δuᵀ Ḡ Δu ≈ ‖ΔI‖², c\'est-à-dire le terme décodé de Krauss au '
                      'premier ordre, sans rasteriser dans le graphe BPTT. '
                      'Équivaut à config.LNN_ROLLOUT_METRIC=True.')
_ap.add_argument('--no-metric', dest='metric', action='store_false',
                 help='force la loss isotrope même si config.LNN_ROLLOUT_METRIC=True')
_ap.add_argument('--metric-ridge', type=float, default=None,
                 help='ρ de A ∝ Ḡ + ρ·λmax·I (déf. config.LNN_ROLLOUT_METRIC_RIDGE=0.01). '
                      'Borne le rapport de pondération entre modes à ~1/ρ ; la '
                      'transposition des λ officiels de Krauss donne ρ ≈ 0.36.')
_ap.add_argument('--init-lnn', type=str, default=None,
                 help='checkpoint LNN de départ dans SAVE_DIR (déf. lnn.pt)')
_ap.add_argument('--out', type=str, default=None,
                 help='nom du checkpoint de sortie dans SAVE_DIR (déf. lnn_rollout.pt). '
                      'Le checkpoint de départ n\'est JAMAIS écrasé.')
_ap.add_argument('--lr', type=float, default=None,
                 help='override config.LNN_FINETUNE_LR')
_ap.add_argument('--epochs', type=int, default=None,
                 help='override config.LNN_FINETUNE_EPOCHS')
_ap.add_argument('--batch', type=int, default=None,
                 help='override config.LNN_ROLLOUT_BATCH')
_ap.add_argument('--windows-per-epoch', type=int, default=None,
                 help='nb de fenêtres TIRÉES AU HASARD par époque (<=0 = toutes, '
                      'comportement historique). Même levier que train_lnn_krauss : sans '
                      'lui une époque parcourt les ~90 000 fenêtres de la vidéo, chacune '
                      'en BPTT sur W pas, ce qui rend le coût par époque incomparable '
                      'avec celui du résidu FD. Le tirage change à chaque époque, donc '
                      'toutes les fenêtres finissent vues.')
_ap.add_argument('--seq-stride', type=int, default=None,
                 help='override config.SEQ_STRIDE (espacement des échantillons, en frames)')
_ap.add_argument('--sigma', type=float, default=None,
                 help='override config.SMOOTH_LATENT_SIGMA (lissage gaussien des cibles '
                      'latentes, en FRAMES ; < 0.2 ⟹ noyau identité, donc sans effet)')
_ap.add_argument('--plot-every', type=int, default=None,
                 help='override config.LNN_PLOT_EVERY')
_args, _ = _ap.parse_known_args()


def _apply_cli_overrides():
    """Pose les options CLI sur `config` (appelé en tête de main() seulement).

    Retourne (init_name, out_name) : noms de fichiers dans SAVE_DIR.
    """
    if _args.metric is not None:
        config.LNN_ROLLOUT_METRIC = bool(_args.metric)
    if _args.metric_ridge is not None:
        config.LNN_ROLLOUT_METRIC_RIDGE = float(_args.metric_ridge)
    if _args.lr is not None:
        config.LNN_FINETUNE_LR = float(_args.lr)
    if _args.epochs is not None:
        config.LNN_FINETUNE_EPOCHS = int(_args.epochs)
    if _args.batch is not None:
        config.LNN_ROLLOUT_BATCH = int(_args.batch)
    if _args.windows_per_epoch is not None:
        config.LNN_ROLLOUT_WINDOWS_PER_EPOCH = int(_args.windows_per_epoch)
    if _args.seq_stride is not None:
        config.SEQ_STRIDE = int(_args.seq_stride)
    if _args.sigma is not None:
        config.SMOOTH_LATENT_SIGMA = float(_args.sigma)
        config.SMOOTH_LATENT_MODE = 'gaussian'
        config.SMOOTH_LATENT = True
    if _args.plot_every is not None:
        config.LNN_PLOT_EVERY = int(_args.plot_every)
    init_name = _args.init_lnn or 'lnn.pt'
    out_name = _args.out or 'lnn_rollout.pt'
    assert init_name != out_name, (
        f'--out ({out_name}) écraserait le checkpoint de départ --init-lnn.')
    return init_name, out_name

from dataset import VideoFrameDataset
from models import LNN, build_encoder, WhitenedEncoder, load_latent_whiten
from viz import simulate_rk4, initial_velocity, get_sim_pressure
# Machinerie identique à l'entraînement FD (mêmes fonctions, aucune duplication).
from train_lnn_fixedae import (encode_all, smooth_latents, smooth_pressures,
                               plot_latent_rollout,
                               # Métrique de visibilité Ḡ : DÉFINIE dans
                               # train_lnn_fixedae (le résidu FD la pondère
                               # aussi désormais), importée ici — une seule
                               # implémentation pour les deux étages.
                               load_rollout_metric, rollout_loss, metric_mse)


# ─────────────────────────────────────────────────────────────────────────────
#  Physique batchée DIFFÉRENTIABLE (le pendant de viz.simulate_rk4.dvdt, sans le
#  .detach() d'inférence : le graphe est conservé pour back-propager le rollout).
# ─────────────────────────────────────────────────────────────────────────────
def _accel_batched(lnn, z, v, p):
    """Accélération q̈ = M⁻¹(−∇E − friction + F_P), batchée (B,D)→(B,D).

    Signe et termes IDENTIQUES au résidu FD (models.LNN._residual_order2 :
    résidu = q̈ + Minv·(∇E + friction − F_P)) et à viz.simulate_rk4.dvdt. Chemin
    métrique courbe → lnn.accel (Coriolis + Rayleigh + pression inclus). En mode
    lnn.train() les gradients (autograd/grad_E analytique) remontent aux poids.
    """
    if getattr(lnn, 'metric', None) is not None:
        return lnn.accel(z, v, p)                       # (B,D), différentiable en train()
    grad = lnn.dE_dz(z)                                 # (B,D)  ∂E/∂z
    friction = torch.zeros_like(v)
    if getattr(lnn, 'Gamma', None) is not None:
        friction = friction + v @ lnn.Gamma.T           # visqueux matriciel SPD
    elif getattr(lnn, 'log_gamma', None) is not None:
        friction = friction + lnn.gamma * v             # visqueux scalaire
    if getattr(lnn, 'Beta', None) is not None:
        vn = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        friction = friction + (v / vn) @ lnn.Beta.T     # Coulomb matriciel
    elif getattr(lnn, 'log_beta', None) is not None:
        vn = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        friction = friction + lnn.beta * v / vn         # Coulomb scalaire
    a = -grad - friction
    if p is not None and getattr(lnn, 'use_pressure', False):
        a = a + lnn.pressure_force(z, p)                # F_P = b(q)ᵀP (même signe que le résidu)
    if getattr(lnn, 'Minv', None) is not None:
        a = a @ lnn.Minv.T                              # M⁻¹ (SPD, symétrique)
    return a


def _verlet_step(lnn, z, v, p, h=1.0):
    """Un pas velocity-Verlet semi-implicite différentiable (2 évals de force)."""
    a1 = _accel_batched(lnn, z, v, p)
    v_half = v + 0.5 * h * a1
    z_new = z + h * v_half
    a2 = _accel_batched(lnn, z_new, v_half, p)          # frottement semi-implicite (v_half)
    return z_new, v_half + 0.5 * h * a2


def _rk4_step(lnn, z, v, p, h=1.0):
    """Un pas RK4 différentiable (4 évals de force), pression tenue constante."""
    k1v = _accel_batched(lnn, z, v, p);                       k1z = v
    k2v = _accel_batched(lnn, z + .5*h*k1z, v + .5*h*k1v, p); k2z = v + .5*h*k1v
    k3v = _accel_batched(lnn, z + .5*h*k2z, v + .5*h*k2v, p); k3z = v + .5*h*k2v
    k4v = _accel_batched(lnn, z + h*k3z,   v + h*k3v,   p);   k4z = v + h*k3v
    return (z + (h/6)*(k1z + 2*k2z + 2*k3z + k4z),
            v + (h/6)*(k1v + 2*k2v + 2*k3v + k4v))


def rollout(lnn, z0, v0, n_steps, p_seq, integrator='verlet', tbptt=0, h=1.0):
    """Rollout différentiable → (B, n_steps, D). p_seq : (B, n_steps, n_c) ou None
    (maintien d'ordre zéro par pas, comme simulate_rk4). tbptt>0 : detach l'état
    tous les k pas (BPTT tronqué, borne mémoire/gradient).
    h : pas d'intégration (= SEQ_STRIDE : la fenêtre échantillonne toutes les k frames,
    on avance donc k frames par pas ; l'ODE reste PAR FRAME, cf. v0 en unités par-frame)."""
    step = _rk4_step if integrator == 'rk4' else _verlet_step
    z, v = z0, v0
    zs = [z0]
    for i in range(n_steps - 1):
        p_i = None if p_seq is None else p_seq[:, i]
        z, v = step(lnn, z, v, p_i, h)
        zs.append(z)
        if tbptt and (i + 1) % tbptt == 0:
            z, v = z.detach(), v.detach()
    return torch.stack(zs, dim=1)                       # (B, n_steps, D)


def build_rollout_windows(z_all, video_lengths, W, pressures=None, stride=1):
    """Fenêtres de W échantillons ESPACÉS de `stride` frames (indices s, s+k, …,
    s+(W-1)k) NE chevauchant PAS deux vidéos, + SÉQUENCE de pression aux MÊMES
    échantillons. Retourne Z (Nw,W,d) et P (Nw,W,n_c) ou None. stride=1 → frames
    adjacentes (comportement d'origine)."""
    k = max(int(stride), 1)
    span = (W - 1) * k                         # dernier échantillon = s + span
    starts, off = [], 0
    for vlen in video_lengths:
        if vlen > span:
            starts += [off + s for s in range(vlen - span)]
        off += vlen
    assert starts, (f'Aucune vidéo > (W-1)·stride={span} frames (SEQ_LEN+LNN_ROLLOUT_STEPS, '
                    f'stride={k}) : réduire LNN_ROLLOUT_STEPS ou SEQ_STRIDE. '
                    f'Longueurs={list(video_lengths)}')
    Z = torch.stack([z_all[s:s + span + 1:k] for s in starts])           # (Nw,W,d)
    P = None
    if pressures is not None:
        P = torch.stack([torch.from_numpy(pressures[s:s + span + 1:k]) for s in starts])  # (Nw,W,n_c)
    return Z, P


def _batched_initial_velocity(Z, stride=1):
    """v0 PAR FRAME par différence avant d'ordre 2 (3 points), batché : (B,W,d)→(B,d).
    Les échantillons de la fenêtre sont espacés de `stride` frames, donc la différence
    avant est divisée par stride pour donner une vitesse par-frame (cohérente avec un
    rollout intégré à h=stride). stride=1 → identique à viz.initial_velocity."""
    return (-3.0 * Z[:, 0] + 4.0 * Z[:, 1] - Z[:, 2]) / (2.0 * max(int(stride), 1))


def main():
    init_name, out_name = _apply_cli_overrides()
    config.SAVE_DIR.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    assert getattr(config, 'LNN_FREEZE_ENCODER', False), \
        'finetune_lnn_fixedae suppose LNN_FREEZE_ENCODER=True (AE figé).'
    _use_pressure = getattr(config, 'LNN_PRESSURE', False)

    n_roll   = int(getattr(config, 'LNN_ROLLOUT_STEPS', 16))
    W        = config.SEQ_LEN + n_roll                       # fenêtre = FD + rollout
    epochs   = int(getattr(config, 'LNN_FINETUNE_EPOCHS', 40))
    lr       = float(getattr(config, 'LNN_FINETUNE_LR', 1e-4))
    bs       = int(getattr(config, 'LNN_ROLLOUT_BATCH', 256))
    tbptt    = int(getattr(config, 'LNN_ROLLOUT_TBPTT', 0))
    curric   = bool(getattr(config, 'LNN_ROLLOUT_CURRICULUM', True))
    integ    = getattr(config, 'LNN_ROLLOUT_INTEGRATOR', 'verlet')
    assert W >= 3, f'W={W} < 3 : SEQ_LEN+LNN_ROLLOUT_STEPS trop petit pour v0 (3 points).'
    print(f'Rollout finetune : W={W} (SEQ_LEN={config.SEQ_LEN}+{n_roll}), '
          f'intégrateur={integ}, TBPTT={tbptt or "off"}, curriculum={curric}')

    # ── Données (frames + frontières vidéos + pressions) ─────────────────────
    fds = VideoFrameDataset(
        video_dir=config.VIDEO_DIR, img_size=config.IMG_SIZE,
        n_channels=3 if config.ENC_COLOR else 1,
        rest_video=config.REST_VIDEO, rest_n_frames=config.REST_N_FRAMES,
        rest_first_n_frames=getattr(config, 'REST_FIRST_N_FRAMES', 0),
        crop=getattr(config, 'CROP', None),
        exclude_videos=[config.VAL_VIDEO] if config.VAL_VIDEO else None,
        load_pressure=_use_pressure, pressure_dir=getattr(config, 'PRESSURE_DIR', None),
        pressure_cols=getattr(config, 'PRESSURE_COLS', None),
        pressure_norm=getattr(config, 'PRESSURE_NORM', 101325.0), pressure_dt=config.DT,
        pressure_sync_offsets=getattr(config, 'PRESSURE_SYNC_OFFSETS', None),
        store_uint8=True)

    # ── Encodeur gelé (+ blanchiment latent figé) ────────────────────────────
    enc = build_encoder(config.IMG_SIZE, config.ENC_HIDDEN, config.LATENT_DIM,
                        n_channels=3 if config.ENC_COLOR else 1,
                        normalize=getattr(config, 'ENC_NORMALIZE', False)).to(device)
    enc_path = config.SAVE_DIR / 'encoder_ae.pt'
    if not enc_path.exists():
        enc_path = config.SAVE_DIR / 'encoder.pt'
    assert enc_path.exists(), \
        f'encodeur introuvable : ni encoder_ae.pt ni encoder.pt dans {config.SAVE_DIR}'
    enc.load_state_dict(torch.load(enc_path, map_location=device))
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    print(f'Encodeur chargé et figé : {enc_path.name}')

    whiten = load_latent_whiten(config.SAVE_DIR, device, config.LATENT_DIM)
    if whiten is not None:
        enc = WhitenedEncoder(enc, whiten).to(device).eval()
        print('Encodeur enveloppé : sortie en espace latent blanchi (LatentWhiten).')

    # ── Précalcul de tous les q + lissages (identiques à train_lnn_fixedae) ───
    print('Encodage de toutes les frames (une fois)…')
    z_all = encode_all(enc, fds.frames, device)                        # (N,d) CPU
    if getattr(config, 'SMOOTH_LATENT', False):
        z_all = smooth_latents(
            z_all, fds.video_lengths,
            getattr(config, 'SMOOTH_LATENT_WINDOW', 13),
            getattr(config, 'SMOOTH_LATENT_POLY', 3),
            getattr(config, 'SMOOTH_LATENT_MODE', 'savgol'),
            getattr(config, 'SMOOTH_LATENT_SIGMA', 10.0))
        print(f'Lissage latent ({getattr(config, "SMOOTH_LATENT_MODE", "savgol")}).')
    if _use_pressure and getattr(config, 'SMOOTH_PRESSURE', False):
        fds.pressures = smooth_pressures(fds.pressures, fds.video_lengths)
        print('Pression lissée (même filtre que q).')

    _seq_stride = int(getattr(config, 'SEQ_STRIDE', 1))
    Z, P = build_rollout_windows(z_all, fds.video_lengths, W,
                                 fds.pressures if _use_pressure else None,
                                 stride=_seq_stride)
    Z = Z.to(device)
    P = P.to(device) if P is not None else None
    print(f'  {len(fds)} frames → {Z.shape[0]} fenêtres (W={W}, d={config.LATENT_DIM}'
          + (f', stride={_seq_stride}, h={_seq_stride}' if _seq_stride > 1 else '')
          + (f', pression {P.shape[2]} chambre(s)' if P is not None else '') + ')')

    # ── LNN : construction + CHARGEMENT de lnn.pt (fine-tuning, pas de reset) ─
    lnn = LNN(config.LATENT_DIM, config.LNN_HIDDEN).to(device)
    mode = getattr(config, 'Z_REST_MODE', 'barycenter')
    if mode == 'rest_frame':
        rf = torch.from_numpy(fds.rest_frame).unsqueeze(0).to(device)
        with torch.no_grad():
            zr = enc(rf).squeeze(0)
        lnn.energy.set_z_rest(zr, learnable=False)
    else:
        lnn.energy.set_z_rest(torch.zeros(config.LATENT_DIM, device=device), learnable=True)
    lnn_path = config.SAVE_DIR / init_name
    assert lnn_path.exists(), (f'{init_name} introuvable dans {config.SAVE_DIR} — lancer '
                               f'train_lnn_fixedae.py AVANT le fine-tuning rollout.')
    state = torch.load(lnn_path, map_location=device)
    try:
        lnn.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f'⚠ load_state_dict strict échoué ({e}); repli strict=False.')
        lnn.load_state_dict(state, strict=False)
    # Le LNN raffiné est sauvegardé dans un fichier DÉDIÉ (lnn.pt, le LNN ODE, n'est
    # JAMAIS écrasé → baseline préservée, sélection explicite en aval).
    out_path = config.SAVE_DIR / out_name
    print(f'LNN ODE chargé pour fine-tuning : {lnn_path.name} '
          f'(use_pressure={getattr(lnn, "use_pressure", False)})')
    print(f'Sortie (LNN raffiné) : {out_path.name} — {lnn_path.name} conservé intact.')

    opt = torch.optim.Adam(lnn.parameters(), lr=lr)
    print(f'LNN : {sum(p.numel() for p in lnn.parameters() if p.requires_grad):,} '
          f'paramètres — LR {lr:g}, {epochs} époques, batch {bs}')

    # Métrique de la loss de rollout (None = euclidienne, comportement historique).
    metric_A = load_rollout_metric(device)

    # ── Validation (held-out) chargée une fois (plots debug + held-out final) ─
    def _load_val_data():
        val_path = config.VIDEO_DIR / config.VAL_VIDEO \
            if config.VIDEO_DIR.is_dir() else config.VIDEO_DIR
        vds = VideoFrameDataset(
            video_dir=val_path, img_size=config.IMG_SIZE,
            n_channels=3 if config.ENC_COLOR else 1,
            rest_video=None, rest_n_frames=0, crop=getattr(config, 'CROP', None),
            load_pressure=_use_pressure, pressure_dir=getattr(config, 'PRESSURE_DIR', None),
            pressure_cols=getattr(config, 'PRESSURE_COLS', None),
            pressure_norm=getattr(config, 'PRESSURE_NORM', 101325.0), pressure_dt=config.DT,
            pressure_sync_offsets=getattr(config, 'PRESSURE_SYNC_OFFSETS', None),
            store_uint8=True)
        if _use_pressure and getattr(config, 'SMOOTH_PRESSURE', False):
            vds.pressures = smooth_pressures(vds.pressures, vds.video_lengths)
        return vds, encode_all(enc, vds.frames, device).to(device)

    val_data = None
    if config.VAL_VIDEO:
        try:
            val_data = _load_val_data()
            print(f'Validation (plots + held-out) : {config.VAL_VIDEO} '
                  f'({val_data[1].shape[0]} frames)')
        except Exception as e:
            print(f'Chargement validation échoué ({type(e).__name__}: {e}).')

    # ── Plots intermédiaires (rollout d'INFÉRENCE simulate_rk4, comme fixedae) ─
    LNN_PLOT_EVERY = getattr(config, 'LNN_PLOT_EVERY', 0)
    debug_dir = config.SAVE_DIR / f'debug_plots_{out_path.stem}'
    if LNN_PLOT_EVERY > 0:
        debug_dir.mkdir(exist_ok=True)
    splits = np.cumsum([0] + list(fds.video_lengths))
    s0, e0 = int(splits[0]), int(splits[1])

    def _rollout_plot(z_e, sim_ds, start, n, title, save_path):
        _max = getattr(config, 'VIZ_MAX_FRAMES', None)
        if _max is not None and n > _max:
            n, z_e = _max, z_e[:_max]
        v0  = initial_velocity(z_e)
        p_s = get_sim_pressure(lnn, sim_ds, start, n, device) if _use_pressure else None
        z_s = simulate_rk4(lnn, z_e[0], v0, n_steps=n, dt=1.0, pressure=p_s)
        return plot_latent_rollout(z_e.cpu().numpy(), z_s.cpu().numpy(), config.DT,
                                   title, save_path)

    def _plot_intermediate(ep):
        lnn.eval()
        tag = f'epoch_{ep + 1:04d}'
        _rollout_plot(z_all[s0:e0].to(device), fds, s0, e0 - s0,
                      f'epoch {ep + 1} — vidéo 0',
                      debug_dir / f'{tag}_rollout_train0.png')
        if val_data is not None:
            vds_val, z_v = val_data
            _rollout_plot(z_v, vds_val, 0, len(z_v),
                          f'epoch {ep + 1} — val {config.VAL_VIDEO}',
                          debug_dir / f'{tag}_rollout_val.png')
        lnn.train()

    # ── Boucle de fine-tuning rollout ────────────────────────────────────────
    N = Z.shape[0]
    wpe = int(getattr(config, 'LNN_ROLLOUT_WINDOWS_PER_EPOCH', 0))
    H_min = min(W, config.SEQ_LEN + 2)     # horizon de départ du curriculum
    if 0 < wpe < N:
        print(f'{wpe} fenêtres tirées par époque (sur {N}) — tirage renouvelé à chaque '
              f'époque, {int(np.ceil(wpe / bs))} lot(s)/époque.')
    losses = []
    _t_start = time.time()
    for ep in range(epochs):
        # Curriculum : horizon court → W sur la 1re moitié des époques.
        if curric and epochs > 1:
            frac = min(1.0, ep / max(1, epochs // 2))
            H = int(round(H_min + frac * (W - H_min)))
            H = max(3, min(W, H))
        else:
            H = W
        perm = torch.randperm(N, device=device)
        if 0 < wpe < N:
            perm = perm[:wpe]
        lnn.train()
        ep_loss, nb, nskip = 0.0, 0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            Zb = Z[idx]                                          # (b,W,d)
            z0 = Zb[:, 0]
            v0 = _batched_initial_velocity(Zb, _seq_stride)
            p_seq = P[idx][:, :H] if P is not None else None
            try:
                z_sim = rollout(lnn, z0, v0, H, p_seq, integ, tbptt, h=float(_seq_stride))   # (b,H,d)
            except torch.linalg.LinAlgError:
                nskip += 1
                continue                                        # M̃ singulière (métrique courbe) → skip lot
            loss = rollout_loss(z_sim, Zb[:, :H], metric_A)
            if not torch.isfinite(loss):
                nskip += 1
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lnn.parameters(), config.LNN_GRAD_CLIP)
            opt.step()
            ep_loss += loss.item() * len(idx)
            nb += len(idx)
        ep_loss = ep_loss / nb if nb else float('nan')
        losses.append(ep_loss)
        if ep % max(1, epochs // 50) == 0 or ep == epochs - 1:
            print(f'  epoch {ep:4d}  H={H:3d}  rollout MSE = {ep_loss:.4e}'
                  f'  {(time.time() - _t_start) / (ep + 1):.1f}s/ep'
                  + (f'  ({nskip} lot(s) sautés)' if nskip else ''))
        if LNN_PLOT_EVERY > 0 and (ep + 1) % LNN_PLOT_EVERY == 0:
            try:
                _plot_intermediate(ep)
            except Exception as e:
                print(f'  [epoch {ep + 1}] plot intermédiaire échoué '
                      f'({type(e).__name__}: {e})')
        # Checkpoint atomique à chaque époque → lnn_rollout.pt reflète le dernier état
        # raffiné (lnn.pt intact). Écriture atomique.
        _tmp = out_path.with_suffix('.pt.tmp')
        torch.save(lnn.state_dict(), _tmp)
        _tmp.replace(out_path)

    print(f'LNN raffiné (rollout) sauvegardé : {out_path} '
          f'(LNN ODE original intact : {lnn_path.name})')

    # ── Courbe de perte + held-out final ─────────────────────────────────────
    lnn.eval()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, color='tomato')
    ax.set_yscale('log'); ax.set_xlabel('époque'); ax.set_ylabel('rollout MSE')
    ax.set_title('Fine-tuning LNN par rollout'); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_png = config.SAVE_DIR / f'{out_path.stem}_loss.png'
    fig.savefig(loss_png, dpi=130)
    plt.close(fig)
    print(f'Courbe de perte : {loss_png}')

    if config.VAL_VIDEO:
        try:
            vds, z_val = val_data if val_data is not None else _load_val_data()
            z0, v0 = z_val[0], initial_velocity(z_val)
            p_sim = get_sim_pressure(lnn, vds, 0, len(z_val), device) if _use_pressure else None
            z_sim = simulate_rk4(lnn, z0, v0, n_steps=len(z_val), dt=1.0, pressure=p_sim)
            mse = plot_latent_rollout(
                z_val.cpu().numpy(), z_sim.cpu().numpy(), config.DT,
                f'Held-out {config.VAL_VIDEO} (LNN rollout)',
                config.SAVE_DIR / f'{out_path.stem}_heldout.png')
            print(f'Held-out {config.VAL_VIDEO} : rollout {len(z_val)} pas, '
                  f'MSE(z_sim,z_enc)={mse:.4e}, fini={torch.isfinite(z_sim).all().item()}')
        except Exception as e:
            print(f'Held-out ignoré ({type(e).__name__}: {e})')


if __name__ == '__main__':
    main()
