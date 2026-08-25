"""
train_lnn_fixedae.py — réentraîne le LNN avec AUTOENCODEUR GELÉ et q(t) PRÉCALCULÉ.

Contrairement à `train_lnn.py` (qui ré-encode chaque batch à chaque époque, même encodeur
gelé), on encode TOUTES les frames UNE SEULE FOIS → `z_all`, puis on forme les fenêtres
directement dans l'espace latent. Aucun appel encodeur ni décodeur dans la boucle
d'entraînement → seul le LNN tourne. Pensé pour la métrique courbe figée
(`LNN_METRIC_FROM_DECODER=True`, masse fixe non apprise), mais marche pour tout LNN à AE figé.

Le décodeur est gelé par construction : il n'intervient que via la géométrie précalculée
(`metric_geom.pt`, cf. `precompute_metric_geom.py`), chargée comme buffers par le LNN.

Lancer :  py train_lnn_fixedae.py --config ../cases/krauss2026_2seg_npz/config.py
Prérequis : encoder.pt + (si métrique courbe) metric_geom.pt dans SAVE_DIR.
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

# ── Pondération du résidu par la métrique de visibilité Ḡ (opt-in) ──────────
# Mêmes clés que l'étage rollout (LNN_ROLLOUT_METRIC / _FILE / _RIDGE) : un seul
# interrupteur couvre les deux étages, et `load_rollout_metric` est partagée.
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--metric', dest='metric', action='store_true', default=None,
                 help='pondère le résidu d\'Euler-Lagrange par la métrique de VISIBILITÉ '
                      'Ḡ (visibility_metric.pt, cf. compute_visibility_metric.py) au lieu '
                      'd\'une MSE latente isotrope. Équivaut à config.LNN_ROLLOUT_METRIC=True.')
_ap.add_argument('--no-metric', dest='metric', action='store_false',
                 help='force le résidu isotrope même si config.LNN_ROLLOUT_METRIC=True')
_ap.add_argument('--metric-ridge', type=float, default=None,
                 help='ρ de A ∝ Ḡ + ρ·λmax·I (déf. config.LNN_ROLLOUT_METRIC_RIDGE=0.01)')
_args, _ = _ap.parse_known_args()
if _args.metric is not None:
    config.LNN_ROLLOUT_METRIC = bool(_args.metric)
if _args.metric_ridge is not None:
    config.LNN_ROLLOUT_METRIC_RIDGE = float(_args.metric_ridge)

from dataset import VideoFrameDataset
from models import LNN, build_encoder, WhitenedEncoder, load_latent_whiten
from viz import (simulate_rk4, initial_velocity, get_sim_pressure,
                 plot_lnn_training, plot_trajectories,
                 plot_trajectory_validation, plot_energy_map)


@torch.no_grad()
def encode_all(encoder, frames_np, device, bs=128):
    """Encode toutes les frames une fois → (N, d) sur CPU."""
    zs = []
    for i in range(0, len(frames_np), bs):
        x = torch.from_numpy(frames_np[i:i + bs]).to(device)
        if x.dtype == torch.uint8:        # frames stockées en uint8 [0,255] (store_uint8)
            x = x.float().div_(255.0)
        zs.append(encoder(x).cpu())
    return torch.cat(zs, 0)


def smooth_latents(z_all, video_lengths, window, poly, mode='savgol', sigma=None):
    """Lisse q(t) PAR VIDÉO (aucun lissage à cheval sur deux vidéos).

    z_all : (N, d) tensor CPU — trajectoires latentes concaténées.
    mode  : 'savgol'   → Savitzky-Golay (fit polynomial glissant, `window`/`poly`) ;
            'gaussian' → convolution par noyau gaussien (passe-bas monotone, `sigma`
                         en frames, bords 'nearest'). Plus fort/plus « propre » que
                         savgol à support comparable (pas de ripple), comme le lissage
                         des pressions voulu par l'utilisateur.
    window : fenêtre savgol (frames) ramenée à un entier impair ≥ 5, plafonnée à vlen.
    sigma  : écart-type gaussien (frames) ; si None, ignoré.
    Retourne (N, d) tensor CPU lissé. Motif repris de select_truncation.py:114-120.
    """
    z_np = z_all.numpy().copy()
    off = 0
    for vlen in video_lengths:
        seg = z_np[off:off + vlen]
        if mode == 'gaussian':
            from scipy.ndimage import gaussian_filter1d
            if sigma is not None and sigma > 0 and vlen > 1:
                z_np[off:off + vlen] = gaussian_filter1d(
                    seg, float(sigma), axis=0, mode='nearest')
        else:
            from scipy.signal import savgol_filter
            win = min(int(window), vlen if vlen % 2 == 1 else vlen - 1)
            win = max(5, win if win % 2 == 1 else win - 1)
            if win > poly and win <= vlen:
                z_np[off:off + vlen] = savgol_filter(seg, win, poly, axis=0)
        off += vlen
    return torch.from_numpy(z_np)


def smooth_pressures(press_np, video_lengths):
    """Lisse la pression frame-alignée (N, n_c) avec EXACTEMENT le même filtre que
    q(t) (SMOOTH_LATENT_MODE / _SIGMA / _WINDOW / _POLY), PAR VIDÉO. Opt-in via
    SMOOTH_PRESSURE. No-op si SMOOTH_PRESSURE=False ou press_np None. Cohérence du
    forçage b(q)ᵀP entre le résidu et les rollouts (build_windows + get_sim_pressure).
    Retourne (N, n_c) numpy (dtype préservé)."""
    if press_np is None or not getattr(config, 'SMOOTH_PRESSURE', False):
        return press_np
    smode = getattr(config, 'SMOOTH_LATENT_MODE', 'savgol')
    win   = getattr(config, 'SMOOTH_LATENT_WINDOW', 13)
    poly  = getattr(config, 'SMOOTH_LATENT_POLY', 3)
    # σ dédié à la pression ; repli sur celui de q si absent/None.
    sigma = getattr(config, 'SMOOTH_PRESSURE_SIGMA', None)
    if sigma is None:
        sigma = getattr(config, 'SMOOTH_LATENT_SIGMA', 10.0)
    out = smooth_latents(torch.from_numpy(np.ascontiguousarray(press_np)),
                         video_lengths, win, poly, smode, sigma).numpy()
    return out.astype(press_np.dtype, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Métrique de VISIBILITÉ Ḡ des losses latentes (opt-in, LNN_ROLLOUT_METRIC)
#  Utilisée par le résidu FD (ici) ET par le rollout (finetune_lnn_fixedae,
#  qui importe ces deux fonctions) : une seule implémentation, mêmes clés.
# ─────────────────────────────────────────────────────────────────────────────
def load_rollout_metric(device):
    """Retourne `A` (d,d) SPD, la métrique de la loss de rollout, ou None (=identité).

    Par défaut la loss est `‖Δu‖²`, EUCLIDIENNE : en espace blanchi chaque direction
    latente pèse pareil, puisque `LatentWhiten` impose cov ≈ I. Mais « variance unité »
    n'est pas « visibilité unité » : un mode peut être de variance 1 dans les données
    tout en ne déplaçant presque aucun pixel, et la MSE isotrope lui fait alors ajuster
    surtout du bruit d'encodage.

    Avec `LNN_ROLLOUT_METRIC=True` on mesure l'écart en métrique de VISIBILITÉ
    `Ḡ = E[(∂I/∂u)ᵀ(∂I/∂u)]` (précalculée par `compute_visibility_metric.py`), qui est
    au premier ordre la MSE image de Krauss et al. 2026 (VON) sans rasteriser :

        Δuᵀ Ḡ Δu  ≈  ‖I(u+Δu) − I(u)‖²

    RIDGE : on n'utilise jamais `Ḡ` nue. Ses directions quasi nulles ne contraindraient
    plus rien, alors qu'un mode invisible peut être RAIDE : il divergerait sans coût
    immédiat, puis contaminerait les modes visibles par le couplage (M et C ne sont pas
    diagonales). D'où `A ∝ Ḡ + ρ·λmax·I`, `ρ = LNN_ROLLOUT_METRIC_RIDGE`, qui borne le
    rapport de pondération à ~1/ρ.

    NORMALISATION : `trace(A) = d`, donc valeur propre moyenne 1. La loss garde
    l'échelle de la version isotrope (à laquelle elle se réduit exactement si `Ḡ ∝ I`),
    et LR / grad-clip restent valides sans réglage.
    """
    if not getattr(config, 'LNN_ROLLOUT_METRIC', False):
        return None
    fname = getattr(config, 'LNN_ROLLOUT_METRIC_FILE', 'visibility_metric.pt')
    path = config.SAVE_DIR / fname
    assert path.exists(), (
        f'{fname} introuvable dans {config.SAVE_DIR} — lancer '
        f'compute_visibility_metric.py AVANT (ou LNN_ROLLOUT_METRIC=False).')
    blob = torch.load(path, map_location=device)
    G = blob['G'].to(device).double()
    assert G.shape == (config.LATENT_DIM, config.LATENT_DIM), (
        f'métrique {tuple(G.shape)} incompatible avec LATENT_DIM={config.LATENT_DIM} '
        f'— la recalculer pour ce cas.')
    G = 0.5 * (G + G.T)                                   # symétrisation numérique
    lam = torch.linalg.eigvalsh(G)
    rho = float(getattr(config, 'LNN_ROLLOUT_METRIC_RIDGE', 0.01))
    A = G + rho * lam.max() * torch.eye(G.shape[0], dtype=G.dtype, device=device)
    A = A * (G.shape[0] / torch.diagonal(A).sum())        # trace(A) = d
    lam_A = torch.linalg.eigvalsh(A)
    print(f'Métrique de rollout : {fname} (source={blob.get("source", "?")}, '
          f'{blob.get("n_samples", "?")} points, rendu {blob.get("res", "?")}²)')
    print(f'  λ(Ḡ) : {[f"{v:.4g}" for v in lam.flip(0).tolist()]}  '
          f'(cond {lam.max() / lam.clamp(min=1e-30).min():.4g})')
    print(f'  ridge ρ={rho:g}·λmax  →  λ(A) normalisée : '
          f'{[f"{v:.4g}" for v in lam_A.flip(0).tolist()]}  '
          f'(cond {lam_A.max() / lam_A.min():.4g})')
    return A.float()


def rollout_loss(z_sim, z_ref, A):
    """MSE de rollout, en métrique `A` si fournie (sinon euclidienne à l'identique).

    Isotrope : mean over (B,H,d) de Δ².     Métrique : mean over (B,H) de ΔᵀAΔ / d.
    Les deux coïncident exactement pour A = I, d'où une bascule sans changement
    d'échelle (cf. normalisation trace(A)=d dans `load_rollout_metric`).
    """
    return metric_mse(z_sim - z_ref, A)


def metric_mse(delta, A):
    """MSE d'un écart latent `delta` (..., d), pondérée par `A` (d,d) SPD ou isotrope.

    Isotrope (A=None) : moyenne de Δ² sur TOUTES les dimensions.
    Métrique          : moyenne sur les axes de tête de ΔᵀAΔ / d.
    Les deux coïncident EXACTEMENT pour A = I (trace(A)=d), d'où une bascule sans
    changement d'échelle : LR et grad-clip restent valides.

    Générique en forme : sert à l'écart de rollout (z_sim − z_ref) comme au résidu
    d'Euler-Lagrange en différences finies, qui vit dans le même espace latent et
    souffre du même biais isotrope (un mode invisible y pèse autant qu'un mode visible).
    """
    if A is None:
        return (delta ** 2).mean()
    return ((delta @ A) * delta).sum(-1).mean() / delta.shape[-1]



def build_windows(z_all, video_lengths, T, pressures=None, stride=1):
    """Fenêtres de T échantillons ESPACÉS de `stride` frames (indices s, s+k, …,
    s+(T-1)k), NE chevauchant PAS deux vidéos ; pression à l'échantillon central.
    stride=1 → frames adjacentes (comportement d'origine)."""
    k = max(int(stride), 1)
    span = (T - 1) * k                        # dernière frame = s + span
    starts, off = [], 0
    for vlen in video_lengths:
        starts += [off + s for s in range(vlen - span)]
        off += vlen
    Z = torch.stack([z_all[s:s + span + 1:k] for s in starts])          # (Nw, T, d)
    P = None
    if pressures is not None:
        P = torch.stack([torch.from_numpy(pressures[s + (T // 2) * k]) for s in starts])  # (Nw, n_c)
    return Z, P


def plot_latent_rollout(z_enc_np, z_sim_np, dt, title, save_path):
    """Plot latent-space : enc(x) vs rollout RK4, une sous-figure par dimension.

    Aucune reconstruction d'image — on compare uniquement les coordonnées q(t).
    `z_enc_np`, `z_sim_np` : (T, d). Retourne la MSE pour affichage.
    """
    d    = z_enc_np.shape[1]
    t    = np.arange(len(z_enc_np)) * dt
    mse  = float(((z_sim_np - z_enc_np) ** 2).mean())
    ncol = min(d, 2); nrow = int(np.ceil(d / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 3 * nrow), squeeze=False)
    for k in range(d):
        ax = axes[k // ncol][k % ncol]
        ax.plot(t, z_enc_np[:, k], color='steelblue', lw=1.3, label='enc(x)')
        ax.plot(t, z_sim_np[:, k], color='tomato', lw=1.3, ls='--', label='RK4')
        ax.set_xlabel('Temps (s)'); ax.set_ylabel(f'z[{k}]')
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    for k in range(d, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.suptitle(f'{title} — MSE(z_sim, z_enc)={mse:.3e}')
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return mse


def main():
    config.SAVE_DIR.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    assert getattr(config, 'LNN_FREEZE_ENCODER', False), \
        'train_lnn_fixedae suppose LNN_FREEZE_ENCODER=True (AE figé).'
    _use_pressure = getattr(config, 'LNN_PRESSURE', False)

    # ── Données (frames + frontières vidéos + pressions) ─────────────────────
    # store_uint8=True : frames uint8 [0,255] (~4× moins de RAM ; conversion float
    # par batch dans encode_all). Indispensable en 256×256 (float32 ≈ 41 GB → OOM).
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

    # ── Encodeur gelé ────────────────────────────────────────────────────────
    enc = build_encoder(config.IMG_SIZE, config.ENC_HIDDEN, config.LATENT_DIM,
                        n_channels=3 if config.ENC_COLOR else 1,
                        normalize=getattr(config, 'ENC_NORMALIZE', False)).to(device)
    # AE figé ⟹ on prend l'encodeur de l'autoencodeur conjoint (train_ae.py) si présent,
    # sinon l'encodeur pré-entraîné seul.
    enc_path = config.SAVE_DIR / 'encoder_ae.pt'
    if not enc_path.exists():
        enc_path = config.SAVE_DIR / 'encoder.pt'
    assert enc_path.exists(), \
        f'encodeur introuvable : ni encoder_ae.pt ni encoder.pt dans {config.SAVE_DIR}'
    enc.load_state_dict(torch.load(enc_path, map_location=device))
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    print(f'Encodeur chargé et figé : {enc_path}')

    # ── Blanchiment latent post-hoc (figé) ───────────────────────────────────
    # Si LATENT_WHITEN actif (+ latent_whiten.pt présent), on enveloppe l'encodeur
    # ⟹ enc(x) renvoie u = W(z−μ). z_all, z_rest, validation et plots vivent donc
    # tous en espace blanchi, cohérent avec metric_geom transporté en u
    # (precompute_metric_geom.py). None ⟹ comportement inchangé (espace z brut).
    whiten = load_latent_whiten(config.SAVE_DIR, device, config.LATENT_DIM)
    if whiten is not None:
        enc = WhitenedEncoder(enc, whiten).to(device).eval()
        print('Encodeur enveloppé : sortie en espace latent blanchi (LatentWhiten).')

    # ── PRÉCALCUL de tous les q (une seule fois) + fenêtres en latent ────────
    print('Encodage de toutes les frames (une fois)…')
    z_all = encode_all(enc, fds.frames, device)                        # (N, d) CPU
    # ── Lissage (une seule fois) des trajectoires latentes q(t) ──────────────
    # AE figé ⟹ q(t) précalculé une fois : on peut le lisser ici, PAR VIDÉO, pour
    # atténuer le bruit d'encodage avant de former les fenêtres du résidu EL.
    if getattr(config, 'SMOOTH_LATENT', False):
        smode = getattr(config, 'SMOOTH_LATENT_MODE', 'savgol')
        win   = getattr(config, 'SMOOTH_LATENT_WINDOW', 13)
        poly  = getattr(config, 'SMOOTH_LATENT_POLY', 3)
        sigma = getattr(config, 'SMOOTH_LATENT_SIGMA', 10.0)
        z_all = smooth_latents(z_all, fds.video_lengths, win, poly, smode, sigma)
        if smode == 'gaussian':
            print(f'Lissage latent (gaussien) : sigma {sigma} frames, '
                  f'{len(fds.video_lengths)} vidéo(s).')
        else:
            print(f'Lissage latent (Savitzky-Golay) : fenêtre {win}, ordre {poly}, '
                  f'{len(fds.video_lengths)} vidéo(s).')
    # ── Lissage de la pression (même filtre que q, opt-in SMOOTH_PRESSURE) ────
    # In-place sur fds.pressures ⟹ propagé aux fenêtres du résidu ET aux rollouts
    # (get_sim_pressure relit fds.pressures).
    if _use_pressure and getattr(config, 'SMOOTH_PRESSURE', False):
        fds.pressures = smooth_pressures(fds.pressures, fds.video_lengths)
        smode = getattr(config, 'SMOOTH_LATENT_MODE', 'savgol')
        sig   = getattr(config, 'SMOOTH_PRESSURE_SIGMA', None)
        if sig is None:
            sig = getattr(config, 'SMOOTH_LATENT_SIGMA', 10.0)
        win   = getattr(config, 'SMOOTH_LATENT_WINDOW', 13)
        print(f'Lissage pression ({smode}) : '
              + (f'sigma {sig} frames' if smode == 'gaussian' else f'fenêtre {win}')
              + f', {len(fds.video_lengths)} vidéo(s).')
    T = config.SEQ_LEN
    _seq_stride = int(getattr(config, 'SEQ_STRIDE', 1))
    Z, P = build_windows(z_all, fds.video_lengths, T,
                         fds.pressures if _use_pressure else None,
                         stride=_seq_stride)
    if _seq_stride > 1:
        print(f'  Sous-échantillonnage FD : SEQ_STRIDE={_seq_stride} '
              f'(fenêtres de {T} échantillons espacés de {_seq_stride}, dt={_seq_stride})')
    Z = Z.to(device)
    P = P.to(device) if P is not None else None
    print(f'  {len(fds)} frames → {Z.shape[0]} fenêtres (T={T}, d={config.LATENT_DIM})'
          + (f', pression {P.shape[1]} chambre(s)' if P is not None else ''))

    # ── LNN ──────────────────────────────────────────────────────────────────
    lnn = LNN(config.LATENT_DIM, config.LNN_HIDDEN).to(device)
    mode = getattr(config, 'Z_REST_MODE', 'barycenter')
    if mode == 'rest_frame':
        rf = torch.from_numpy(fds.rest_frame).unsqueeze(0).to(device)
        with torch.no_grad():
            zr = enc(rf).squeeze(0)
        lnn.energy.set_z_rest(zr, learnable=False)
        print(f'z_rest depuis rest_frame : {zr.cpu().numpy().round(3)}')
    else:
        lnn.energy.set_z_rest(torch.zeros(config.LATENT_DIM, device=device), learnable=True)
        print('z_rest = 0 (barycentre, apprenable)')
    if lnn.metric is not None:
        print(f'Métrique COURBE active — m={float(lnn.metric.log_m.exp()):.3f} (FIXE), '
              f'α={float(lnn.log_alpha_ray.exp()):.5f}, Linv_raw={lnn.Linv_raw}')

    # Métrique de la loss : Ḡ (visibilité) si LNN_ROLLOUT_METRIC, sinon isotrope (None).
    metric_A = load_rollout_metric(device)

    opt = torch.optim.Adam(lnn.parameters(), lr=config.LNN_LR)
    print(f'LNN : {sum(p.numel() for p in lnn.parameters() if p.requires_grad):,} '
          f'paramètres entraînables')

    # ── Chargement de la vidéo de validation (held-out) ──────────────────────
    # VAL_VIDEO est EXCLUE de fds (donc absente de z_all) : on la charge/encode UNE
    # fois ici pour la réutiliser dans les plots de debug ET le held-out final.
    def _load_val_data():
        """Charge VAL_VIDEO → (vds, z_val sur device). Encodage une seule fois."""
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
        # Même lissage de pression que l'entraînement (rollout val cohérent).
        if _use_pressure and getattr(config, 'SMOOTH_PRESSURE', False):
            vds.pressures = smooth_pressures(vds.pressures, vds.video_lengths)
        return vds, encode_all(enc, vds.frames, device).to(device)

    val_data = None
    if config.VAL_VIDEO:
        try:
            val_data = _load_val_data()
            print(f'Vidéo de validation (plots debug + held-out) : {config.VAL_VIDEO} '
                  f'({val_data[1].shape[0]} frames)')
        except Exception as e:
            print(f'Chargement validation échoué ({type(e).__name__}: {e}) '
                  f'— plots debug sur vidéo train 0')

    # ── Plots intermédiaires (latent-space only, aucun décodeur) ─────────────
    # Gated par LNN_PLOT_EVERY (0 = off). DEUX rollouts par époque : la 1re vidéo
    # d'ENTRAÎNEMENT (vidéo 0) ET la vidéo de VALIDATION (held-out, si disponible),
    # un fichier chacun. Pas de ré-encodage par époque : z_all / z_val déjà encodés,
    # juste une intégration RK4 du LNN.
    LNN_PLOT_EVERY = getattr(config, 'LNN_PLOT_EVERY', 0)
    debug_dir = config.SAVE_DIR / 'debug_plots_lnn_fixedae'
    if LNN_PLOT_EVERY > 0:
        debug_dir.mkdir(exist_ok=True)
    splits = np.cumsum([0] + list(fds.video_lengths))       # frontières vidéos dans z_all
    s0, e0 = int(splits[0]), int(splits[1])                 # 1re vidéo d'entraînement

    def _rollout_plot(z_e, sim_ds, start, n, title, save_path):
        """Un rollout RK4 depuis z_e[0] sur n pas + plot enc(x) vs RK4."""
        # Plafond VIZ_MAX_FRAMES (comme train_lnn._get_z_sim) : sur la vidéo Krauss
        # (~55 230 frames) un rollout pleine longueur diverge (NaN) sur un LNN à peine
        # entraîné → torch.linalg.solve « singular ». On borne enc(x) ET le rollout.
        _max = getattr(config, 'VIZ_MAX_FRAMES', None)
        if _max is not None and n > _max:
            n, z_e = _max, z_e[:_max]
        v0  = initial_velocity(z_e)
        p_s = (get_sim_pressure(lnn, sim_ds, start, n, device) if _use_pressure else None)
        z_s = simulate_rk4(lnn, z_e[0], v0, n_steps=n, dt=1.0, pressure=p_s)
        plot_latent_rollout(z_e.cpu().numpy(), z_s.cpu().numpy(), config.DT,
                            title, save_path)

    def _plot_intermediate(ep):
        lnn.eval()
        tag = f'epoch_{ep + 1:04d}'
        # (1) vidéo d'entraînement 0
        _rollout_plot(z_all[s0:e0].to(device), fds, s0, e0 - s0,
                      f'epoch {ep + 1} — vidéo 0',
                      debug_dir / f'{tag}_rollout_train0.png')
        # (2) vidéo de validation (held-out), si disponible
        if val_data is not None:
            vds_val, z_v = val_data                          # z_v déjà sur device
            _rollout_plot(z_v, vds_val, 0, len(z_v),
                          f'epoch {ep + 1} — val {config.VAL_VIDEO}',
                          debug_dir / f'{tag}_rollout_val.png')
        lnn.train()

    # ── Boucle d'entraînement (LNN seul, aucun encodeur/décodeur) ────────────
    N, bs = Z.shape[0], config.LNN_BATCH
    losses = []
    for ep in range(config.LNN_EPOCHS):
        perm = torch.randperm(N, device=device)
        lnn.train()
        ep_loss, nb = 0.0, 0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            r = lnn.residual(Z[idx], dt=float(_seq_stride),
                             pressure=(P[idx] if P is not None else None))
            loss = metric_mse(r, metric_A)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lnn.parameters(), config.LNN_GRAD_CLIP)
            opt.step()
            ep_loss += loss.item() * len(idx)
            nb += len(idx)
        ep_loss /= nb
        losses.append(ep_loss)
        if ep % max(1, config.LNN_EPOCHS // 50) == 0 or ep == config.LNN_EPOCHS - 1:
            print(f'  epoch {ep:4d}  résidu² = {ep_loss:.4e}')
        if LNN_PLOT_EVERY > 0 and (ep + 1) % LNN_PLOT_EVERY == 0:
            try:
                _plot_intermediate(ep)
            except Exception as e:
                print(f'  [epoch {ep + 1}] plot intermédiaire échoué '
                      f'({type(e).__name__}: {e})')
        # Checkpoint à CHAQUE époque : lnn.pt reflète toujours le dernier état
        # entraîné (le run ne sauvait qu'à la toute fin → aucun poids exploitable
        # en cours de route ni en cas d'interruption). Écriture atomique.
        # Sous Windows, os.replace échoue (WinError 5) si un tiers — antivirus,
        # indexeur, explorateur — tient le fichier ouvert au même instant. C'est
        # transitoire, mais sur 10 000 époques × 1 sauvegarde/époque ça finit par
        # tomber, et ça tuait tout le run. On réessaie brièvement ; en dernier
        # recours on saute CETTE sauvegarde (l'époque suivante réécrira).
        _ckpt = config.SAVE_DIR / 'lnn.pt'
        _tmp = _ckpt.with_suffix('.pt.tmp')
        torch.save(lnn.state_dict(), _tmp)
        for _try in range(5):
            try:
                _tmp.replace(_ckpt)
                break
            except PermissionError:
                time.sleep(0.2 * (_try + 1))
        else:
            print(f'  [epoch {ep + 1}] checkpoint verrouillé — sauvegarde sautée')

    torch.save(lnn.state_dict(), config.SAVE_DIR / 'lnn.pt')
    print(f'LNN sauvegardé : {config.SAVE_DIR / "lnn.pt"}')

    # ── Visualisations ───────────────────────────────────────────────────────
    # Mêmes plots que train_lnn.py (réutilisés depuis viz), à AE figé : la
    # trajectoire/énergie sont évaluées en ré-encodant les frames d'entraînement.
    lnn.eval()
    out = config.SAVE_DIR

    fig_train = plot_lnn_training(losses)
    fig_train.savefig(out / 'lnn_fixedae_loss.png', dpi=130)
    plt.close(fig_train)
    print(f'Courbe de perte       : {out / "lnn_fixedae_loss.png"}')

    z_all_np = z_all.cpu().numpy()                                     # (N, d)
    fig_traj = plot_trajectories(z_all_np, fds.video_lengths, fds.indices)
    fig_traj.savefig(out / 'lnn_fixedae_trajectories.png', dpi=130)
    plt.close(fig_traj)
    print(f'Trajectoires latentes : {out / "lnn_fixedae_trajectories.png"}')

    # Trajectoire RK4 simulée vs encodée (vidéos d'entraînement 0/1)
    try:
        fig_val = plot_trajectory_validation(
            lnn, enc, fds, device, dt=config.DT, video_idx=0,
            max_frames=getattr(config, 'VIZ_MAX_FRAMES', None))
        fig_val.savefig(out / 'lnn_fixedae_validation.png', dpi=130)
        plt.close(fig_val)
        print(f'Validation rollout    : {out / "lnn_fixedae_validation.png"}')
    except Exception as e:
        print(f'Plot validation ignoré ({type(e).__name__}: {e})')

    # Carte d'énergie E(z) (seulement lisible en d ≤ 2)
    if config.LATENT_DIM <= 2:
        try:
            T0 = fds.video_lengths[0]
            v0 = initial_velocity(z_all[:T0].to(device))
            z_sim_dbg = simulate_rk4(lnn, z_all[0].to(device), v0,
                                     n_steps=T0, dt=1.0).cpu().numpy()
            fig_E = plot_energy_map(lnn, z_all_np, fds.video_lengths, device,
                                    z_sim=z_sim_dbg)
            fig_E.savefig(out / 'lnn_fixedae_energy.png', dpi=130)
            plt.close(fig_E)
            print(f'Carte d\'énergie       : {out / "lnn_fixedae_energy.png"}')
        except Exception as e:
            print(f'Plot énergie ignoré ({type(e).__name__}: {e})')

    # ── Validation : rollout sur VAL_VIDEO (best-effort) ─────────────────────
    if config.VAL_VIDEO:
        try:
            # Réutilise la vidéo de val déjà chargée/encodée (sinon recharge).
            vds, z_val = val_data if val_data is not None else _load_val_data()
            lnn.eval()
            z0, v0 = z_val[0], initial_velocity(z_val)
            p_sim = get_sim_pressure(lnn, vds, 0, len(z_val), device) if _use_pressure else None
            z_sim = simulate_rk4(lnn, z0, v0, n_steps=len(z_val), dt=1.0, pressure=p_sim)
            mse = plot_latent_rollout(
                z_val.cpu().numpy(), z_sim.cpu().numpy(), config.DT,
                f'Validation held-out {config.VAL_VIDEO}',
                config.SAVE_DIR / 'lnn_fixedae_heldout.png')
            print(f'Validation {config.VAL_VIDEO} : rollout {len(z_val)} pas, '
                  f'MSE(z_sim, z_enc) = {mse:.4e}, fini={torch.isfinite(z_sim).all().item()}')
            print(f'Plot held-out         : {config.SAVE_DIR / "lnn_fixedae_heldout.png"}')
        except Exception as e:
            print(f'Validation ignorée ({type(e).__name__}: {e})')


if __name__ == '__main__':
    main()
