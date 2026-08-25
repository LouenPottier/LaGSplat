"""
eval_freq_rest_cycles.py — trois critères d'erreur PHYSIQUES du LNN, mesurés dans
l'espace latent contre la référence `q_enc(t) = enc(x_t)`.

Motivation : la MSE image à 0.5 s (`eval_multistep_mse.py`) est la métrique du
head-to-head Krauss, mais elle mélange tout (rendu, phase, amplitude, dérive) sur
un horizon trop court pour dire si le modèle a la bonne PHYSIQUE. Les trois
critères ci-dessous isolent chacun une propriété vérifiable, et prennent tous
`enc(x)` comme vérité terrain — c'est la seule référence disponible (pas de
capteur d'état sur ces systèmes) :

  (1) ERREUR DE FRÉQUENCE du MODE VISIBLE, en oscillation LIBRE.
      Sur chaque intervalle où la pression d'entrée est CONSTANTE (pour un cas
      non forcé : toute la vidéo), on estime la fréquence dominante de `q_enc(t)`
      et celle du rollout parti de la même condition initiale, avec le MÊME
      estimateur (FFT fenêtrée Hann + interpolation parabolique du pic sur le
      log-spectre). Erreur relative |f_sim − f_enc| / f_enc.
      C'est le test du POTENTIEL et de la MASSE : ω² = eig(M⁻¹∇²V).

      ⚠️ DEUX PRÉCAUTIONS, sans lesquelles le chiffre ne veut rien dire :
      - **Pondérer par la VISIBILITÉ.** Les `d` composantes latentes ne s'agrègent
        PAS à poids égal : en espace blanchi chaque direction a variance 1 mais
        pas visibilité 1 (rapport mesuré jusqu'à 341× entre modes, cf.
        `visibility_metric.pt`). Une somme isotrope peut donc retourner la
        fréquence d'un mode qui ne déplace quasiment aucun pixel. On agrège donc
        par la forme quadratique de `Ḡ = E[(∂I/∂u)ᵀ(∂I/∂u)]` (métrique de
        visibilité du DÉCODEUR, `compute_visibility_metric.py`), qui est au
        premier ordre le spectre de puissance en espace IMAGE. Contrôle
        indépendant reporté à côté : la fréquence de la seule projection sur le
        1ᵉʳ vecteur propre de `Ḡ` (le mode le plus visible).
      - **Vraiment LIBRE.** Mesurer sur un intervalle où `P` varie encore
        donnerait la fréquence du FORÇAGE, pas un mode propre. La détection
        d'intervalle impose donc deux conditions (|ΔP| par frame ET excursion
        totale de `P`), sur la pression BRUTE, et l'excursion réellement obtenue
        est affichée pour être vérifiable.

  (2) ERREUR DE POSITION DE REPOS.
      `enc(x)` constant ⟹ position d'équilibre. Par intervalle libre, on prend
      `q_eq_enc` = moyenne de `q_enc` sur la plus longue QUEUE quasi-statique
      (vitesse sous le quantile `--static-quantile` des vitesses de la vidéo) ;
      à défaut (oscillation encore vive), moyenne sur un nombre ENTIER de
      périodes en fin d'intervalle, qui annule l'oscillation au 1er ordre
      (marqué `cycle-mean` dans la sortie).
      Côté modèle, deux estimateurs de l'équilibre sous la MÊME pression :
        - `relax` : relaxation suramortie `q ← q + h·accel(q, q̇=0, P)` à pas
          adaptatif, partie de `q_eq_enc` ⟹ équilibre STABLE le plus proche.
          Mesure donc directement « `q_eq_enc` est-il une racine de la force ? ».
        - `roll`  : moyenne sur la dernière période d'un rollout dynamique long
          (`--relax-periods` périodes) sous pression constante ⟹ vérifie en plus
          que le modèle S'Y POSE (dissipation correcte), au lieu d'osciller
          indéfiniment ou de diverger.
      C'est le test du POTENTIEL et du FORÇAGE `b(q)ᵀP` (équilibre chargé).

  (3) ERREUR MOYENNE SUR ~5 PÉRIODES.
      MSE latente `‖q_sim − q_enc‖²/d` d'un rollout de `--n-periods` périodes
      (période mesurée sur `enc(x)`), forcé par la pression MESURÉE, moyennée sur
      `--n-windows` départs tirés au hasard. Horizon ~20× plus long que le 0.5 s
      du benchmark, donc sensible à la dérive de phase et à l'enveloppe
      d'amortissement, que 0.5 s ne voit pas.
      Rapportée aussi en NRMSE (normalisée par l'écart-type de `q_enc`), seul
      chiffre comparable d'un cas test à l'autre : l'échelle du latent est
      arbitraire (blanchiment), donc une MSE latente brute ne se compare pas
      entre deux cas.

⚠️ COUPLER LE CHECKPOINT ET SES RÉGLAGES. `--sigma` / `--sigma-pressure` /
`--cq-eps` doivent valoir ceux de l'ENTRAÎNEMENT du checkpoint évalué (le lissage
fixe les conditions initiales, `c₀` fait partie de la dynamique). Le point de
fonctionnement rapporté est celui du README (section « Operating point ») ;
`scripts/run_2seg_npz.py` le repasse tel quel à cette évaluation.

Lancer :
    py eval_freq_rest_cycles.py --config ../cases/krauss2026_2seg_npz/config.py \
       --lnn lnn_lr1e-3_c1_s1_500ep_seed0.pt --sigma 1 --sigma-pressure 1 \
       --cq-eps 1 --video scr_2seg_32x32_val20.npz --json freq2T_seed0.json

Sorties : impression du récapitulatif, `<SAVE_DIR>/freq_rest_cycles[_tag].png`
(4 panneaux de diagnostic) et un JSON si `--json`.
"""
import argparse
import json

from _bootstrap import load_config
config = load_config()

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--lnn', type=str, default='auto',
                 help="checkpoint LNN dans SAVE_DIR ('auto' = lnn_rollout.pt sinon lnn.pt)")
_ap.add_argument('--encoder', type=str, default='auto',
                 help="checkpoint encodeur ('auto' = encoder_ae.pt sinon encoder.pt)")
_ap.add_argument('--video', type=str, default=None,
                 help='vidéo/NPZ à évaluer (défaut : VIDEO_DIR). Passer la held-out.')
_ap.add_argument('--frames', type=int, default=12000, help='nb de frames chargées/encodées')
_ap.add_argument('--skip-sec', type=float, default=8.0,
                 help='tête statique ignorée (avant le 1ᵉʳ échelon de pression ; déf. 8 s)')
# ── (1) fréquence ───────────────────────────────────────────────────────────
_ap.add_argument('--min-free-sec', type=float, default=2.0,
                 help='durée minimale (s) d\'un intervalle à pression constante (déf. 2)')
_ap.add_argument('--n-free', type=int, default=20,
                 help='nb max d\'intervalles libres analysés, les plus longs d\'abord (déf. 20)')
_ap.add_argument('--metric', type=str, default='visibility',
                 choices=['visibility', 'isotropic'],
                 help="métrique d'agrégation des d composantes latentes. "
                      "'visibility' (défaut) = forme quadratique de Ḡ "
                      '(visibility_metric.pt) ⟹ on mesure le mode VISIBLE, au '
                      'premier ordre le spectre en espace image. '
                      "'isotropic' = poids uniforme (diagnostic ; faux dès que Ḡ "
                      'est mal conditionnée)')
_ap.add_argument('--min-osc-frac', type=float, default=0.05,
                 help='amplitude minimale de q_enc dans une fenêtre pour que sa '
                      'fréquence ait un sens, en fraction du RMS de q_enc (déf. 0.05)')
_ap.add_argument('--freq-periods', type=float, default=8.0,
                 help='longueur de la fenêtre d\'analyse spectrale, en périodes (déf. 8)')
_ap.add_argument('--pressure-tol', type=float, default=1e-2,
                 help='|ΔP| par frame sous lequel P est dite constante, en fraction '
                      'de l\'amplitude de P (déf. 1e-2)')
_ap.add_argument('--pressure-span', type=float, default=0.05,
                 help='excursion TOTALE tolérée de P sur un intervalle libre, en '
                      'fraction de l\'amplitude (déf. 0.05) ; élimine la dérive lente')
# ── (2) position de repos ───────────────────────────────────────────────────
_ap.add_argument('--static-quantile', type=float, default=0.10,
                 help='quantile des vitesses ‖q̇_enc‖ définissant « quasi-statique » (déf. 0.10)')
_ap.add_argument('--rest-from-last', type=int, default=0,
                 help="référence d'équilibre = moyenne des N DERNIÈRES frames de la "
                      'vidéo, au lieu de la queue quasi-statique / moyenne par cycles. '
                      'À utiliser quand on SAIT que la vidéo se termine au repos '
                      '(ex. rock1_val.mp4) : c\'est alors le seul cas où le critère (2) '
                      'mesure vraiment une position de repos observée.')
_ap.add_argument('--relax-periods', type=float, default=20.0,
                 help='durée du rollout de relaxation, en périodes (déf. 20)')
_ap.add_argument('--relax-iters', type=int, default=2000,
                 help='itérations max de la relaxation suramortie (déf. 2000)')
# ── (3) erreur moyenne sur N périodes ───────────────────────────────────────
_ap.add_argument('--n-periods', type=float, default=5.0,
                 help='horizon du critère (3), en périodes (déf. 5)')
_ap.add_argument('--n-windows', type=int, default=50, help='nb de départs tirés (déf. 50)')
_ap.add_argument('--stride', type=int, default=0,
                 help='0 (défaut) = tirage ALÉATOIRE de --n-windows départs ; ≥1 = énumération '
                      'EXHAUSTIVE de tous les départs de [skip, n−H] avec ce pas (1 = tout le '
                      'held-out ; --n-windows est alors ignoré)')
_ap.add_argument('--period-s', type=float, default=None,
                 help='FORCE la période de référence (s) du critère (3) au lieu de la '
                      'déduire de q_enc. Indispensable pour comparer des ARCHITECTURES '
                      'entre elles : l\'horizon doit être fixé par la physique du '
                      'système, pas par ce que chaque modèle croit mesurer. '
                      'Réf. mesurée sur les vidéos _step : 0.2496 s (1-seg, 4.006 Hz), '
                      '0.6870 s (2-seg, 1.4556 Hz).')
_ap.add_argument('--seed', type=int, default=0)
# ── réglages à accorder au checkpoint ───────────────────────────────────────
_ap.add_argument('--sigma', type=float, default=None, help='override SMOOTH_LATENT_SIGMA')
_ap.add_argument('--sigma-pressure', type=float, default=None, help='override SMOOTH_PRESSURE_SIGMA')
_ap.add_argument('--cq-eps', type=float, default=None, help='override LNN_RAYLEIGH_CQ_EPS')
_ap.add_argument('--integrator', type=str, default=None,
                 choices=['verlet', 'rk4', 'gen_alpha'], help='override LNN_INTEGRATOR')
_ap.add_argument('--thresh-factor', type=float, default=2.0,
                 help='divergence si ‖q‖ > facteur × max‖q_enc‖ des données (déf. 2)')
_ap.add_argument('--decoder', type=str, default='auto',
                 help="décodeur GS pour l'espace IMAGE ('auto' = decoder2dpt_ae.pt "
                      'sinon decoder2dpt.pt)')
_ap.add_argument('--eval-res', type=int, default=32,
                 help='résolution de comparaison image (déf. 32, comme Krauss)')
_ap.add_argument('--no-image', action='store_true',
                 help='ne mesurer QUE l\'espace latent (saute le décodage, plus rapide)')
_ap.add_argument('--v0', type=str, default='fd2', choices=['fd2', 'auto'],
                 help="estimateur de la vitesse initiale des rollouts : 'fd2' = "
                      "différence avant d'ordre 2 sur 3 frames (historique), 'auto' = "
                      "régression polynomiale unilatérale dont la fenêtre est CHOISIE "
                      "sur les données, sans modèle (viz.select_v0_estimator). "
                      "'auto' retombe sur fd2 quand aucune fenêtre plus large ne fait "
                      "mieux (cas sous-échantillonnés type dp).")
_ap.add_argument('--keep-dup-lead', action='store_true',
                 help="ne PAS couper les frames de tête dupliquées de la vidéo "
                      "(défaut : coupées ; une frame dupliquée annule la vitesse au "
                      "1er pas et fait sortir la différence avant du mauvais signe, "
                      "cf. dp1.mp4/dp2.mp4)")
_ap.add_argument('--json', type=str, default=None, help='fichier JSON de sortie')
_ap.add_argument('--out', type=str, default=None, help='nom du PNG de sortie')
_args, _ = _ap.parse_known_args()

if _args.sigma is not None:
    config.SMOOTH_LATENT_SIGMA = float(_args.sigma)
if _args.sigma_pressure is not None:
    config.SMOOTH_PRESSURE_SIGMA = float(_args.sigma_pressure)
if _args.cq_eps is not None:
    config.LNN_RAYLEIGH_CQ_EPS = float(_args.cq_eps)
if _args.integrator is not None:
    config.LNN_INTEGRATOR = _args.integrator

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch.nn.functional as F

from dataset import VideoFrameDataset
from models import LNN, build_encoder, WhitenedEncoder, load_latent_whiten
from models_2pt import build_decoder2pt
from train_lnn_fixedae import encode_all, smooth_latents, smooth_pressures
from viz import (simulate_rk4, initial_velocity, initial_velocity_poly,
                 select_v0_estimator, get_sim_pressure)


def to_float01(frames_np):
    """(T,C,H,W) uint8[0,255] ou float[0,1] → tensor float [0,1] CPU."""
    t = torch.from_numpy(np.ascontiguousarray(frames_np))
    return t.float().div_(255.0) if t.dtype == torch.uint8 else t.float()


def _cfg(name, default=None):
    return getattr(config, name, default)


def load_visibility_metric(d):
    """
    Métrique de visibilité `Ḡ` du décodeur (`compute_visibility_metric.py`),
    normalisée à `trace(Ḡ) = d` (valeur propre moyenne 1) pour que les erreurs
    en norme `Ḡ` gardent l'échelle des erreurs isotropes.

    ⚠️ **Pas de ridge ici**, contrairement à `finetune_lnn_fixedae`. Là-bas la
    ridge `ρ·λmax·I` est indispensable parce qu'on OPTIMISE : sans elle les
    directions quasi invisibles ne seraient plus contraintes du tout et
    pourraient diverger. Ici on MESURE : ajouter une ridge remettrait
    précisément le poids isotrope qu'on cherche à retirer. On veut `Ḡ` nue.

    Retourne (G (d,d) float64 ndarray, evecs, evals décroissants) ou (None,…).
    """
    fname = _cfg('LNN_ROLLOUT_METRIC_FILE', 'visibility_metric.pt')
    path = config.SAVE_DIR / fname
    if not path.exists():
        return None, None, None
    blob = torch.load(path, map_location='cpu', weights_only=False)
    G = blob['G'].double()
    if tuple(G.shape) != (d, d):
        print(f'⚠ {fname} est en dimension {tuple(G.shape)} ≠ d={d} : ignorée '
              f'(la recalculer pour ce cas).')
        return None, None, None
    G = 0.5 * (G + G.T)
    lam, V = torch.linalg.eigh(G)
    order = torch.argsort(lam, descending=True)
    lam, V = lam[order], V[:, order]
    G = G * (d / torch.diagonal(G).sum())                # trace(G) = d
    lam_n = lam * (d / lam.sum())
    print(f'Métrique de visibilité : {fname} (source={blob.get("source","?")}, '
          f'décodeur={blob.get("decoder","?")}, rendu {blob.get("res","?")}²)')
    print(f'  λ(Ḡ) normalisées : {[f"{v:.4g}" for v in lam_n.tolist()]}  '
          f'(cond {lam.max()/lam.clamp(min=1e-30).min():.4g})')
    return G.numpy(), V.numpy(), lam_n.numpy()


def gnorm2(X, G):
    """‖x‖²_G ligne à ligne. X : (T,d) → (T,). G=None ⟹ norme euclidienne."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return (X * X).sum(1) if G is None else np.einsum('ti,ij,tj->t', X, G, X)


# ════════════════════════════════════════════════════════════════════════════
# Estimateur de fréquence (commun à enc(x) et au rollout — c'est le point clé :
# tout biais de l'estimateur s'annule dans l'erreur relative)
# ════════════════════════════════════════════════════════════════════════════
def dominant_freq(q, dt, G=None, min_cycles=1.5):
    """
    Fréquence dominante (Hz) d'une trajectoire multi-dimensionnelle, **pondérée
    par la visibilité**.

    q : (N, d) ndarray. Spectre agrégé sur les `d` composantes latentes. Le poids
    de l'agrégation est le point critique :

      - `G = None` : somme ISOTROPE `Σ_j |F_j|²`. **Faux en général.** En espace
        blanchi chaque direction a variance 1 mais PAS visibilité 1 : un mode qui
        ne déplace presque aucun pixel y pèse autant qu'un mode bien visible, et
        le pic peut être celui d'un mode invisible (rapport de visibilité mesuré :
        jusqu'à 341× entre modes, cf. `visibility_metric.pt`). Conservé pour
        diagnostic seulement.
      - `G` = métrique de visibilité `Ḡ = E[(∂I/∂u)ᵀ(∂I/∂u)]` du décodeur : on
        prend la forme quadratique `F(f)^H Ḡ F(f)`, qui est **au premier ordre le
        spectre de puissance en espace IMAGE** (`Δuᵀ Ḡ Δu ≈ ‖I(u+Δu) − I(u)‖²`).
        C'est la fréquence du **mode visible**, la seule qui ait un sens ici.

    Fenêtre de Hann ; pic raffiné par interpolation parabolique sur le
    LOG-spectre (un pic de FFT fenêtrée est ~gaussien en amplitude, donc
    parabolique en log ⟹ précision sous-bin, ce qui compte : sur 8 périodes un
    bin vaut déjà 12 % de f).

    `min_cycles` écarte la dérive basse fréquence (< 1.5 cycle dans la fenêtre),
    qui est une tendance et non une oscillation.

    Retourne (f_hz, freqs, power) ; f_hz = nan si la fenêtre est trop courte.
    """
    q = np.asarray(q, dtype=np.float64)
    n = len(q)
    if n < 8:
        return float('nan'), np.zeros(0), np.zeros(0)
    x = q - q.mean(axis=0, keepdims=True)
    w = np.hanning(n)[:, None]
    F = np.fft.rfft(x * w, axis=0)                       # (nf, d) complexe
    if G is None:
        p = (np.abs(F) ** 2).sum(axis=1)
    else:
        # Re(F^H Ḡ F) = Re(F)ᵀḠRe(F) + Im(F)ᵀḠIm(F)  (Ḡ symétrique réelle)
        p = (np.einsum('fi,ij,fj->f', F.real, G, F.real)
             + np.einsum('fi,ij,fj->f', F.imag, G, F.imag))
        p = np.maximum(p, 0.0)                           # Ḡ SPD ⟹ p ≥ 0 (arrondi)
    freqs = np.fft.rfftfreq(n, d=dt)
    p_masked = p.copy()
    p_masked[freqs < min_cycles / (n * dt)] = 0.0
    if not np.any(p_masked > 0):
        return float('nan'), freqs, p
    k = int(np.argmax(p_masked))
    f = freqs[k]
    # Raffinement sous-bin (impossible aux extrémités ou si le sommet est plat).
    if 0 < k < len(p) - 1 and p[k - 1] > 0 and p[k + 1] > 0:
        a, b, c = np.log(p[k - 1]), np.log(p[k]), np.log(p[k + 1])
        den = a - 2 * b + c
        if den < -1e-12:                      # concavité stricte = vrai sommet
            delta = 0.5 * (a - c) / den
            if abs(delta) <= 0.5:
                f = freqs[k] + delta * (freqs[1] - freqs[0])
    return float(f), freqs, p


# ════════════════════════════════════════════════════════════════════════════
# Détection des intervalles à pression CONSTANTE (= oscillation libre)
# ════════════════════════════════════════════════════════════════════════════
def free_segments(P_raw, lo, hi, min_len, tol_frac, span_frac):
    """
    Intervalles [a, b) ⊂ [lo, hi) où la pression est CONSTANTE, donc où le système
    oscille librement (ni forçage variable, ni transitoire d'échelon).

    Double critère, les deux nécessaires :
      - `tol_frac` : incrément par frame |ΔP| < tol_frac × amplitude ⟹ exclut les
        fronts d'échelon et les rampes rapides ;
      - `span_frac` : excursion TOTALE sur l'intervalle ptp(P) < span_frac ×
        amplitude ⟹ exclut la dérive lente, qu'un seuil par frame laisse passer
        (280 frames sous le seuil peuvent parcourir toute la plage). Sans lui, une
        entrée « smooth » lentement variable serait déclarée libre à tort.

    P_raw : (N, n_c) ndarray BRUT (non lissé : le lissage transforme les échelons
    en rampes et ferait disparaître les fronts qu'on cherche à exclure).
    P_raw = None (cas non forcé) ⟹ un seul intervalle, tout [lo, hi).
    """
    if P_raw is None:
        return [(lo, hi)] if hi - lo >= min_len else []
    amp = float(np.ptp(P_raw[lo:hi], axis=0).max())
    if amp <= 0:                               # pression rigoureusement constante
        return [(lo, hi)] if hi - lo >= min_len else []
    step = np.abs(np.diff(P_raw[lo:hi], axis=0)).max(axis=1)   # (hi-lo-1,)
    flat = step < tol_frac * amp
    runs, a = [], None
    for i, ok in enumerate(flat):
        if ok and a is None:
            a = i
        elif not ok and a is not None:
            runs.append((lo + a, lo + i))
            a = None
    if a is not None:
        runs.append((lo + a, lo + len(flat)))
    segs = []
    for (s, e) in runs:
        if e - s < min_len:
            continue
        if float(np.ptp(P_raw[s:e], axis=0).max()) > span_frac * amp:
            continue
        segs.append((s, e))
    return segs


# ════════════════════════════════════════════════════════════════════════════
# Équilibre du modèle : relaxation suramortie q ← q + h·accel(q, 0, P)
# ════════════════════════════════════════════════════════════════════════════
def accel_at_rest(lnn, q, p):
    """
    Accélération à VITESSE NULLE, `q` : (1,d) → (1,d).

    Réplique le dispatch de `viz.simulate_rk4.dvdt` : `lnn.accel` n'existe que
    pour les LNN à métrique apprise (`LNN_MASS_LEARNED`) ; les autres ont
    `lnn.metric is None` et passent par
    `Minv·(−∂E/∂z + b(q)ᵀP)`. Appeler `lnn.accel` sans vérifier lève
    `AttributeError: 'NoneType' object has no attribute 'log_m'`.

    À `v = 0` tous les termes dissipatifs s'annulent (visqueux `Γv`, Coulomb
    `Β·v/‖v‖` → 0), et la Coriolis aussi : il ne reste que le conservatif et le
    forçage, ce qui est exactement la condition d'équilibre cherchée.
    """
    v = torch.zeros_like(q)
    if getattr(lnn, 'metric', None) is not None:
        return lnn.accel(q, v, p)
    a = -lnn.dE_dz(q)
    if p is not None and getattr(lnn, 'use_pressure', False):
        a = a + lnn.pressure_force(q, p)
    Minv = getattr(lnn, 'Minv', None)
    if Minv is not None:
        a = a @ Minv.T
    return a


def relax_to_equilibrium(lnn, q0, p_const, n_iters, tol=1e-10):
    """
    Équilibre STABLE le plus proche de `q0` sous pression constante `p_const`.

    On suit le flot suramorti `q̇ = M(q)⁻¹F(q, q̇=0)`, c.-à-d. exactement `accel`
    à vitesse nulle : ses points fixes sont les racines de la force, et le flot
    ne converge que vers les racines STABLES (les seules qui ont un sens
    physique ici). Pas adaptatif : on double tant que ‖a‖ décroît, on divise par
    deux et on rejette le pas sinon. Aucune dérivée de `accel` n'est requise, ce
    qui évite le piège du `create_graph=self.training` (en `eval()` le graphe de
    `accel` vis-à-vis de `q` n'est pas construit, donc un Newton/LBFGS naïf sur
    ‖accel‖² serait silencieusement faux).

    Retourne (q_eq (d,), ‖accel‖ final, n_iters effectués, converged: bool).
    """
    q = q0.detach().clone().reshape(1, -1)
    v = torch.zeros_like(q)
    p = None if p_const is None else p_const.reshape(1, -1)

    def force_norm(qq):
        a = accel_at_rest(lnn, qq, p)
        return a.detach(), float(a.detach().norm())

    a, na = force_norm(q)
    if not np.isfinite(na):
        return q0.detach().clone(), float('nan'), 0, False
    # Pas initial : ramène le 1ᵉʳ déplacement à ~1 % de l'échelle de q0.
    scale = max(float(q0.norm()), 1.0)
    h = 0.01 * scale / max(na, 1e-12)
    it = 0
    for it in range(1, n_iters + 1):
        q_try = q + h * a
        a_try, na_try = force_norm(q_try)
        if np.isfinite(na_try) and na_try < na:
            q, a, na = q_try, a_try, na_try
            h *= 1.5
        else:
            h *= 0.5
            if h < 1e-18 * scale:
                break
        if na < tol * scale:
            break
    return q.reshape(-1).detach(), float(na), it, bool(na < 1e-6 * scale)


# ════════════════════════════════════════════════════════════════════════════
def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dt = config.DT
    D = config.LATENT_DIM
    n_ch = 3 if _cfg('ENC_COLOR', True) else 1
    use_p = _cfg('LNN_PRESSURE', False)

    # ── Vidéo évaluée ────────────────────────────────────────────────────────
    # VIDEO_DIR est tantôt un fichier (Krauss : un .npz), tantôt un dossier
    # (rock : videos_split/) ⟹ on résout `--video` dans les deux cas.
    video_dir = config.VIDEO_DIR
    if _args.video is not None:
        base = config.VIDEO_DIR if config.VIDEO_DIR.is_dir() else config.VIDEO_DIR.parent
        video_dir = base / _args.video
    print(f'Device   : {dev}   |  DT = {dt:.6g} s ({1/dt:.2f} fps)  |  d = {D}')
    print(f'Vidéo    : {video_dir.name}'
          + ('  (VIDEO_DIR, in-sample)' if _args.video is None else '  (--video)'))

    ds = VideoFrameDataset(
        video_dir=video_dir, img_size=config.IMG_SIZE, n_channels=n_ch,
        rest_video=None, rest_first_n_frames=_cfg('REST_FIRST_N_FRAMES', 0),
        crop=_cfg('CROP', None),
        load_pressure=use_p, pressure_dir=_cfg('PRESSURE_DIR', None),
        pressure_cols=_cfg('PRESSURE_COLS', None),
        pressure_norm=_cfg('PRESSURE_NORM', 101325.0), pressure_dt=dt,
        pressure_sync_offsets=_cfg('PRESSURE_SYNC_OFFSETS', None),
        max_frames=_args.frames, store_uint8=True)
    n = len(ds.frames)
    print(f'{n} frames chargées ({n*dt:.1f} s).')

    # ── Frames de TÊTE dupliquées (défaut de la vidéo source) ────────────────
    # Une frame dupliquée impose une vitesse nulle au 1er pas ET décale d'une
    # frame la base de temps de tout le reste du clip. Mesuré : dp1.mp4 et
    # dp2.mp4 commencent par deux images identiques (MSE 7e-7 contre 4e-3 de
    # médiane). Ailleurs dans une vidéo, deux frames voisines se ressemblent
    # LÉGITIMEMENT près d'un rebroussement : on ne coupe donc que la tête.
    if not _args.keep_dup_lead and n > 8:
        _f = ds.frames[:6].astype(np.float32)
        if ds.frames.dtype == np.uint8:
            _f = _f / 255.0
        _d = [float(((_f[i + 1] - _f[i]) ** 2).mean()) for i in range(len(_f) - 1)]
        _ref = float(np.median(_d))
        _nd = 0
        while _nd < len(_d) - 1 and _d[_nd] < _ref / 50:
            _nd += 1
        if _nd:
            print(f'⚠ {_nd} frame(s) de tête dupliquée(s) (MSE {_d[0]:.2e} contre '
                  f'{_ref:.2e}) : coupées. --keep-dup-lead pour les garder.')
            ds.frames = ds.frames[_nd:]
            if getattr(ds, 'pressures', None) is not None:
                ds.pressures = ds.pressures[_nd:]
            if getattr(ds, 'video_lengths', None):
                ds.video_lengths[0] -= _nd
            n -= _nd

    # ── Encodeur gelé (+ blanchiment latent figé) ────────────────────────────
    enc = build_encoder(config.IMG_SIZE, config.ENC_HIDDEN, D, n_channels=n_ch,
                        normalize=_cfg('ENC_NORMALIZE', False)).to(dev)
    if _args.encoder == 'auto':
        enc_path = config.SAVE_DIR / 'encoder_ae.pt'
        if not enc_path.exists():
            enc_path = config.SAVE_DIR / 'encoder.pt'
    else:
        enc_path = config.SAVE_DIR / _args.encoder
    enc.load_state_dict(torch.load(enc_path, map_location=dev))
    enc.eval()
    for prm in enc.parameters():
        prm.requires_grad_(False)
    wh = load_latent_whiten(config.SAVE_DIR, dev, D)
    if wh is not None:
        enc = WhitenedEncoder(enc, wh).to(dev).eval()
    print(f'Encodeur : {enc_path.name}' + ('  + LatentWhiten' if wh is not None else ''))

    # ── LNN ──────────────────────────────────────────────────────────────────
    if _args.lnn == 'auto':
        lnn_path = config.SAVE_DIR / 'lnn_rollout.pt'
        if not lnn_path.exists():
            lnn_path = config.SAVE_DIR / 'lnn.pt'
    else:
        lnn_path = config.SAVE_DIR / _args.lnn
    lnn = LNN(D, config.LNN_HIDDEN).to(dev)
    state = torch.load(lnn_path, map_location=dev)
    try:
        lnn.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f'⚠ strict load échoué ({e}) ; repli strict=False.')
        lnn.load_state_dict(state, strict=False)
    lnn.eval()
    print(f'LNN      : {lnn_path.name}  (pression={getattr(lnn, "use_pressure", False)}, '
          f'intégrateur={_cfg("LNN_INTEGRATOR", "verlet")}, '
          f'c₀={_cfg("LNN_RAYLEIGH_CQ_EPS", None)})')
    print(f'Lissage  : q σ={_cfg("SMOOTH_LATENT_SIGMA", None)} | '
          f'P σ={_cfg("SMOOTH_PRESSURE_SIGMA", None)} frames '
          f'(doivent être ceux de l\'entraînement du checkpoint)')

    # ── Décodeur GS : espace IMAGE ───────────────────────────────────────────
    # Référence des erreurs image = les VRAIES frames (convention A, celle de
    # eval_multistep_mse.py) ⟹ l'erreur inclut le plancher de l'autoencodeur, qui
    # est reporté à côté. C'est la seule convention équitable entre architectures
    # (chacune a son décodeur) et elle rend les chiffres directement comparables
    # à la MSE 0.5 s du tableau de l'article.
    decoder = whiten_dec = None
    R = int(_args.eval_res)
    n_ch_dec = 3 if _cfg('DEC_COLOR', True) else 1
    gt_frames = ds.frames
    if not _args.no_image:
        dp = (config.SAVE_DIR / 'decoder2dpt_ae.pt') if _args.decoder == 'auto' \
            else (config.SAVE_DIR / _args.decoder)
        if _args.decoder == 'auto' and not dp.exists():
            dp = config.SAVE_DIR / 'decoder2dpt.pt'
        if not dp.exists():
            print(f'⚠ décodeur introuvable ({dp.name}) : espace image non mesuré.')
        else:
            st = torch.load(dp, map_location=dev)
            decoder = build_decoder2pt(
                latent_dim=D, n_gaussians=st['mu_raw'].shape[0],
                img_size=config.IMG_SIZE, n_channels=n_ch_dec).to(dev)
            decoder.load_state_dict(st); decoder.eval()
            print(f'Décodeur : {dp.name} ({st["mu_raw"].shape[0]} gaussiennes) '
                  f'→ MSE image à {R}×{R}')
            if n_ch_dec != n_ch:
                gt_frames = VideoFrameDataset(
                    video_dir=video_dir, img_size=config.IMG_SIZE, n_channels=n_ch_dec,
                    rest_video=None, rest_first_n_frames=_cfg('REST_FIRST_N_FRAMES', 0),
                    crop=_cfg('CROP', None), max_frames=_args.frames,
                    store_uint8=True).frames

    def decode(z_bd):
        """(B,d) latent (blanchi) → (B,C,R,R) [0,1] sur device, par blocs."""
        out = []
        with torch.no_grad():
            for i in range(0, len(z_bd), 64):
                zb = z_bd[i:i + 64].to(dev)
                if wh is not None:
                    zb = wh.inverse(zb)
                im = decoder(zb).clamp(0, 1)
                if R < im.shape[-1]:
                    im = F.interpolate(im, size=(R, R), mode='area')
                out.append(im)
        return torch.cat(out, 0)

    def gt_at(idx):
        """Frames réelles aux indices `idx` → (B,C,R,R) [0,1] sur device."""
        im = to_float01(gt_frames[idx]).to(dev)
        return F.interpolate(im, size=(R, R), mode='area') if R < im.shape[-1] else im

    # ── q_enc(t) : la référence ──────────────────────────────────────────────
    print('Encodage…')
    z_raw = encode_all(enc, ds.frames, dev)                      # (n, d) CPU
    z = z_raw
    if _cfg('SMOOTH_LATENT', False):
        z = smooth_latents(z_raw.clone(), [n],
                           _cfg('SMOOTH_LATENT_WINDOW', 13), _cfg('SMOOTH_LATENT_POLY', 3),
                           _cfg('SMOOTH_LATENT_MODE', 'savgol'),
                           _cfg('SMOOTH_LATENT_SIGMA', 10.0))
    P_raw = np.asarray(ds.pressures) if (use_p and getattr(ds, 'pressures', None) is not None) else None
    if P_raw is not None and _cfg('SMOOTH_PRESSURE', False):
        # `ds.pressures` (lissée) sert au FORÇAGE ; `P_raw` (brute) à la détection
        # des fronts d'échelon — le lissage les transformerait en rampes.
        P_raw = P_raw.copy()
        ds.pressures = smooth_pressures(ds.pressures, ds.video_lengths)

    z_np = z.cpu().numpy()
    z_d = z.to(dev)

    # ── Estimateur de la vitesse initiale des rollouts ───────────────────────
    # 'fd2' = 3 points (historique) ; 'auto' = fenêtre choisie SUR LES DONNÉES
    # contre une dérivée centrée de référence (aucun modèle appris n'intervient).
    _v0_name, _v0_w, _v0_deg = 'fd2', 3, 2
    if _args.v0 == 'auto':
        _v0_name, _v0_w, _v0_deg, _ = select_v0_estimator(z_np)

    # ── Métrique de VISIBILITÉ : c'est elle qui définit « le mode visible » ───
    G, Vvis, lam_vis = load_visibility_metric(D)
    if _args.metric == 'isotropic':
        G = None
        print('⚠ --metric isotropic : agrégation à poids UNIFORME sur les d '
              'composantes.\n  En espace blanchi, variance 1 ≠ visibilité 1 : le pic '
              'peut être celui d\'un\n  mode qui ne déplace aucun pixel. Diagnostic '
              'seulement.')
    elif G is None and D == 1:
        # d=1 : une seule direction latente ⟹ toute pondération positive donne le
        # même spectre à une constante près, donc le même pic. La question du
        # « mode visible » ne se pose pas.
        print('d=1 : métrique de visibilité sans objet (une seule direction '
              'latente).')
    elif G is None:
        print('⚠ visibility_metric.pt ABSENT pour ce cas : repli sur l\'agrégation '
              'isotrope.\n  Les fréquences rapportées ne sont PAS celles du mode '
              'visible, et les erreurs (2)\n  et (3) comptent autant les directions '
              'invisibles que les visibles. Lancer\n  `py compute_visibility_metric.py '
              '--config … --source data` puis relancer.')

    # Toutes les erreurs sont mesurées dans la MÊME métrique que les fréquences,
    # et normalisées par le RMS de q_enc dans cette métrique ⟹ invariantes à
    # l'échelle globale de Ḡ comme à celle du latent.
    q_std = float(np.sqrt(gnorm2(z_np - z_np.mean(0), G).mean()))   # amplitude RMS
    q_peak = float(np.sqrt(gnorm2(z_np - z_np.mean(0), G).max()))   # amplitude crête
    r_max = float(np.linalg.norm(z_np, axis=1).max())    # divergence : euclidien
    thresh = _args.thresh_factor * r_max
    speed = np.sqrt(gnorm2(np.gradient(z_np, axis=0), G))         # ‖q̇_enc‖_G
    static_thr = float(np.quantile(speed, _args.static_quantile))

    lo = min(int(round(_args.skip_sec / dt)), max(0, n - 10))
    hi = n
    min_free = int(round(_args.min_free_sec / dt))
    segs = free_segments(P_raw, lo, hi, min_free, _args.pressure_tol, _args.pressure_span)
    segs.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
    segs = segs[:_args.n_free]
    print(f'\nIntervalles à pression constante : {len(segs)} '
          f'(≥ {_args.min_free_sec} s), durées '
          f'{[round((b-a)*dt, 1) for a, b in segs[:8]]}{"…" if len(segs) > 8 else ""} s')
    # Contrôle explicite du caractère LIBRE : mesurer une fréquence sur un
    # intervalle où P bouge encore n'aurait aucun sens (on lirait la fréquence du
    # forçage, pas un mode propre). On affiche donc l'excursion réelle de P.
    p_span_max = 0.0
    if P_raw is not None and segs:
        amp_P = float(np.ptp(P_raw[lo:hi], axis=0).max())
        spans = [float(np.ptp(P_raw[s:e], axis=0).max()) / max(amp_P, 1e-12)
                 for s, e in segs]
        p_span_max = max(spans)
        print(f'  excursion de P sur ces intervalles : médiane '
              f'{100*float(np.median(spans)):.2f} %, max {100*p_span_max:.2f} % '
              f'de l\'amplitude totale (seuil --pressure-span '
              f'{100*_args.pressure_span:.0f} %)')
    if not segs:
        # Deux causes très différentes, à ne pas confondre dans le message.
        if P_raw is None:
            print(f'  ⚠ vidéo trop COURTE : {n} frames = {n*dt:.1f} s < --min-free-sec '
                  f'({_args.min_free_sec} s).\n    Ce cas n\'a pas de forçage, toute la '
                  f'vidéo est libre — il faut juste baisser\n    --min-free-sec (et '
                  f'--n-periods pour le critère (3)).')
        else:
            print('  ⚠ aucun intervalle à pression constante : critères (1) et (2) NON '
                  'MESURABLES\n    sur cette vidéo (entrée continûment variable). Seul (3) '
                  'est rapporté. Utiliser\n    un enregistrement à échelons (step input), '
                  'ou relâcher --pressure-tol/--pressure-span.')

    def rollout(s, n_steps, const_pressure=False):
        """Rollout depuis la frame `s`. Retourne (n_steps, d) sur device, ou None."""
        if s + 3 > n or n_steps < 1:
            return None
        z0 = z_d[s]
        v0 = (initial_velocity(z_d[s:s + 3]) if s + _v0_w > n
              else initial_velocity_poly(z_d[s:s + _v0_w], _v0_w, _v0_deg))
        p = get_sim_pressure(lnn, ds, s, n_steps, dev)
        if p is not None and const_pressure:
            p = p[:1].expand(n_steps, -1).contiguous()
        with torch.no_grad():
            traj = simulate_rk4(lnn, z0, v0, n_steps=n_steps, dt=1.0, pressure=p)
        if not torch.isfinite(traj).all() or float(traj.norm(dim=1).max()) > thresh:
            return None
        return traj

    # ════════════════════════════════════════════════════════════════════════
    # (1) FRÉQUENCE + (2) POSITION DE REPOS, par intervalle libre
    # ════════════════════════════════════════════════════════════════════════
    rows, spectra = [], None
    for (a, b) in segs:
        seg_len = b - a
        # 1er passage : fréquence de enc(x) sur tout l'intervalle → ordre de grandeur.
        f0, _, _ = dominant_freq(z_np[a:b], dt, G)
        if not np.isfinite(f0) or f0 <= 0:
            continue
        # 2e passage : fenêtres d'analyse de `--freq-periods` périodes (même
        # fenêtre pour enc et sim ⟹ même résolution spectrale, biais annulé).
        # Un intervalle libre long est PAVÉ en plusieurs fenêtres, chacune étant
        # une mesure indépendante (rollout ré-initialisé sur enc(x)) : sans quoi
        # un cas à un seul intervalle (rock : une vidéo, pas de pression) ne
        # donnerait qu'un point et aucune erreur-type.
        w = min(seg_len, max(16, int(round(_args.freq_periods / (f0 * dt)))))
        f_list, s_list, amp_list, coh_list, v1_list = [], [], [], [], []
        for t0 in range(a, b - w + 1, w):
            ze_w = z_np[t0:t0 + w]
            # Amplitude d'oscillation DANS la fenêtre (RMS après retrait de la
            # moyenne). Si enc(x) ne bouge pas, sa « fréquence dominante » est
            # celle du bruit d'encodeur : la fenêtre est écartée du critère (1)
            # (elle reste utile au critère (2), qui veut justement du repos).
            amp_e = float(np.sqrt(gnorm2(ze_w - ze_w.mean(0), G).mean()))
            if amp_e < _args.min_osc_frac * q_std:
                continue
            f_e, fr_e, sp_e = dominant_freq(ze_w, dt, G)
            tr_w = rollout(t0, w, const_pressure=True)
            if tr_w is None or not np.isfinite(f_e) or f_e <= 0:
                continue
            sim_w = tr_w.cpu().numpy()
            f_s, fr_s, sp_s = dominant_freq(sim_w, dt, G)
            amp_s = float(np.sqrt(gnorm2(sim_w - sim_w.mean(0), G).mean()))
            # Le pic DOMINANT du rollout peut être une dérive lente alors que le
            # bon mode est bien présent, plus bas dans le spectre. On mesure donc
            # aussi la puissance du rollout À la fréquence de enc(x), rapportée à
            # son maximum : ~1 = le mode est là mais n'est pas dominant, ≪1 = il
            # est absent. Sans ce chiffre, une erreur de fréquence élevée ne dit
            # pas laquelle des deux situations on a.
            coh = float('nan')
            if sp_s.size and sp_s.max() > 0:
                coh = float(sp_s[int(np.argmin(np.abs(fr_s - f_e)))] / sp_s.max())
            # Contrôle explicite : fréquence de la PROJECTION sur la direction la
            # plus visible (1er vecteur propre de Ḡ). Le spectre pondéré ci-dessus
            # agrège toutes les directions ; celui-ci ne regarde que le mode que
            # l'image montre le plus. Les deux doivent concorder.
            f_e1 = f_s1 = float('nan')
            if Vvis is not None:
                v1 = Vvis[:, 0]
                f_e1, _, _ = dominant_freq((ze_w @ v1)[:, None], dt)
                f_s1, _, _ = dominant_freq((sim_w @ v1)[:, None], dt)
            v1_list.append((f_e1, f_s1))
            f_list.append((f_e, f_s))
            coh_list.append(coh)
            amp_list.append(amp_s / amp_e if amp_e > 0 else float('nan'))
            s_list.append((fr_e, sp_e, fr_s, sp_s, t0, w, ze_w, sim_w, f_e, f_s))
        if not f_list:
            rows.append(dict(a=int(a), b=int(b), dur_s=seg_len * dt, f_enc=f0,
                             diverged=True))
            continue
        f_enc = float(np.mean([e for e, _ in f_list]))
        f_sim = float(np.nanmean([s for _, s in f_list]))
        if spectra is None:                     # garde le 1ᵉʳ (segment le plus long)
            spectra = s_list[0]

        # ── (2) position d'équilibre observée sur enc(x) ─────────────────────
        # Plus longue QUEUE quasi-statique de l'intervalle.
        # ⚠️ BUG CORRIGÉ (2026-08-03) : la version précédente balayait
        #     `while tail > 0 and speed[b - tail] < static_thr: tail -= 1`
        # dont l'indice `b - tail` part de `a` et AVANCE vers `b` : elle comptait
        # donc les frames lentes du DÉBUT de l'intervalle, alors que la moyenne
        # qui suit porte sur celles de la FIN. Or un intervalle de relaxation
        # commence RAPIDE (juste après l'échelon) et finit LENT : la boucle
        # s'arrêtait aussitôt, `n_static = 0`, et la branche `static` n'était
        # JAMAIS empruntée (vérifié : mode `cycle-mean` sur les 7 cas). Le mode
        # qui rend ce critère fidèle à son nom était donc désactivé.
        n_static = 0                            # frames quasi-statiques EN FIN
        while n_static < seg_len and speed[b - 1 - n_static] < static_thr:
            n_static += 1
        T_frames = 1.0 / (f_enc * dt) if np.isfinite(f_enc) and f_enc > 0 else float('inf')
        if _args.rest_from_last > 0:
            # Repos CONNU en fin de vidéo : référence directe, sans détection.
            nlast = min(int(_args.rest_from_last), n)
            n_static = nlast
            q_eq_enc = z_np[n - nlast:n].mean(0)
            mode = f'last-{nlast}'
        elif n_static >= max(8, 0.5 * T_frames):
            q_eq_enc = z_np[b - n_static:b].mean(0)
            mode = 'static'
        else:
            # Pas de vrai palier : moyenne sur un nombre ENTIER de périodes en
            # fin d'intervalle (l'oscillation s'y annule au 1er ordre).
            n_cyc = int(seg_len // T_frames) if np.isfinite(T_frames) else 0
            if n_cyc < 1:
                q_eq_enc, mode = z_np[a:b].mean(0), 'seg-mean'
            else:
                q_eq_enc, mode = z_np[b - int(n_cyc * T_frames):b].mean(0), 'cycle-mean'

        p_const = None
        if use_p and getattr(ds, 'pressures', None) is not None:
            p_const = torch.from_numpy(np.asarray(ds.pressures[a], dtype=np.float32)).to(dev)
        q0 = torch.from_numpy(q_eq_enc.astype(np.float32)).to(dev)
        with torch.enable_grad():
            q_relax, fnorm, n_it, conv = relax_to_equilibrium(
                lnn, q0, p_const, _args.relax_iters)
        e_relax = float(np.sqrt(gnorm2(q_relax.cpu().numpy() - q_eq_enc, G)[0]))

        # Équilibre dynamique : rollout long sous pression constante, moyenne sur
        # la dernière période.
        n_rel = int(round(_args.relax_periods * T_frames)) if np.isfinite(T_frames) else 0
        e_roll = float('nan')
        tr = rollout(a, n_rel, const_pressure=True) if n_rel > 8 else None
        if tr is not None:
            last = max(8, int(round(T_frames)))
            q_roll = tr[-last:].mean(0).cpu().numpy()
            e_roll = float(np.sqrt(gnorm2(q_roll - q_eq_enc, G)[0]))

        # ── (2) en espace IMAGE ─────────────────────────────────────────────
        # Cible = moyenne des VRAIES frames du palier (les mêmes que celles qui
        # définissent q_eq_enc). On décode l'équilibre du MODÈLE et l'équilibre
        # ENCODÉ : le second donne le plancher AE, c.-à-d. la part de l'erreur
        # imputable au décodeur et non à la dynamique.
        ei_relax = ei_floor = ei_dyn = float('nan')
        if decoder is not None:
            if mode.startswith('last-'):
                idx_rest = np.arange(n - n_static, n)      # les frames de repos elles-mêmes
            elif mode == 'static':
                idx_rest = np.arange(b - n_static, b)
            else:
                idx_rest = np.arange(a, b)
            with torch.no_grad():
                I_obs = gt_at(idx_rest).mean(0, keepdim=True)
                I_sim = decode(q_relax.reshape(1, -1).cpu())
                I_enc = decode(torch.from_numpy(q_eq_enc.astype(np.float32)).reshape(1, -1))
                ei_relax = float((I_sim - I_obs).pow(2).mean())
                ei_floor = float((I_enc - I_obs).pow(2).mean())
                # Part DYNAMIQUE pure : on retire l'autoencodeur en comparant les
                # deux décodages entre eux plutôt qu'aux frames. Mesuré sur rock
                # (repos réel) : 6.5e-7 contre un plancher AE de 1.0e-4, soit
                # 160× plus petit ⟹ `ei_relax` recopie le plancher et ne
                # discrimine pas. C'est CETTE colonne qui mesure le modèle.
                ei_dyn = float((I_sim - I_enc).pow(2).mean())

        rows.append(dict(
            a=int(a), b=int(b), dur_s=seg_len * dt, win_s=w * dt,
            e_img_relax=ei_relax, e_img_floor=ei_floor, e_img_dyn=ei_dyn,
            f_enc=f_enc, f_sim=f_sim, n_freq_windows=len(f_list),
            f_windows=[[float(e), float(s)] for e, s in f_list],
            amp_ratio=float(np.nanmean(amp_list)) if amp_list else float('nan'),
            amp_windows=[float(x) for x in amp_list],
            coh_windows=[float(x) for x in coh_list],
            v1_windows=[[float(e), float(s)] for e, s in v1_list],
            f_relerr=abs(f_sim - f_enc) / f_enc if f_enc > 0 else float('nan'),
            rest_mode=mode, n_static_s=n_static * dt,
            e_relax=e_relax, e_relax_rel=e_relax / q_std,
            relax_fnorm=fnorm, relax_iters=n_it, relax_conv=conv,
            e_roll=e_roll, e_roll_rel=e_roll / q_std,
            diverged=False))

    ok = [r for r in rows if not r['diverged']]
    if segs and not ok:
        print('  ⚠ tous les intervalles libres ont divergé : (1) et (2) non mesurés.')

    def agg(key):
        v = np.array([r[key] for r in ok if np.isfinite(r.get(key, np.nan))])
        if v.size == 0:
            return float('nan'), float('nan'), 0
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0, v.size

    # La fréquence s'agrège sur TOUTES les fenêtres d'analyse (pavage des
    # intervalles), pas sur les intervalles : c'est là qu'est la statistique.
    f_all = np.array([abs(s - e) / e for r in ok for (e, s) in r.get('f_windows', [])
                      if e > 0 and np.isfinite(s)])
    f_rel_m = float(f_all.mean()) if f_all.size else float('nan')
    f_rel_se = float(f_all.std(ddof=1) / np.sqrt(f_all.size)) if f_all.size > 1 else 0.0
    f_n = int(f_all.size)
    # Rapport d'amplitude d'oscillation sim/enc sur les mêmes fenêtres. Une
    # erreur de fréquence ne se lit QU'AVEC lui : si le rollout n'oscille pas
    # (rapport ≪ 1), son « pic spectral » est la fuite d'une décroissance
    # monotone, pas une fréquence propre, et l'erreur de fréquence est un
    # plancher optimiste — le vrai défaut est l'absence d'oscillation.
    a_all = np.array([x for r in ok for x in r.get('amp_windows', []) if np.isfinite(x)])
    amp_m = float(a_all.mean()) if a_all.size else float('nan')
    amp_se = float(a_all.std(ddof=1) / np.sqrt(a_all.size)) if a_all.size > 1 else 0.0
    c_all = np.array([x for r in ok for x in r.get('coh_windows', []) if np.isfinite(x)])
    coh_m = float(c_all.mean()) if c_all.size else float('nan')
    v1_all = np.array([[e, s] for r in ok for (e, s) in r.get('v1_windows', [])
                       if e > 0 and np.isfinite(s)])
    if v1_all.size:
        v1_rel = np.abs(v1_all[:, 1] - v1_all[:, 0]) / v1_all[:, 0]
        v1_err = float(v1_rel.mean())
        v1_se = float(v1_rel.std(ddof=1) / np.sqrt(v1_rel.size)) if v1_rel.size > 1 else 0.0
    else:
        v1_err, v1_se = float('nan'), float('nan')
    er_m, er_se, er_n = agg('e_relax_rel')
    ed_m, ed_se, ed_n = agg('e_roll_rel')
    if _args.period_s is not None:
        T_med, T_src = float(_args.period_s), 'IMPOSÉE (--period-s)'
    elif ok:
        T_med = float(np.median([1.0 / r['f_enc'] for r in ok if r['f_enc'] > 0]))
        T_src = 'oscillation libre'
    else:
        # Pas d'intervalle libre : la « période » du critère (3) est celle du mode
        # dominant de q_enc sur toute la vidéo (forcée, donc c'est la période du
        # mouvement observé, pas une fréquence propre — dit tel quel dans la sortie).
        f_glob, _, _ = dominant_freq(z_np[lo:hi], dt, G)
        assert np.isfinite(f_glob) and f_glob > 0, 'aucune fréquence dominante dans q_enc'
        T_med, T_src = 1.0 / f_glob, 'mode dominant de q_enc (forcé)'

    # ════════════════════════════════════════════════════════════════════════
    # (3) ERREUR MOYENNE SUR ~N PÉRIODES (pression MESURÉE, départs au hasard)
    # ════════════════════════════════════════════════════════════════════════
    H = int(round(_args.n_periods * T_med / dt))
    rng = np.random.default_rng(_args.seed)
    hi_w = n - H - 3
    assert hi_w > lo, (f'pas assez de frames ({n}) pour {_args.n_periods} périodes '
                       f'({H} frames) ; réduire --n-periods ou --skip-sec')
    if _args.stride > 0:
        starts = list(range(lo, hi_w, _args.stride))
        print(f'Départs ÉNUMÉRÉS dans [{lo}, {hi_w}] au pas {_args.stride} : '
              f'{len(starts)} fenêtres (couverture exhaustive du held-out chargé)')
    else:
        starts = rng.integers(lo, hi_w, size=min(_args.n_windows, hi_w - lo)).tolist()
    per_win, curves, n_div = [], [], 0
    per_win_img, per_win_floor = [], []
    # Amplitude et erreur relative EN RÉGIME FORCÉ. Le rapport d'amplitude du
    # critère (1) est mesuré en oscillation LIBRE (P gelée, intervalles à P
    # constante) : sur un actionneur pneumatique c'est le régime le moins
    # intéressant, la question étant l'amplitude de la réponse à un échelon.
    # Ici P est la pression MESURÉE et les départs sont tirés partout, échelons
    # compris. `nrmse_win` normalise en outre l'erreur par l'amplitude de LA
    # FENÊTRE (et non par le RMS global du clip) ⟹ comparable d'un cas à l'autre
    # sans dépendre du régime d'excitation de la vidéo entière.
    per_win_amp, per_win_nrmse = [], []
    for s in starts:
        # ⚠️ ALIGNEMENT TEMPOREL (corrigé le 2026-08-04, le décalage d'une frame
        # pénalisait nos chiffres de ~20 % à horizon égal). `viz.simulate_rk4`
        # renvoie `traj = [z0, …]` : `traj[i]` est l'état à `s + i`, et `traj[0]`
        # EST la condition initiale. On confronte donc `traj[i]` à la frame `s+i`.
        # C'est la convention du notebook des auteurs (cellule 19 de
        # `Latent_dynamics_learning.ipynb` : `z_pred_seq = [mu_seq[0:1]]` puis n-1
        # pas), donc celle de `eval_multistep_mse.py`. Le point
        # `i=0` est trivial en latent (erreur nulle) mais vaut le plancher AE en
        # image ; il est inclus des DEUX côtés, c'est ce qui rend la comparaison
        # inter-architectures licite.
        tr = rollout(s, H, const_pressure=False)
        if tr is None:
            n_div += 1
            continue
        # Erreur en métrique de visibilité, divisée par d pour rester à l'échelle
        # d'une MSE par composante (trace(Ḡ)=d ⟹ se réduit à la MSE si Ḡ=I).
        idx = np.arange(s, s + H)
        err = gnorm2(tr.cpu().numpy() - z_np[idx], G) / D                      # (H,)
        curves.append(err)
        per_win.append(err.mean())
        sim_w = tr.cpu().numpy()
        enc_w = z_np[idx]
        amp_e_w = float(np.sqrt(gnorm2(enc_w - enc_w.mean(0), G).mean()))
        amp_s_w = float(np.sqrt(gnorm2(sim_w - sim_w.mean(0), G).mean()))
        if amp_e_w > 0:
            per_win_amp.append(amp_s_w / amp_e_w)
            per_win_nrmse.append(float(np.sqrt(err.mean() * D)) / amp_e_w)
        # ── (3) en espace IMAGE : MSE [0,1] à R², même convention que
        # eval_multistep_mse.py : l'état `traj[i]` (instant `s+i`) est confronté à
        # la frame `s+i`, point initial inclus (cf. la note d'alignement ci-dessus).
        if decoder is not None:
            with torch.no_grad():
                gt = gt_at(idx)
                per_win_img.append(float((decode(tr.cpu()) - gt).pow(2)
                                         .mean(dim=(1, 2, 3)).mean()))
                # Plancher AE sur le latent BRUT (non lissé), comme
                # eval_multistep_mse.py : décoder le latent lissé ferait payer au
                # plancher le coût du lissage (mesuré : 6.2e-5 contre 1.25e-5),
                # et le rapport MSE/plancher ne serait plus comparable au tableau
                # 0.5 s de l'article.
                per_win_floor.append(float((decode(z_raw[idx]) - gt).pow(2)
                                           .mean(dim=(1, 2, 3)).mean()))
    assert per_win, 'tous les rollouts du critère (3) ont divergé.'
    per_win = np.asarray(per_win)
    mse_lat = float(per_win.mean())
    mse_se = float(per_win.std(ddof=1) / np.sqrt(len(per_win))) if len(per_win) > 1 else 0.0
    # q_std est un RMS par composante dans la même métrique ⟹ le rapport est
    # invariant à l'échelle de Ḡ comme à celle du latent.
    nrmse = float(np.sqrt(mse_lat * D) / q_std)

    def _ms(v):
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            return float('nan'), float('nan')
        return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(v.size))
                                 if v.size > 1 else 0.0)
    amp_f_m, amp_f_se = _ms(per_win_amp)
    nrmse_w_m, nrmse_w_se = _ms(per_win_nrmse)
    curve = np.mean(np.stack(curves), axis=0)
    mse_img = float(np.mean(per_win_img)) if per_win_img else float('nan')
    mse_img_se = (float(np.std(per_win_img, ddof=1) / np.sqrt(len(per_win_img)))
                  if len(per_win_img) > 1 else float('nan'))
    floor_img = float(np.mean(per_win_floor)) if per_win_floor else float('nan')

    # ════════════════════════════════════════════════════════════════════════
    # Rapport
    # ════════════════════════════════════════════════════════════════════════
    print('\n' + '═' * 74)
    print(f'  {video_dir.name}   |   {lnn_path.name}')
    print('═' * 74)
    print(f'Référence enc(x) : période médiane {T_med:.3f} s ({T_med/dt:.1f} frames), '
          f'f = {1/T_med:.4f} Hz   [{T_src}]')
    print(f'Échelle latente  : RMS(q_enc − q̄) = {q_std:.4f}   max‖q_enc‖ = {r_max:.3f}')
    print('-' * 74)
    if ok:
        print(f'(1) ERREUR DE FRÉQUENCE DU MODE VISIBLE (oscillation libre, '
              f'{f_n} fenêtres sur {len(ok)} intervalles)')
        if v1_all.size:
            print(f'   ▸ mode le plus visible (projection sur le 1ᵉʳ vec. propre de Ḡ) '
                  f'— LE CHIFFRE À CITER')
            print(f'      f_enc = {v1_all[:,0].mean():.4f} Hz   '
                  f'f_sim = {v1_all[:,1].mean():.4f} Hz')
            print(f'      erreur relative  {100*v1_err:6.2f} %   ± {100*v1_se:.2f} '
                  f'(erreur-type)')
        print(f'   ▸ spectre agrégé, pondéré par Ḡ (toutes directions visibles)')
        print(f'      f_enc = {np.mean([r["f_enc"] for r in ok]):.4f} Hz   '
              f'f_sim = {np.nanmean([r["f_sim"] for r in ok]):.4f} Hz')
        print(f'      erreur relative  {100*f_rel_m:6.2f} %   ± {100*f_rel_se:.2f} '
              f'(erreur-type)')
        if v1_all.size and np.isfinite(f_rel_m) and abs(f_rel_m - v1_err) > 0.15:
            print(f'      ⚠ les deux diffèrent de {100*abs(f_rel_m-v1_err):.0f} points : '
                  f'l\'agrégat est dominé par une\n        direction AUTRE que la plus '
                  f'visible. Citer la ligne « mode le plus visible ».')
        # N'annoter que si la fréquence est FAUSSE : à erreur faible, le pic du
        # rollout est déjà à f_enc et ce chiffre vaut 1 trivialement.
        note = ''
        if np.isfinite(coh_m) and np.isfinite(f_rel_m) and f_rel_m > 0.15:
            note = ('\n        ⟹ le bon mode EST présent dans le rollout, mais son pic '
                    'dominant est ailleurs\n           (typiquement une dérive plus lente) : '
                    'défaut de HIÉRARCHIE des modes'
                    if coh_m > 0.3 else
                    '\n        ⟹ le bon mode est ABSENT du rollout : défaut de raideur '
                    'ou excès de dissipation')
        print(f'      puissance du rollout à f_enc / son max : {coh_m:.3f}{note}')
        print(f'      amplitude sim/enc {amp_m:6.3f}     ± {amp_se:.3f}'
              + ('   ⚠ le rollout n\'oscille quasiment pas : la fréquence ci-dessus'
                 '\n        est celle d\'une décroissance monotone, pas un mode propre'
                 if np.isfinite(amp_m) and amp_m < 0.3 else ''))
        print('-' * 74)
        print(f'(2) ERREUR DE POSITION DE REPOS ({er_n} intervalles ; '
              f'modes : { ", ".join(sorted({r["rest_mode"] for r in ok})) })')
        print(f'      (normalisée par l\'amplitude RMS de q_enc = {q_std:.4f} ; '
              f'amplitude crête = {q_peak:.4f})')
        print(f'      équilibre statique (relaxation)  {er_m:.4f} × ampl.RMS '
              f'± {er_se:.4f}   ({agg("e_relax")[0]/q_peak:.4f} × ampl.crête, '
              f'absolu {agg("e_relax")[0]:.4f})')
        print(f'      équilibre dynamique (rollout)    {ed_m:.4f} × ampl.RMS '
              f'± {ed_se:.4f}   ({agg("e_roll")[0]/q_peak:.4f} × ampl.crête, '
              f'absolu {agg("e_roll")[0]:.4f})')
        ei_m, _, ei_n = agg('e_img_relax')
        ef_m, _, _ = agg('e_img_floor')
        ed_img, _, _ = agg('e_img_dyn')
        if ei_n:
            print(f'      IMAGE : part dynamique MSE(decode(q_sim), decode(q_enc)) = '
                  f'{ed_img:.3e}')
            print(f'              contre frames réelles {ei_m:.3e} | plancher AE '
                  f'{ef_m:.3e} (rapport {ei_m/ef_m:.2f}×, ~1 ⟹ noyé dans l\'AE)')
    else:
        print('(1) ERREUR DE FRÉQUENCE      : non mesurable (aucune oscillation libre)')
        print('(2) ERREUR DE POSITION DE REPOS : non mesurable (aucun palier de pression)')
    print('-' * 74)
    print(f'(3) ERREUR MOYENNE SUR {_args.n_periods:g} PÉRIODES '
          f'({H} frames = {H*dt:.2f} s, {len(per_win)}/{len(starts)} départs finis)')
    print(f'      MSE latente  {mse_lat:.4e}  ± {mse_se:.2e}')
    print(f'      NRMSE        {nrmse:.4f}   (= RMSE / RMS de q_enc ; 1.0 = prédire la moyenne)')
    if per_win_amp:
        print(f'      RÉGIME FORCÉ (P mesurée, {len(per_win_amp)} fenêtres) :')
        print(f'        amplitude sim/enc  {amp_f_m:.3f}  ± {amp_f_se:.3f}   '
              f'(contre {amp_m:.3f} en oscillation libre)'
              if np.isfinite(amp_m) else
              f'        amplitude sim/enc  {amp_f_m:.3f}  ± {amp_f_se:.3f}')
        print(f'        NRMSE par fenêtre  {nrmse_w_m:.4f}  ± {nrmse_w_se:.4f}   '
              f'(erreur / amplitude de LA FENÊTRE, pas du clip entier)')
    if per_win_img:
        print(f'      IMAGE : MSE {mse_img:.4e} ± {mse_img_se:.2e}   '
              f'plancher AE {floor_img:.4e}   rapport {mse_img/floor_img:.1f}×')
        print(f'        (MSE image [0,1] à {R}², contre les VRAIES frames — même '
              f'convention que le tableau 0.5 s)')
    if n_div:
        print(f'      ⚠ {n_div}/{len(starts)} rollouts divergents (exclus de la moyenne)')
    print('═' * 74)

    # ── Plot de diagnostic ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    if spectra is not None:
        fr_e, sp_e, fr_s, sp_s, a0, w0, ze, zs, fe, fs = spectra
        t = np.arange(w0) * dt
        ax = axes[0, 0]
        for j in range(min(D, 4)):
            ax.plot(t, ze[:, j], color=f'C{j}', lw=1.2,
                    label='enc(x)' if j == 0 else None)
            ax.plot(t, zs[:, j], color=f'C{j}', lw=1.0, ls='--',
                    label='rollout' if j == 0 else None)
        ax.set_xlabel('temps (s)'); ax.set_ylabel('q')
        ax.set_title(f'(1) oscillation libre — intervalle @ {a0*dt:.1f} s')
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        ax.semilogy(fr_e, sp_e / max(sp_e.max(), 1e-30), color='C0', label='enc(x)')
        ax.semilogy(fr_s, sp_s / max(sp_s.max(), 1e-30), color='C3', ls='--', label='rollout')
        ax.axvline(fe, color='C0', alpha=0.4); ax.axvline(fs, color='C3', ls='--', alpha=0.4)
        ax.set_xlim(0, min(fr_e.max(), 8 * max(fe, 1e-6)))
        ax.set_ylim(1e-6, 2)
        ax.set_xlabel('fréquence (Hz)'); ax.set_ylabel('puissance (norm.)')
        ax.set_title(f'(1) spectres — f_enc={fe:.3f} Hz, f_sim={fs:.3f} Hz '
                     f'({100*abs(fs-fe)/fe:.1f} %)')
        ax.legend(fontsize=8)

    ax = axes[1, 0]
    idx = np.arange(len(ok))
    ax.bar(idx - 0.2, [r['e_relax_rel'] for r in ok], width=0.4, label='relaxation (statique)')
    ax.bar(idx + 0.2, [r['e_roll_rel'] for r in ok], width=0.4, label='rollout (dynamique)')
    ax.axhline(er_m, color='C0', ls=':', lw=1)
    ax.set_xlabel('intervalle libre'); ax.set_ylabel('‖Δq_eq‖ / RMS(q_enc)')
    ax.set_title('(2) erreur de position de repos')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(np.arange(H) * dt, curve, color='C2')
    ax.axhline(q_std ** 2 / D, color='k', ls=':', lw=1,
               label='variance de q_enc (prédire la moyenne)')
    for kp in range(1, int(_args.n_periods) + 1):
        ax.axvline(kp * T_med, color='gray', alpha=0.25, lw=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('temps (s)'); ax.set_ylabel('MSE latente')
    ax.set_title(f'(3) erreur sur {_args.n_periods:g} périodes — moyenne '
                 f'{mse_lat:.2e} (NRMSE {nrmse:.2f})')
    ax.legend(fontsize=8)

    fig.suptitle(f'{video_dir.name} — {lnn_path.name}', fontsize=10)
    fig.tight_layout()
    tag = _args.out or f'freq_rest_cycles_{lnn_path.stem}.png'
    out = config.SAVE_DIR / tag
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f'Plot : {out}')

    # ── JSON ─────────────────────────────────────────────────────────────────
    res = {
        'case': str(config.SAVE_DIR.parent.name), 'video': video_dir.name,
        'lnn': lnn_path.name, 'encoder': enc_path.name,
        'd': D, 'dt': dt, 'n_frames': n,
        'sigma_q': _cfg('SMOOTH_LATENT_SIGMA', None),
        'sigma_p': _cfg('SMOOTH_PRESSURE_SIGMA', None),
        'cq_eps': _cfg('LNN_RAYLEIGH_CQ_EPS', None),
        'v0_estimator': _v0_name, 'integrator': _cfg('LNN_INTEGRATOR', 'verlet'),
        'q_rms': q_std, 'q_peak': q_peak, 'q_norm_max': r_max,
        'rest_err_relax_rel_peak': agg('e_relax')[0] / q_peak,
        'rest_err_roll_rel_peak': agg('e_roll')[0] / q_peak, 'T_median_s': T_med, 'T_source': T_src,
        'n_free_segments': len(ok), 'n_freq_windows': f_n,
        'freq_err_rel': f_rel_m, 'freq_err_rel_se': f_rel_se,
        'amp_ratio_sim_enc': amp_m, 'amp_ratio_sim_enc_se': amp_se,
        'sim_power_at_f_enc': coh_m,
        'metric': _args.metric if G is not None else 'isotropic (Ḡ absente)',
        'visibility_evals': None if lam_vis is None else [float(x) for x in lam_vis],
        'freq_err_rel_top_visible_mode': v1_err,
        'freq_err_rel_top_visible_mode_se': v1_se,
        'freq_enc_hz_top_visible': float(v1_all[:, 0].mean()) if v1_all.size else None,
        'freq_sim_hz_top_visible': float(v1_all[:, 1].mean()) if v1_all.size else None,
        'pressure_span_max_frac': p_span_max,
        'freq_enc_hz': float(np.mean([r['f_enc'] for r in ok])) if ok else None,
        'freq_sim_hz': float(np.nanmean([r['f_sim'] for r in ok])) if ok else None,
        'rest_err_relax_rel': er_m, 'rest_err_relax_rel_se': er_se,
        'rest_err_relax_abs': agg('e_relax')[0],
        'rest_err_roll_rel': ed_m, 'rest_err_roll_rel_se': ed_se,
        'rest_err_roll_abs': agg('e_roll')[0],
        'n_periods': _args.n_periods, 'horizon_frames': H, 'horizon_s': H * dt,
        'mse_latent': mse_lat, 'mse_latent_se': mse_se, 'nrmse': nrmse,
        'amp_ratio_forced': amp_f_m, 'amp_ratio_forced_se': amp_f_se,
        'nrmse_per_window': nrmse_w_m, 'nrmse_per_window_se': nrmse_w_se,
        'n_windows_forced': int(len(per_win_amp)),
        'eval_res': R,
        'mse_image': mse_img, 'mse_image_se': mse_img_se, 'ae_floor_image': floor_img,
        'rest_err_image': agg('e_img_relax')[0], 'rest_ae_floor_image': agg('e_img_floor')[0],
        'rest_err_image_dyn': agg('e_img_dyn')[0],
        'n_windows_ok': int(len(per_win)), 'n_windows_div': int(n_div),
        'segments': rows,
    }
    if _args.json:
        with open(_args.json, 'w') as fh:
            json.dump(res, fh, indent=1)
        print(f'JSON : {_args.json}')


if __name__ == '__main__':
    main()
