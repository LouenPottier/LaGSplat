"""
Cas Krauss 2026 — SCR souple **2-segments**, sur LEURS NPZ et LEUR SPLIT.

Objet : comparaison directe avec Krauss et al. 2026 (VON, RA-L), en prenant
exactement leurs données et leur découpe, pour que la seule variable soit le
modèle :

  - données   : ``scr_dataset_raw_2seg_32x32_59fps.npz`` (processed_data.zip),
                images 32×32 RGB, pressions déjà alignées par les auteurs
                (événement de dépressurisation, cf. ``SCR_data_processing.ipynb``)
                ⟹ NI CSV, NI recalage maison (voir dataset.load_npz_frames) ;
  - split     : premiers 80 % = train, derniers 20 % = val, découpe CONTIGUË
                (``train_val_ratio: 0.80``, ``n_train = int(0.8·n_total)``).
                Matérialisé en deux fichiers par ``make_krauss_npz_split.py``
                (43 872 / 10 968 frames) ;
  - horloge   : 59.94 fps ⟹ ``DT = 1001/60000``. L'horizon 0.5 s du tableau fait
                donc **30 pas**, comme chez eux.

Réglages de ce run : ``IMG_SIZE=(32,32)``, ``d=4``, ``SEQ_STRIDE=2``, lissage
gaussien σ=1 frame (q ET pression). Structure du LNN : masse apprise M(q),
dissipation pleine C(q), potentiel invexe, résidu intégral, blanchiment latent
post-hoc.

Sélection :  py train_ae.py --config ../cases/krauss2026_2seg_npz/config.py
``_HERE`` et ``Path`` sont injectés par le bootstrap ; ``SAVE_DIR`` est ancré sur
``_HERE/checkpoints``.
"""

# ── Données (NPZ Krauss, split officiel 80/20) ─────────────────────────────
VIDEO_DIR = _HERE / 'data' / 'scr_2seg_32x32_train80.npz'
VAL_VIDEO = 'scr_2seg_32x32_val20.npz'    # même dossier ; eval_multistep_mse --video
IMG_SIZE  = (32, 32)
ENC_COLOR = True
DEC_COLOR = True

# 59.94 fps (NTSC), pas 60 : les NPZ sont ré-échantillonnés depuis les MP4 120000/1001.
DT = 1001 / 60000

# Frame de repos = médiane des 100 premières frames du dataset (ancrage de z_rest).
REST_FIRST_N_FRAMES = 100

# ── Dimension latente (DDL) ────────────────────────────────────────────────
TSNE_DIM   = 4
LATENT_DIM = TSNE_DIM

# ── Fenêtres du résidu FD : 1 échantillon sur 2 ────────────────────────────
# Décimation ×2 des fenêtres (dt du résidu = 2 frames ≈ 33 ms). L'ODE reste PAR
# FRAME (v en unités de position/frame) ; c'est le SNR de la dérivée seconde qui
# remonte (∝ k²), pas l'échelle de temps du modèle.
SEQ_STRIDE = 2

# ── Lissage temporel gaussien σ=1 frame (~17 ms à 59.94 fps) ───────────────
SMOOTH_LATENT         = True
SMOOTH_LATENT_MODE    = 'gaussian'
SMOOTH_LATENT_SIGMA   = 1.0
SMOOTH_PRESSURE       = True
SMOOTH_PRESSURE_SIGMA = 1.0

# ── Forçage de pression pneumatique ────────────────────────────────────────
# Les pressions vivent DANS le NPZ (clés p1..p4, seg.1 = p1,p2 ; seg.2 = p3,p4),
# déjà échantillonnées par frame et déjà divisées par 101 325 Pa ⟹ NORM = 1.0.
# PRESSURE_DIR / PRESSURE_SYNC_OFFSETS sont sans objet sur une source .npz.
LNN_PRESSURE      = True
LNN_PRESSURE_MODE = 'invex'
PRESSURE_COLS     = ['p1', 'p2', 'p3', 'p4']
PRESSURE_NORM     = 1.0
PRESSURE_DIR      = None
PRESSURE_SYNC_OFFSETS = None

# ── Encodeur / espace latent ───────────────────────────────────────────────
# Whitening par batch coupé (instable sur ce cas) ; blanchiment latent FIGÉ
# post-AE à la place (compute_latent_whiten.py → latent_whiten.pt).
ENC_NORMALIZE      = False
LATENT_WHITEN      = True
LATENT_WHITEN_MODE = 'pca'

# ── LNN : masse apprise M(q), dissipation pleine C(q), potentiel invexe ─────
LNN_FREEZE_ENCODER         = True     # AE figé (requis par train_lnn_fixedae / _krauss)
LNN_METRIC_FROM_DECODER    = False
LNN_MASS_LEARNED           = True
LNN_POTENTIAL_FROM_DECODER = False
LNN_MASS_LEARNED_HIDDEN    = [64, 64]
LNN_MASS_LEARNED_EPS       = 0.1
LNN_METRIC_MASS_INIT       = 2.0

LNN_RAYLEIGH_CQ       = True
LNN_RAYLEIGH_PULLBACK = False
LNN_RAYLEIGH_CQ_INIT  = 0.2
LNN_RAYLEIGH_CQ_EPS   = 5e-3

LNN_LOSS_MODE = 'integral'   # résidu inverse-intégral (sans q̈)
LNN_INVEX     = True         # potentiel de déformation invexe

LNN_EPOCHS     = 500
LNN_PLOT_EVERY = 1
# Loss de Krauss (train_lnn_krauss.py) : le terme décodé se compare à 32², soit la
# résolution NATIVE des données ET celle de la métrique du papier — aucun
# rééchantillonnage (le défaut 64 aurait suréchantillonné les cibles).
LNN_KRAUSS_DEC_RES = 32
VIZ_MAX_FRAMES = 5000        # ~83 s à 59.94 fps

# ── Autoencodeur conjoint (train_ae.py) ────────────────────────────────────
# 43 872 frames 32×32 : une époque = ~685 steps à AE_BATCH=64. Rendu 32² très
# léger ⟹ on garde 2048 splats (plancher AE le plus bas possible, c'est lui qui
# borne la MSE multi-step comparée à Krauss).
AE_EPOCHS       = 100
AE_BATCH        = 64
AE_PRINT_EVERY  = 1
AE_PLOT_EVERY   = 5
AE_PRUNE_WARMUP = 20
AE_PRUNE_EVERY  = 20
AE_ENC_LR       = 5.e-4
AE_DEC_LR       = 5.e-4
AE_N_GAUSSIANS            = 2048
DEC2PT_GSPLAT_N_GAUSSIANS = 2048

DEC_PLOT_EVERY = 5
