"""
Pipeline 2 — Configuration centrale.
Modifier ici tous les hyperparamètres.
"""
from pathlib import Path

# ── Données ────────────────────────────────────────────────────────────────
VIDEO_DIR     = Path('../cases/krauss2026_2seg_npz/data/scr_2seg_32x32_train80.npz')
VAL_VIDEO     = None  # nom de fichier dans VIDEO_DIR à utiliser uniquement pour la validation
CROP          = None  # None = pas de crop, sinon (x, y, w, h) en pixels avant resize
IMG_SIZE      = (128, 128)   # (H, W)
ENC_COLOR     = True         # True = encodeur en RGB (3 canaux), False = niveaux de gris
DEC_COLOR     = True         # True = décodeur reconstruit en RGB, False = niveaux de gris
REST_VIDEO    = None   # vidéo de repos : None = utilise les frames du dataset principal
REST_N_FRAMES = 60     # nb de dernières frames de REST_VIDEO (si REST_FIRST_N_FRAMES=0)
REST_FIRST_N_FRAMES = 1  # > 0 : médiane des N premières frames (REST_VIDEO ou dataset)
                          # 0   : médiane des REST_N_FRAMES dernières frames (REST_VIDEO)
                          #       ou médiane globale si REST_VIDEO=None
SUBTRACT_REST = False  # legacy : ignoré (conservé pour compatibilité d'appels)


# ── Mode d'initialisation de z_rest ───────────────────────────────────────────
# 'rest_frame' : z_rest = enc(rest_frame), non apprenable, remis à jour après
#                chaque mise à jour des poids de l'encodeur.
# 'barycenter' : z_rest = barycentre de tous les z, apprenable.
Z_REST_MODE = 'rest_frame'# ou 'barycenter'

# ── Résolution automatique des chemins ─────────────────────────────────────
# Si on tourne depuis un dossier checkpoint (présence de .checkpoint),
# VIDEO_DIR est résolu depuis le dossier parent et SAVE_DIR = dossier courant.
_HERE = Path(__file__).parent.resolve()
_SENTINEL = _HERE / '.checkpoint'
if _SENTINEL.exists():
    # On est dans un dossier checkpoint — vidéos dans le dossier parent
    VIDEO_DIR = _HERE.parent / VIDEO_DIR
    if REST_VIDEO is not None:
        # REST_VIDEO reste un nom de fichier, résolu dans VIDEO_DIR
        pass

# ── TSNE ───────────────────────────────────────────────────────────────────
TSNE_DIM         = 3           # 1, 2 ou 3 ou 4
TSNE_PERPLEXITY  = 100
TSNE_MAX_ITER    = 1000
TSNE_RANDOM_STATE = 42

# ── Encodeur ───────────────────────────────────────────────────────────────
# ENC_NORMALIZE=True  : blanchiment par covariance intégré à l encodeur
#   - forward() applique le whitening PAR BATCH dans le graphe (anti-collapse actif)
#   - running stats (μ, E[zzᵀ]) accumulées en moyenne mobile (ENC_WHITEN_MOMENTUM)
#     SUR LE BATCH COURANT (jamais une passe sur toute la base) → estiment le
#     blanchiment global, utilisées en inférence et pour z_rest (cf. _global_W).
#     Coût ≈ un batch (+ eigh D×D). Varient avec les poids de l'encodeur.
#   - checkpoints incluent running_mean/running_W → cohérence inférence/entraînement
# ENC_NORMALIZE=False : z bruts (comportement original)
ENC_NORMALIZE = True
# Moment de la moyenne mobile des stats globales de blanchiment (μ, E[zzᵀ]).
# Petit ⟹ moyennage sur ~1/momentum batches (≈ base entière, peu de bruit) ;
# grand ⟹ suit plus vite les poids de l'encodeur mais plus bruité. 0.05 ≈ 20 batches.
ENC_WHITEN_MOMENTUM = 0.05
# Plancher (clamp) des valeurs propres de la covariance dans WhiteningLayer.
# W = eigvecs · clamp(eigvals, min=ENC_WHITENING_EPS)^{-1/2}. Si les valeurs propres
# du z BRUT tombent sous ce plancher (cas CpAE : latent collapsé, λ ~ 1e-7), le clamp
# domine → W ≈ (1/√eps)·I = scaling constant, et le gradient anti-collapse à travers
# eigh s'annule. Le descendre sous l'échelle brute (1e-7) rétablit un vrai blanchiment
# (sortie std≈1) ET la pression anti-collapse. (Avant : hard-codé 1e-4.)
ENC_WHITENING_EPS = 1.e-7
ENC_HIDDEN  = [64, 32, 32]

# ── Blanchiment latent POST-HOC (figé, appris après l'autoencodeur) ──────────
# Alternative à ENC_NORMALIZE (WhiteningLayer par batch, instable à d élevé) :
# un blanchiment FIGÉ appris UNE fois sur les stats globales du dataset après
# l'entraînement de l'AE (compute_latent_whiten.py → latent_whiten.pt), inséré
# entre l'encodeur GELÉ et le LNN. Le LNN (+ métrique pull-back) travaille alors
# dans un espace u équilibré (cov ≈ I) SANS réentraîner encodeur ni décodeur ;
# l'inverse est appliqué avant le décodeur. Transformation LINÉAIRE ⟹ la
# géométrie du décodeur (J, μ_z, Σ_zz⁻¹) se transporte de façon covariante
# (LatentWhiten.transform_geom, appliqué dans precompute_metric_geom.py).
# Opt-in : False ⟹ comportement strictement inchangé. Typiquement on met
# ENC_NORMALIZE=False ET LATENT_WHITEN=True (les deux sont indépendants).
# Pipeline (AE figé) : train_ae → compute_latent_whiten → precompute_metric_geom
#                      → train_lnn_fixedae → generate_video.
LATENT_WHITEN      = False
LATENT_WHITEN_MODE = 'pca'      # 'pca' (décorrélé, var=1, axes=CP) | 'zca' (rotation min.)
LATENT_WHITEN_EPS  = 1.e-6      # plancher des valeurs propres de la covariance latente

# ── Anti-effondrement du latent par divergence KL (Castañeda et al., CVPR 2025) ──
# *Learning Physics From Video* (castaneda2025learning).
# ALTERNATIVE au whitening (ENC_NORMALIZE) pour empêcher l'effondrement de l'espace
# latent (E_θ(x)→0 rendant le résidu Euler-Lagrange trivialement nul) pendant train_lnn.
# Au lieu de blanchir la sortie de l'encodeur DANS le graphe, on AJOUTE à la loss du
# résidu le terme L₂ de Castañeda, qui pousse la distribution empirique (par batch, par
# dimension) des latents encodés vers un prior N(0,1) :
#     L_KL = −(1/d) Σⱼ [ 1 + ln σ²ⱼ − μ²ⱼ − σ²ⱼ ]   (= (2/d)·KL(N(μ,σ²)‖N(0,1)) ≥ 0)
# où μⱼ, σ²ⱼ sont la moyenne / variance de la j-ème coordonnée latente sur le batch.
# Minimiser pousse μⱼ→0 (centrage) et σ²ⱼ→1 (variance unité) ⟹ interdit le collapse
# (σ²ⱼ→0 ⟹ −ln σ²ⱼ→+∞, coût explosif). Contrairement au whitening, ce terme ne
# décorrèle PAS les axes (pas de blanchiment de covariance) : il ne contraint que les
# moments MARGINAUX de chaque dimension. Usage prévu : ENC_NORMALIZE=False ET
# LNN_ANTICOLLAPSE_KL=True (z bruts → LNN, la KL tient l'échelle latente). Opt-in ;
# False ⟹ comportement inchangé. Lu par train_lnn.py uniquement.
LNN_ANTICOLLAPSE_KL     = False
LNN_ANTICOLLAPSE_LAMBDA = 1.0     # poids λ du terme KL ajouté à la loss du résidu
LNN_ANTICOLLAPSE_EPS    = 1.e-6   # plancher numérique de σ²ⱼ (évite ln 0 au collapse)

LATENT_DIM  = TSNE_DIM          # doit correspondre à TSNE_DIM
ENC_LR      = 1.e-3
ENC_EPOCHS  = 20
ENC_BATCH   = 64
ENC_GP_LAMBDA = 0          # poids du gradient penalty (0 = désactivé)
ENC_CPAE          = True         # True = CpAEEncoder (CNN grands filtres, Zhu et al. 2025)
ENC_CPAE_LAMBDA_J = 1.0            # poids pénalité nonlocale des filtres (éq. 6)
ENC_CPAE_N_LARGE  = 3              # L* : nb de couches à grands filtres
ENC_CPAE_KERNEL_L = 12             # taille du grand filtre (premières L* couches)
ENC_CPAE_KERNEL_S = 4              # taille du petit filtre (couches suivantes)
ENC_CPAE_CHANNELS = (2, 2, 2)  # canaux par couche conv

# ── LNN ────────────────────────────────────────────────────────────────────
LNN_FD_ORDER = 4                # ordre du schéma de différences finies : 2 ou 4
SEQ_LEN      = {2: 3, 4: 5}[LNN_FD_ORDER]   # dérivé automatiquement
# Sous-échantillonnage temporel des fenêtres FD (VideoSeqDataset) : chaque fenêtre prend
# SEQ_LEN frames ESPACÉES de SEQ_STRIDE frames (indices s, s+k, …, s+(SEQ_LEN-1)k) au lieu
# de frames adjacentes. Le résidu est alors évalué avec dt=SEQ_STRIDE (dérivées PAR FRAME,
# physiques) → compatible avec le rollout à dt=1 frame. Pour une dynamique LENTE devant le
# fps (pendule à 60 fps : ω·dt≈0.04 rad/frame), les frames adjacentes donnent une accélération
# FD minuscule NOYÉE dans le bruit de l'encodeur (SNR ∝ k²) : k>1 remonte le signal physique
# au-dessus du plancher de bruit. 1 = désactivé (frames adjacentes, comportement legacy).
SEQ_STRIDE   = 1
# Forme de la loss EL d'entraînement :
#   'accel'    : résidu d'ACCÉLÉRATION (DeLaN) r = q̈_FD − accel(q,q̇). Utilise la
#                différence SECONDE q̈_FD (bruitée) et inverse
#                la masse M̃⁻¹. Comportement historique.
#   'integral' : résidu inverse-INTÉGRAL SANS q̈ (Laiche et al. 2025, laiche2025noaccel).
#                Forme faible d'Euler-Lagrange intégrée pas à pas : la dérivée totale
#                s'intègre en différence de moments conjugués
#                   p_{k+1} − p_k = ∫(∂L/∂q − ∂D/∂q̇ + Q)dt ≈ (Δt/2)(g_k+g_{k+1}).
#                Ne forme JAMAIS q̈ (q̇ par différence PREMIÈRE) ni n'inverse la masse ;
#                Coriolis capturée gratuitement par Δp (chemin courbe). Exige SEQ_LEN≥4
#                (⟹ LNN_FD_ORDER=4, SEQ_LEN=5).
LNN_LOSS_MODE = 'accel'
# Potentiel pull-back par le décodeur (cf. metrique_latente_decodeur.md) :
#   False : V(q) = ICNN libre DANS la carte latente q (convexité imposée en q —
#           dépendante de la carte, non invariante).
#   True  : Ṽ(q) = V_ICNN(μ(q)), V_ICNN ICNN convexe DANS l'espace décodé (positions μ),
#           μ(q)=μ_xy+J(q−μ_z) affine (géométrie figée du décodeur, même J=∂μ/∂q que
#           M̃=JᵀMJ et Jᵀf). Gradient ∂Ṽ/∂q=Jᵀ∇_μV, Hessienne Jᵀ(∇²V)J : la courbure du
#           potentiel est CONFINÉE à range(Jᵀ) (directions physiques) — neutralise les
#           modes parasites de jauge quand J est rang-déficient (multi-DDL). Exige la
#           géométrie precompute_metric_geom.py (clés J, mu_xy, mu_z).
LNN_POTENTIAL_FROM_DECODER = False
DT          = 1001 / 120000     # pas de temps (s) — Krauss 2026 : 120000/1001 ≈ 119.88 fps (NTSC "120p").
                                # /!\ PAS 1/120 : les MP4 sources sont à 120000/1001 fps (cf. ffprobe _256).
                                # Les CSV de pression sont en secondes RÉELLES → utiliser 1/120 dilatait
                                # l'axe temps de 1001/1000 (≈0.46 s de décalage pression/frame en fin de
                                # demi-vidéo). _128_half est re-taggé 120/1 mais le contenu reste 119.88 fps.
LNN_HIDDEN  = [64, 64 ]
LNN_ICNN = True
# Divergence de Bregman (ICNN uniquement) : E(z) = g(z) − g(z_r) − ∇g(z_r)ᵀ(z−z_r)
# → z_rest = argmin V garanti par construction (gradient nul en z_rest), pas seulement
#   E(z_rest)=0. Sans effet si LNN_ICNN=False. Voir EnergyNet.forward.
LNN_BREGMAN = True
# Plancher fortement convexe : E ← E + ½·ε·‖z−z_rest‖² → minimiseur unique, Hessien SPD.
# 0.0 = désactivé (convexité simple).
LNN_EPS_STRONG = 1e-2
# ── Potentiel INVEX (convexe ∘ difféomorphisme) ────────────────────────────
# Si l'énergie de déformation réelle est INVEX (un seul équilibre = min global,
# lignes de niveau CONNEXES mais NON convexes — p. ex. Krauss) et non strictement
# convexe, l'ICNN seul est trop rigide (sous-ensembles de niveau convexes imposés).
# LNN_INVEX=True compose le cœur convexe g avec un difféomorphisme appris Φ
# (i-ResNet bi-Lipschitz) : E(z) = D_g(Φ(z), Φ(z_rest)). Φ bijectif à Jacobien
# partout non singulier ⟹ unique point stationnaire = min global en z_rest préservé
# (E≥0, E(z_rest)=0, ∇E(z_rest)=0), mais sous-ensembles de niveau = Φ⁻¹(convexes)
# → connexes, non convexes ⟹ potentiel invex. Sans effet si LNN_ICNN=False.
# NB : avec Φ, ∂E/∂z n'a plus la forme analytique softplus ⟹ repli autograd auto.
LNN_INVEX         = False
LNN_DIFFEO_HIDDEN = 64      # largeur du MLP résiduel de chaque bloc i-ResNet
LNN_DIFFEO_BLOCKS = 2       # nb de blocs résiduels composés (Φ = (I+r_K)∘…∘(I+r_1))
LNN_DIFFEO_COEFF  = 0.9     # cap Lipschitz du résidu (<1) → I+r inversible (i-ResNet)
LNN_LR      = 1e-2
LNN_EPOCHS  = 100
LNN_BATCH   = 256
# Encodage par segments contigus pour le fine-tuning de l'encodeur dans le LNN.
# Le dataset de fenêtres glissantes (stride 1) ré-encode chaque frame ~SEQ_LEN fois
# par époque ; avec un encodeur CNN lourd (CpAE) c'est le coût dominant. En encodant
# un segment contigu de LNN_CHUNK_LEN frames UNE fois puis en formant toutes les
# fenêtres de longueur SEQ_LEN dans l'espace latent, chaque frame n'est encodée qu'une
# fois → ~SEQ_LEN× moins de passages encodeur, pour des résidus identiques.
# 0 (ou ≤ SEQ_LEN) = désactivé (mode legacy : fenêtres glissantes).
LNN_CHUNK_LEN   = 2        # longueur des segments contigus (frames)
LNN_CHUNK_BATCH = 64        # nb de segments par step (≈ LNN_CHUNK_BATCH × LNN_CHUNK_LEN frames/step)
LNN_MASS_MATRIX = True
# Échelle initiale de la matrice de masse inverse : Minv ≈ s·I à l'init (L = √s·I).
# s < 1 ⟹ masse plus lourde ⟹ fréquence propre initiale ω = √eig(Minv·∇²V) plus basse.
# Levier le plus direct si l'init du potentiel ICNN est trop raide (ω trop élevée).
# Reste apprenable (n'affecte que le point de départ). 1.0 = comportement historique.
LNN_MINV_INIT = 1.0
# Si True : Minv est FIGÉ à s·I (buffer, hors optimiseur) au lieu d'être appris. La
# fréquence propre est alors portée UNIQUEMENT par le potentiel (ω=√eig(Minv·∇²V), Minv
# constant) : utile quand on veut fixer l'échelle de masse et ne laisser bouger que ∇²V.
# False = Minv = LLᵀ+εI appris (comportement historique).
LNN_MASS_FIXED = False

# ── Métrique latente COURBE dérivée du décodeur ────────────────────────────
# Si True : la matrice de masse n'est plus le Minv libre mais M̃(q) = m·Σᵢ ŵᵢ(q) JᵢᵀJᵢ
# (forme A), précalculée par `precompute_metric_geom.py` à partir du décodeur GELÉ. Active
# le résidu LNN-plein (Coriolis dérivée de M̃(q) par autograd, pas de réseau libre) et la
# dissipation de Rayleigh α·M̃(q)·q̇ (+ β·∇²V, β inactif pour l'instant). Suppose l'AE figé.
# Pose Linv_raw=None (masse libre désactivée). Défaut False → comportement historique.
LNN_METRIC_FROM_DECODER = False
LNN_METRIC_GEOM   = 'metric_geom.pt'   # fichier dans SAVE_DIR (sortie de precompute_metric_geom.py)
LNN_METRIC_MASS_INIT = 1.0             # m initial FIXE (échelle de masse, non apprise ; calibre la
                                       # fréquence propre initiale ω = √eig((1/m)·M̂⁻¹∇²V), ω ∝ 1/√m)
                                       # Sert aussi d'échelle isotrope initiale s de la masse APPRISE
                                       # (LNN_MASS_LEARNED : M̂(q)≡s·I au départ).
# ── Masse q-dépendante APPRISE (alternative à la masse-métrique décodeur) ────
# Si True : M̃(q) n'est PAS dérivée du décodeur mais APPRISE — M̂(q)=L(q)L(q)ᵀ+εI SPD,
# L(q) facteur de Cholesky produit par un MLP (LatentMetricLearned). Même machinerie
# courbe que LNN_METRIC_FROM_DECODER (Coriolis par autograd, résidu plein/intégral,
# accel inverse M̃(q)), mais l'inertie q-dépendante est estimée par le LNN (couplages
# inter-DDL via l'hors-diagonale de L(q)), indépendamment de la cinématique q↦μ figée.
# Ne requiert PAS metric_geom.pt. EXCLUSIF de LNN_METRIC_FROM_DECODER. Dissipation
# recommandée : LNN_RAYLEIGH_CQ (C(q) apprise ; le pull-back exige la géométrie décodeur).
LNN_MASS_LEARNED        = False
LNN_MASS_LEARNED_HIDDEN = [64, 64]     # MLP φ(q) → facteur de Cholesky L(q) de M̂(q)
LNN_MASS_LEARNED_EPS    = 1e-4         # plancher SPD ε de M̂(q) = L(q)L(q)ᵀ + εI
# ── Garde-fous de M̂(q) HORS du support des données (opt-in, défauts neutres) ──
# Diagnostic Krauss 2-seg (2026-07-28) : le MLP de Cholesky extrapole librement
# hors des données. Mesuré le long d'un rayon, de ‖q‖=0 à 256 : ‖∂M/∂q‖ passe de
# 386 à 5.4e5 et cond(M̂) de 8 à 6.6e8 (une valeur propre collée au plancher ε).
# Comme la Coriolis vaut (∂p/∂q)q̇ avec p = M̂(q)q̇, elle est QUADRATIQUE en vitesse
# et proportionnelle à ‖∂M/∂q‖ : elle domine la dissipation (linéaire) dès
# ‖q̇‖ ≈ 0.02–0.09, soit SOUS la vitesse typique des données (0.15/frame), et fait
# diverger le rollout long (‖q‖ → 1e15 en 50 s, contre ‖q‖ ≤ 8.9 dans les données).
# Vérifié : en gelant M̂ à sa moyenne, les mêmes rollouts restent bornés (‖q‖ ≤ 9.1).
#   LNN_MASS_FAR_CONST : L(q) = L∞ + g(q)·(φ(q) − L∞) avec g gaussienne valant 1
#     pour ‖q‖ ≤ R0 puis décroissant sur l'échelle TAU ⟹ M̂ CONSTANTE loin des
#     données (∂M/∂q → 0, Coriolis → 0), inchangée sur le support. L∞ est appris.
#   LNN_MASS_COND_MAX = κ : ridge proportionnel à la trace ⟹ cond(M̂) ≤ κ pour tout q.
# R0 se règle sur les données : en espace blanchi, p99 de ‖q‖ ≈ 4–5 (Krauss 2-seg).
LNN_MASS_FAR_CONST = False     # True = masse constante hors du support des données
LNN_MASS_GATE_R0   = 4.0       # rayon (espace latent blanchi) où la porte commence à fermer
LNN_MASS_GATE_TAU  = 2.0       # largeur de fermeture de la porte
LNN_MASS_COND_MAX  = 0.0       # κ : borne du conditionnement de M̂(q) (0 = désactivé)
LNN_RAYLEIGH_LOG_ALPHA_INIT = -3.0     # log(α) Rayleigh proportionnel masse : dissipation α·M̃·q̇
LNN_RAYLEIGH_BETA = 0.0                # β proportionnel raideur (∇²V) — non câblé (laisser 0)
# Dissipation par CONGRUENCE C̃(q) = M̂^{1/2}(q)·C·M̂^{1/2}(q), C = LLᵀ+εI (d×d SPD apprise).
# Étend α·M̃ (mass-prop.) à un couplage d×d SPD arbitraire en conjuguant un cœur C appris par
# la racine de la métrique : taux dissipé q̇ᵀC̃q̇ = (M̂^{1/2}q̇)ᵀC(M̂^{1/2}q̇) ≥ 0 ssi C SPD ⟹
# dissipation garantie. C=s·I redonne EXACTEMENT s·M̂ (nest le mode mass-prop.). Opt-in.
LNN_RAYLEIGH_C = False                 # True = dissipation par congruence M̂^{1/2}·C·M̂^{1/2}
LNN_RAYLEIGH_C_INIT = 0.05             # échelle init de C (C≈s·I, L=√s·I)
LNN_RAYLEIGH_C_EPS = 1e-6              # plancher SPD ε de C = LLᵀ+εI
# Q-dépendance du cœur C (le sandwich M̂·C·M̂ est déjà q-dép. via M̂) :
#   'const'  : C = C₀ constante (cf. ci-dessus).
#   'scalar' : Rung 1 — porte scalaire C(q) = softplus(φ(q))·C₀, φ MLP ℝᵈ→ℝ. Seule
#              l'INTENSITÉ d'amortissement varie avec q (anisotropie figée par C₀) ;
#              softplus ≥ 0 ⟹ SPD préservée. Init neutre (gate≈1 ⟹ = 'const' au départ).
LNN_RAYLEIGH_C_MODE = 'const'          # 'const' | 'scalar' (Rung 1)
LNN_RAYLEIGH_C_GATE_HIDDEN = [32, 32]  # MLP de la porte φ(q) (mode 'scalar')
# Dissipation visqueuse anisotrope par PULL-BACK : C̃(q)=Σᵢ ŵᵢ(q) Jᵢᵀ C_amb Jᵢ, où
# C_amb=L_ambL_ambᵀ+εI est une dissipation AMBIANTE (espace de rendu, pos×pos) anisotrope
# APPRISE. Exact analogue de la masse M̂(q)=Σᵢ ŵᵢ JᵢᵀJᵢ (= pull-back de l'inertie ambiante
# I), avec C_amb à la place de I : LINÉAIRE en q̇, ANISOTROPE (C_amb pleine), q̇ᵀC̃q̇=
# Σᵢ ŵᵢ‖√C_amb Jᵢq̇‖²≥0 garanti. Même présence ŵᵢ(q)/géométrie Jᵢ que la masse ⟹ cohérence.
# PRIME sur (et exclut) LNN_RAYLEIGH_C / α·M̃. Exige LNN_METRIC_FROM_DECODER. Opt-in.
LNN_RAYLEIGH_PULLBACK      = False     # True = dissipation pull-back Σᵢ ŵᵢ Jᵢᵀ C_amb Jᵢ
LNN_RAYLEIGH_PULLBACK_INIT = 0.05      # échelle init de C_amb (C_amb≈s·I, L_amb=√s·I)
LNN_RAYLEIGH_PULLBACK_EPS  = 1e-6      # plancher SPD ε de C_amb = L_ambL_ambᵀ+εI
# Dissipation de Rayleigh q-dépendante PLEINE : C̃(q) = C(q) = L(q)L(q)ᵀ + εI (d×d SPD).
# Forme la plus générale (« version C complet ») : un MLP φ : ℝᵈ → ℝ^{d(d+1)/2} produit le
# facteur de Cholesky L(q) (triangulaire inf., diagonale > 0 via softplus) ⟹ C(q) SPD pour
# TOUT q. Force dissipative = C(q)q̇, linéaire en q̇, q̇ᵀC(q)q̇ ≥ 0 garanti. PRIME sur (et
# exclut) pull-back / LNN_RAYLEIGH_C / α·M̃. Init NEUTRE : MLP dernière couche poids=0, biais
# diag=softplus⁻¹(√s) ⟹ C(q) ≡ s·I au départ (= mass-prop. isotrope d'intensité s).
# ⚠️ C'est la forme la plus EXPRESSIVE, donc la plus exposée au sur-apprentissage : une
# matrice d'amortissement pleine q-dépendante peut absorber la misspécification de la
# métrique/du potentiel (résidu bas mais physique non extrapolante). À utiliser sous
# régularisation / contrôle. Exige LNN_METRIC_FROM_DECODER. Opt-in, défaut off.
LNN_RAYLEIGH_CQ        = False         # True = dissipation pleine C̃(q)=C(q)=L(q)L(q)ᵀ+εI
LNN_RAYLEIGH_CQ_HIDDEN = [64, 64]      # MLP φ(q) → facteur de Cholesky L(q)
LNN_RAYLEIGH_CQ_INIT   = 0.05          # échelle init s (C(q) ≡ s·I au départ)
# ⚠️ `eps` n'est PAS qu'un garde-fou numérique : c'est le PLANCHER D'AMORTISSEMENT
# ISOTROPE du modèle, et il fixe donc la stabilité du rollout long. Trop bas ⟹ une
# direction de C(q) reste non amortie et la Coriolis quadratique diverge ; trop haut
# ⟹ l'amortissement isotrope remplace la dissipation apprise. Règle de choix : eps de
# l'ordre de 30–50 % de la valeur propre MÉDIANE de C(q) sur les données.
LNN_RAYLEIGH_CQ_EPS    = 1e-6          # plancher SPD ET plancher d'amortissement isotrope
LNN_METRIC_RIDGE  = 1e-4               # plancher de masse (ridge) sur M̃(q) avant inversion
                                       # ⚠️ ABSOLU : à régler EN UNITÉS DE M̃ = m·M̂. Le défaut
                                       # 1e-4 suppose M̂~O(1) et m~1 ; il devient dérisoire dès
                                       # que m ≫ 1 (cas Krauss NPZ recalé, m = 70–459).
# Conditionnement borné de la métrique DÉCODEUR (LatentMetricA), analogue de
# LNN_MASS_COND_MAX pour la masse apprise : ridge proportionnel à la trace, ε_eff=tr(M̂)/κ
# ⟹ cond(M̂) ≤ κ pour TOUT q. Nécessaire parce que chaque JᵀJ est de rang ≤ pos (=2) : loin
# des données la présence softmax sature sur une gaussienne, M̂ retombe au rang 2 et M̃⁻¹
# explose (λmin ×10⁻⁴ mesuré ⟹ rollout divergent). Choisir κ ≫ cond(M̂) sur les données
# (Krauss NPZ : ~2 en 1-seg, ~110 en 2-seg) pour ne pas déformer la région des données.
# 0 = désactivé (comportement historique).
LNN_METRIC_COND_MAX = 0.0

LNN_VISCOUS_MATRIX = True
LNN_COULOMB_MATRIX = True
LNN_VISCOUS        = True          # True = frottement visqueux γ·v
LNN_LOG_GAMMA_INIT = -5.0          # log(γ) initial (ignoré si LNN_VISCOUS=False)
LNN_COULOMB        = False      # True = frottement de Coulomb β·v/‖v‖
LNN_LOG_BETA_INIT  = -6.0          # log(β) initial (ignoré si LNN_COULOMB=False)
LNN_NEURAL_DISS = False     # activer ici
LNN_DISS_HIDDEN = [64, 64]   # architecture du réseau
LNN_FREEZE_ENCODER = False      # True = encodeur figé pendant LNN
LNN_GRAD_CLIP = 100. #
LNN_ENC_LR    = 1e-4          # LR encodeur pendant fine-tuning LNN (< LNN_LR)
LNN_GP_LAMBDA = 0            # poids du rank penalty sur le Jacobien de l'encodeur (0 = désactivé)
NN_VIDEO_K        = 5    # nb de voisins pour la médiane pour la reconstruction vidéo knn
NN_VIDEO_N_FRAMES = 500    # 0 = durée originale, > T_orig = extrapolation
LNN_PLOT_EVERY = 10   # plots intermédiaires train_lnn (0 = désactivé)
VIZ_MAX_FRAMES = 200 # limite les trajectoires de visualisation (None = pas de limite)

# ── Intégration de l'ODE lagrangienne à l'inférence ──────────────────────────
# Le résidu d'entraînement (LNN.residual) reste en différences finies (LNN_FD_ORDER) ;
# ces flags ne concernent QUE le rollout/simulation (intégration en avant).
#
# Rollout par lot (viz.simulate_rk4 → generate_*, train_lnn, plot de validation) :
#   'verlet' : velocity-Verlet semi-implicite (symplectique, ordre 2). 1 éval de force
#              par pas (réutilisation hors pression/contact) ⟹ ~4× moins de backward
#              autograd que RK4, et énergie non dérivante en long horizon (extrapolation).
#   'rk4'    : Runge-Kutta 4 (ordre 4, 4 évals/pas) — legacy, pour comparaison.
#   'gen_alpha' : generalized-α (Chung & Hulbert 1993), IMPLICITE. Dissipation
#              numérique SÉLECTIVE en fréquence (atténue les HF, préserve les BF,
#              ordre 2). Réglée par LNN_RHO_INF (rayon spectral à ω→∞). Résout
#              a_{n+1} par Newton modifié (Jacobien FD gelé, ~2·D + LNN_GENALPHA_ITERS
#              évals de force/pas) ⟹ stable jusqu'aux modes raides. Coût > Verlet ;
#              opt-in pour rollouts « propres »/stabilisés. NB : dissipation
#              NUMÉRIQUE (dépend de dt·ω), pas physique — la vraie atténuation
#              physique reste dans la dissipation apprise (Gamma/D).
LNN_INTEGRATOR = 'verlet'
# generalized-α : rayon spectral à fréquence infinie ρ∞ ∈ [0, 1]. 1.0 = aucune
# dissipation (= Verlet trapézoïdal) ; <1 amortit les HF (0.8–0.9 = filtrage doux) ;
# 0.0 = annihilation asymptotique des HF. Sans effet si LNN_INTEGRATOR != 'gen_alpha'.
LNN_RHO_INF = 1.0
# Newton COMPLET par pas gen_alpha (Jacobien FD rafraîchi au point milieu à chaque
# itération). LNN_GENALPHA_ITERS = nombre MAX d'itérations ; arrêt anticipé dès que
# ‖résidu‖ < LNN_GENALPHA_TOL·(‖a_n‖+1). 8 max couvre large (converge en ~2–4 sur un
# système raide) ; monter si divergence, descendre pour accélérer un système ~linéaire.
LNN_GENALPHA_ITERS = 8
LNN_GENALPHA_TOL   = 1e-8
# Le Jacobien FD (2·D évals) est calculé au point milieu prédicteur et GELÉ sur le pas
# (Newton modifié) ; rafraîchi tous les JAC_REFRESH iters seulement si un pas raide n'a
# pas convergé (repli). 3 = bon compromis vitesse/robustesse ; 1 = Newton complet (lent),
# grand = jamais rafraîchi.
LNN_GENALPHA_JAC_REFRESH = 3
# Apps interactives gradio (un pas physique par tick d'UI, temps réel) :
#   'semi_implicit' : Euler symplectique (1 éval/pas, robuste, naturellement dissipatif).
#   'verlet'        : velocity-Verlet semi-implicite (2 évals/pas, plus précis).
#   'rk4'           : RK4 (4 évals/pas) — legacy.
LNN_INTEGRATOR_LIVE = 'semi_implicit'
# Gradient ∂E/∂z analytique de l'EnergyNet ICNN (Jacobien forward softplus→sigmoid)
# au lieu de torch.autograd.grad ⟹ supprime la construction de graphe par appel.
# N'a d'effet que si LNN_ICNN=True ; repli automatique sur autograd sinon (ou si False).
LNN_ANALYTIC_GRAD = True

# ── Fine-tuning du LNN par ROLLOUT (finetune_lnn_fixedae.py UNIQUEMENT) ───────
# On part d'un LNN déjà entraîné sur la loss ODE (résidu FD, train_lnn_fixedae) et
# on le raffine sur une loss de TRAJECTOIRE : MSE(rollout différentiable, q(t) vrai)
# sur un horizon court. Corrige exactement ce que le résidu FD ne contraint pas —
# la stabilité hors-variété et l'enveloppe d'amortissement long-horizon (cf. le mode
# d'échec du bug 'invex' et le déficit de dissipation Gamma sur Krauss).
#
# ⚠️ TOUTES ces clés ne sont lues QUE par finetune_lnn_fixedae.py : aucune n'affecte
#    train_lnn(_fixedae), train_ae, les décodeurs ni les générateurs — le fine-tuning
#    est purement additif (charge lnn.pt, sauvegarde lnn_rollout.pt sans écraser lnn.pt).
LNN_ROLLOUT_STEPS   = 16      # pas déroulés EN PLUS de la fenêtre FD : W = SEQ_LEN + LNN_ROLLOUT_STEPS
LNN_FINETUNE_EPOCHS = 40      # époques de fine-tuning rollout
LNN_FINETUNE_LR     = 1e-3    # LR (< LNN_LR : on raffine, on ne ré-apprend pas)
LNN_ROLLOUT_BATCH   = 256     # taille de lot (rollout = BPTT ⟹ plus lourd que le résidu)
LNN_ROLLOUT_TBPTT   = 0       # >0 : BPTT tronqué (detach état tous les k pas) ; 0 = BPTT complet
LNN_ROLLOUT_CURRICULUM = True # rampe l'horizon (court → LNN_ROLLOUT_STEPS) sur la 1re moitié des époques
LNN_ROLLOUT_INTEGRATOR = 'verlet'  # 'verlet' (symplectique, différentiable, recommandé) | 'rk4'
# ── Métrique de la loss de rollout : VISIBILITÉ plutôt qu'euclidienne ────────
# Par défaut la loss de rollout est ‖Δu‖², donc ISOTROPE : en espace blanchi chaque
# direction latente pèse pareil (LatentWhiten impose cov ≈ I). Or « variance unité »
# n'est pas « visibilité unité » : un mode peut être de variance 1 dans les données tout
# en ne déplaçant presque aucun pixel. La MSE isotrope lui fait alors ajuster surtout du
# bruit d'encodage, et surpondère un mode que la métrique d'évaluation (MSE image) ignore.
# Mesuré sur Krauss 1-seg (d=2) : les deux modes diffèrent d'un facteur ~470 en visibilité
# (RMS pixel pour 1σ de u : 0.114 contre 0.0053) et reçoivent pourtant le même poids.
#   True ⟹ Δuᵀ A Δu avec A ∝ Ḡ + ρ·λmax·I, Ḡ = E[(∂I/∂u)ᵀ(∂I/∂u)] précalculée par
#   compute_visibility_metric.py. C'est, au premier ordre, le terme de reconstruction
#   décodée de Krauss et al. 2026 (VON) SANS rasteriser dans le graphe BPTT.
#   La ridge ρ est indispensable : sans elle les directions quasi invisibles ne seraient
#   plus contraintes du tout, or un mode invisible peut être RAIDE et diverger sans coût
#   immédiat, puis contaminer les modes visibles par le couplage (M, C non diagonales).
#   A est normalisée à trace(A)=d ⟹ même échelle de loss qu'en isotrope (LR/clip inchangés).
# ⚠️ Lues UNIQUEMENT par finetune_lnn_fixedae.py (+ compute_visibility_metric.py pour les
#    VIS_METRIC_*). Défauts NEUTRES : False ⟹ comportement strictement identique à avant.
LNN_ROLLOUT_METRIC       = False   # True = loss de rollout en métrique de visibilité
LNN_ROLLOUT_METRIC_FILE  = 'visibility_metric.pt'   # produit par compute_visibility_metric.py
LNN_ROLLOUT_METRIC_RIDGE = 0.01    # ρ, en fraction de λmax ⟹ borne le rapport de poids à ~1/ρ
# Fenêtres tirées par époque dans finetune_lnn_fixedae (0 = toutes, historique). Même
# levier que --windows-per-epoch de train_lnn_krauss : une vidéo Krauss donne ~90 000
# fenêtres, chacune déroulée en BPTT sur W pas ⟹ sans plafond, une époque de rollout
# coûte deux ordres de grandeur de plus qu'une époque de résidu FD.
LNN_ROLLOUT_WINDOWS_PER_EPOCH = 0
# ── Précalcul de la métrique de visibilité (compute_visibility_metric.py) ────
VIS_METRIC_RES     = 64        # résolution de rendu de la jacobienne (n'affecte que l'échelle,
                               #   pas le spectre relatif, seul utilisé en aval)
VIS_METRIC_SAMPLES = 64        # nb de points u où moyenner G(u)
VIS_METRIC_SOURCE  = 'auto'    # 'auto' | 'data' (u=whiten(enc(frames)), le plus fidèle)
                               #   | 'normal' (u~N(0,I), justifié par le blanchiment, sans images)
# ── Lissage temporel de l'encodeur (finetune_lnn.py, encodeur NON figé) ──────
# Pénalise la non-régularité de q(t)=enc(x) le long de la fenêtre (échantillons espacés de
# SEQ_STRIDE frames) ⟹ pousse l'encodeur à produire une trajectoire latente lisse. Opt-in,
# ajouté à la loss de rollout (transparent pour les autres entraînements). Lu par finetune_lnn.py.
LNN_SMOOTH_ENC        = False # True = ajoute λ·‖Δ^order z‖² à la loss de rollout
LNN_SMOOTH_ENC_LAMBDA = 1.0   # poids λ (à calibrer : ordre de grandeur du terme dépend de SEQ_STRIDE)
LNN_SMOOTH_ENC_ORDER  = 2     # 2 = courbure/accélération (laisse passer les rampes) | 1 = vitesse (plus dur)

# ── Forçage de pression pneumatique (Krauss 2026) ──────────────────────────
# Injecte la pression interne de l'actionneur dans le résidu d'Euler-Lagrange :
#     d/dt ∂L/∂q̇ − ∂L/∂q + ∂D/∂q̇ = b(q)ᵀ P        (espace latent)
# où b(q) = ∂V/∂q est la sensibilité-volume latente (matrice d'actionnement).
# Deux formes, sélectionnées par LNN_PRESSURE_MODE :
#   'constant'  (niveau 0) : b(q) ≡ B constant appris (n_c × d). Actionnement
#                            linéaire — cas Krauss/Koopman. F_P = P @ B.
#   'potential' (niveau 1) : potentiel de pression V_P(q) = −Pᵀ ν_φ(q), avec
#                            ν_φ : ℝ^d → ℝ^{n_c} un MLP (fonction de volume
#                            latente). b(q) = ∂ν_φ/∂q, F_P = ∂(Pᵀν_φ)/∂q.
#                            Naturel pour les step inputs (équilibre déplacé par P).
#   'invex'     (niveau 1) : idem 'potential' mais ν_φ = −(convexe ∘ Φ) CONCAVE —
#                            ICNN à n_c têtes (W_h/W_out ≥ 0) NÉGATIVÉ, composé d'un
#                            difféomorphisme Φ appris (i-ResNet, réutilise LNN_DIFFEO_*).
#                            Le signe rend V_P=−Pᵀν_φ INVEXE à min unique ET V_eff=V−Pᵀν_φ
#                            coercif ⟹ équilibre chargé UNIQUE et STABLE (une ν convexe
#                            ferait un V_P concave/maximum → rollout divergent). Lignes de
#                            niveau connexes non convexes. Init Φ≈Id ⟹ proche de 'constant'.
# La pression est un INPUT mesuré (pas un DDL) : elle n'entre que par ce second
# membre, jamais par l'encodeur. Les CSV sont alignés sur les frames par
# interpolation temporelle (voir dataset.load_pressure_frames).
LNN_PRESSURE      = False         # True = activer le forçage de pression
LNN_PRESSURE_MODE = 'constant'    # 'constant' (niveau 0) | 'potential' | 'invex' (niveau 1)
PRESSURE_DIR      = None          # dossier des CSV de pression (sources .mp4 seulement ;
                                  # sans objet sur une source .npz, qui porte ses pressions)
PRESSURE_COLS     = ['p_is_4', 'p_is_6']   # colonnes-chambres du CSV ; n_c = len(...)
                                           # p_is_* = pression mesurée (physique).
                                           # 1-segment Krauss : p4 et p6 (2 DDL planaires)
PRESSURE_NORM     = 101325.0      # normalisation (Pa atmosphérique), cf. NPZ Krauss
PRESSURE_HIDDEN   = [32, 32]      # MLP de ν_φ(q) (mode 'potential' uniquement)

# ── Lissage des trajectoires latentes q(t) (train_lnn_fixedae uniquement) ──────
# AE figé ⟹ q(t) précalculé une seule fois : on peut le lisser (Savitzky-Golay,
# PAR VIDÉO) avant de former les fenêtres du résidu EL, pour atténuer le bruit de
# l'encodeur. Neutre par défaut (opt-in par cas test).
SMOOTH_LATENT        = False      # True = lisser q(t) une fois après encode_all
SMOOTH_LATENT_MODE   = 'savgol'   # 'savgol' (fit polynomial glissant) | 'gaussian' (noyau gaussien)
SMOOTH_LATENT_WINDOW = 13         # savgol : fenêtre (frames) — impair ≥ 5 ; ~0.1 s @120fps
SMOOTH_LATENT_POLY   = 3          # savgol : ordre du polynôme Savitzky-Golay
SMOOTH_LATENT_SIGMA  = 10.0       # gaussian : écart-type du noyau (frames) ; coupe ≈0.132·fps/σ
SMOOTH_PRESSURE      = False      # True = lisser AUSSI la pression p(t) avec le MÊME filtre que q
SMOOTH_PRESSURE_SIGMA = None      # gaussian : σ dédié à la pression (frames) ; None = même σ que q

# Recalage temporel pression↔vidéo (s). Le capteur de pression et la caméra ne
# démarrent PAS au même instant : chez Krauss la pression est loggée AVANT la caméra,
# donc la frame 0 de la vidéo correspond à un temps `offset > 0` du CSV (en secondes
# réelles). Sans ce recalage, on supposerait offset=0 → décalage pression/frame de
# plusieurs dizaines de secondes (l'effet DOMINANT, bien devant la dilatation fps).
# Dict {stem_vidéo_ORIGINALE: offset_s} ; l'offset se PROPAGE automatiquement à toutes
# les variantes (256, 128_half, 256_skip4, debug300…) car elles partagent la frame 0
# (seul DT change). Stem normalisé par dataset._video_base_stem. Clé absente → offset 0
# (+ warning). Mesures 1-segment :
#   - smooth : offset 28.25 s — vérité-terrain via NPZ aligné officiel (corr 0.997).
#   - step   : offset 13.15 s — RAFFINÉ 2026-07-06 par le repère du 1ᵉʳ échelon (même
#              méthode que le 2-seg, 17.8→17.9). Onset du 1ᵉʳ échelon de pression brute
#              (p_is_4/p_is_6) à t_CSV = 20.29 s ; onset du 1ᵉʳ saut de q=enc(x)
#              (z_enc.npy, encodeur figé) à la frame 853 / t_frame = 7.165 s ⟹ offset
#              ≥ 20.29 − 7.165 = 13.13 s pour que q SUIVE P (déformation après cause
#              pneumatique). L'ancien 13.1 s faisait précéder q sur P de ~0.026 s
#              (non-physique) ; 13.15 s laisse q suivre P de ~0.02 s. La corr. globale
#              mouvement↔magnitude-P a un optimum plat 13.0–13.2 s (0.81), cohérent, mais
#              c'est le repère franc du 1ᵉʳ échelon qui tranche. Diagnostic :
#              pressures/sync_check_1seg_step.png (fronts de P et sauts de q superposés).
PRESSURE_SYNC_OFFSETS = {
    'smooth_input_rand_1_segment_15min_2Hz_max':   28.25,
    'step_input_random_1_segment_15_min_90kPa_max': 13.15,   # raffiné 1ᵉʳ échelon (q suit P)
}

# Résolution du chemin pression depuis un dossier checkpoint (cf. VIDEO_DIR)
if _SENTINEL.exists():
    PRESSURE_DIR = _HERE.parent / PRESSURE_DIR

# ── Sauvegarde ─────────────────────────────────────────────────────────────
# ── Répertoire de sauvegarde ───────────────────────────────────────────────
if _SENTINEL.exists():
    SAVE_DIR = _HERE   # on sauvegarde dans le dossier courant
else:
    _vid_path = Path(VIDEO_DIR)
    if _vid_path.is_file():
        _ckpt_name = f'checkpoints_{_vid_path.parent.name}_{_vid_path.stem}'
    else:
        _ckpt_name = f'checkpoints_{_vid_path.name}'
    SAVE_DIR = Path(_ckpt_name)
 

# ── Décodeur 2DGS ──────────────────────────────────────────────────────────
DEC_N_GAUSSIANS = 150  # nombre de gaussiennes
DEC_SMART_BIAS   = True           # True = biais init depuis image, False = Kaiming
DEC_HIDDEN      = [128, 128]     # couches cachées du MLP décodeur
DEC_LR          = 1e-3
DEC_LR_FACTOR   = 0.5            # facteur de réduction du LR
DEC_LR_PATIENCE = 200             # epochs sans amélioration avant réduction
DEC_LR_MIN      = 1e-6           # LR minimal
DEC_EPOCHS      = 15000
DEC_BATCH       = 32
DEC_SERIAL      = False          # True = rasterisation en série (économise la VRAM)
DEC_L1_W        = 0.08          # poids L1  (0 = désactivé)
DEC_MSE_W       = 0.0            # poids MSE (0 = désactivé)
DEC_SSIM_W      = 0.02            # poids SSIM (0 = désactivé)
DEC_ANISO_W     = 0.01           # poids pénalité anisotropie des gaussiennes (0 = désactivé)
DEC_DELTA_W     = 0.5            # poids loss delta temporel L1 (0 = désactivé)
DEC_SSIM_WIN    = 7              # taille fenêtre SSIM (doit être impair)
DEC_PLOT_EVERY  = 5              # plot interactif toutes les N epochs (0 = désactivé)

# ── Décodeur 2D(+t)GS  (GaussianSplatDecoder2pt) ──────────────────────────
# Backend de rendu, lu par models_2pt.build_decoder2pt (tous les scripts) :
#   'auto'   → gsplat si installé ET CUDA disponible, sinon repli sur le décodeur
#              torch pur, avec un avertissement de DÉPRÉCIATION appuyé ;
#   'gsplat' → impose gsplat (ImportError s'il manque). À poser quand un
#              résultat publié est en jeu : évite un repli silencieux ;
#   'torch'  → impose le décodeur maison, sans avertissement.
# ⚠️ Le repli n'est PAS équivalent : ~100× plus lent (O(K·H·W), pas de tuilage)
# et compositing différent (somme normalisée vs alpha front-to-back), donc les
# POIDS NE SONT PAS INTERCHANGEABLES entre les deux backends. Il sert à faire
# tourner la chaîne sans CUDA, pas à reproduire les chiffres.
DEC2PT_BACKEND = 'auto'

# Ces paramètres s'ajoutent à la section "Décodeur 2DGS" de config.py.
# Les autres clés DEC_* (LR, EPOCHS, BATCH, L1_W, etc.) sont réutilisées.
# DEC_HIDDEN et DEC_SMART_BIAS sont ignorés (pas de MLP dans ce décodeur).

DEC2PT_SIGMA_XY  = 0.05   # écart-type spatial initial des Gaussiennes (coordonnées normalisées [0,1])
DEC2PT_SIGMA_Z   = 1.0    # multiplicateur de std(z_all) pour l'écart-type latent initial
DEC2PT_SIGMA_Z_FLOOR = 0.5  # plancher de σ_z du gate conditionnel au smart_init : garantit
                            # une porte w_z assez large pour survivre au déploiement du latent
                            # quand l'AE démarre sur un encodeur non entraîné (sinon rendu noir +
                            # gradient nul). Neutre si l'encodeur a déjà un latent étalé (std ≳ 0.5).

# ── Décodeur 2D+t via gsplat (train_decoder2dpt_gsplat.py) ────────────────
# N_GAUSSIANS large possible grâce à la rasterisation CUDA de gsplat
DEC2PT_GSPLAT_N_GAUSSIANS = 2048   # >> 150 du décodeur 2D original
DEC2PT_ALPHA_THRESH        = 0.02  # seuil d'opacité pour le pruning
DEC2PT_PRUNE_EVERY         = 200   # pruning tous les N epochs
DEC2PT_PRUNE_WARMUP        = 500   # epochs avant d'activer le pruning
DEC2PT_ALPHA_W             = 0.0   # pression douce vers le bas sur les opacités
                                   # (pendant de AE_ALPHA_W / DEC3PT_ALPHA_W) : pénalise
                                   # mean(sigmoid(log_alpha)) pour que les gaussiennes
                                   # inutiles atteignent DEC2PT_ALPHA_THRESH et soient
                                   # recyclées par le prune/réinit. Garder faible (1e-3) :
                                   # départage entre primitives équivalentes, pas un
                                   # objectif. 0 ⟹ loss inchangée bit à bit.

# ── Décodeur 2D à gaussiennes PILOTÉES PAR UN MLP EN q ────────────────────
# (models_2dmlp.py + train_decoder2dmlp_gsplat.py — encodeur FIGÉ)
# Variante du décodeur 2pt : au lieu d'ellipsoïdes fixes en (2+d)D coupés à z=q,
# K gaussiennes purement 2D dont (μ, Σ [, couleur, opacité]) sortent d'un MLP
# résiduel en q (tête initialisée à ZÉRO ⟹ scène statique au départ). Réutilise
# les clés DEC_* (LR/EPOCHS/BATCH/L1_W/SSIM_W/ANISO_W/PLOT_EVERY) et les clés de
# pruning DEC2PT_ALPHA_* / DEC2PT_PRUNE_* / DEC2PT_SIGMA_XY, pour rester
# directement comparable à train_decoder2dpt_gsplat.py.
DEC2DMLP_N_GAUSSIANS  = 2048       # K (à défaut : DEC2PT_GSPLAT_N_GAUSSIANS)
DEC2DMLP_HIDDEN       = [128, 128] # couches cachées du MLP q → paramètres
DEC2DMLP_LR           = None       # LR du MLP ; None ⟹ DEC_LR (même LR que la scène)
DEC2DMLP_PE_FREQS     = 0          # fréquences de Fourier sur q (0 = q brut)
DEC2DMLP_DEFORM_COLOR = False      # la couleur dépend-elle de q ?
DEC2DMLP_DEFORM_ALPHA = False      # l'opacité dépend-elle de q ?
DEC2DMLP_DMU_MAX      = 1.0        # borne de |Δμ| (coord. normalisées, soft-clamp tanh)
DEC2DMLP_DL_MAX       = 3.0        # borne de |ΔL| (espace brut, pré-softplus)
# Bornes des échelles envoyées au rasteriseur (coordonnées [0,1] de l'image).
# ⚠️ Garde-fou, pas un réglage cosmétique : Σ(q) sortant d'un MLP, un pas
# d'optimisation peut produire une gaussienne géante (elle intersecte alors toutes
# les tuiles de gsplat) ou dégénérée, ce qui fait tomber le noyau CUDA en « illegal
# memory access » DANS le backward — crash non rattrapable (contexte CUDA perdu).
DEC2DMLP_SCALE_MIN    = 1e-3       # σ_xy min (≈ 0.26 px à 256)
DEC2DMLP_SCALE_MAX    = 0.5        # σ_xy max (la moitié de l'image)
DEC2DMLP_BACKEND      = None       # None ⟹ suit DEC2PT_BACKEND ('auto'|'gsplat'|'torch')


DEC3PT_N_GAUSSIANS   = 15000
DEC3PT_COLOR         = True
DEC3PT_LR            = 1e-3
DEC3PT_LR_FACTOR     = 0.5
DEC3PT_LR_PATIENCE   = 200
DEC3PT_LR_MIN        = 1e-6
DEC3PT_EPOCHS        = 15000
DEC3PT_BATCH         = 1


DEC3PT_L1_W          = 0.08
DEC3PT_SSIM_W        = 0.02
DEC3PT_ANISO_W       = 0.01
DEC3PT_NV_L1_W       = 0.8
DEC3PT_NV_SSIM_W     = 0.2
DEC3PT_SCALE_W = 1e-2   # à tuner

DEC3PT_SIGMA_XYZ     = 0.025
DEC3PT_SIGMA_Z       = 1.0
DEC3PT_Z_MEAN        = 2.0
DEC3PT_Z_SPREAD      = 1.6
DEC3PT_PLOT_EVERY    = 2



DEC3PT_ALPHA_W      = 1e-3  # très faible — juste pour tuer les inutiles
DEC3PT_ALPHA_THRESH = 0.025
DEC3PT_PRUNE_EVERY  = 5
DEC3PT_PRUNE_WARMUP = 0




SHARP_N_VIEWS    = 65




# ── Autoencodeur conjoint (train_ae.py) ────────────────────────────────────
# Entraîne l'encodeur ET le décodeur 2D+t gsplat ENSEMBLE par reconstruction
# pure (sans LNN). C'est la phase « warmup » de train_all sortie en script
# autonome. Init et hyperparamètres cohérents avec :
#   - train_encoder.py            → build_encoder + ENC_* (archi, LR, régul.)
#   - train_decoder2dpt_gsplat.py → GaussianSplatDecoder2pt_gsplat + DEC_*/DEC2PT_*
# Par défaut, chaque clé AE_* hérite de la clé encodeur/décodeur correspondante
# (modifier ici pour découpler).
AE_EPOCHS       = DEC_EPOCHS              # budget d'époques (réutilise celui du décodeur)
AE_BATCH        = DEC_BATCH              # taille de batch (frames)
AE_ENC_LR       = ENC_LR                  # LR encodeur   (cf. train_encoder.py)
AE_DEC_LR       = DEC_LR                  # LR décodeur   (cf. train_decoder2dpt_gsplat.py)
AE_LR_FACTOR    = DEC_LR_FACTOR           # ReduceLROnPlateau (comme le décodeur)
AE_LR_PATIENCE  = DEC_LR_PATIENCE
AE_LR_MIN       = DEC_LR_MIN
AE_N_GAUSSIANS  = DEC2PT_GSPLAT_N_GAUSSIANS
# Loss de reconstruction : réutilise DEC_L1_W / DEC_SSIM_W / DEC_ANISO_W.
# Régularisation encodeur : ENC_CPAE_LAMBDA_J (CpAE) / ENC_GP_LAMBDA (MLP), mise à
# l'échelle par AE_ENC_REG_W. 0 = coupée (défaut) : en AE pur, la supervision
# recon est faible, donc la pénalité nonlocale à λ_J=1.0 dominerait et aplatirait
# les filtres (encodeur effondré) ; l'anti-collapse latent reste assuré par le
# whitening (ENC_NORMALIZE). C'est le comportement de la phase warmup de train_all.
AE_ENC_REG_W    = 0.0
# Pression DOUCE vers le bas sur les opacités : + AE_ALPHA_W · mean(sigmoid(log_alpha)).
# Pendant 2D du DEC3PT_ALPHA_W de train_decoder3pt.py. Sert de départage entre
# primitives : une gaussienne utile compense ce coût par sa L1/SSIM, une gaussienne
# inutile descend jusqu'à passer sous AE_ALPHA_THRESH, où prune_and_reinit_2d la
# recycle vers une zone de forte erreur. SANS ce terme, rien ne pousse les α vers le
# bas et le pruning peut ne JAMAIS se déclencher (cas `dp` : min α = 0.25 sur 2048
# gaussiennes, seuil 0.02). Garder faible (1e-3) : c'est un a priori de parcimonie,
# pas un objectif. 0 = désactivé (loss inchangée bit à bit).
AE_ALPHA_W      = 0.0
AE_ALPHA_THRESH = DEC2PT_ALPHA_THRESH     # pruning des gaussiennes mortes (cf. décodeur)
AE_PRUNE_EVERY  = DEC2PT_PRUNE_EVERY
AE_PRUNE_WARMUP = DEC2PT_PRUNE_WARMUP
AE_PLOT_EVERY   = DEC_PLOT_EVERY          # debug plots reconstruction (0 = désactivé)
AE_PRINT_EVERY  = 50

ALL_EPOCHS        = 150
ALL_WARMUP_EPOCHS = 0
ALL_LR_LNN = 1.e-2
ALL_LR_ENC = 1.e-5
ALL_LR_DEC = 1.e-3
ALL_ENC_LR_DEC    = 1.e-3   # (legacy, ignoré par le nouveau train_all conjoint à 1 optimiseur)
ALL_PHYSICS_W     = 1.0    # (legacy — la physique passe maintenant par ALL_ROLLOUT_W)
ALL_PLOT_EVERY = 10
ALL_PLOT_DEC = 10

# ── train_all conjoint (rollout latent + recon + KL) ─────────────────────────
# La physique n'est plus le résidu FD single-step mais une loss de ROLLOUT
# différentiable dans l'espace latent (mêmes forces que viz.simulate_rk4). Un seul
# optimiseur, une loss combinée : L = recon + ALL_ROLLOUT_W·rollout + KL Castañeda
# (LNN_ANTICOLLAPSE_*).
ALL_ROLLOUT_STEPS   = LNN_ROLLOUT_STEPS   # W = SEQ_LEN + ALL_ROLLOUT_STEPS (fenêtre déroulée)
ALL_ROLLOUT_W       = 1.0                 # poids de la loss de rollout latent
ALL_ROLLOUT_INTEGRATOR = LNN_ROLLOUT_INTEGRATOR  # 'verlet' (diff., recommandé) | 'rk4'
ALL_ROLLOUT_TBPTT   = LNN_ROLLOUT_TBPTT   # >0 : BPTT tronqué (detach état tous les k pas)
ALL_ROLLOUT_CURRICULUM = True             # rampe l'horizon court → W sur la 1re moitié conjointe
ALL_BATCH           = 64                  # fenêtres par pas (rollout = BPTT ⟹ plus lourd)
ALL_RECON_SAMPLES   = DEC_BATCH           # frames décodées/pas pour la loss recon (réutilise q déjà encodé)
# KL anti-effondrement de train_all = KL de Castañeda (mêmes clés LNN_ANTICOLLAPSE_* que
# train_lnn.py) : gatée par LNN_ANTICOLLAPSE_KL, pondérée par LNN_ANTICOLLAPSE_LAMBDA.
