import config
"""
Pipeline 2 — Modèles PyTorch.

Encoder    : frame (1, H, W) → z ∈ ℝ^D
EnergyNet  : z ∈ ℝ^D → E ∈ ℝ
             E(z) = ½ zᵀ A z  +  MLP(z)
             A = LLᵀ + εI  (SPD, initialisée à I)
             MLP est un ICNN (convexe) si LNN_ICNN=True, sinon MLP libre
LNN        : enveloppe EnergyNet + termes de frottement optionnels
             résidu Euler-Lagrange :  a + γ·v + β·v/‖v‖ + dE/dz = 0
"""
import math

import torch
import torch.nn as nn


class WhiteningLayer(nn.Module):
    """
    Couche de blanchiment différentiable.

    Mode entraînement (self.training=True) :
        Whitening calculé depuis le batch courant, DANS le graphe.
        μ et Σ^{-1/2} recalculés à chaque forward — gradient traverse eigh,
        pression anti-collapse directe. Les buffers running_mean/running_W
        sont mis à jour (détachés) à chaque forward pour l'inférence.

    Mode inférence (self.training=False) :
        Utilise running_mean/running_W tels qu'ils sont dans le checkpoint.

    Paramètres
    ----------
    latent_dim : int
    eps        : float — plancher sur les valeurs propres (stabilité)
    """

    def __init__(self, latent_dim: int, eps: float = 1e-4,
                 momentum: float = 0.05):
        super().__init__()
        self.latent_dim = latent_dim
        self.eps        = eps
        # Moyenne mobile des stats GLOBALES (toute la base) pour l'inférence.
        # momentum petit ⟹ moyennage sur ~1/momentum batches → estimation base
        # entière (et non « dernier batch »). Cf. _global_W.
        self.momentum   = float(momentum)

        self.register_buffer('running_mean',  torch.zeros(latent_dim))
        self.register_buffer('running_W',     torch.eye(latent_dim))
        self.register_buffer('initialized',   torch.tensor(False))
        # Moment d'ordre 2 non centré E[zzᵀ] (EMA) : permet d'estimer la
        # covariance GLOBALE = E[zzᵀ] − μμᵀ (variance inter-batch incluse, à la
        # différence d'une EMA de covariances par batch). Non persistant : pas
        # sauvé dans le checkpoint (running_W l'est) → compat. checkpoints existants,
        # ré-accumulé en début d'entraînement. _stats_warm : amorçage dans CE process.
        self.register_buffer('running_M2', torch.eye(latent_dim), persistent=False)
        self.register_buffer('_stats_warm', torch.tensor(False), persistent=False)

    def _global_W(self) -> torch.Tensor:
        """Matrice de blanchiment depuis les stats globales EMA (cov = M2 − μμᵀ)."""
        cov = self.running_M2 - torch.outer(self.running_mean, self.running_mean)
        cov = cov + self.eps * torch.eye(
            self.latent_dim, device=cov.device, dtype=cov.dtype)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        return eigvecs * eigvals.clamp(min=self.eps).rsqrt().unsqueeze(0)

    @torch.no_grad()
    def set_global(self, mean: torch.Tensor, M2: torch.Tensor):
        """
        Pose les stats de blanchiment depuis une estimation GLOBALE (toute la base) :
        moyenne `mean` (D,) et moment d'ordre 2 non centré `M2 = E[zzᵀ]` (D, D).
        Met à jour TOUS les buffers de façon cohérente (running_mean, running_M2,
        running_W, flags) → la moyenne mobile par batch CONTINUE ensuite depuis ce
        point stable (au lieu de réamorcer sur ~1 batch, instable à d élevé). Utilisé
        par le priming global de train_ae / train_lnn / train_all.
        """
        self.running_mean.copy_(mean.detach())
        self.running_M2.copy_(M2.detach())
        self._stats_warm.fill_(True)
        self.running_W.copy_(self._global_W())
        self.initialized.fill_(True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z : (..., D) → z_white : (..., D)"""
        if self.training:
            shape  = z.shape
            z_flat = z.reshape(-1, self.latent_dim)
            N      = z_flat.shape[0]
            # Garde dégénérée N<2 : un seul échantillon → mu = z lui-même →
            # z_c = 0 → sortie EXACTEMENT 0 (et covariance indéfinie). On ne met
            # alors PAS à jour les stats et on blanchit par les running stats si
            # disponibles (sinon identité). Évite que enc(rest_frame) (1 frame)
            # renvoie 0 et n'écrase les buffers. Cf. z_rest dans train_lnn/all.
            if N < 2:
                if self.initialized.item():
                    return ((z_flat - self.running_mean) @ self.running_W
                            ).reshape(shape)
                return z
            mu     = z_flat.mean(0)
            z_c    = z_flat - mu
            cov    = z_c.T @ z_c / max(N - 1, 1)
            cov    = cov + self.eps * torch.eye(
                self.latent_dim, device=z.device, dtype=z.dtype)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            W = eigvecs * eigvals.clamp(min=self.eps).rsqrt().unsqueeze(0)
            # ── Stats globales (inférence / z_rest) par moyenne mobile ──────
            # Calculé UNIQUEMENT sur le batch courant (μ et 2ⁿᵈ moment E[zzᵀ] de
            # z_flat), puis accumulé en EMA. On ne repasse JAMAIS sur toute la base :
            # l'EMA *estime* le blanchiment global (comme les running stats d'un
            # BatchNorm) pour ~le coût d'un batch (+ un eigh D×D). Cela évite le
            # biais « dernier batch » de l'ancien copy_(). Amorçage direct au 1ᵉʳ pas.
            with torch.no_grad():
                mu_d = mu.detach()
                M2_d = (z_flat.detach().T @ z_flat.detach()) / N      # E[zzᵀ] batch
                if not self._stats_warm.item():
                    self.running_mean.copy_(mu_d)
                    self.running_M2.copy_(M2_d)
                    self._stats_warm.fill_(True)
                else:
                    m = self.momentum
                    self.running_mean.mul_(1 - m).add_(mu_d, alpha=m)
                    self.running_M2.mul_(1 - m).add_(M2_d, alpha=m)
                self.running_W.copy_(self._global_W())
                self.initialized.fill_(True)
            # Sortie : blanchiment PAR BATCH (décorrélation active, anti-collapse).
            return (z_c @ W).reshape(shape)
        else:
            if not self.initialized.item():
                return z
            return (z - self.running_mean) @ self.running_W

    def extra_repr(self) -> str:
        return f'latent_dim={self.latent_dim}, eps={self.eps}' 


class Encoder(nn.Module):
    """
    MLP (ELU) : frame aplatie → vecteur latent z.

    Si normalize=True, une WhiteningLayer est attachée et son forward()
    est appliqué automatiquement en sortie du MLP.  Les running stats
    Les buffers running_mean/running_W sont mis à jour à chaque forward en training.

    L'option normalize est contrôlée par config.ENC_NORMALIZE (défaut False
    si absent de config).
    """

    def __init__(self, img_size, hidden_dims, latent_dim, n_channels=1,
                 normalize: bool = False,
                 whitening_eps: float = 1e-4):
        super().__init__()
        self.normalize = normalize
        in_dim = n_channels * img_size[0] * img_size[1]
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

        if normalize:
            self.whitening = WhiteningLayer(
                latent_dim, eps=whitening_eps,
                momentum=getattr(config, 'ENC_WHITEN_MOMENTUM', 0.05))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, H, W)  →  z : (B, D)  [blanchi si normalize=True]"""
        z = self.net(x.flatten(1))
        if self.normalize:
            z = self.whitening(z)
        return z


class CpAEEncoder(nn.Module):
    """
    Encodeur CpAE (Zhu et al., ICLR 2025) : CNN à grands filtres dans les
    premières couches (défaut : 12×12 sur L*=3 couches) pour garantir la
    continuité temporelle de q(t).

    Réf : Thm. 3.1 — si les filtres des L* premières couches sont Lipschitz
    continus dans l'espace du filtre, alors enc(I(t)) évolue continûment avec
    la dynamique sous-jacente.  La pénalité nonlocale (éq. 6) promeut ce
    lissage ; elle est exposée via nonlocal_penalty() et doit être ajoutée à
    la loss (pondérée par config.ENC_CPAE_LAMBDA_J).

    Si normalize=True, une WhiteningLayer est appliquée en sortie (anti-collapse,
    cohérent avec le mode ENC_NORMALIZE de l'encodeur MLP).
    """

    def __init__(self, img_size, latent_dim, n_channels=1,
                 normalize=False, whitening_eps=1e-4,
                 n_large_layers=3, large_kernel=12, small_kernel=4,
                 channels=(32, 64, 128)):
        super().__init__()
        self.n_large_layers = n_large_layers
        self.normalize      = normalize

        conv_layers = []
        in_ch = n_channels
        for i, out_ch in enumerate(channels):
            k = large_kernel if i < n_large_layers else small_kernel
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k,
                                         stride=2, padding=(k - 2) // 2))
            in_ch = out_ch

        self.conv_layers = nn.ModuleList(conv_layers)
        self.act  = nn.ELU()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc   = nn.Linear(in_ch * 16, latent_dim)

        if normalize:
            self.whitening = WhiteningLayer(
                latent_dim, eps=whitening_eps,
                momentum=getattr(config, 'ENC_WHITEN_MOMENTUM', 0.05))

        self._build_nonlocal_kernel(large_kernel)

    def _build_nonlocal_kernel(self, J, sigma=1.0):
        """Précompute K[a,b] = exp(−‖pos_a − pos_b‖² / σ²), taille (J², J²)."""
        ij   = torch.stack(torch.meshgrid(
                   torch.arange(J, dtype=torch.float32),
                   torch.arange(J, dtype=torch.float32), indexing='ij'),
               dim=-1).reshape(J * J, 2)
        diff = ij.unsqueeze(0) - ij.unsqueeze(1)             # (J², J², 2)
        K    = torch.exp(-(diff ** 2).sum(-1) / sigma ** 2)  # (J², J²)
        self.register_buffer('_nonlocal_K', K)

    def nonlocal_penalty(self) -> torch.Tensor:
        """
        Pénalité de lissage des filtres des L* premières couches (éq. 6).
          = Σ_{l ≤ L*} Σ_{a,b} K[a,b] · (W_l[a] − W_l[b])²
        = trace(W_flat @ Laplacien @ W_flat^T) par couche, sommé sur les couches.
        Minimiser ceci force les poids à varier doucement spatialement
        → encodeur Lipschitz → q(t) continu dans le temps.
        """
        K     = self._nonlocal_K
        L_lap = torch.diag(K.sum(1)) - K                      # laplacien (J², J²)
        penalty = K.new_zeros(())
        for conv in self.conv_layers[:self.n_large_layers]:
            W      = conv.weight                               # (C_out, C_in, J, J)
            W_flat = W.reshape(-1, W.shape[-1] * W.shape[-2]) # (C_out·C_in, J²)
            penalty = penalty + (W_flat @ L_lap * W_flat).sum()
        return penalty

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, H, W)  →  z : (B, D)"""
        for conv in self.conv_layers:
            x = self.act(conv(x))
        z = self.fc(self.pool(x).flatten(1))
        if self.normalize:
            z = self.whitening(z)
        return z


def build_encoder(img_size, hidden_dims, latent_dim, n_channels=1,
                  normalize=False, whitening_eps=None):
    """
    Construit l'encodeur selon config.ENC_CPAE :
      True  → CpAEEncoder (CNN grands filtres + pénalité nonlocale, Zhu 2025)
      False → Encoder (MLP, comportement original)

    whitening_eps : plancher des valeurs propres de la WhiteningLayer. None →
    lu depuis config.ENC_WHITENING_EPS (défaut 1e-4 si absent).
    """
    if whitening_eps is None:
        whitening_eps = getattr(config, 'ENC_WHITENING_EPS', 1e-4)
    if getattr(config, 'ENC_CPAE', False):
        return CpAEEncoder(
            img_size       = img_size,
            latent_dim     = latent_dim,
            n_channels     = n_channels,
            normalize      = normalize,
            whitening_eps  = whitening_eps,
            n_large_layers = getattr(config, 'ENC_CPAE_N_LARGE',   3),
            large_kernel   = getattr(config, 'ENC_CPAE_KERNEL_L',  12),
            small_kernel   = getattr(config, 'ENC_CPAE_KERNEL_S',  4),
            channels       = getattr(config, 'ENC_CPAE_CHANNELS',  (32, 64, 128)),
        )
    return Encoder(img_size, hidden_dims, latent_dim, n_channels=n_channels,
                   normalize=normalize, whitening_eps=whitening_eps)


# ─────────────────────────────────────────────────────────────────────────────
# Blanchiment latent POST-HOC (figé, appris UNE fois après l'encodeur)
# ─────────────────────────────────────────────────────────────────────────────

class LatentWhiten(nn.Module):
    r"""
    Reparamétrisation latente AFFINE figée  u = (z − μ) W  (et inverse z = μ + u W⁻¹).

    Contrairement à `WhiteningLayer` (blanchiment PAR BATCH dans le graphe de
    l'encodeur, recalculé à chaque forward — instable à d élevé, cf. cas Krauss
    2-seg), `LatentWhiten` est calculée UNE SEULE FOIS après l'entraînement de
    l'autoencodeur (`compute_latent_whiten.py`) sur les stats GLOBALES de tout le
    dataset, puis GELÉE. Elle s'insère entre l'encodeur figé et la dynamique LNN :

        frame ──enc──▶ z ──whiten──▶ u ──(LNN en espace u)──▶ u ──inverse──▶ z ──dec──▶ image

    Le LNN (et la métrique pull-back du décodeur) travaillent donc dans un espace
    `u` ÉQUILIBRÉ (covariance ≈ I), sans réentraîner ni l'encodeur ni le décodeur.
    Comme la transformation est LINÉAIRE, la géométrie du décodeur (J, μ_z, Σ_zz⁻¹)
    se transporte de façon COVARIANTE : cf. `transform_geom`.

    Conventions (vecteurs LIGNE, batch (N, d)) :
        whiten  :  u = (z − mean) @ W          W     (d, d)
        inverse :  z = mean + u @ W_inv        W_inv = W⁻¹  (d, d)

    Modes (`fit`) :
        'pca' : W = V diag(λ^−1/2)        → u décorrélé, variances = 1, axes = CP
        'zca' : W = V diag(λ^−1/2) Vᵀ     → idem mais rotation minimale (axes ≈ z)
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer('mean',        torch.zeros(latent_dim))
        self.register_buffer('W',           torch.eye(latent_dim))
        self.register_buffer('W_inv',       torch.eye(latent_dim))
        self.register_buffer('initialized', torch.tensor(False))

    @torch.no_grad()
    def fit(self, z_all: torch.Tensor, mode: str = 'pca', eps: float = 1e-6):
        """Ajuste mean/W/W_inv sur z_all (N, d). Renvoie self (gèle la transformée)."""
        z = z_all.detach().to(torch.float64)
        mean = z.mean(0)
        zc   = z - mean
        cov  = zc.T @ zc / max(len(z) - 1, 1)
        cov  = cov + eps * torch.eye(self.latent_dim, dtype=cov.dtype)
        eigvals, eigvecs = torch.linalg.eigh(cov)          # λ croissants, V colonnes
        inv_sqrt = eigvals.clamp(min=eps).rsqrt()
        if mode == 'zca':
            W = eigvecs @ torch.diag(inv_sqrt) @ eigvecs.T
        elif mode == 'pca':
            W = eigvecs @ torch.diag(inv_sqrt)
        else:
            raise ValueError(f"LATENT_WHITEN_MODE inconnu : {mode!r} (attendu 'pca'|'zca')")
        self.mean.copy_(mean.float())
        self.W.copy_(W.float())
        self.W_inv.copy_(torch.linalg.inv(W).float())
        self.initialized.fill_(True)
        return self

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z : (..., d) → u : (..., d) blanchi."""
        return (z - self.mean) @ self.W

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        """u : (..., d) → z : (..., d) espace d'origine (entrée du décodeur)."""
        return u @ self.W_inv + self.mean

    @torch.no_grad()
    def transform_geom(self, geom: dict) -> dict:
        r"""
        Transporte la géométrie FIGÉE du décodeur (precompute_metric_geom.py) de
        l'espace z (où elle est extraite) vers l'espace u où vit le LNN. La
        reparamétrisation u = W(z−μ) ⟹ z = μ + W_inv·u (colonne), ∂z/∂u = W_invᵀ.

            J_u    = J · W_invᵀ                       (∂μ_xy/∂u = ∂μ_xy/∂z · ∂z/∂u)
            JtJ_u  = J_uᵀ J_u
            μz_u   = (μ_z − μ) W                       (centre latent en u)
            Szzi_u = W_inv · Σ_zz⁻¹ · W_invᵀ          (forme quadratique du gating)

        μ_xy, a, latent_dim, pos_dim, … : inchangés (espace image / scalaires).
        """
        Wt_inv = self.W_inv.t()                                  # (d, d)
        g = dict(geom)
        J    = geom['J'].float()                                 # (K, pos, d)
        J_u  = J @ Wt_inv                                        # (K, pos, d)
        g['J']    = J_u
        g['JtJ']  = J_u.transpose(-1, -2) @ J_u                  # (K, d, d)
        g['mu_z'] = (geom['mu_z'].float() - self.mean) @ self.W  # (K, d)
        Szzi = geom['Szzi'].float()                              # (K, d, d)
        g['Szzi'] = self.W_inv @ Szzi @ Wt_inv                   # (K, d, d)
        g['latent_whiten'] = True
        return g


class WhitenedEncoder(nn.Module):
    """
    Enveloppe figée encodeur + `LatentWhiten` : `forward(x) = whiten(encoder(x))`.

    Permet de passer un encodeur « en espace u » aux scripts/visualisations sans
    les modifier (ils appellent `enc(x)` et reçoivent directement u). Délègue les
    attributs inconnus (p. ex. `normalize`, `nonlocal_penalty`) à l'encodeur sous-
    jacent. `whiten` reste accessible via `.whiten` pour l'inverse (décodeur).
    """

    def __init__(self, encoder: nn.Module, whiten: LatentWhiten):
        super().__init__()
        self.encoder = encoder
        self.whiten  = whiten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.whiten(self.encoder(x))

    def __getattr__(self, name):
        # nn.Module.__getattr__ gère d'abord les sous-modules/buffers enregistrés
        # (encoder, whiten). En dernier recours, on délègue à l'encodeur.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.encoder, name)


def load_latent_whiten(save_dir, device, latent_dim,
                       fname: str = 'latent_whiten.pt'):
    """
    Charge `<save_dir>/<fname>` en `LatentWhiten` figée si `config.LATENT_WHITEN`
    est actif ET le fichier existe ; sinon renvoie None (comportement inchangé).
    """
    if not getattr(config, 'LATENT_WHITEN', False):
        return None
    path = save_dir / fname
    if not path.exists():
        return None
    whiten = LatentWhiten(latent_dim).to(device)
    whiten.load_state_dict(torch.load(path, map_location=device))
    whiten.eval()
    for p in whiten.buffers():
        p.requires_grad_(False)
    return whiten


def fit_latent_whiten(z_all, latent_dim, device, mode: str = 'pca',
                      eps: float = 1e-6, save_path=None):
    """Ajuste une `LatentWhiten` FIGÉE sur `z_all` (N, d), la gèle, la sauve
    optionnellement (`save_path`), et la renvoie prête à envelopper l'encodeur
    (`WhitenedEncoder`).

    Usage : entraînements à ENCODEUR NON FIGÉ (train_lnn, train_all, finetune_lnn)
    quand `ENC_NORMALIZE=False`. On calcule ce blanchiment UNE FOIS, sur les stats
    globales de l'encodeur AVANT l'epoch 0 (mêmes maths que compute_latent_whiten.py),
    puis on l'applique FIXE tout du long (≠ `WhiteningLayer` recalculée par batch). Deux
    effets : (1) q(t) démarre en espace ≈ N(0, I) ⟹ la loss KL anti-effondrement part
    déjà basse dès l'epoch 0 ; (2) comme la transformée est FIXE (constante affine), le
    gradient passe au travers vers l'encodeur qui reste librement entraîné, et la
    transformée reste valide même quand l'encodeur bouge — on l'applique à ses sorties,
    en entraînement comme en aval (`load_latent_whiten` reconstruit le même wrap)."""
    whiten = LatentWhiten(latent_dim).fit(z_all, mode=mode, eps=eps).to(device)
    whiten.eval()
    for b in whiten.buffers():
        b.requires_grad_(False)
    if save_path is not None:
        torch.save(whiten.state_dict(), save_path)
    return whiten


def get_or_fit_latent_whiten(save_dir, latent_dim, device, encode_fn,
                             mode: str = 'pca', eps: float = 1e-6,
                             fname: str = 'latent_whiten.pt'):
    """Renvoie `(whiten, created)`. CHARGE `<save_dir>/<fname>` s'il existe (⟹ COHÉRENCE
    avec un LNN déjà entraîné sur CETTE MÊME transformée — c'est le point critique : un LNN
    entraîné en espace u ne doit être raffiné qu'en espace u) ; sinon l'AJUSTE via
    `encode_fn()` — qui DOIT renvoyer `z_all (N,d)` calculé en mode EVAL — et le sauve.

    NE gate PAS sur `LATENT_WHITEN` (à l'appelant de décider). Utilisé par les entraînements
    à ENCODEUR NON FIGÉ (train_lnn / train_all / finetune_lnn) : préférer CHARGER garantit que
    tous les maillons de la chaîne (compute_latent_whiten → train_lnn_fixedae → finetune_lnn)
    partagent la même définition de u ; on ne réajuste que si le fichier manque encore."""
    path = save_dir / fname
    if path.exists():
        whiten = LatentWhiten(latent_dim).to(device)
        whiten.load_state_dict(torch.load(path, map_location=device))
        whiten.eval()
        for b in whiten.buffers():
            b.requires_grad_(False)
        return whiten, False
    whiten = fit_latent_whiten(encode_fn(), latent_dim, device,
                               mode=mode, eps=eps, save_path=path)
    return whiten, True


class InvexDiffeo(nn.Module):
    """
    Difféomorphisme bi-Lipschitz Φ : ℝᴰ → ℝᴰ de type i-ResNet (Behrmann 2019).

        Φ = (I + r_K) ∘ … ∘ (I + r_1)

    Chaque bloc résiduel r a Lip(r) < 1 (linéaires spectralement normalisés ×
    coeff, activations tanh 1-Lipschitz) ⟹ I + r est inversible et son Jacobien
    est partout non singulier. Composer un potentiel CONVEXE g avec Φ donne un
    potentiel INVEX : ∇(g∘Φ)(z) = JΦ(z)ᵀ ∇g(Φ(z)) s'annule ssi ∇g(Φ(z))=0
    (JΦ inversible), soit en l'unique z = Φ⁻¹(argmin g). Donc un seul point
    stationnaire = min global, mais sous-ensembles de niveau = Φ⁻¹(convexes)
    → connexes, non convexes. r est borné (tanh) ⟹ Φ propre (coercivité héritée
    du cœur convexe). À l'init r≈0 ⟹ Φ≈Id (démarre du cas convexe).
    """

    def __init__(self, dim: int, hidden: int, n_blocks: int, coeff: float = 0.9):
        super().__init__()
        try:
            from torch.nn.utils.parametrizations import spectral_norm
        except ImportError:                      # torch plus ancien
            from torch.nn.utils import spectral_norm
        self.coeff      = float(coeff)
        self.blocks     = nn.ModuleList()
        self.raw_scales = nn.ParameterList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                spectral_norm(nn.Linear(dim, hidden)),
                nn.Tanh(),
                spectral_norm(nn.Linear(hidden, dim)),
            ))
            # Échelle bornée du résidu, init 0 ⟹ Φ ≈ Id au départ (cas convexe).
            self.raw_scales.append(nn.Parameter(torch.zeros(())))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Chaque linéaire a σ_max ≤ 1 (spectral_norm) et tanh est 1-Lipschitz ⟹
        # Lip(block) ≤ 1. L'échelle α = coeff·tanh(raw) ∈ (−coeff, coeff) garde
        # Lip(r) = |α| < coeff < 1 ⟹ I + r inversible (i-ResNet), JΦ partout
        # non singulier. raw=0 à l'init ⟹ α=0 ⟹ Φ = Id (démarre du cas convexe).
        for block, raw in zip(self.blocks, self.raw_scales):
            alpha = self.coeff * torch.tanh(raw)
            z = z + alpha * block(z)
        return z


class InvexVolume(nn.Module):
    """
    Fonction de volume latente INVEXE ν_φ : ℝ^d → ℝ^{n_c} pour le forçage de
    pression (mode LNN_PRESSURE_MODE='invex').

    Chaque composante-chambre est CONCAVE : ν_c(q) = −C_c(Φ(q)) où
      • Φ : difféomorphisme i-ResNet bi-Lipschitz (InvexDiffeo), Jacobien partout
        non singulier, Φ ≈ Id à l'init ;
      • C : ℝ^d → ℝ^{n_c} CONVEXE par sortie — ICNN partagé (Amos et al. 2017) :
        h_{k+1}=softplus(W_z^k u + W_h^k h_k) avec W_h^k ≥ 0, têtes de sortie
        C = W_out h_K avec W_out ≥ 0 (softplus). u = Φ(q). ν = −C.

    Le SIGNE est essentiel. La pression entre par V_eff(q) = V(q) − Pᵀ ν_φ(q)
    (cf. residual/accel : M q̈ = −∂/∂q[V − Pᵀν]). Pour que le potentiel de pression
    V_P = −Pᵀ ν_φ soit INVEXE à MINIMUM unique (⟹ équilibre chargé unique) ET que
    V_eff reste coercif (équilibre STABLE), il faut, avec P ≥ 0 :
        ν_φ = −(convexe ∘ Φ)  (concave)
      ⟹ V_P = −Pᵀ ν_φ = +Pᵀ(convexe ∘ Φ)  = invexe à min unique,
        V_eff = V + Pᵀ(convexe ∘ Φ)          = convexe coercif (V ICNN + Σ≥0 convexes).
    Une ν CONVEXE ferait de V_P une CONCAVE (stationnaire = MAXIMUM) et retirerait de
    la courbure à V_eff ⟹ équilibre instable ⟹ rollout divergent (résidu FD faible mais
    dynamique off-manifold instable). Lignes de niveau connexes non convexes (plus
    souple que l'ICNN nu). À l'init Φ≈Id, W_h≈0 ⟹ ν quasi-linéaire (≈ régime 'constant').
    """

    def __init__(self, latent_dim, n_c, hidden_dims,
                 diffeo_hidden=64, diffeo_blocks=2, diffeo_coeff=0.9):
        super().__init__()
        self.diffeo      = InvexDiffeo(latent_dim, diffeo_hidden,
                                       diffeo_blocks, diffeo_coeff)
        self.hidden_dims = list(hidden_dims)
        dims = [latent_dim] + self.hidden_dims
        # ICNN convexe : passthrough W_z^k(u) libre (u = Φ(q) à CHAQUE couche),
        # récurrence W_h^k ≥ 0 (softplus), têtes de sortie ≥ 0.
        self.Wz_layers = nn.ModuleList()
        self.Wh_raw    = nn.ParameterList()
        self.biases    = nn.ParameterList()
        for k in range(len(self.hidden_dims)):
            self.Wz_layers.append(nn.Linear(latent_dim, dims[k + 1], bias=False))
            self.Wh_raw.append(None if k == 0
                               else nn.Parameter(torch.full((dims[k + 1], dims[k]), -3.0)))
            self.biases.append(nn.Parameter(torch.zeros(dims[k + 1])))
        # n_c têtes convexes : W_out ≥ 0 via softplus (init softplus(-1)≈0.31)
        self.w_out_raw = nn.Parameter(torch.full((n_c, self.hidden_dims[-1]), -1.0))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        u = self.diffeo(q)                                   # Φ(q)
        h = None
        for k in range(len(self.hidden_dims)):
            out = self.Wz_layers[k](u) + self.biases[k]       # passthrough libre en u
            if self.Wh_raw[k] is not None and h is not None:
                Wh  = torch.nn.functional.softplus(self.Wh_raw[k])   # (h_out,h_in) ≥ 0
                out = out + torch.nn.functional.linear(h, Wh)
            h = torch.nn.functional.softplus(out)             # convexe croissant
        W = torch.nn.functional.softplus(self.w_out_raw)      # (n_c, H) ≥ 0
        C = torch.nn.functional.linear(h, W)                  # (B, n_c) CONVEXE en u
        return -C                                             # ν = −C : CONCAVE (V_P invexe/min, V_eff coercif)


class EnergyNet(nn.Module):
    """
    Potentiel appris E(z), deux modes contrôlés par config.LNN_ICNN :

    ── Mode MLP libre (LNN_ICNN=False) ────────────────────────────────────
        E(z) = ‖ φ(z) - φ(z_rest) ‖²
    où φ : ℝ^D → ℝ^H est un MLP ELU libre.
    E ≥ 0 et E(z_rest) = 0 par construction.

    ── Mode ICNN convexe (LNN_ICNN=True) ──────────────────────────────────
    Architecture ICNN scalaire (Amos et al. 2017) :
        h_0     = z
        h_{k+1} = softplus( W_z^k z  +  W_h^k h_k )    avec W_h^k ≥ 0
        E(z)    = w^T h_K  -  w^T h_K(z_rest)           avec w ≥ 0
    La convexité de E en z est garantie par :
        - softplus convexe croissante
        - W_h^k ≥ 0 (composition préserve la convexité)
        - w ≥ 0 (combinaison linéaire positive de fonctions convexes)
    E(z_rest) = 0 par soustraction explicite.

    ── Ancrage de l'argmin (LNN_BREGMAN, ICNN uniquement) ──────────────────
    La simple soustraction de constante g(z)−g(z_rest) annule la VALEUR en z_rest
    mais pas le GRADIENT : le minimiseur de g reste où ∇g=0, sans raison de coïncider
    avec z_rest. Pour forcer z_rest = argmin V, on soustrait le Taylor d'ordre 1
    (divergence de Bregman) :
        E(z) = g(z) − g(z_rest) − ∇g(z_rest)ᵀ(z − z_rest)
    ⟹ E ≥ 0 (convexité), E(z_rest)=0 ET ∇E(z_rest)=0 ⟹ z_rest minimiseur global.

    ── Convexité stricte (LNN_EPS_STRONG > 0) ──────────────────────────────
    E ← E + ½·ε·‖z − z_rest‖² rend E fortement convexe → minimiseur UNIQUE,
    Hessien SPD à l'équilibre (modes propres bien définis).

    z_rest peut être mis à jour en cours d'entraînement via set_z_rest().
    """

    def __init__(self, latent_dim: int, hidden_dims: list, mu_map=None):
        super().__init__()
        self.icnn = config.LNN_ICNN
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        # Mode Bregman : z_rest = argmin garanti (gradient nul en z_rest).
        # Plancher fortement convexe ε·½‖z−z_rest‖² → minimiseur unique.
        self.bregman    = getattr(config, 'LNN_BREGMAN', False)
        self.eps_strong = float(getattr(config, 'LNN_EPS_STRONG', 0.0))

        # ── Pull-back par le décodeur (LNN_POTENTIAL_FROM_DECODER) ──────────────
        # Si mu_map est fourni : le potentiel s'évalue sur les positions décodées
        # x = μ(q) (affine, figé) et NON sur q. L'ICNN/MLP est donc construit avec
        # in_dim = μ-dim (= K·pos), tandis que z_rest reste un point de l'espace q
        # (dim latent_dim) — mappé en interne par μ. Convexité imposée en μ (physique) ;
        # gradient/Hessienne récupèrent automatiquement la structure Jᵀ(·)J (μ affine).
        self.mu_map = mu_map
        in_dim = mu_map.out_dim if mu_map is not None else latent_dim

        if not self.icnn:
            # ── MLP libre : φ : ℝ^in → ℝ^H  ──────────────────────────────
            layers = []
            prev = in_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.ELU()]
                prev = h
            self.phi = nn.Sequential(*layers)
            self.out_dim = prev

        else:
            # ── ICNN scalaire : h_{k+1} = softplus(W_x^k x + W_h^k h_k), W_h^k ≥ 0
            # (x = q, ou x = μ(q) si pull-back). Couche de sortie : w^T h_K, w ≥ 0.
            # Contrainte positivité : softplus(raw) au lieu de clamp(min=0)
            # → gradient toujours non nul, pas de poids morts
            dims = [in_dim] + list(hidden_dims)
            self.Wz_layers = nn.ModuleList()      # projections directes x→h_k (libres)
            self.Wh_raw    = nn.ParameterList()   # paramètres bruts pour Wh (avant softplus) ; None à k=0
            self.biases    = nn.ParameterList()

            for k in range(len(hidden_dims)):
                h_in  = dims[k]
                h_out = dims[k + 1]
                self.Wz_layers.append(nn.Linear(in_dim, h_out, bias=False))
                if k == 0:
                    self.Wh_raw.append(None)   # pas de récurrence à la première couche
                else:
                    # Initialisation : softplus_inv(ε) ≈ -5 pour démarrer proche de 0
                    # mais avec gradient non nul
                    wh_raw = nn.Parameter(torch.full((h_out, h_in), -3.0))
                    self.Wh_raw.append(wh_raw)
                self.biases.append(nn.Parameter(torch.zeros(h_out)))

            # Couche de sortie scalaire : w ≥ 0 via softplus
            # Initialisation : softplus(-1) ≈ 0.31 → contributions modérées
            self.w_out_raw = nn.Parameter(torch.zeros(hidden_dims[-1]))
            self.out_dim = 1

        # ── Difféomorphisme Φ (potentiel INVEX) ─────────────────────────────
        # Si LNN_INVEX et ICNN : on compose le cœur convexe avec Φ ⟹ potentiel
        # invex (un seul équilibre = min global, lignes de niveau connexes mais
        # non convexes). Φ ≈ Id à l'init. Avec Φ, ∂E/∂z perd sa forme analytique
        # softplus (chaîne par JΦ) ⟹ analytic_ok=False (repli autograd dans dE_dz).
        self.diffeo = None
        if self.icnn and getattr(config, 'LNN_INVEX', False):
            self.diffeo = InvexDiffeo(
                dim      = latent_dim,
                hidden   = getattr(config, 'LNN_DIFFEO_HIDDEN', 64),
                n_blocks = getattr(config, 'LNN_DIFFEO_BLOCKS', 2),
                coeff    = getattr(config, 'LNN_DIFFEO_COEFF', 0.9),
            )
        # Warp invex (Φ) et pull-back décodeur (μ) sont deux transformations
        # ALTERNATIVES de l'entrée de l'ICNN : jamais activées ensemble (Φ opère en
        # ℝᵈ, μ en ℝ^{K·pos} — dimensions incompatibles). Exclusivité garantie ici.
        assert not (self.diffeo is not None and self.mu_map is not None), (
            'LNN_INVEX et LNN_POTENTIAL_FROM_DECODER sont mutuellement exclusifs.')
        # ∂E/∂z analytique possible seulement sans transformation d'entrée
        # (ni Φ invex, ni pull-back μ) ; sinon repli autograd dans dE_dz.
        # Témoin harmonique : potentiel quadratique pur (cf. forward). Court-circuite
        # ICNN/difféo/MLP ⟹ la voie analytique softplus de grad_E ne s'applique pas
        # (repli autograd, exact et trivial sur une forme quadratique).
        self.harmonic = bool(getattr(config, 'LNN_HARMONIC', False))
        self.analytic_ok = (self.icnn and self.diffeo is None
                            and self.mu_map is None and not self.harmonic)

        # z_rest : buffer non-entraînable (espace q), mis à jour depuis l'extérieur
        self.register_buffer('z_rest', torch.zeros(latent_dim))

    def _in(self, q: torch.Tensor) -> torch.Tensor:
        """Entrée de l'ICNN/MLP : q si pas de pull-back, sinon μ(q) (affine, figé)."""
        return q if self.mu_map is None else self.mu_map(q)

    # ── Helpers ICNN ───────────────────────────────────────────────────────

    def _clamp_icnn_weights(self):
        """Obsolète — la contrainte ≥ 0 est maintenant assurée par softplus."""
        pass

    def _icnn(self, z: torch.Tensor) -> torch.Tensor:
        """
        Évalue l'ICNN scalaire.
        z : (B, D) → E : (B,)

        Wh et w_out sont contraints ≥ 0 via softplus(raw) :
        gradient toujours non nul, pas de poids morts contrairement à clamp.
        """
        h = None
        for k in range(len(self.hidden_dims)):
            wz  = self.Wz_layers[k]
            wh_raw = self.Wh_raw[k]
            b   = self.biases[k]
            out = wz(z) + b
            if wh_raw is not None and h is not None:
                Wh = torch.nn.functional.softplus(wh_raw)   # (h_out, h_in) ≥ 0
                out = out + torch.nn.functional.linear(h, Wh)
            h = torch.nn.functional.softplus(out)   # (B, h_k)
        # Sortie scalaire : softplus(w_raw)^T h ≥ 0
        w = torch.nn.functional.softplus(self.w_out_raw)   # (H,) ≥ 0
        return (h * w).sum(dim=-1)                          # (B,)

    def _icnn_grad(self, z: torch.Tensor) -> torch.Tensor:
        """
        Gradient analytique ∂g/∂z de l'ICNN scalaire, par propagation forward du
        Jacobien J = dh/dz à travers les couches softplus (softplus'(x)=sigmoid(x)).
        z : (B, D) → (B, D).

        Évite torch.autograd.grad (pas de construction de graphe par appel). Reste
        construit en ops torch ⟹ différentiable vis-à-vis des poids (entraînement).
        """
        B = z.shape[0]
        h = None
        J = None                                  # dh/dz : (B, h_dim, D)
        for k in range(len(self.hidden_dims)):
            wz     = self.Wz_layers[k]            # Linear(D, h_out, bias=False)
            wh_raw = self.Wh_raw[k]
            b      = self.biases[k]
            out  = wz(z) + b                                  # (B, h_out)
            dout = wz.weight.unsqueeze(0).expand(B, -1, -1)   # (B, h_out, D)
            if wh_raw is not None and h is not None:
                Wh   = torch.nn.functional.softplus(wh_raw)   # (h_out, h_in) ≥ 0
                out  = out + torch.nn.functional.linear(h, Wh)
                dout = dout + Wh.unsqueeze(0) @ J             # (B, h_out, D)
            sig = torch.sigmoid(out)             # softplus'(out)
            h   = torch.nn.functional.softplus(out)
            J   = sig.unsqueeze(-1) * dout       # (B, h_out, D)
        w = torch.nn.functional.softplus(self.w_out_raw)      # (H,) ≥ 0
        return (w.view(1, -1, 1) * J).sum(dim=1)              # (B, D)

    def grad_E(self, z: torch.Tensor) -> torch.Tensor:
        """
        Gradient analytique ∂E/∂z (chemin ICNN uniquement). z : (B, D) → (B, D).
        Réplique exactement forward() :
            Bregman      ⟹ E = g(z) − g(z_r) − ∇g(z_r)ᵀ(z−z_r) → ∇g(z) − ∇g(z_r)
            (sinon)      ⟹ E = g(z) − g(z_r)                    → ∇g(z)
            eps_strong>0 ⟹ + ½ε‖z−z_r‖²                         → + ε(z−z_r)
        """
        if not self.icnn:
            raise RuntimeError('grad_E analytique disponible uniquement pour ICNN')
        if self.diffeo is not None:
            raise RuntimeError('grad_E analytique incompatible avec le difféomorphisme '
                               'invex (chaîne JΦ) — utiliser le repli autograd.')
        # Tout est calculé dans l'espace d'entrée de l'ICNN (x = q, ou x = μ(q) si
        # pull-back), puis chaîné vers q par ∂E/∂q = (∂E/∂x)·A  (μ = A q + b ⟹ ∂x/∂q = A).
        x   = self._in(z)                                     # (B, in)
        x_r = self._in(self.z_rest.unsqueeze(0))              # (1, in)
        g = self._icnn_grad(x)                                # (B, in)
        if self.bregman:
            g = g - self._icnn_grad(x_r)                      # ∇g(x) − ∇g(x_r)
        if self.eps_strong > 0.0:
            g = g + self.eps_strong * (x - x_r)
        if self.mu_map is not None:
            g = g @ self.mu_map.A                             # (B, in)@(in, D) = (B, D)
        return g

    # ── API publique ────────────────────────────────────────────────────────

    def set_z_rest(self, z_rest: torch.Tensor, learnable: bool = False):
        """
        Définit z_rest.
        - learnable=False (défaut) : buffer fixe, non optimisé
        - learnable=True           : paramètre optimisé
        """
        z = z_rest.detach().to(next(self.parameters()).device)
        if learnable:
            if hasattr(self, 'z_rest') and not isinstance(self.z_rest, nn.Parameter):
                del self._buffers['z_rest']
            self.z_rest = nn.Parameter(z)
        else:
            if hasattr(self, 'z_rest') and isinstance(self.z_rest, nn.Parameter):
                del self._parameters['z_rest']
            self.register_buffer('z_rest', z)

    def forward(self, z):
        """z : (B, D)  →  E : (B,)  (potentiel évalué sur x = μ(z) si pull-back)"""
        # ── Témoin HARMONIQUE : potentiel strictement quadratique ────────────────
        # E(z) = ½·k·‖z − z_rest‖², k = LNN_EPS_STRONG. Aucun réseau : ni ICNN, ni
        # difféo, ni MLP libre ⟹ raideur CONSTANTE, donc oscillateur linéaire pur
        # (la fréquence ne dépend plus de l'amplitude). Sert de plancher de
        # comparaison aux potentiels convexe / invexe. La généralité n'est pas
        # perdue en fixant k : ω² = Minv·k avec Minv appris, et l'amortissement
        # reste libre (Gamma ou C(q)) — seule la FORME du potentiel est contrainte.
        # Opt-in ; False (défaut) ⟹ comportement strictement inchangé.
        if self.harmonic:
            dz = z - self.z_rest.unsqueeze(0)
            return 0.5 * self.eps_strong * dz.pow(2).sum(dim=-1)
        if self.icnn:
            # Entrée de l'ICNN : x = q, ou x = μ(q) (affine figé) si pull-back décodeur.
            x   = self._in(z)                                          # (B, in)
            x_r = self._in(self.z_rest.unsqueeze(0))                   # (1, in)
            # ── Warp invex : u = Φ(x), u_r = Φ(x_r) ─────────────────────────
            # Le cœur convexe (ICNN + Bregman + plancher ε) opère en coordonnée u.
            # Φ bijectif ⟹ E = D_g(Φ(x),Φ(x_r)) est invex (un seul équilibre = min
            # global, lignes de niveau warpées). Sans diffeo : u=x (cas convexe).
            # Φ et pull-back μ sont exclusifs (cf. assert dans __init__).
            if self.diffeo is not None:
                u, u_r = self.diffeo(x), self.diffeo(x_r)
            else:
                u, u_r = x, x_r
            if self.bregman:
                # Divergence de Bregman de g=icnn au point u_r (espace ICNN, warpé si invex) :
                #   E = g(u) − g(u_r) − ∇g(u_r)ᵀ(u − u_r)
                # → E ≥ 0, E(z_r)=0 ET ∇E(z_r)=0 ⟹ z_rest = argmin par construction.
                # ∇g(u_r) (= grad_rest) est calculé ANALYTIQUEMENT (_icnn_grad, ops pures)
                # plutôt que par autograd.grad imbriqué : pas de graphe d'ordre supérieur
                # (⟹ pas de double-backward fragile, robuste train/éval, et plus rapide).
                # Valable aussi avec Φ : ∇g(u_r) est le gradient de l'ICNN au point u_r,
                # exactement ce que _icnn_grad calcule (la chaîne JΦ est portée par du).
                # u_r DÉTACHÉ (∂/∂z_rest passe par du) mais différentiable vis-à-vis des
                # poids. Sous torch.no_grad() (cartes d'énergie), E n'a pas de grad_fn.
                ur_d      = u_r.detach()                                # u_rest détaché
                g_rest    = self._icnn(ur_d)                            # (1,)
                grad_rest = self._icnn_grad(ur_d)                       # (1, in)  ∇g(u_r) analytique
                g_u = self._icnn(u)                                    # (B,)
                du  = u - u_r                                          # (B, in)
                E   = g_u - g_rest - (grad_rest * du).sum(dim=-1)      # (B,)
            else:
                # Soustraction de constante : garantit E(z_r)=0 mais PAS argmin=z_rest.
                E = self._icnn(u) - self._icnn(u_r)                     # (B,)
            if self.eps_strong > 0.0:
                # Plancher fortement convexe en u (= Bregman de ½ε‖u‖²) → unicité
                # du minimiseur (préservée par le difféo). u=x sans diffeo.
                E = E + 0.5 * self.eps_strong * (u - u_r).pow(2).sum(dim=-1)
            return E                                                    # (B,)
        else:
            phi_z    = self.phi(self._in(z))                           # (B, H)
            phi_rest = self.phi(self._in(self.z_rest.unsqueeze(0)))    # (1, H)
            return (phi_z - phi_rest).pow(2).sum(dim=-1)                # (B,)


class MuMap(nn.Module):
    """
    Cinématique de position AFFINE figée μ(q) = A q + b, dérivée de la géométrie du
    décodeur (precompute_metric_geom.py). Pour la gaussienne i :

        μᵢ(q) = μ_xy,i + Jᵢ (q − μ_z,i),   Jᵢ = ∂μ_cond,i/∂q  (le MÊME J que Jᵀf et M̃=JᵀMJ)

    empilé sur les K gaussiennes ⟹ A = stack(Jᵢ) ∈ ℝ^{(K·pos)×d},
    b_i = μ_xy,i − Jᵢ μ_z,i. Sert de front au potentiel pull-back Ṽ(q)=V_ICNN(μ(q)) :
    ∂Ṽ/∂q = Aᵀ ∇_μV (covecteur transporté par Jᵀ), ∇²Ṽ = Aᵀ(∇²V)A (courbure confinée
    à range(Aᵀ)=range(Jᵀ)). Buffers NON entraînables (géométrie figée).
    """

    def __init__(self, J, mu_z, mu_xy):
        super().__init__()
        # DÉTACHER : la géométrie est figée. Certains champs de metric_geom.pt sont
        # sauvés avec requires_grad=True (sorties du décodeur) → sans detach, le buffer
        # b traînerait un grad_fn (graphe vers mu_z), libéré au 1er backward puis réutilisé
        # au suivant ⟹ « backward through the graph a second time ».
        J, mu_z, mu_xy = J.detach(), mu_z.detach(), mu_xy.detach()
        K, pos, d = J.shape
        A = J.reshape(K * pos, d).contiguous()                 # (K·pos, d)
        # b_i = μ_xy,i − Jᵢ μ_z,i  → empilé (K·pos,)
        b = (mu_xy - torch.einsum('kpd,kd->kp', J, mu_z)).reshape(K * pos).contiguous()
        self.register_buffer('A', A.float())                   # (out, d)
        self.register_buffer('b', b.float())                   # (out,)
        self.out_dim = K * pos

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """q : (B, d) → μ(q) : (B, K·pos)."""
        return q @ self.A.t() + self.b


class LatentMetricA(nn.Module):
    """
    Métrique cinétique latente COURBE (forme A), dérivée de la géométrie FIGÉE du décodeur
    :

        M̃(q) = m · M̂(q),    M̂(q) = Σᵢ ŵᵢ(q) Jᵢᵀ Jᵢ,
        ŵᵢ(q) = aᵢ wᵢ(q) / Σⱼ aⱼ wⱼ(q),
        wᵢ(q) = exp(−½ (q−μ_z,i)ᵀ Σ_zz,i⁻¹ (q−μ_z,i)).

    La géométrie (JtJ, μ_z, Σ_zz⁻¹, a) sont des buffers NON entraînables (sortie de
    `precompute_metric_geom.py`, décodeur gelé). Seule l'échelle `m = exp(log_m)` est
    apprise (calibre la fréquence propre). M̂(q) est différentiable en `q` (⟹ Coriolis).
    """

    def __init__(self, geom: dict, mass_init: float = 1.0, cond_max: float = 0.0):
        super().__init__()
        self.latent_dim = int(geom['latent_dim'])
        # ── Conditionnement borné hors du support des données (opt-in) ────────
        # Chaque JᵀJ individuel est de rang ≤ `pos` (2 : μ vit dans le plan image),
        # donc rang-DÉFICIENT dès que d > 2. Sur les données, M̂ = Σᵢ ŵᵢ JᵢᵀJᵢ est
        # bien conditionnée parce que la présence ŵ mélange des gaussiennes dont les
        # sous-espaces diffèrent ; mais loin des données le softmax de `presence`
        # SATURE sur une seule gaussienne et M̂ retombe au rang 2. Mesuré (Krauss NPZ,
        # métrique recalée) le long d'un rayon sortant : λmin(M̃) passe de 429 à
        # 7.5e-3 (1-seg) et de 10.2 à 6.2e-4 (2-seg) entre ‖q‖=0 et 50. Comme
        # q̈ = M̃⁻¹·force, l'accélération est amplifiée ×10⁴–10⁵ hors variété ⟹
        # rollout divergent (‖q‖ → 1e5 en ~7 s) alors que les données tiennent dans
        # ‖q‖ ≤ 4.5. `LNN_METRIC_RIDGE` (plancher ABSOLU, défaut 1e-4) ne protège pas :
        # il est calibré pour M̂~O(1) et devient dérisoire dès que m ≫ 1.
        # Remède, identique à celui de LatentMetricLearned : ridge proportionnel à la
        # trace, ε_eff = tr(M̂)/κ ⟹ λmin ≥ tr/κ ≥ λmax/κ, donc cond(M̂) ≤ κ pour TOUT q.
        # Il suit l'échelle LOCALE de la masse (il ne fausse donc pas la magnitude) et
        # ne demande pas d'eigh dans le graphe. Choisir κ ≫ cond(M̂) sur les données
        # ⟹ région des données quasi inchangée, seule l'extrapolation est bornée.
        self.cond_max = float(cond_max)
        self.register_buffer('JtJ',  geom['JtJ'].float())    # (K, d, d)
        self.register_buffer('J',    geom['J'].float())      # (K, pos, d) — pull-back dissipation
        self.register_buffer('mu_z', geom['mu_z'].float())   # (K, d)
        self.register_buffer('Szzi', geom['Szzi'].float())   # (K, d, d)
        self.register_buffer('a',    geom['a'].float())      # (K,)
        # Échelle de masse FIXE (non apprise) : m = exp(log_m), buffer (jamais optimisé).
        # La jauge de masse est figée par precompute_metric_geom.py + mass_init ; on ne
        # laisse pas l'optimiseur la bouger (cf. demande utilisateur). Calibrée hors-ligne
        # pour viser une fréquence propre initiale cible (cf. LNN_METRIC_MASS_INIT).
        self.register_buffer('log_m', torch.tensor(math.log(float(mass_init))))

    def presence(self, q):
        """
        q : (B, d) → ŵ : (B, K) — gating gaussien normalisé ŵᵢ ∝ aᵢ exp(−½‖q−μ_z,i‖²_Σ⁻¹).
        Calculé via softmax stabilisé (log-sum-exp) : reste une distribution valide même
        loin de toutes les gaussiennes (sinon underflow → 0/0 → NaN en extrapolation).
        """
        dz     = q.unsqueeze(1) - self.mu_z.unsqueeze(0)              # (B, K, d)
        quad   = torch.einsum('bki,kij,bkj->bk', dz, self.Szzi, dz)  # (B, K)
        logits = -0.5 * quad + torch.log(self.a.clamp(min=1e-30)).unsqueeze(0)
        return torch.softmax(logits, dim=1)                          # (B, K)

    def Mhat(self, q):
        """q : (B, d) → M̂(q) : (B, d, d) (sans l'échelle m ; différentiable en q).

        Si `cond_max` = κ > 0, ajoute le ridge proportionnel à la trace décrit dans
        `__init__` : M̂ ← M̂ + (tr(M̂)/κ)·I, qui borne cond(M̂) ≤ κ pour tout q et
        empêche la dégénérescence de rang loin des données.
        """
        wh = self.presence(q)                                         # (B, K)
        M  = torch.einsum('bk,kij->bij', wh, self.JtJ)               # (B, d, d)
        if self.cond_max > 0:
            tr  = M.diagonal(dim1=-2, dim2=-1).sum(-1).reshape(-1, 1, 1)
            eye = torch.eye(self.latent_dim, device=q.device, dtype=q.dtype)
            M   = M + (tr / self.cond_max) * eye
        return M

    def M(self, q):
        """Métrique complète M̃(q) = m · M̂(q)."""
        return self.log_m.exp() * self.Mhat(q)


class LatentMetricLearned(nn.Module):
    """
    Métrique cinétique latente COURBE **APPRISE** (masse q-dépendante, NON dérivée du
    décodeur) :

        M̂(q) = L(q) L(q)ᵀ + εI   (d×d SPD, différentiable en q),

    où le facteur de Cholesky L(q) (triangulaire inf., diagonale > 0 via softplus,
    hors-diagonale libre) est produit par un MLP φ : ℝᵈ → ℝ^{d(d+1)/2}. Même
    construction que la dissipation pleine `_C_q` — SPD garantie pour tout q.

    Interface IDENTIQUE à `LatentMetricA` (latent_dim, log_m, Mhat, M) ⟹ branchée
    telle quelle dans TOUTE la machinerie courbe : énergie cinétique T=½q̇ᵀM̃(q)q̇,
    Coriolis (∂p/∂q)q̇ obtenue par AUTOGRAD à travers M̂(q) (résidu `_residual_curved`
    / `_pg_curved`, accélération `accel` qui inverse M̃(q) en D×D), rollout `viz`.
    Contrairement à `LatentMetricA`, l'inertie est APPRISE par le LNN (couplages
    inter-DDL via l'hors-diagonale de L(q)), pas imposée par la cinématique q↦μ figée.

    Échelle de masse : m = exp(log_m) est fixée à 1 (buffer) — la magnitude de la masse
    est ENTIÈREMENT portée par le réseau (init isotrope M̂(q)≡s·I, s = mass_init), ce qui
    évite un double paramétrage redondant. Init NEUTRE : dernière couche poids=0, biais
    diag = softplus⁻¹(√s) ⟹ M̂(q) ≡ s·I au départ (fréquence propre initiale calibrée par s,
    ω ∝ 1/√s). NB : on apprend M(q) (pas Minv(q)) car le résidu intégral utilise le moment
    p=M(q)q̇ et la Coriolis via Δp ; l'inversion D×D de `accel` est négligeable (d ≤ 8).
    """

    def __init__(self, latent_dim: int, mass_init: float = 1.0,
                 hidden=(64, 64), eps: float = 1e-4,
                 far_const: bool = False, gate_r0: float = 4.0,
                 gate_tau: float = 2.0, cond_max: float = 0.0):
        super().__init__()
        self.latent_dim = int(latent_dim)
        d = self.latent_dim
        self.eps = float(eps)
        # ── Masse CONSTANTE loin des données + conditionnement borné ──────────
        # Mesuré sur Krauss 2-seg : hors du support, le MLP de Cholesky diverge
        # (‖∂M/∂q‖ : 386 → 5e5 entre ‖q‖=0 et 256) et M̂ se dégénère (une valeur
        # propre collée au plancher ε=0.1, l'autre à 6.6e7, cond 6.6e8). Comme la
        # Coriolis vaut (∂p/∂q)q̇ avec p=M̂(q)q̇, elle explose avec ‖∂M/∂q‖ et fait
        # diverger le rollout long (‖q‖ → 1e15 en 50 s) alors que les données
        # tiennent dans ‖q‖ ≤ 8.9. Deux garde-fous, tous deux opt-in :
        #   far_const : L(q) = L∞ + g(q)·(φ(q) − L∞), porte radiale g décroissant de
        #     1 (‖q‖ ≤ r0, données) à 0 (loin) ⟹ M̂ CONSTANTE hors du support, donc
        #     ∂M/∂q → 0 et Coriolis → 0 exactement. L∞ est APPRIS (init = biais neutre).
        #   cond_max : ridge proportionnel à la trace, ε_eff = (d/κ)·tr(LLᵀ)/d ⟹
        #     λmin ≥ tr/κ ≥ λmax/κ, donc cond(M̂) ≤ κ pour TOUT q, sans eigh dans le
        #     graphe (la trace suffit, différentiable et stable).
        self.far_const = bool(far_const)
        self.gate_r0 = float(gate_r0)
        self.gate_tau = max(float(gate_tau), 1e-6)
        self.cond_max = float(cond_max)
        # m fixé à 1 : l'échelle est dans le réseau (buffer, jamais optimisé).
        self.register_buffer('log_m', torch.zeros(()))
        tri = torch.tril_indices(d, d)                       # (2, n_tri)
        self.register_buffer('tril_i', tri[0])
        self.register_buffer('tril_j', tri[1])
        self.register_buffer('diag_mask', tri[0] == tri[1])  # (n_tri,) bool
        n_tri = tri.shape[1]
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        last = nn.Linear(prev, n_tri)
        nn.init.zeros_(last.weight)
        bias = torch.zeros(n_tri)
        bias[self.diag_mask] = math.log(math.expm1(mass_init ** 0.5))  # softplus⁻¹(√s)
        with torch.no_grad():
            last.bias.copy_(bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)                    # φ : ℝᵈ → ℝ^{d(d+1)/2}
        # Valeur ASYMPTOTIQUE (loin des données) du vecteur de Cholesky, apprise.
        # Init = même biais neutre que la dernière couche ⟹ M̂ ≡ s·I partout au départ,
        # donc activer far_const ne change rien à l'initialisation.
        self.far_raw = nn.Parameter(bias.clone())

    def gate(self, q):
        """g(q) ∈ (0,1] : 1 sur le support des données (‖q‖ ≤ r0), → 0 au-delà.

        Écart au-delà de r0 seulement (relu) ⟹ g ≡ 1 exactement dans la région
        des données : le comportement appris y est INCHANGÉ, seul l'extérieur est
        contraint. Gaussienne en (‖q‖−r0)/τ : lisse, donc M̂ reste C¹ (Coriolis
        bien définie partout).
        """
        r = q.norm(dim=-1, keepdim=True)                     # (N, 1)
        x = torch.relu(r - self.gate_r0) / self.gate_tau
        return torch.exp(-0.5 * x * x)

    def _L(self, q):
        raw  = self.net(q)                                   # (N, n_tri)
        if self.far_const:
            g = self.gate(q)                                 # (N, 1)
            raw = self.far_raw.unsqueeze(0) + g * (raw - self.far_raw.unsqueeze(0))
        vals = torch.where(self.diag_mask,
                           torch.nn.functional.softplus(raw), raw)
        L = q.new_zeros(q.shape[0], self.latent_dim, self.latent_dim)
        L[:, self.tril_i, self.tril_j] = vals                # remplit la tri. inf.
        return L

    def Mhat(self, q):
        """q : (B, d) → M̂(q) : (B, d, d) SPD, différentiable en q (⟹ Coriolis).

        Si `cond_max` = κ > 0, on ajoute un ridge proportionnel à la trace :
        M̂ = LLᵀ + (tr(LLᵀ)/κ)·I + εI. Comme λmax ≤ tr et λmin ≥ tr/κ, le
        conditionnement est borné par κ POUR TOUT q — y compris là où le réseau
        extrapole. Le ridge suit l'échelle locale de la masse (proportionnel à la
        trace), il ne fausse donc pas la magnitude comme le ferait un ε absolu.
        """
        L   = self._L(q)
        eye = torch.eye(self.latent_dim, device=q.device, dtype=q.dtype)
        M = L @ L.transpose(-1, -2)
        if self.cond_max > 0:
            tr = M.diagonal(dim1=-2, dim2=-1).sum(-1).reshape(-1, 1, 1)
            M = M + (tr / self.cond_max) * eye
        return M + self.eps * eye

    def M(self, q):
        """Métrique complète M̃(q) = m · M̂(q) (m ≡ 1)."""
        return self.log_m.exp() * self.Mhat(q)


class LNN(nn.Module):
    """
    Lagrangian Neural Network.

    Minimise le résidu de l'équation d'Euler-Lagrange discrète :
        a_t  +  γ · v_t  +  β · v_t / ‖v_t‖  +  dE/dz(z_t)  =  0

    Les termes de frottement sont optionnels :
        γ · v      : frottement visqueux  (config.LNN_VISCOUS)
        β · v/‖v‖  : frottement de Coulomb (config.LNN_COULOMB)
    ‖v‖ est régularisé par un ε pour éviter la division par zéro.

    Le résidu est calculé sur les frames intérieures (indices 1…T-2).
    """

    def __init__(self, latent_dim, hidden_dims):
        super().__init__()

        # ── Géométrie figée du décodeur (partagée métrique courbe + potentiel pull-back)
        # Chargée UNE fois si l'un des deux usages est actif (precompute_metric_geom.py).
        need_geom = (getattr(config, 'LNN_METRIC_FROM_DECODER', False)
                     or getattr(config, 'LNN_POTENTIAL_FROM_DECODER', False))
        geom = None
        if need_geom:
            geom_path = config.SAVE_DIR / getattr(config, 'LNN_METRIC_GEOM', 'metric_geom.pt')
            assert geom_path.exists(), (
                f'Géométrie décodeur introuvable : {geom_path}. Cette voie '
                f'(LNN_METRIC_FROM_DECODER / LNN_POTENTIAL_FROM_DECODER) demande un '
                f'précalcul de géométrie du décodeur absent de ce dépôt ; le '
                f'protocole publié passe par LNN_MASS_LEARNED.')
            geom = torch.load(geom_path, map_location='cpu')
            assert geom['latent_dim'] == latent_dim, (
                f"latent_dim incohérent : geom={geom['latent_dim']} vs LNN={latent_dim}")

        # ── Potentiel pull-back Ṽ(q)=V_ICNN(μ(q)) (LNN_POTENTIAL_FROM_DECODER) ──────
        mu_map = None
        if getattr(config, 'LNN_POTENTIAL_FROM_DECODER', False):
            for k in ('J', 'mu_z', 'mu_xy'):
                assert k in geom, (
                    f"Clé '{k}' absente de {geom_path} : la géométrie doit contenir "
                    f"J, mu_z et mu_xy.")
            mu_map = MuMap(geom['J'].float(), geom['mu_z'].float(), geom['mu_xy'].float())
        self.energy = EnergyNet(latent_dim, hidden_dims, mu_map=mu_map)

        # ── Frottement visqueux ───────────────────────────────────────────
        if config.LNN_VISCOUS:
            if getattr(config, 'LNN_VISCOUS_MATRIX', False):
                # Gamma = L @ Lᵀ + εI  (SPD pleine). Init L = exp(½·LOG_GAMMA_INIT)·I
                # ⟹ Gamma ≈ exp(LNN_LOG_GAMMA_INIT)·I : LNN_LOG_GAMMA_INIT règle
                # l'échelle initiale comme en diagonal. Reste apprenable, SPD préservée.
                sg = math.exp(0.5 * float(config.LNN_LOG_GAMMA_INIT))
                self.Lgamma_raw = nn.Parameter(torch.eye(latent_dim) * sg)
                self.register_parameter('log_gamma', None)
            else:
                # Diagonal : un coefficient par dimension
                self.log_gamma = nn.Parameter(
                    torch.full((latent_dim,), float(config.LNN_LOG_GAMMA_INIT))
                )
                self.Lgamma_raw = None
        else:
            self.register_parameter('log_gamma', None)
            self.Lgamma_raw = None

        # ── Frottement de Coulomb ─────────────────────────────────────────
        if config.LNN_COULOMB:
            if getattr(config, 'LNN_COULOMB_MATRIX', False):
                # Beta = L @ Lᵀ + εI  (SPD pleine). Init L = exp(½·LOG_BETA_INIT)·I
                # ⟹ Beta ≈ exp(LNN_LOG_BETA_INIT)·I : LNN_LOG_BETA_INIT règle
                # l'échelle initiale comme en diagonal. Reste apprenable, SPD préservée.
                sb = math.exp(0.5 * float(config.LNN_LOG_BETA_INIT))
                self.Lbeta_raw = nn.Parameter(torch.eye(latent_dim) * sb)
                self.register_parameter('log_beta', None)
            else:
                # Diagonal : un coefficient par dimension
                self.log_beta = nn.Parameter(
                    torch.full((latent_dim,), float(config.LNN_LOG_BETA_INIT))
                )
                self.Lbeta_raw = None
        else:
            self.register_parameter('log_beta', None)
            self.Lbeta_raw = None

        # ── Matrice de masse inverse ──────────────────────────────────────
        if getattr(config, 'LNN_MASS_MATRIX', False):
            # Minv = L @ Lᵀ + εI  (SPD garantie). Init L = √s · I ⟹ Minv ≈ s·I,
            # avec s = LNN_MINV_INIT l'échelle initiale de la masse inverse :
            # s < 1 ⟹ masse plus lourde ⟹ fréquence propre initiale ω = √eig(Minv·∇²V)
            # plus basse (utile quand l'init du potentiel ICNN est trop raide). Reste
            # apprenable ; n'affecte que le point de départ.
            s = float(getattr(config, 'LNN_MINV_INIT', 1.0))
            _Linv0 = torch.eye(latent_dim) * (s ** 0.5)
            if getattr(config, 'LNN_MASS_FIXED', False):
                # Minv figé à s·I : buffer (hors optimiseur, pas de gradient). La fréquence
                # propre est portée entièrement par le potentiel ∇²V (ω=√eig(Minv·∇²V)).
                self.register_buffer('Linv_raw', _Linv0)
            else:
                self.Linv_raw = nn.Parameter(_Linv0)
        else:
            self.Linv_raw = None

        # ── Métrique latente COURBE : masse q-dépendante (Coriolis, résidu plein) ──
        # Deux sources EXCLUSIVES de M̃(q), toutes deux branchées dans la MÊME machinerie
        # courbe (accel/résidu/rollout) :
        #   • LNN_METRIC_FROM_DECODER : M̃(q)=m·Σᵢŵᵢ(q)JᵢᵀJᵢ dérivée de la géométrie FIGÉE
        #     du décodeur (LatentMetricA, m fixe non appris).
        #   • LNN_MASS_LEARNED        : M̂(q)=L(q)L(q)ᵀ+εI APPRISE (LatentMetricLearned,
        #     Cholesky neural) — inertie q-dépendante estimée par le LNN, indépendante du
        #     décodeur. Ne requiert PAS metric_geom.pt.
        # Dans les deux cas : Linv_raw désactivé (la masse vient de la métrique), résidu
        # LNN-plein (Coriolis) et dissipation de Rayleigh disponibles.
        metric_decoder = getattr(config, 'LNN_METRIC_FROM_DECODER', False)
        metric_learned = getattr(config, 'LNN_MASS_LEARNED', False)
        assert not (metric_decoder and metric_learned), (
            'LNN_METRIC_FROM_DECODER et LNN_MASS_LEARNED sont exclusifs '
            '(deux sources de M̃(q)).')
        self.metric = None
        if metric_decoder:
            # geom déjà chargé en tête de __init__ (need_geom).
            self.metric = LatentMetricA(
                geom, mass_init=getattr(config, 'LNN_METRIC_MASS_INIT', 1.0),
                cond_max=getattr(config, 'LNN_METRIC_COND_MAX', 0.0))
        elif metric_learned:
            self.metric = LatentMetricLearned(
                latent_dim,
                mass_init=getattr(config, 'LNN_METRIC_MASS_INIT', 1.0),
                hidden=getattr(config, 'LNN_MASS_LEARNED_HIDDEN', [64, 64]),
                eps=getattr(config, 'LNN_MASS_LEARNED_EPS', 1e-4),
                far_const=getattr(config, 'LNN_MASS_FAR_CONST', False),
                gate_r0=getattr(config, 'LNN_MASS_GATE_R0', 4.0),
                gate_tau=getattr(config, 'LNN_MASS_GATE_TAU', 2.0),
                cond_max=getattr(config, 'LNN_MASS_COND_MAX', 0.0))
        # ── Ablation : masse APPRISE mais FIGÉE (M̂(q) ≡ s·I constante) ───────────
        # L'init de LatentMetricLearned est neutre (dernière couche à poids nuls, biais
        # diag = softplus⁻¹(√s)) ⟹ geler ses paramètres laisse M̂(q) = s·I pour TOUT q,
        # donc une masse CONSTANTE et une Coriolis identiquement nulle, tout en gardant
        # la machinerie COURBE. C'est le seul moyen de conserver la dissipation apprise
        # C(q) (LNN_RAYLEIGH_CQ), qui n'est câblée que dans le chemin courbe : repasser
        # au chemin plat la remplacerait par Gamma, constante. Sert à isoler l'apport de
        # la q-dépendance de la MASSE, à dissipation apprise inchangée.
        # Opt-in ; False (défaut) ⟹ comportement strictement inchangé.
        if metric_learned and getattr(config, 'LNN_MASS_LEARNED_FREEZE', False):
            for _p in self.metric.parameters():
                _p.requires_grad_(False)

        if self.metric is not None:
            self.Linv_raw = None   # masse libre (Minv constant) désactivée : elle vient de M̃(q)
            # Dissipation de Rayleigh proportionnelle masse : C̃(q) = α·M̃(q).
            self.log_alpha_ray = nn.Parameter(torch.tensor(
                float(getattr(config, 'LNN_RAYLEIGH_LOG_ALPHA_INIT', -3.0))))
            self.rayleigh_beta = float(getattr(config, 'LNN_RAYLEIGH_BETA', 0.0))
            # ── Dissipation par CONGRUENCE C̃(q)=M̃^{1/2}(q)·C·M̃^{1/2}(q), C d×d SPD ──
            # Métrique COMPLÈTE M̃=m·M̂. Cœur C = LLᵀ+εI appris (init L=√s·I ⟹ C≈s·I ⟹
            # C̃≈s·M̃ = Rayleigh prop. masse). La congruence NEST le défaut α·M̃ (cas C=αI)
            # à TOUT m, taux d'amortissement ζ=s/(2ω) indép. de m ; et l'étend à un couplage
            # d×d SPD arbitraire ; q̇ᵀC̃q̇=(M̃^{1/2}q̇)ᵀC(M̃^{1/2}q̇)≥0 (C SPD) ⟹ dissipation
            # garantie. (M̂CM̂ — pull-back double, drag-on-moment — abandonné : ne nest PAS α·M̃.)
            self.use_rayleigh_C = bool(getattr(config, 'LNN_RAYLEIGH_C', False))
            self.rayleigh_c_mode = str(getattr(config, 'LNN_RAYLEIGH_C_MODE', 'const'))
            if self.use_rayleigh_C:
                s = float(getattr(config, 'LNN_RAYLEIGH_C_INIT', 0.05))
                self.LC_raw = nn.Parameter(torch.eye(latent_dim) * (s ** 0.5))
                self.rayleigh_C_eps = float(getattr(config, 'LNN_RAYLEIGH_C_EPS', 1e-6))
                # ── Rung 1 : porte scalaire C(q)=softplus(φ(q))·C₀ (intensité q-dép.) ──
                # φ : MLP ℝᵈ→ℝ. Init NEUTRE : dernière couche poids=0, biais=softplus⁻¹(1)
                # ⟹ gate(q)≡1 au départ ⟹ dissipation identique au mode 'const'.
                if self.rayleigh_c_mode == 'scalar':
                    hidden = getattr(config, 'LNN_RAYLEIGH_C_GATE_HIDDEN', [32, 32])
                    layers, prev = [], latent_dim
                    for h in hidden:
                        layers += [nn.Linear(prev, h), nn.ELU()]
                        prev = h
                    last = nn.Linear(prev, 1)
                    nn.init.zeros_(last.weight)
                    nn.init.constant_(last.bias, math.log(math.e - 1.0))  # softplus⁻¹(1)
                    layers.append(last)
                    self.rayleigh_gate = nn.Sequential(*layers)   # φ : ℝ^d → ℝ
                elif self.rayleigh_c_mode != 'const':
                    raise ValueError(
                        f"LNN_RAYLEIGH_C_MODE inconnu : {self.rayleigh_c_mode!r} "
                        f"(attendu 'const' ou 'scalar')")
                else:
                    self.rayleigh_gate = None
            else:
                self.register_parameter('LC_raw', None)
                self.rayleigh_gate = None

            # ── Dissipation visqueuse ANISOTROPE par PULL-BACK (LNN_RAYLEIGH_PULLBACK) ──
            # C̃(q)=Σᵢ ŵᵢ(q) Jᵢᵀ C_amb Jᵢ : pull-back d'une dissipation AMBIANTE (espace de
            # rendu, pos×pos) anisotrope C_amb=L_ambL_ambᵀ+εI apprise. Exact analogue de la
            # masse M̂(q)=Σᵢ ŵᵢ JᵢᵀJᵢ (pull-back de l'inertie ambiante I), avec C_amb au lieu
            # de I. Init L_amb=√s·I ⟹ C_amb≈s·I (isotrope au départ, anisotropie apprise).
            # PRIME sur use_rayleigh_C (les deux à True ⟹ pull-back gagne).
            # ⚠️ Exige la géométrie décodeur (J, pos_dim) ⟹ INDISPONIBLE avec une masse
            # apprise (LNN_MASS_LEARNED, geom=None) : dans ce cas on l'ignore (utiliser CQ).
            _pb_req = bool(getattr(config, 'LNN_RAYLEIGH_PULLBACK', False))
            assert not (_pb_req and geom is None), (
                'LNN_RAYLEIGH_PULLBACK exige la géométrie décodeur (LatentMetricA) et est '
                'incompatible avec LNN_MASS_LEARNED — utiliser LNN_RAYLEIGH_CQ (C(q) appris).')
            self.use_rayleigh_pullback = _pb_req and geom is not None
            if self.use_rayleigh_pullback:
                pos = int(geom['pos_dim'])
                s   = float(getattr(config, 'LNN_RAYLEIGH_PULLBACK_INIT', 0.05))
                self.LC_amb_raw      = nn.Parameter(torch.eye(pos) * (s ** 0.5))
                self.rayleigh_pb_eps = float(getattr(config, 'LNN_RAYLEIGH_PULLBACK_EPS', 1e-6))
                self.use_rayleigh_C  = False   # exclusif
            else:
                self.LC_amb_raw = None

            # ── Dissipation q-dépendante PLEINE C̃(q)=C(q)=L(q)L(q)ᵀ+εI (LNN_RAYLEIGH_CQ) ──
            # « Version C complet » : un MLP φ produit le facteur de Cholesky L(q) (tri. inf.,
            # diagonale > 0 via softplus) ⟹ C(q) SPD pour tout q. Force = C(q)q̇ (forme
            # littérale C(q)q̇), linéaire en q̇, q̇ᵀC(q)q̇ ≥ 0 garanti. PRIME sur pull-back ET
            # use_rayleigh_C (les exclut). Init neutre : dernière couche poids=0, biais diag
            # = softplus⁻¹(√s) ⟹ L(q)≡√s·I ⟹ C(q)≡s·I (= mass-prop. isotrope au départ).
            self.use_rayleigh_cq = bool(getattr(config, 'LNN_RAYLEIGH_CQ', False))
            if self.use_rayleigh_cq:
                d = latent_dim
                self.rayleigh_cq_dim = d
                self.rayleigh_cq_eps = float(getattr(config, 'LNN_RAYLEIGH_CQ_EPS', 1e-6))
                tri = torch.tril_indices(d, d)               # (2, n_tri)
                self.register_buffer('cq_tril_i', tri[0])
                self.register_buffer('cq_tril_j', tri[1])
                self.register_buffer('cq_diag_mask', tri[0] == tri[1])   # (n_tri,) bool
                n_tri = tri.shape[1]
                hidden = getattr(config, 'LNN_RAYLEIGH_CQ_HIDDEN', [64, 64])
                layers, prev = [], d
                for h in hidden:
                    layers += [nn.Linear(prev, h), nn.ELU()]
                    prev = h
                last = nn.Linear(prev, n_tri)
                nn.init.zeros_(last.weight)
                s = float(getattr(config, 'LNN_RAYLEIGH_CQ_INIT', 0.05))
                bias = torch.zeros(n_tri)
                bias[self.cq_diag_mask] = math.log(math.expm1(s ** 0.5))  # softplus⁻¹(√s)
                with torch.no_grad():
                    last.bias.copy_(bias)
                layers.append(last)
                self.rayleigh_cq_net = nn.Sequential(*layers)   # φ : ℝᵈ → ℝ^{d(d+1)/2}
                self.use_rayleigh_C        = False   # exclusif
                self.use_rayleigh_pullback = False   # exclusif (CQ prime)
            else:
                self.rayleigh_cq_net = None
        else:
            self.use_rayleigh_pullback = False
            self.use_rayleigh_cq = False
            self.LC_amb_raw = None
            self.rayleigh_cq_net = None

        # ── Forçage de pression pneumatique ───────────────────────────────
        # F_P(q, P) = b(q)ᵀ P, ajouté au second membre de l'équation EL.
        #   'constant'  : b(q) ≡ B appris (n_c × d)         → F_P = P @ B
        #   'potential' : V_P(q) = −Pᵀ ν_φ(q), ν_φ : ℝ^d → ℝ^{n_c} (MLP libre)
        #                 → F_P = ∂(Pᵀ ν_φ)/∂q = b(q)ᵀ P
        #   'invex'     : idem 'potential' mais ν_φ CONCAVE (−convexe ∘ difféo Φ,
        #                 InvexVolume) ⟹ V_P=−Pᵀν_φ INVEXE à min unique et
        #                 V_eff=V−Pᵀν_φ coercif : équilibre chargé unique ET stable.
        self.use_pressure  = getattr(config, 'LNN_PRESSURE', False)
        self.pressure_mode = getattr(config, 'LNN_PRESSURE_MODE', 'constant')
        if self.use_pressure:
            n_c = len(config.PRESSURE_COLS)
            if self.pressure_mode == 'constant':
                # B init à 0 → forçage nul au départ (le LNN part du cas non forcé)
                self.B_pressure = nn.Parameter(torch.zeros(n_c, latent_dim))
                self.nu_net = None
            elif self.pressure_mode == 'potential':
                hidden = getattr(config, 'PRESSURE_HIDDEN', [32, 32])
                layers, prev = [], latent_dim
                for h in hidden:
                    layers += [nn.Linear(prev, h), nn.ELU()]
                    prev = h
                layers += [nn.Linear(prev, n_c)]
                self.nu_net = nn.Sequential(*layers)   # ν_φ : ℝ^d → ℝ^{n_c}
                self.B_pressure = None
            elif self.pressure_mode == 'invex':
                # ν_φ INVEXE : convexe (ICNN n_c têtes) ∘ difféomorphisme Φ (i-ResNet).
                # Réutilise les hyperparamètres LNN_DIFFEO_* du potentiel de déformation.
                self.nu_net = InvexVolume(
                    latent_dim    = latent_dim,
                    n_c           = n_c,
                    hidden_dims   = getattr(config, 'PRESSURE_HIDDEN', [32, 32]),
                    diffeo_hidden = getattr(config, 'LNN_DIFFEO_HIDDEN', 64),
                    diffeo_blocks = getattr(config, 'LNN_DIFFEO_BLOCKS', 2),
                    diffeo_coeff  = getattr(config, 'LNN_DIFFEO_COEFF', 0.9),
                )
                self.B_pressure = None
            else:
                raise ValueError(
                    f'LNN_PRESSURE_MODE inconnu : {self.pressure_mode!r} '
                    f"(attendu 'constant', 'potential' ou 'invex')")
        else:
            self.B_pressure = None
            self.nu_net = None

    @property
    def gamma(self):
        return self.log_gamma.exp() if self.log_gamma is not None else None

    @property
    def beta(self):
        return self.log_beta.exp() if self.log_beta is not None else None

    @property
    def Gamma(self):
        """
        Matrice de frottement visqueux SPD : Gamma = L @ Lᵀ + εI
        Retourne None si LNN_VISCOUS_MATRIX=False ou LNN_VISCOUS=False.
        """
        if self.Lgamma_raw is None:
            return None
        L = torch.tril(self.Lgamma_raw)
        return L @ L.T + 1e-6 * torch.eye(
            L.shape[0], device=L.device, dtype=L.dtype)

    @property
    def Beta(self):
        """
        Matrice de frottement de Coulomb SPD : Beta = L @ Lᵀ + εI
        Retourne None si LNN_COULOMB_MATRIX=False ou LNN_COULOMB=False.
        """
        if self.Lbeta_raw is None:
            return None
        L = torch.tril(self.Lbeta_raw)
        return L @ L.T + 1e-6 * torch.eye(
            L.shape[0], device=L.device, dtype=L.dtype)

    @property
    def Minv(self):
        """
        Matrice de masse inverse SPD : Minv = L @ Lᵀ + ε I
        avec L = tril(Linv_raw) (triangulaire inférieure).
        Retourne None si LNN_MASS_MATRIX=False.
        """
        if self.Linv_raw is None:
            return None
        L = torch.tril(self.Linv_raw)          # (D, D) triangulaire inférieure
        return L @ L.T + 1e-6 * torch.eye(
            L.shape[0], device=L.device, dtype=L.dtype)  # (D, D) SPD

    def _C_amb(self):
        """Dissipation visqueuse AMBIANTE anisotrope C_amb = L_amb L_ambᵀ + εI (pos×pos SPD)."""
        L = torch.tril(self.LC_amb_raw)
        eye = torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
        return L @ L.t() + self.rayleigh_pb_eps * eye

    def _C_q(self, q):
        """
        Matrice de dissipation pleine q-dépendante C(q) = L(q)L(q)ᵀ + εI (N, D, D) SPD.
        L(q) = facteur de Cholesky triangulaire inférieur produit par le MLP rayleigh_cq_net,
        diagonale > 0 via softplus (hors-diagonale libre) ⟹ C(q) symétrique définie positive
        pour tout q. q : (N, D).
        """
        d   = self.rayleigh_cq_dim
        raw = self.rayleigh_cq_net(q)                              # (N, n_tri)
        # diagonale → softplus(>0), hors-diagonale → brute (pas d'op in-place)
        vals = torch.where(self.cq_diag_mask,
                           torch.nn.functional.softplus(raw), raw)  # (N, n_tri)
        L = q.new_zeros(q.shape[0], d, d)
        L[:, self.cq_tril_i, self.cq_tril_j] = vals                # remplit la tri. inf.
        eye = torch.eye(d, device=q.device, dtype=q.dtype)
        return L @ L.transpose(-1, -2) + self.rayleigh_cq_eps * eye

    def _rayleigh_diss(self, q, v, Mhat):
        """Dissipation C̃(q)q̇, mise à l'échelle par `LNN_DISS_SCALE` (défaut 1.0 = neutre).

        Knob de DIAGNOSTIC : multiplie TOUT le covecteur de dissipation par un facteur
        constant, sans réentraîner. Permet de tester la sensibilité du rollout à
        l'amortissement appris (ex. 0.5 = moitié moins d'amortissement) là où le modèle
        est soupçonné d'en manquer ou d'en avoir trop. Comme `_rayleigh_diss` est le
        point d'entrée UNIQUE de la dissipation (résidu `residual` ET accélération
        `accel`, donc tous les intégrateurs), le facteur s'applique partout de façon
        cohérente. ⚠️ À 0.5 le facteur touche AUSSI le plancher SPD εI, donc le
        C̃ effectif reste SPD (produit d'une SPD par un scalaire > 0) et la dissipation
        reste physiquement admissible (q̇ᵀC̃q̇ ≥ 0).
        """
        diss = self._rayleigh_diss_raw(q, v, Mhat)
        s = float(getattr(config, 'LNN_DISS_SCALE', 1.0))
        return diss if s == 1.0 else s * diss

    def _rayleigh_diss_raw(self, q, v, Mhat):
        """
        Covecteur de dissipation de Rayleigh −∂D/∂q̇ = C̃(q)·q̇ (renvoie C̃(q)q̇, N×D).
        Chemin métrique courbe ; quatre formes, par ordre de priorité :
          • cq (use_rayleigh_cq) : C̃(q)=C(q)=L(q)L(q)ᵀ+εI, matrice d'amortissement PLEINE
            q-dépendante (Cholesky L(q) par MLP, diag>0). Forme littérale C(q)q̇, la plus
            expressive ; q̇ᵀC(q)q̇≥0 garanti (C(q) SPD). « Version C complet ».
          • pull-back (use_rayleigh_pullback) : C̃(q)=Σᵢ ŵᵢ(q) Jᵢᵀ C_amb Jᵢ — pull-back d'une
            dissipation visqueuse AMBIANTE anisotrope C_amb (pos×pos SPD apprise). LINÉAIRE
            en q̇, ANISOTROPE, q̇ᵀC̃q̇=Σᵢ ŵᵢ‖√C_amb Jᵢq̇‖²≥0. Même présence ŵᵢ(q)/géométrie Jᵢ
            que la masse M̂(q)=Σᵢ ŵᵢ JᵢᵀJᵢ (= pull-back de I). Aucune matrice C̃ matérialisée.
          • use_rayleigh_C : C̃(q)=M̂(q)·C·M̂(q), C d×d SPD (+ porte φ(q) si mode 'scalar').
          • défaut : C̃(q)=α·M̃(q) (Rayleigh proportionnel masse, isotrope dans la métrique).
        q, v : (N, D) ; Mhat : (N, D, D).
        """
        if getattr(self, 'use_rayleigh_cq', False):
            C = self._C_q(q)                                     # (N, D, D) SPD
            return (C @ v.unsqueeze(-1)).squeeze(-1)            # C(q) q̇   (N, D)
        if getattr(self, 'use_rayleigh_pullback', False):
            C  = self._C_amb()                                   # (pos, pos) SPD
            wh = self.metric.presence(q)                         # (N, K)  présence ŵᵢ(q)
            J  = self.metric.J                                   # (K, pos, d)
            Jq    = torch.einsum('kpd,nd->nkp', J, v)            # Jᵢ q̇          (N, K, pos)
            CJq   = torch.einsum('pr,nkr->nkp', C, Jq)           # C_amb Jᵢ q̇    (N, K, pos)
            JtCJq = torch.einsum('kpd,nkp->nkd', J, CJq)         # Jᵢᵀ C_amb Jᵢ q̇ (N, K, d)
            return torch.einsum('nk,nkd->nd', wh, JtCJq)         # Σᵢ ŵᵢ(...)q̇   (N, D)
        if getattr(self, 'use_rayleigh_C', False):
            L   = self.LC_raw
            eye = torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
            C   = L @ L.t() + self.rayleigh_C_eps * eye          # (D, D) SPD
            Mv0 = (Mhat @ v.unsqueeze(-1)).squeeze(-1)           # M̂ q̇
            CMv = Mv0 @ C.t()                                    # C(M̂ q̇)
            diss = (Mhat @ CMv.unsqueeze(-1)).squeeze(-1)        # M̂ C M̂ q̇
            if getattr(self, 'rayleigh_gate', None) is not None:
                gate = torch.nn.functional.softplus(self.rayleigh_gate(q))  # (N, 1)
                diss = gate * diss
            return diss
        alpha = self.log_alpha_ray.exp()
        Mfull = self.metric.log_m.exp() * Mhat                   # M̃(q)
        return alpha * (Mfull @ v.unsqueeze(-1)).squeeze(-1)

    def dE_dz(self, z):
        """
        Gradient ∂E/∂z.
        z : (B, D) — doit pouvoir propager les gradients.
        Retourne : (B, D)

        Si config.LNN_ANALYTIC_GRAD et EnergyNet ICNN sans transformation d'entrée
        (ni difféo invex Φ, ni pull-back μ) → forme analytique (EnergyNet.grad_E,
        Jacobien forward) ; sinon repli sur torch.autograd.grad (VJP économe).

        NB : les deux transformations sont exclues de la voie analytique —
        • difféo invex (Φ∘g) : la chaîne JΦ casse la forme softplus fermée ;
        • pull-back décodeur (mu_map) : grad_E matérialiserait un Jacobien forward en
          espace μ (dim K·pos, lourd) → autograd VJP préférable.
        EnergyNet.forward n'a plus d'autograd.grad imbriqué (Bregman analytique) et les
        buffers de MuMap sont détachés ⟹ pas de double-backward fragile.
        (self.energy.analytic_ok = icnn ∧ pas de Φ ∧ pas de μ.)
        """
        if getattr(config, 'LNN_ANALYTIC_GRAD', False) and self.energy.analytic_ok:
            return self.energy.grad_E(z)
        z = z.requires_grad_(True)
        E = self.energy(z)
        grad = torch.autograd.grad(E.sum(), z, create_graph=self.training)[0]
        return grad

    def pressure_force(self, z, p):
        """
        Force généralisée de pression F_P = b(q)ᵀ P au point z.

        z : (B, D) — état latent central
        p : (B, n_c) — pression normalisée (n_c chambres)
        Retourne : (B, D)

        'constant'         : F_P = P @ B     (B constant n_c×D)
        'potential'/'invex': F_P = ∂(Pᵀ ν_φ)/∂z = Σ_c P_c ∂ν_c/∂z   (autograd)
                             ('invex' : ν_φ = −(convexe ∘ Φ), concave ⟹ V_P invexe/min)
        """
        if self.pressure_mode == 'constant':
            return p @ self.B_pressure                         # (B, D)
        # 'potential' / 'invex' — même RHS ∂(Pᵀν_φ)/∂z (ν_φ diffère seulement)
        z = z.requires_grad_(True)
        nu = self.nu_net(z)                                    # (B, n_c)
        scal = (nu * p).sum()                                  # Σ_{b,c} P_c ν_c
        grad = torch.autograd.grad(scal, z, create_graph=self.training)[0]
        return grad                                            # (B, D)

    def residual(self, z_seq, dt, pressure=None):
        """
        Calcule le résidu Euler-Lagrange par différences finies.

        z_seq    : (B, T, D)  avec T = SEQ_LEN (3 pour ordre 2, 5 pour ordre 4)
        dt       : float, pas de temps
        pressure : (B, n_c) ou None — pression à la frame centrale. Si fournie
                   (et self.use_pressure), le forçage −Minv·b(q)ᵀP est soustrait.

        Retourne residual : (B, T-2, D)  —  1 point pour ordre 2, 1 point pour ordre 4
        """
        if pressure is not None and not self.use_pressure:
            pressure = None   # forçage désactivé : on ignore une pression fournie
        if getattr(config, 'LNN_LOSS_MODE', 'accel') == 'integral':
            return self.residual_integral(z_seq, dt, pressure)
        if self.metric is not None:
            return self._residual_curved(z_seq, dt, pressure)
        if config.LNN_FD_ORDER == 4:
            return self._residual_order4(z_seq, dt, pressure)
        else:
            return self._residual_order2(z_seq, dt, pressure)

    def _residual_order2(self, z_seq, dt, pressure=None):
        """Différences finies centrées ordre 2 — T=3, 1 point de résidu."""
        v = (z_seq[:, 2:] - z_seq[:, :-2])                              / (2 * dt)
        a = (z_seq[:, 2:] - 2 * z_seq[:, 1:-1] + z_seq[:, :-2])        / (dt ** 2)

        z_mid  = z_seq[:, 1:-1]            # (B, T-2, D)
        B, Tm2, D = z_mid.shape
        grad_E = self.dE_dz(z_mid.reshape(B * Tm2, D)).reshape(B, Tm2, D)

        rhs = grad_E
        if self.Gamma is not None:
            # Gamma : (D, D) — frottement visqueux matriciel
            rhs = rhs + v @ self.Gamma.T
        elif self.log_gamma is not None:
            rhs = rhs + self.gamma * v
        if self.Beta is not None:
            # Beta : (D, D) — frottement de Coulomb matriciel
            v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            rhs = rhs + (v / v_norm) @ self.Beta.T
        elif self.log_beta is not None:
            v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            rhs = rhs + self.beta * v / v_norm
        # Forçage de pression : second membre EL  →  −F_P côté forces
        if pressure is not None:
            F_P = self.pressure_force(z_mid.reshape(B * Tm2, D),
                                      pressure).reshape(B, Tm2, D)
            rhs = rhs - F_P
        if self.Minv is not None:
            rhs = rhs @ self.Minv.T
        return a + rhs   # (B, T-2, D)

    def _residual_order4(self, z_seq, dt, pressure=None):
        """Différences finies centrées ordre 4 — T=5, 1 point de résidu (centre)."""
        z0, z1, z2, z3, z4 = [z_seq[:, i] for i in range(5)]   # chacun (B, D)

        v = (-z4 + 8*z3 - 8*z1 + z0) / (12 * dt)
        a = (-z4 + 16*z3 - 30*z2 + 16*z1 - z0) / (12 * dt**2)

        grad_E = self.dE_dz(z2)   # (B, D)

        rhs = grad_E
        if self.Gamma is not None:
            rhs = rhs + v @ self.Gamma.T
        elif self.log_gamma is not None:
            rhs = rhs + self.gamma * v
        if self.Beta is not None:
            v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            rhs = rhs + (v / v_norm) @ self.Beta.T
        elif self.log_beta is not None:
            v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            rhs = rhs + self.beta * v / v_norm
        # Forçage de pression : second membre EL  →  −F_P côté forces
        if pressure is not None:
            rhs = rhs - self.pressure_force(z2, pressure)   # (B, D)
        if self.Minv is not None:
            rhs = rhs @ self.Minv.T
        return (a + rhs).unsqueeze(1)   # (B, 1, D) — cohérent avec (B, T-2, D)

    # ── Résidu inverse-INTÉGRAL (SANS q̈ — Laiche et al. 2025, laiche2025noaccel) ──
    def residual_integral(self, z_seq, dt, pressure=None):
        """
        Forme FAIBLE (intégrée) d'Euler-Lagrange, sans q̈. On intègre
            d/dt(∂L/∂q̇) − ∂L/∂q + ∂D/∂q̇ = Q
        sur chaque pas [t_k, t_{k+1}] : la dérivée totale s'intègre EXACTEMENT en
        différence de moments conjugués, et q̈ disparaît :

            p_{k+1} − p_k  ≈  (Δt/2)(g_k + g_{k+1}),
            p = ∂L/∂q̇  (moment conjugué),   g = ∂L/∂q − ∂D/∂q̇ + Q.

        q̇ par différence PREMIÈRE centrée (jamais q̈). On n'inverse JAMAIS la masse :
          • chemin diagonal (M̃ const) : forme INCRÉMENT DE VITESSE — on pose p≡q̇ et
            g≡M̃⁻¹·(force) (= « accélération », via Minv déjà disponible) ;
          • chemin courbe M̃(q) : forme INCRÉMENT DE MOMENT — p = M̃(q)q̇, g = force.
            La Coriolis −(∂p/∂q)q̇ est capturée GRATUITEMENT et exactement par Δp.

        Pression : maintien d'ordre 0 (pression centrale appliquée à toute la fenêtre).

        z_seq : (B, T, D), T = SEQ_LEN ≥ 4. Retourne residual : (B, T-3, D).
        """
        B, T, D = z_seq.shape
        if T < 4:
            raise ValueError(
                f'LNN_LOSS_MODE="integral" exige SEQ_LEN≥4 (vitesses consécutives) ; '
                f'SEQ_LEN={T}. Mettre LNN_FD_ORDER=4 (SEQ_LEN=5).')
        # Vitesses (diff. PREMIÈRE centrée) et positions aux frames intérieures k=1..T-2
        v  = (z_seq[:, 2:] - z_seq[:, :-2]) / (2 * dt)        # (B, Ni, D)
        q  = z_seq[:, 1:-1]                                   # (B, Ni, D)
        Ni = T - 2
        p_exp = None
        if pressure is not None:
            p_exp = pressure.unsqueeze(1).expand(B, Ni, -1).reshape(B * Ni, -1)
        if self.metric is not None:
            p, g = self._pg_curved(q.reshape(B * Ni, D), v.reshape(B * Ni, D), p_exp)
        else:
            p, g = self._pg_diagonal(q.reshape(B * Ni, D), v.reshape(B * Ni, D), p_exp)
        p = p.reshape(B, Ni, D)
        g = g.reshape(B, Ni, D)
        # Résidu intégral par intervalle [j, j+1] : Δp − (Δt/2)(g_j + g_{j+1})
        dp   = p[:, 1:] - p[:, :-1]                           # (B, Ni-1, D)
        trap = 0.5 * dt * (g[:, 1:] + g[:, :-1])              # (B, Ni-1, D)
        return dp - trap                                     # (B, T-3, D)

    def _pg_diagonal(self, q, v, p_press):
        """
        (p, g) du résidu intégral, chemin masse CONSTANTE. q, v : (N, D).
        Forme incrément de vitesse : p ≡ q̇, g ≡ M̃⁻¹·force (= accélération), avec
        force = ∂L/∂q − ∂D/∂q̇ + Q = −∂V/∂q − (Γq̇ + β q̇/‖q̇‖) + F_P. On réutilise Minv
        (jamais d'inversion) : signe cohérent avec _residual_order2 (rhs = +∂V/∂q + diss − F_P).
        """
        rhs = self.dE_dz(q)                                   # +∂V/∂q   (N, D)
        if self.Gamma is not None:
            rhs = rhs + v @ self.Gamma.T
        elif self.log_gamma is not None:
            rhs = rhs + self.gamma * v
        if self.Beta is not None:
            vn = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            rhs = rhs + (v / vn) @ self.Beta.T
        elif self.log_beta is not None:
            vn = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            rhs = rhs + self.beta * v / vn
        if p_press is not None:
            rhs = rhs - self.pressure_force(q, p_press)
        if self.Minv is not None:
            rhs = rhs @ self.Minv.T
        return v, -rhs                                        # p = q̇,  g = −M̃⁻¹·rhs

    def _rayleigh_C_diss(self, Mhat, v, q_g):
        """
        Dissipation de Rayleigh par CONGRUENCE : C̃(q) = M̃^{1/2}(q)·C·M̃^{1/2}(q),
        avec la métrique COMPLÈTE M̃ = m·M̂ (m = échelle de masse), C = LLᵀ+εI (d×d SPD
        apprise). Retourne le covecteur C̃ q̇ : (N, D).

        Pourquoi M̃^{1/2} (et NON M̂^{1/2}) : le mode par défaut est α·M̃ = α·m·M̂. Avec
        M̃^{1/2}, C=αI redonne EXACTEMENT α·M̃ ⟹ nest le défaut à TOUT m, et le taux
        d'amortissement ζ = s/(2ω) est INDÉPENDANT de m (avec M̂^{1/2}, ζ ∝ 1/m : un grand
        m diluait la dissipation — bug corrigé 2026-06-27). SPD garantie :
        q̇ᵀC̃q̇ = (M̃^{1/2}q̇)ᵀC(M̃^{1/2}q̇) ≥ 0 (C SPD). M̃^{1/2}=√m·M̂^{1/2}, M̂^{1/2} via
        eigh symétrique (d petit, différentiable en q et en les poids).
        Rung 1 : porte scalaire c(q)=softplus(φ(q)) ≥ 0 module l'intensité (SPD préservée).
        """
        L   = self.LC_raw
        eye = torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
        C   = L @ L.t() + self.rayleigh_C_eps * eye            # (D, D) SPD
        # racine SPD M̃^{1/2} = √m · M̂^{1/2}, M̂^{1/2} = V diag(√λ) Vᵀ. clamp(min>0) :
        # M̂ peut être rang-déficient (cf. ridge §3.4) → borne √λ pour gradient fini en λ→0.
        lam, V = torch.linalg.eigh(Mhat)                       # (N, D), (N, D, D)
        sq     = (self.metric.log_m.exp() * lam.clamp(min=1e-8)).sqrt()  # √(m·λ)  (N, D)
        sqrtM  = (V * sq.unsqueeze(-2)) @ V.transpose(-1, -2)  # M̃^{1/2}   (N, D, D)
        s    = (sqrtM @ v.unsqueeze(-1)).squeeze(-1)           # M̃^{1/2} q̇         (N, D)
        Cs   = s @ C.t()                                       # C (M̃^{1/2} q̇)
        diss = (sqrtM @ Cs.unsqueeze(-1)).squeeze(-1)          # M̃^{1/2} C M̃^{1/2} q̇
        if getattr(self, 'rayleigh_gate', None) is not None:
            gate = torch.nn.functional.softplus(self.rayleigh_gate(q_g))  # (N, 1)
            diss = gate * diss
        return diss

    def _pg_curved(self, q, v, p_press):
        """
        (p, g) du résidu intégral, chemin métrique COURBE M̃(q)=m·M̂(q). q, v : (N, D).
        p = M̃(q)q̇ (moment conjugué) ; g = ∂L/∂q − ∂D/∂q̇ + Q (SANS le terme de Coriolis :
        il est porté par Δp). ∂L/∂q = m·∂_q(½q̇ᵀM̂q̇) − ∂V/∂q. Aucune inversion de masse.
        """
        cg   = self.training
        q_g  = q if q.requires_grad else q.detach().requires_grad_(True)
        m    = self.metric.log_m.exp()
        Mhat = self.metric.Mhat(q_g)                          # (N, D, D)
        Mv   = (Mhat @ v.unsqueeze(-1)).squeeze(-1)           # M̂ q̇
        p    = m * Mv                                         # moment conjugué M̃(q)q̇
        Tk   = 0.5 * (v * Mv).sum(-1)                         # ½ q̇ᵀM̂q̇
        dT_dq = torch.autograd.grad(Tk.sum(), q_g,
                                    create_graph=cg, retain_graph=True)[0]  # (N, D)
        # ∂L/∂q = m·∂T/∂q − ∂V/∂q   (V = potentiel ICNN, pull-back éventuel inclus)
        g = m * dT_dq - self.dE_dz(q_g)
        # −∂D/∂q̇ : dissipation de Rayleigh (helper partagé avec accel)
        g = g - self._rayleigh_diss(q_g, v, Mhat)
        if p_press is not None:
            g = g + self.pressure_force(q_g, p_press)
        return p, g

    # ── Résidu LNN-plein avec métrique courbe M̃(q) (Coriolis dérivée) ─────────
    def _residual_curved(self, z_seq, dt, pressure=None):
        """
        Résidu d'Euler-Lagrange complet pour L = ½ q̇ᵀM̃(q)q̇ − V(q), avec M̃(q) dérivée
        du décodeur (forme A). La Coriolis est obtenue par autograd à travers M̃(q) :

            M̃(q) q̈ = ∂L/∂q − (∂p/∂q)q̇ − C̃(q)q̇ + F_P ,    p = M̃(q)q̇ .

        Renvoie le résidu d'ACCÉLÉRATION r = q̈ − a_pred (même convention d'échelle que
        les autres résidus). Compatible LNN_FD_ORDER ∈ {2, 4}.
        """
        if config.LNN_FD_ORDER == 4:
            z0, z1, z2, z3, z4 = [z_seq[:, i] for i in range(5)]
            v = (-z4 + 8*z3 - 8*z1 + z0) / (12 * dt)
            a = (-z4 + 16*z3 - 30*z2 + 16*z1 - z0) / (12 * dt**2)
            r = self._curved_core(z2, v, a, pressure)        # (B, D)
            return r.unsqueeze(1)                            # (B, 1, D)
        else:
            v = (z_seq[:, 2:] - z_seq[:, :-2])                       / (2 * dt)
            a = (z_seq[:, 2:] - 2 * z_seq[:, 1:-1] + z_seq[:, :-2]) / (dt ** 2)
            q = z_seq[:, 1:-1]                               # (B, T-2, D)
            B, Tm2, D = q.shape
            r = self._curved_core(q.reshape(B * Tm2, D),
                                  v.reshape(B * Tm2, D),
                                  a.reshape(B * Tm2, D), pressure)
            return r.reshape(B, Tm2, D)

    def _curved_core(self, q, v, a, pressure=None):
        """Résidu d'accélération courbe : r = q̈ − accel(q, q̇). q, v, a : (N, D)."""
        return a - self.accel(q, v, pressure)

    def accel(self, q, v, pressure=None):
        """
        Accélération du système lagrangien COURBE (métrique dérivée du décodeur) :

            q̈ = M̃(q)⁻¹ [ ∂L/∂q − (∂p/∂q)q̇ − C̃(q)q̇ + F_P ] ,   p = M̃(q)q̇ ,

        avec L = ½ q̇ᵀM̃(q)q̇ − V(q), C̃(q) = α·M̃(q) (Rayleigh prop. masse). La Coriolis
        (∂p/∂q)q̇ et le terme cinétique de ∂L/∂q sont obtenus par autograd à travers M̃(q).
        q, v : (N, D) → (N, D). Source unique de la physique : utilisée par le résidu
        d'entraînement (q̈−accel) ET les intégrateurs d'inférence (viz.simulate_rk4).
        Si l'encodeur est entraînable (q.requires_grad), le graphe est conservé ; sinon
        on crée une feuille (AE figé, cas nominal).
        """
        cg   = self.training
        q_g  = q if q.requires_grad else q.detach().requires_grad_(True)
        m    = self.metric.log_m.exp()

        Mhat = self.metric.Mhat(q_g)                          # (N, D, D)  (sans m)
        Mv   = (Mhat @ v.unsqueeze(-1)).squeeze(-1)           # M̂(q)v      (N, D)
        T    = 0.5 * (v * Mv).sum(-1)                         # ½ vᵀM̂v     (N,)

        # ∂T/∂q (partie cinétique de ∂L/∂q, sans m)
        dT_dq = torch.autograd.grad(T.sum(), q_g,
                                    create_graph=cg, retain_graph=True)[0]   # (N, D)
        # Jacobien ∂(M̂v)/∂q contracté avec v  ⟹  (∂p̂/∂q)v  (sans m)
        cols = []
        for k in range(self.metric.latent_dim):
            gk = torch.autograd.grad(Mv[:, k].sum(), q_g,
                                     create_graph=cg, retain_graph=True)[0]  # (N, D)
            cols.append(gk)
        Jp   = torch.stack(cols, dim=1)                       # (N, D, D) : [n,k,j]=∂Mv_k/∂q_j
        pq_v = torch.einsum('nkj,nj->nk', Jp, v)              # (N, D)  (sans m)

        Mfull = m * Mhat                                      # M̃(q)
        # ∂L/∂q = m·∂T/∂q − ∂V/∂q   (V = potentiel ICNN)
        dL_dq = m * dT_dq - self.dE_dz(q_g)
        # conservatif + Coriolis : ∂L/∂q − (∂p/∂q)q̇
        force = dL_dq - m * pq_v
        # dissipation de Rayleigh : −∂D/∂q̇ = −C̃(q)q̇ (pull-back / M̂CM̂ / α·M̃ — helper)
        force = force - self._rayleigh_diss(q_g, v, Mhat)
        if self.rayleigh_beta > 0.0:
            raise NotImplementedError(
                'Rayleigh β>0 (proportionnel raideur, ∇²V) pas encore câblé — laisser β=0.')
        # forçage de pression : +F_P
        if pressure is not None:
            force = force + self.pressure_force(q_g, pressure)

        # q̈ = M̃(q)⁻¹ · force   (résolution D×D). Plancher de masse (ridge) : évite la
        # singularité quand une seule gaussienne domine (M̂ rang-déficient, cf. doc §3.4/§5)
        # — négligeable vs M̂~O(1) (géométrie normalisée par precompute_metric_geom.py).
        ridge = float(getattr(config, 'LNN_METRIC_RIDGE', 1e-4))
        eye = torch.eye(Mfull.shape[-1], device=Mfull.device, dtype=Mfull.dtype)
        return torch.linalg.solve(Mfull + ridge * eye, force.unsqueeze(-1)).squeeze(-1)


class GaussianDecoder(nn.Module):
    """
    Décodeur 2DGS : z ∈ ℝ^D → image rasterisée (1, H, W).

    Pour chaque gaussienne k, le MLP prédit :
        μ_k  ∈ [0,1]²     — centre (coordonnées normalisées)
        σ_k  ∈ ℝ²₊        — demi-axes (log-paramétrés)
        θ_k  ∈ [0, π)     — angle de rotation
        α_k  ∈ (0,1)      — opacité
        c_k  ∈ [0,1]      — intensité (niveaux de gris)

    Rasterisation : alpha compositing sur grille (H, W).
        I(p) = Σ_k  c_k · α_k · G_k(p)
    où G_k(p) = exp(-½ (p-μ_k)ᵀ Σ_k⁻¹ (p-μ_k)), clampé en [0,1].

    Pas de fond précalculé : le fond est implicitement appris via
    les gaussiennes (opacité faible + grande variance).
    """

    def __init__(self, latent_dim: int, n_gaussians: int,
                 hidden_dims: list, img_size: tuple, n_channels: int = 3):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.n_channels  = n_channels
        self.img_size    = img_size   # (H, W)

        # MLP : z → paramètres bruts de toutes les gaussiennes
        # Sortie par gaussienne : μx, μy, log_sx, log_sy, θ_raw, logit_α, c×n_channels
        #                          2  +    2     +   1   +    1  +  n_channels
        out_dim = n_gaussians * (6 + n_channels)
        layers  = []
        prev    = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)

        # Grille de pixels fixe (H, W, 2), coordonnées dans [0,1]
        H, W = img_size
        ys = torch.linspace(0, 1, H)
        xs = torch.linspace(0, 1, W)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)
        grid = torch.stack([grid_x, grid_y], dim=-1)             # (H, W, 2)
        self.register_buffer('grid', grid)                        # non-entraînable
        # Ordre de compositing par mobilité (mis à jour chaque epoch)
        self.register_buffer('mobility_order',
                            torch.arange(n_gaussians))           # (K,) défaut : ordre naturel

    def forward(self, z: torch.Tensor, serial: bool = False) -> torch.Tensor:
        """
        z      : (B, D)
        serial : si True, rasterise un élément à la fois (économise la VRAM)
        out    : (B, C, H, W)  valeurs dans [0, 1]  (C = n_channels)
        """
        if serial:
            return torch.stack([self._rasterize_one(z[i:i+1]).squeeze(0) for i in range(z.shape[0])])
        B    = z.shape[0]
        H, W = self.img_size
        K    = self.n_gaussians

        # ── Prédiction des paramètres ──────────────────────────────────
        raw = self.mlp(z).reshape(B, K, 6 + self.n_channels)

        # μ dans [-0.1, 1.1] : autorise 10% de débordement hors image
        mu    = torch.sigmoid(raw[..., 0:2]) * 1.2 - 0.1  # (B,K,2)  ∈ [-0.1,1.1]
        sigma = torch.exp(raw[..., 2:4]).clamp(1e-3, 0.5)  # (B,K,2)  demi-axes > 0
        theta = torch.tanh(raw[..., 4]) * (torch.pi / 2)   # (B,K)    ∈ (-π/2, π/2)
        alpha = torch.sigmoid(raw[..., 5])                      # (B,K)
        color = torch.sigmoid(raw[..., 6:6+self.n_channels])       # (B,K,C)

        # ── Matrice de covariance inverse ─────────────────────────────
        # Rotation R(θ), échelle S = diag(σx, σy)
        # Σ = R S² Rᵀ  →  Σ⁻¹ = R S⁻² Rᵀ

        cos_t = torch.cos(theta)   # (B, K)
        sin_t = torch.sin(theta)

        sx2_inv = 1.0 / sigma[..., 0].pow(2)   # (B, K)
        sy2_inv = 1.0 / sigma[..., 1].pow(2)

        # Éléments de Σ⁻¹ (symétrique 2×2)
        A = cos_t.pow(2) * sx2_inv + sin_t.pow(2) * sy2_inv   # (B, K)
        B_ = cos_t * sin_t * (sx2_inv - sy2_inv)               # (B, K)
        C = sin_t.pow(2) * sx2_inv + cos_t.pow(2) * sy2_inv   # (B, K)

        # ── Évaluation des gaussiennes sur la grille ──────────────────
        # grid : (H, W, 2)  →  (1, 1, H, W, 2)
        grid = self.grid.unsqueeze(0).unsqueeze(0)              # (1, 1, H, W, 2)

        # mu   : (B, K, 2)  →  (B, K, 1, 1, 2)
        mu_  = mu.unsqueeze(2).unsqueeze(3)

        # Déplacement (B, K, H, W, 2)
        d    = grid - mu_
        dx   = d[..., 0]   # (B, K, H, W)
        dy   = d[..., 1]

        # Forme quadratique : dᵀ Σ⁻¹ d
        A_   = A.unsqueeze(-1).unsqueeze(-1)    # (B, K, 1, 1)
        B__  = B_.unsqueeze(-1).unsqueeze(-1)
        C_   = C.unsqueeze(-1).unsqueeze(-1)

        quad = A_ * dx.pow(2) + 2 * B__ * dx * dy + C_ * dy.pow(2)  # (B,K,H,W)
        G    = torch.exp(-0.5 * quad)                                  # (B,K,H,W)

        # ── Alpha compositing ─────────────────────────────────────────
        C      = self.n_channels
        alpha_ = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1, 1)
        color_ = color.reshape(B, K, C, 1, 1)                     # (B, K, C, 1, 1)
        G_     = G.unsqueeze(2)                                    # (B, K, 1, H, W)

        # I(p) = Σ_k c_k · α_k · G_k(p)  →  (B, C, H, W)
        weights = (alpha_ * G_).sum(dim=1).clamp(min=1.0)         # (B, 1, H, W)
        image   = (color_ * alpha_ * G_).sum(dim=1) / weights     # (B, C, H, W)
        image   = image.clamp(0, 1)

        return image



    @torch.no_grad()
    def update_mobility_order(self, z_batch: torch.Tensor):
        """
        Recalcule l'ordre de compositing par variance empirique de μ_k sur z_batch.
        z_batch : (N, D) — batch de z représentatifs (ex: tout le dataset)
        Gaussiennes les plus mobiles en premier → occultent le fond.
        """
        raw  = self.mlp(z_batch).reshape(len(z_batch), self.n_gaussians, 6 + self.n_channels)
        mu   = torch.sigmoid(raw[..., 0:2])          # (N, K, 2)
        # Variance sur la dimension batch pour chaque gaussienne
        mobility = mu.var(dim=0).sum(dim=-1)         # (K,)  somme var_x + var_y
        self.mobility_order = torch.argsort(mobility, descending=True)  # (K,)

    def smart_init_from_image(self, ref_image: torch.Tensor, z_ref: torch.Tensor,
                              grad_weight: float = 0.8,
                              init_bias: bool = True):
        """
        Initialise les biais de la dernière couche du MLP à partir d'une image de référence
        et du z correspondant, pour que decoder(z_ref) ≈ smart_init exactement.

        biais = target - W · h(z_ref)
        où h(z_ref) est la sortie de l'avant-dernière couche (poids Kaiming inchangés).

        ref_image : (C, H, W) float32 dans [0, 1]
        z_ref     : (D,) ou (1, D) — z encodé de la même frame
        grad_weight : poids du gradient vs fond uniforme
        """
        import numpy as np

        # ── Normalisation de l'image ───────────────────────────────────────
        if ref_image.ndim == 3 and ref_image.shape[0] in (1, 3):
            img = ref_image.permute(1, 2, 0)   # (H, W, C)
        else:
            img = ref_image                     # déjà (H, W, C)
        H, W, C = img.shape
        K = self.n_gaussians

        # ── 1. Carte de gradient ──────────────────────────────────────────
        gray = img.mean(dim=-1)                             # (H, W)
        gx = torch.nn.functional.pad(gray[:, 1:] - gray[:, :-1], (0, 1))
        gy = torch.nn.functional.pad(gray[1:, :] - gray[:-1, :], (0, 0, 0, 1))
        grad_mag = (gx**2 + gy**2).sqrt()                  # (H, W)

        probs = grad_weight * grad_mag.flatten() + (1 - grad_weight)
        probs = (probs / probs.sum()).cpu().numpy()

        n_sample = min(K, H * W)
        flat_idx = np.random.choice(H * W, size=n_sample, replace=False, p=probs)
        py = flat_idx // W
        px = flat_idx  % W

        # ── 2. Positions dans [0, 1] ──────────────────────────────────────
        mu_x = px / (W - 1)    # [0, 1]
        mu_y = py / (H - 1)    # [0, 1]
        mu = torch.tensor(np.stack([mu_x, mu_y], axis=1), dtype=torch.float32)
        # Inverse de sigmoid(x)*1.2-0.1 = mu  →  x = logit((mu+0.1)/1.2)
        logit_mu = torch.logit(((mu + 0.1) / 1.2).clamp(0.01, 0.99))

        # ── 3. Échelle ∝ distance au plus proche voisin ───────────────────
        with torch.no_grad():
            dists = torch.cdist(mu, mu)
            dists.fill_diagonal_(float("inf"))
            nn_dist = dists.min(dim=1).values               # (K,)
        log_s = torch.log(nn_dist.unsqueeze(1).expand(-1, 2).clamp(1e-3, 0.4))

        # ── 4. Couleur du pixel ───────────────────────────────────────────
        colors = img[py, px, :self.n_channels].clamp(0.01, 0.99)  # (K, C)
        logit_color = torch.logit(colors)

        # ── 5. θ=0, α=0 (sigmoid(0)=0.5) ────────────────────────────────
        theta    = torch.zeros(K, 1)
        logit_alpha = torch.zeros(K, 1)

        # ── 6. Assemblage dans l'ordre du MLP ────────────────────────────
        # ordre : μx, μy, log_sx, log_sy, θ_raw, logit_α, c...
        bias_target = torch.cat([
            logit_mu,     # (K, 2)
            log_s,        # (K, 2)
            theta,        # (K, 1)
            logit_alpha,  # (K, 1)
            logit_color,  # (K, C)
        ], dim=1).flatten()   # (K * (6 + C),)

        # ── 7. Calcul de h(z_ref) via l'avant-dernière couche ────────────
        with torch.no_grad():
            z_in = z_ref.reshape(1, -1).to(next(self.parameters()).device)
            # Forward jusqu'à l'avant-dernière couche (tout sauf mlp[-1])
            h = z_in
            for layer in list(self.mlp.children())[:-1]:
                h = layer(h)                    # (1, hidden_dim)
            h = h.squeeze(0)                    # (hidden_dim,)

            # biais = target - W · h  → decoder(z_ref) = W·h + biais = target
            bias_target = bias_target.to(h.device)
            corrected_bias = bias_target - self.mlp[-1].weight @ h

            if init_bias:
                self.mlp[-1].bias.copy_(corrected_bias)

        print(f"smart_init : {K} gaussiennes initialisées depuis image ({H}×{W}), z_ref shape={z_ref.shape}")

    def _rasterize_one(self, z: torch.Tensor) -> torch.Tensor:
        """z : (1, D) → image : (1, C, H, W)  — forward sans serial"""
        return self.forward(z, serial=False)
