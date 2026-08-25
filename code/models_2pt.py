"""
Pipeline 2 — GaussianSplatDecoder2pt

Décodeur 2D(+t)GS : Gaussiennes ellipsoïdales dans l'espace (x, y, z₁, …, z_d)
où z est la variable latente (coordonnée d'état, pas une profondeur).

Le rendu d'une frame est la coupe de ces ellipsoïdes à z = z_query :
    - La contribution spatiale de chaque Gaussienne k est une Gaussienne 2D
      conditionnelle (Schur complement), modulée par son amplitude en z.
    - Pas de MLP : tous les paramètres sont des nn.Parameter directs.

Différences vs GaussianDecoder :
    - forward(z_query) avec z_query : (B, d) au lieu de z : (B, D)
    - Pas de MLP z→params
    - Covariance complète (2+d)×(2+d) via facteur de Cholesky inférieur
    - smart_init prend en plus z_all (N, d) pour initialiser l'axe latent
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _tril_indices(n: int):
    """Retourne (rows, cols) des indices du triangle inférieur d'une matrice n×n."""
    rows, cols = torch.tril_indices(n, n)
    return rows, cols


class GaussianSplatDecoder2pt(nn.Module):
    """
    Décodeur 2D(+t)GS à paramètres directs.

    Chaque Gaussienne k est un ellipsoïde dans ℝ^(2+d) :
        μ_k   ∈ ℝ^(2+d)               — centre (x, y, z₁…z_d)
        L_k   ∈ ℝ^((2+d)×(2+d))       — facteur de Cholesky inférieur de Σ_k
        α_k   ∈ (0,1)                  — opacité globale
        c_k   ∈ [0,1]^C               — couleur

    Rendu à z_query ∈ ℝ^d :
        1. Extraire les blocs de Σ_k = L_k L_kᵀ :
               Σ_xx (2×2), Σ_xz (2×d), Σ_zz (d×d)
        2. Gaussienne 2D conditionnelle (Schur complement) :
               μ_xy|z  = μ_xy + Σ_xz Σ_zz⁻¹ (z_query - μ_z)
               Σ_xy|z  = Σ_xx - Σ_xz Σ_zz⁻¹ Σ_zx
        3. Amplitude latente :
               w_z = exp(-½ (z_query - μ_z)ᵀ Σ_zz⁻¹ (z_query - μ_z))
        4. Évaluation sur grille pixels, alpha compositing normalisé.
    """

    def __init__(
        self,
        latent_dim: int,       # d — dimension de la variable latente z
        n_gaussians: int,      # K
        img_size: tuple,       # (H, W)
        n_channels: int = 1,   # C
    ):
        super().__init__()
        self.latent_dim  = latent_dim   # d
        self.n_gaussians = n_gaussians  # K
        self.img_size    = img_size     # (H, W)
        self.n_channels  = n_channels   # C
        self.full_dim    = 2 + latent_dim  # 2+d

        n = self.full_dim
        self.n_tril = n * (n + 1) // 2  # nb de params Cholesky par Gaussienne

        # ── Paramètres directs ────────────────────────────────────────────
        # mu     : centres dans ℝ^(2+d), xy dans [0,1] via sigmoid au forward
        self.mu_raw    = nn.Parameter(torch.zeros(n_gaussians, n))

        # L_raw  : triangle inférieur de Cholesky
        #   - éléments diagonaux stockés comme log(diag) → softplus pour SPD
        #   - éléments hors-diagonale libres
        self.L_raw     = nn.Parameter(torch.zeros(n_gaussians, self.n_tril))

        # log_alpha : logit de l'opacité α ∈ (0,1)
        self.log_alpha = nn.Parameter(torch.zeros(n_gaussians))

        # color     : logit de la couleur c ∈ [0,1]^C
        self.color_raw = nn.Parameter(torch.zeros(n_gaussians, n_channels))

        # ── Grille de pixels fixe (H, W, 2), coordonnées dans [0,1] ─────
        H, W = img_size
        ys = torch.linspace(0, 1, H)
        xs = torch.linspace(0, 1, W)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)
        grid = torch.stack([grid_x, grid_y], dim=-1)             # (H, W, 2)
        self.register_buffer('grid', grid)

        # Indices du triangle inférieur (réutilisés dans _build_L)
        rows, cols = _tril_indices(n)
        self.register_buffer('tril_rows', rows)
        self.register_buffer('tril_cols', cols)

        # Indices diagonaux dans le vecteur tril (pour appliquer softplus)
        diag_mask = (rows == cols)
        self.register_buffer('diag_mask', diag_mask)

    # ── Utilitaires internes ──────────────────────────────────────────────

    def _build_L(self) -> torch.Tensor:
        """
        Construit le facteur de Cholesky inférieur L : (K, n, n).
        Diagonale = softplus(L_raw[diag]) pour garantir SPD.
        """
        K = self.n_gaussians
        n = self.full_dim
        L = torch.zeros(K, n, n, device=self.L_raw.device, dtype=self.L_raw.dtype)

        raw = self.L_raw.clone()
        # Diagonale → softplus (strictement positif)
        raw[:, self.diag_mask] = F.softplus(self.L_raw[:, self.diag_mask]) + 1e-4
        L[:, self.tril_rows, self.tril_cols] = raw
        return L   # (K, n, n)

    def _build_Sigma(self) -> torch.Tensor:
        """Σ = L Lᵀ : (K, n, n)."""
        L = self._build_L()
        return L @ L.transpose(-1, -2)   # (K, n, n)

    def _decode_params(self):
        """
        Décode les paramètres bruts en grandeurs physiques.
        Retourne :
            mu_xy  : (K, 2)    centres spatiaux dans [0,1]
            mu_z   : (K, d)    centres latents
            Sigma  : (K, n, n) covariances complètes
            alpha  : (K,)      opacités
            color  : (K, C)    couleurs
        """
        # μ_xy via sigmoid étendu [-0.1, 1.1] comme GaussianDecoder
        mu_xy  = torch.sigmoid(self.mu_raw[:, :2]) * 1.2 - 0.1  # (K, 2)
        mu_z   = self.mu_raw[:, 2:]                               # (K, d)  libre
        Sigma  = self._build_Sigma()                              # (K, n, n)
        alpha  = torch.sigmoid(self.log_alpha)                    # (K,)
        color  = torch.sigmoid(self.color_raw)                    # (K, C)
        return mu_xy, mu_z, Sigma, alpha, color

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, z_query: torch.Tensor) -> torch.Tensor:
        """
        z_query : (B, d)
        Retourne : (B, C, H, W) dans [0, 1]
        """
        B     = z_query.shape[0]
        H, W  = self.img_size
        K     = self.n_gaussians
        d     = self.latent_dim
        C     = self.n_channels

        mu_xy, mu_z, Sigma, alpha, color = self._decode_params()

        # ── Blocs de covariance ───────────────────────────────────────────
        Sigma_xx = Sigma[:, :2, :2]    # (K, 2, 2)
        Sigma_xz = Sigma[:, :2, 2:]    # (K, 2, d)
        Sigma_zz = Sigma[:, 2:, 2:]    # (K, d, d)

        # ── Schur complement et amplitude latente ─────────────────────────
        # dz : (B, K, d)
        dz = z_query.unsqueeze(1) - mu_z.unsqueeze(0)   # (B, K, d)

        # Σ_zz⁻¹ : (K, d, d)
        # Régularisation de Sigma_zz avant inversion : évite σ_z → 0 → gradient explosif
        eye_d = torch.eye(d, device=Sigma_zz.device, dtype=Sigma_zz.dtype).unsqueeze(0)  # (1, d, d)
        Sigma_zz_inv = torch.linalg.inv(Sigma_zz + 1e-4 * eye_d)  # (K, d, d)

        # Σ_xz Σ_zz⁻¹ : (K, 2, d)
        SxzSzzinv = Sigma_xz @ Sigma_zz_inv             # (K, 2, d)

        # Covariance conditionnelle Σ_xy|z = Σ_xx - Σ_xz Σ_zz⁻¹ Σ_zx : (K, 2, 2)
        Sigma_cond = Sigma_xx - SxzSzzinv @ Sigma_xz.transpose(-1, -2)  # (K, 2, 2)
        Sigma_cond_inv = torch.linalg.inv(Sigma_cond)                    # (K, 2, 2)

        # Amplitude latente w_z : (B, K)
        # w_z = exp(-½ dz Σ_zz⁻¹ dz)
        # dz : (B, K, d, 1)
        dz4  = dz.unsqueeze(-1)                          # (B, K, d, 1)
        # Σ_zz⁻¹ : (1, K, d, d)
        Szzi = Sigma_zz_inv.unsqueeze(0)                 # (1, K, d, d)
        quad_z = (dz4.transpose(-1, -2) @ Szzi @ dz4).squeeze(-1).squeeze(-1)  # (B, K)
        w_z = torch.exp(-0.5 * quad_z)                   # (B, K)

        # ── Centre conditionnel μ_xy|z : (B, K, 2) ───────────────────────
        # SxzSzzinv : (K, 2, d), dz : (B, K, d, 1)
        SxzSzzinv_b = SxzSzzinv.unsqueeze(0)             # (1, K, 2, d)
        shift = (SxzSzzinv_b @ dz4).squeeze(-1)          # (B, K, 2)
        mu_cond = mu_xy.unsqueeze(0) + shift              # (B, K, 2)

        # ── Évaluation sur grille ─────────────────────────────────────────
        # Forme quadratique A*dx² + 2B*dx*dy + C*dy² avec (A,B,C) = éléments de Σ_cond_inv.
        # Évite le tenseur 6D (B, K, H, W, 2, 1) qui explose en VRAM avec K large.
        grid_  = self.grid.unsqueeze(0).unsqueeze(0)      # (1, 1, H, W, 2)
        mu_    = mu_cond.unsqueeze(2).unsqueeze(3)        # (B, K, 1, 1, 2)
        dp     = grid_ - mu_                              # (B, K, H, W, 2)
        dx     = dp[..., 0]                               # (B, K, H, W)
        dy     = dp[..., 1]                               # (B, K, H, W)

        # Éléments de Σ_cond_inv (symétrique) : (K,) → (1, K, 1, 1)
        sA = Sigma_cond_inv[:, 0, 0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1,K,1,1)
        sB = Sigma_cond_inv[:, 0, 1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1,K,1,1)
        sC = Sigma_cond_inv[:, 1, 1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1,K,1,1)
        quad_xy = sA * dx.pow(2) + 2 * sB * dx * dy + sC * dy.pow(2)          # (B, K, H, W)

        G = torch.exp(-0.5 * quad_xy)                     # (B, K, H, W)  ∈ (0,1]

        # Pondération par amplitude latente
        w_z_ = w_z.unsqueeze(-1).unsqueeze(-1)            # (B, K, 1, 1)
        G    = G * w_z_                                   # (B, K, H, W)

        # ── Alpha compositing normalisé ───────────────────────────────────
        # G      : (B, K, H, W)
        # alpha  : (K,)      → (1, K, 1, 1)
        # color  : (K, C)    → (1, K, C, 1, 1)
        alpha_ = alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)    # (1, K, 1, 1)
        color_ = color.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)    # (1, K, C, 1, 1)
        aG     = alpha_ * G                                        # (B, K, H, W)

        weights = aG.sum(dim=1, keepdim=True).clamp(min=1.0)      # (B, 1, H, W)
        # color_ : (1,K,C,1,1)  aG : (B,K,H,W) → unsqueeze pour broadcast
        image   = (color_ * aG.unsqueeze(2)).sum(dim=1) / weights  # (B, C, H, W)
        image   = image.clamp(0, 1)

        return image

    # ── Initialisation intelligente ───────────────────────────────────────

    @torch.no_grad()
    def smart_init(
        self,
        ref_image: torch.Tensor,   # (C, H, W) float32 dans [0,1]
        z_all: torch.Tensor,       # (N, d) — tous les z encodés du dataset
        grad_weight: float = 0.8,
        sigma_xy_scale: float = 0.05,  # σ spatial initial (en coordonnées normalisées)
        sigma_z_scale: float = 1.0,    # multiplicateur de l'écart-type latent
        sigma_z_floor: float = 0.5,    # plancher de σ_z du gate conditionnel (robustesse)
    ):
        """
        Initialise les paramètres directs depuis une image de référence et
        la distribution empirique des z.

        - mu_xy  : échantillonnés depuis la carte de gradient de ref_image
        - mu_z   : échantillonnés aléatoirement depuis N(μ_z_emp, σ_z_emp)
        - L      : diagonale initialisée pour σ_xy ≈ sigma_xy_scale,
                   σ_z ≈ sigma_z_scale * std(z_all)
        - alpha  : 0.5 (logit=0)
        - color  : couleur du pixel le plus proche dans ref_image
        """
        K   = self.n_gaussians
        d   = self.latent_dim
        n   = self.full_dim
        H_r, W_r = ref_image.shape[-2], ref_image.shape[-1]

        # ── Image → (H, W, C) ─────────────────────────────────────────────
        if ref_image.ndim == 3 and ref_image.shape[0] in (1, 3):
            img = ref_image.permute(1, 2, 0)   # (H, W, C)
        else:
            img = ref_image
        H_i, W_i, _ = img.shape

        # ── 1. Carte de gradient → positions μ_xy ─────────────────────────
        gray    = img.mean(dim=-1)                                 # (H, W)
        gx      = F.pad(gray[:, 1:] - gray[:, :-1], (0, 1))
        gy      = F.pad(gray[1:, :] - gray[:-1, :], (0, 0, 0, 1))
        grad_mag = (gx**2 + gy**2).sqrt()

        probs = grad_weight * grad_mag.flatten() + (1 - grad_weight)
        probs = (probs / probs.sum()).cpu().numpy()

        # Tirage SANS remise tant qu'il y a plus de pixels que de gaussiennes ; avec
        # remise sinon (petites images : 32² = 1024 pixels < K). L'ancien
        # `min(K, H·W)` produisait alors moins de positions que de gaussiennes et
        # cassait l'assemblage de mu_raw (K lignes attendues).
        flat_idx = np.random.choice(H_i * W_i, size=K, replace=K > H_i * W_i, p=probs)
        py = flat_idx // W_i
        px = flat_idx  % W_i

        mu_x = px / (W_i - 1)   # [0, 1]
        mu_y = py / (H_i - 1)   # [0, 1]
        # Inverse de sigmoid(x)*1.2-0.1=mu
        mu_xy_np = np.stack([mu_x, mu_y], axis=1)                 # (K, 2)
        logit_mu_xy = np.log(
            ((mu_xy_np + 0.1) / 1.2).clip(0.01, 0.99) /
            (1 - ((mu_xy_np + 0.1) / 1.2).clip(0.01, 0.99))
        )  # (K, 2)

        # ── 2. Distribution empirique des z → positions μ_z ───────────────
        z_np   = z_all.cpu().numpy()                               # (N, d)
        z_mean = z_np.mean(axis=0)                                 # (d,)
        z_std  = z_np.std(axis=0).clip(1e-3)                      # (d,)
        # Tirage aléatoire dans la distribution des z
        mu_z_np = z_mean + z_std * np.random.randn(K, d)          # (K, d)

        # ── 3. Assemblage de mu_raw ───────────────────────────────────────
        mu_raw_np = np.concatenate([logit_mu_xy, mu_z_np], axis=1)  # (K, 2+d)
        self.mu_raw.copy_(torch.tensor(mu_raw_np, dtype=torch.float32))

        # ── 4. Initialisation de L_raw ────────────────────────────────────
        # On veut Σ ≈ diag(σ_x², σ_y², σ_z1², …)
        # L = diag(σ_x, σ_y, σ_z1, …) → L_raw[diag] = softplus_inv(σ)
        # σ_z UNIFORME et LARGE sur TOUTES les dims latentes : la plus grande dispersion
        # empirique appliquée à chaque dim (au lieu du z_std par-dim qui effondrait σ_z
        # dans les dims à faible variance → gaussiennes larges partout au départ).
        #
        # PLANCHER `sigma_z_floor` : σ_z fixe la largeur de la porte conditionnelle
        # w_z = exp(−½(z−μ_z)ᵀΣzz⁻¹(z−μ_z)) qui pondère l'opacité (et donc le gradient)
        # de chaque gaussienne. Si l'encodeur est encore NON entraîné au moment du
        # smart_init (cas train_ae from scratch), z_all est quasi constant → z_std ~1e-3 →
        # gate de largeur ~1e-3. Dès que la loss de reconstruction commence à écarter les
        # z (de O(1)), w_z→0 pour toutes les gaussiennes → rendu noir ET gradient nul →
        # l'AE gèle avant que la reconstruction n'ait pu déployer le latent. On plancher
        # donc σ_z pour que la porte reste ouverte tant que le latent se structure. Le
        # gate reste apprenable et se resserre ensuite. Neutre pour un encodeur sain
        # (z_std ≳ floor ⟹ inchangé) ; ne mord que sur un latent dégénéré à l'init.
        sigma_z_init = np.full(d, sigma_z_scale * max(float(z_std.max()), float(sigma_z_floor)))
        sigma_target = np.concatenate([
            np.full(2, sigma_xy_scale),
            sigma_z_init,                   # (d,) — large et uniforme
        ])   # (2+d,)

        def softplus_inv(x):
            # softplus_inv(y) = log(exp(y) - 1)
            return np.log(np.exp(np.clip(x, 1e-4, 30)) - 1 + 1e-6)

        # Vecteur tril initialisé à 0, diagonale à softplus_inv(sigma_target)
        L_raw_np = np.zeros((K, self.n_tril), dtype=np.float32)
        diag_mask_np = self.diag_mask.cpu().numpy()
        for i in range(n):
            # trouver l'index dans le vecteur tril de l'élément (i,i)
            tril_rows_np = self.tril_rows.cpu().numpy()
            tril_cols_np = self.tril_cols.cpu().numpy()
            idx = np.where((tril_rows_np == i) & (tril_cols_np == i))[0][0]
            L_raw_np[:, idx] = softplus_inv(sigma_target[i])

        self.L_raw.copy_(torch.tensor(L_raw_np, dtype=torch.float32))

        # ── 5. Couleur du pixel ───────────────────────────────────────────
        colors = img[py, px, :self.n_channels].clamp(0.01, 0.99)   # (K, C)
        logit_color = torch.log(colors / (1 - colors))
        self.color_raw.copy_(logit_color)

        # ── 6. Alpha : 0.5 (logit=0, déjà initialisé) ────────────────────
        # log_alpha déjà à 0 par défaut

        print(
            f"smart_init 2pt : {K} gaussiennes | "
            f"z_mean={z_mean.round(3)}, z_std={z_std.round(3)} | "
            f"s_xy={sigma_xy_scale:.3f}, s_z={sigma_z_init.round(3)} (uniforme, large)"
        )

    # ── Diagnostic ────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_params_summary(self) -> dict:
        """Retourne un dict de statistiques sur les paramètres courants."""
        mu_xy, mu_z, Sigma, alpha, color = self._decode_params()
        return {
            'mu_xy_mean':  mu_xy.mean(0).cpu().numpy(),
            'mu_z_mean':   mu_z.mean(0).cpu().numpy(),
            'mu_z_std':    mu_z.std(0).cpu().numpy(),
            'alpha_mean':  alpha.mean().item(),
            'sigma_xy':    Sigma[:, :2, :2].diagonal(dim1=-2, dim2=-1).sqrt().mean(0).cpu().numpy(),
            'sigma_z':     Sigma[:, 2:, 2:].diagonal(dim1=-2, dim2=-1).sqrt().mean(0).cpu().numpy(),
        }


# ── Utilitaire pour GaussianSplatDecoder2pt_gsplat ────────────────────────────

def _cov2d_to_quat_scale_pancake(
    Sigma_cond: torch.Tensor,   # (K, 2, 2) covariances conditionnelles
    eps_z: float = 1e-5,
) -> tuple:
    """
    Convertit des covariances 2D en quaternion+scales pour Gaussiennes "pancake" 3D.

    Construit implicitement Sigma_3d = block_diag(Sigma_cond, eps_z) et retourne
    la décomposition q/s nécessaire à gsplat.rasterization :
        Sigma_3d = R @ diag(scales²) @ R^T    avec R = rotation depuis quaternion q.

    Retourne :
        quats  : (K, 4) quaternions wxyz
        scales : (K, 3) échelles (racine des valeurs propres)
    """
    K   = Sigma_cond.shape[0]
    dev = Sigma_cond.device
    dt  = Sigma_cond.dtype

    # Symétrise + plancher SPD du bloc 2D
    S  = (Sigma_cond + Sigma_cond.transpose(-1, -2)) * 0.5
    S  = S + 1e-5 * torch.eye(2, device=dev, dtype=dt).unsqueeze(0)

    # Décomposition propre 2×2 en forme close — évite toute itération LAPACK
    # (torch.linalg.eigh peut échouer sur matrices mal conditionnées en début d'entraînement)
    a, b_el, c = S[:, 0, 0], S[:, 0, 1], S[:, 1, 1]
    m   = (a + c) * 0.5
    p   = (((a - c) * 0.5) ** 2 + b_el ** 2).clamp(min=1e-10).sqrt()
    L2  = torch.stack([m - p, m + p], dim=1).clamp(min=1e-8)   # (K,2) croissant
    scales_xy = L2.sqrt()                                        # (K,2)

    # Vecteur propre de λ₁ (petite) : [-b, (a-c)/2 + p] normalisé
    v1x = -b_el
    v1y = (a - c) * 0.5 + p
    n1  = (v1x ** 2 + v1y ** 2).clamp(min=1e-10).sqrt()
    degen = n1 < 1e-8                              # matrice scalaire / b=0, a=c
    v1x = torch.where(degen, torch.ones_like(v1x),  v1x / n1)
    v1y = torch.where(degen, torch.zeros_like(v1y), v1y / n1)
    # V2 : colonnes = vecteurs propres, det = v1x²+v1y² = +1 par construction
    V2c = torch.stack([
        torch.stack([v1x, v1y], dim=1),     # colonne 0 : vecteur propre de λ₁
        torch.stack([-v1y, v1x], dim=1),    # colonne 1 : vecteur propre de λ₂ (⊥)
    ], dim=2)   # (K, 2, 2)

    # Rotation 3D : bloc XY = V2c, axe Z invariant
    R = torch.zeros(K, 3, 3, device=dev, dtype=dt)
    R[:, :2, :2] = V2c
    R[:, 2, 2]   = 1.0

    # Quaternion wxyz — méthode de Shepperd
    R00=R[:,0,0]; R11=R[:,1,1]; R22=R[:,2,2]
    R21=R[:,2,1]; R12=R[:,1,2]
    R02=R[:,0,2]; R20=R[:,2,0]
    R10=R[:,1,0]; R01=R[:,0,1]
    tr = R00 + R11 + R22

    s0=( tr+1).clamp(1e-10).sqrt()*2; s0c=s0.clamp(1e-8)
    q0=torch.stack([0.25*s0c,(R21-R12)/s0c,(R02-R20)/s0c,(R10-R01)/s0c],1)
    s1=(1+R00-R11-R22).clamp(1e-10).sqrt()*2; s1c=s1.clamp(1e-8)
    q1=torch.stack([(R21-R12)/s1c,0.25*s1c,(R01+R10)/s1c,(R02+R20)/s1c],1)
    s2=(1+R11-R00-R22).clamp(1e-10).sqrt()*2; s2c=s2.clamp(1e-8)
    q2=torch.stack([(R02-R20)/s2c,(R01+R10)/s2c,0.25*s2c,(R12+R21)/s2c],1)
    s3=(1+R22-R00-R11).clamp(1e-10).sqrt()*2; s3c=s3.clamp(1e-8)
    q3=torch.stack([(R10-R01)/s3c,(R02+R20)/s3c,(R12+R21)/s3c,0.25*s3c],1)

    c0=tr>0; c1=(~c0)&(R00>R11)&(R00>R22); c2=(~c0)&(~c1)&(R11>R22)
    q = q3.clone()
    q = torch.where(c2.unsqueeze(1), q2, q)
    q = torch.where(c1.unsqueeze(1), q1, q)
    q = torch.where(c0.unsqueeze(1), q0, q)
    q = F.normalize(q, dim=1)   # (K, 4) wxyz

    scales_z = torch.full((K, 1), eps_z ** 0.5, device=dev, dtype=dt)
    scales   = torch.cat([scales_xy, scales_z], dim=1)  # (K, 3)
    return q, scales


class GaussianSplatDecoder2pt_gsplat(GaussianSplatDecoder2pt):
    """
    Variante de GaussianSplatDecoder2pt qui utilise gsplat.rasterization pour le rendu.

    Architecture identique au parent (même Schur complement, mêmes paramètres,
    même state_dict) ; seul le rendu change :
      - Parent  : évaluation manuelle sur grille H×W — O(K·H·W) en mémoire, lent
      - Ici     : rasterisation CUDA tuilée via gsplat — ~100× plus rapide

    Différence de compositing : le parent utilise un compositing normalisé
    (somme / max), ici gsplat fait un alpha compositing séquentiel front-to-back.
    Le modèle s'adapte pendant l'entraînement ; les poids ne sont pas interchangeables
    entre les deux compositings pour une même scène.

    Astuce géométrique :
      Gaussiennes 2D conditionnelles (mu_cond, Sigma_cond) → Gaussiennes 3D "pancake"
      à Z=1 avec une caméra fictive (fx=W, fy=H, cx=W/2, cy=H/2).
      La projection pinhole mappe exactement [0,1]² → grille H×W pixels :
          u = W * mu_x,  v = H * mu_y
      La covariance projetée ≈ diag(W,H) @ Sigma_cond @ diag(W,H).
    """

    def forward(self, z_query: torch.Tensor) -> torch.Tensor:
        """
        z_query : (B, d)
        Retourne : (B, C, H, W) dans [0, 1]
        """
        try:
            from gsplat import rasterization
        except ImportError:
            raise ImportError(
                'gsplat requis. Installez depuis les wheels précompilés ou '
                'pip install gsplat.'
            )

        B   = z_query.shape[0]
        H, W = self.img_size
        K   = self.n_gaussians
        d   = self.latent_dim
        C   = self.n_channels
        dev = z_query.device

        mu_xy, mu_z, Sigma, alpha, color = self._decode_params()

        # ── Schur complement (identique au parent) ────────────────────────
        Sigma_xx = Sigma[:, :2, :2]
        Sigma_xz = Sigma[:, :2, 2:]
        Sigma_zz = Sigma[:, 2:, 2:]

        eye_d        = torch.eye(d, device=dev, dtype=Sigma_zz.dtype).unsqueeze(0)
        Sigma_zz_inv = torch.linalg.inv(Sigma_zz + 1e-4 * eye_d)
        SxzSzzinv    = Sigma_xz @ Sigma_zz_inv
        Sigma_cond   = Sigma_xx - SxzSzzinv @ Sigma_xz.transpose(-1, -2)  # (K, 2, 2)

        dz  = z_query.unsqueeze(1) - mu_z.unsqueeze(0)   # (B, K, d)
        dz4 = dz.unsqueeze(-1)                            # (B, K, d, 1)

        quad_z = (dz4.transpose(-1, -2) @ Sigma_zz_inv.unsqueeze(0) @ dz4
                  ).squeeze(-1).squeeze(-1).clamp(max=20.0)   # (B, K)
        w_z    = torch.exp(-0.5 * quad_z)                      # (B, K)

        shift    = (SxzSzzinv.unsqueeze(0) @ dz4).squeeze(-1)  # (B, K, 2)
        mu_cond  = mu_xy.unsqueeze(0) + shift                   # (B, K, 2)

        # ── Covariance 3D "pancake" : XY + dim Z plate ────────────────────
        # Sigma_3d = block_diag(Sigma_cond, eps_z) → décomposé en (quats, scales)
        quats, scales = _cov2d_to_quat_scale_pancake(Sigma_cond)  # (K,4), (K,3)

        # ── Caméra fictive : [0,1]² ↔ H×W pixels ──────────────────────────
        # point (mu_x, mu_y) ∈ [0,1]² → 3D (mu_x-0.5, mu_y-0.5, 1.0)
        # projection : u = W*(X/Z) + W/2 = W*(mu_x-0.5)+W/2 = W*mu_x  ✓
        K_cam = torch.tensor(
            [[float(W), 0.0, W / 2.0],
             [0.0,      float(H), H / 2.0],
             [0.0,      0.0,      1.0]],
            device=dev, dtype=z_query.dtype,
        )
        viewmat = torch.eye(4, device=dev, dtype=z_query.dtype)

        rgb_list = []
        for b in range(B):
            # Positions 3D : X, Y ∈ [-0.5, 0.5],  Z = 1
            xyz_b = torch.stack([
                mu_cond[b, :, 0] - 0.5,
                mu_cond[b, :, 1] - 0.5,
                torch.ones(K, device=dev, dtype=z_query.dtype),
            ], dim=1)  # (K, 3)

            opac_b     = (alpha * w_z[b]).clamp(0.0, 0.99)
            background = torch.zeros(C, device=dev, dtype=z_query.dtype)

            renders, _, _ = rasterization(
                means       = xyz_b,
                quats       = quats,
                scales      = scales,
                opacities   = opac_b,
                colors      = color,
                viewmats    = viewmat.unsqueeze(0),
                Ks          = K_cam.unsqueeze(0),
                width       = W,
                height      = H,
                near_plane  = 0.5,
                far_plane   = 2.0,
                sh_degree   = None,
                render_mode = 'RGB',
                backgrounds = background,
            )
            # renders : (1, H, W, C)
            rgb_list.append(renders[0, :, :, :C].permute(2, 0, 1).clamp(0, 1))

        return torch.stack(rgb_list, dim=0)   # (B, C, H, W)


# ── Fabrique de décodeur : gsplat, avec repli torch pur ──────────────────────

_GSPLAT_STATE = {'available': None, 'warned': False}


def gsplat_available() -> bool:
    """
    `True` si `gsplat.rasterization` est importable ET qu'un GPU CUDA est
    disponible : la rasterisation gsplat est un noyau CUDA, un gsplat installé
    sur une machine sans GPU ne rend rien. Résultat mis en cache.
    """
    if _GSPLAT_STATE['available'] is None:
        try:
            from gsplat import rasterization  # noqa: F401
            _GSPLAT_STATE['available'] = torch.cuda.is_available()
        except Exception:
            _GSPLAT_STATE['available'] = False
    return _GSPLAT_STATE['available']


_FALLBACK_WARNING = """
================================================================================
  ATTENTION : gsplat est INDISPONIBLE, repli sur le decodeur torch pur
              (GaussianSplatDecoder2pt). Chemin DEPRECIE.
================================================================================
  1. LENTEUR. Le rendu est evalue sur toute la grille H*W pour chacune des K
     gaussiennes, en O(K*H*W) memoire, sans tuilage. Environ 100 fois plus lent
     que la rasterisation CUDA. Un entrainement d'autoencodeur qui prend des
     heures avec gsplat prend des jours ici.

  2. LES POIDS NE SONT PAS INTERCHANGEABLES. Le compositing differe : somme
     normalisee ici, alpha compositing sequentiel front-to-back dans gsplat.
     Charger un checkpoint entraine avec gsplat dans ce decodeur (ou l'inverse)
     donne des images FAUSSES, sans erreur ni exception. Les chiffres publies
     supposent gsplat.

  Ce repli sert a faire tourner la chaine de bout en bout sans CUDA (lecture du
  code, petits cas de demonstration, integration continue), pas a reproduire les
  resultats. Pour reproduire : installer gsplat.

  Forcer explicitement un backend : config.DEC2PT_BACKEND = 'gsplat' | 'torch'
  ('gsplat' leve une erreur au lieu de replier silencieusement).
================================================================================
"""


def decoder2pt_class(backend: str = 'auto'):
    """
    Retourne la classe de décodeur 2D(+t) à utiliser.

    backend :
      - 'auto'   : gsplat s'il est installé, sinon repli torch pur avec un
                   avertissement de dépréciation appuyé (émis une seule fois) ;
      - 'gsplat' : impose gsplat, lève ImportError s'il manque (à utiliser quand
                   un résultat publié est en jeu, pour ne pas replier en silence) ;
      - 'torch'  : impose le décodeur maison, sans avertissement.
    """
    if backend not in ('auto', 'gsplat', 'torch'):
        raise ValueError(f"DEC2PT_BACKEND inconnu : {backend!r} "
                         f"(attendu 'auto', 'gsplat' ou 'torch')")

    if backend == 'torch':
        return GaussianSplatDecoder2pt

    if backend == 'gsplat':
        if not gsplat_available():
            raise ImportError(
                'DEC2PT_BACKEND="gsplat" mais gsplat est introuvable. '
                'Installer gsplat, ou passer a "auto" pour accepter le repli '
                'torch pur (lent, et poids non interchangeables).')
        return GaussianSplatDecoder2pt_gsplat

    if gsplat_available():
        return GaussianSplatDecoder2pt_gsplat

    if not _GSPLAT_STATE['warned']:
        import warnings
        print(_FALLBACK_WARNING, flush=True)
        warnings.warn('gsplat indisponible : repli sur GaussianSplatDecoder2pt '
                      '(lent, poids non interchangeables avec gsplat).',
                      RuntimeWarning, stacklevel=2)
        _GSPLAT_STATE['warned'] = True
    return GaussianSplatDecoder2pt


def build_decoder2pt(*args, backend: str = None, **kwargs):
    """
    Construit le décodeur 2D(+t) du backend actif. Signature identique aux deux
    classes (même `__init__`, même `state_dict`) : c'est un remplacement direct
    de `GaussianSplatDecoder2pt_gsplat(...)`.

    `backend=None` lit `config.DEC2PT_BACKEND` (défaut 'auto').
    """
    if backend is None:
        try:
            import config
            backend = getattr(config, 'DEC2PT_BACKEND', 'auto')
        except Exception:
            backend = 'auto'
    return decoder2pt_class(backend)(*args, **kwargs)
