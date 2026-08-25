"""
Pipeline 2 — Fonctions de visualisation.
Toutes les fonctions reçoivent les données, affichent et retournent la figure.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (enregistre la projection 3D)
import torch



def _get_whitening_stats(encoder, z_all: 'torch.Tensor'):
    """
    Retourne (z_mean, W_white) pour le whitening.

    Si encoder.normalize=True : lit running_mean/running_W (stats du dernier training step).
    Sinon : calcule depuis z_all.
    """
    if getattr(encoder, 'normalize', False) and encoder.whitening.initialized.item():
        z_mean  = encoder.whitening.running_mean   # (D,)
        W_white = encoder.whitening.running_W      # (D, D)
    else:
        z_mean = z_all.mean(dim=0)
        z_c    = z_all - z_mean
        N      = z_all.shape[0]
        Sigma  = (z_c.T @ z_c) / max(N - 1, 1)
        Sigma  = Sigma + 1e-4 * torch.eye(Sigma.shape[0], device=Sigma.device)
        ev, evec = torch.linalg.eigh(Sigma)
        W_white  = evec * ev.clamp(min=1e-4).sqrt().reciprocal().unsqueeze(0)
    return z_mean, W_white


# ── TSNE ──────────────────────────────────────────────────────────────────

def plot_tsne(z_tsne: np.ndarray, indices: np.ndarray, cmap='viridis') -> plt.Figure:
    """
    Visualise z_tsne colorié par l'index temporel global.

    z_tsne  : (N, D) avec D = 1, 2 ou 3
    indices : (N,) index temporel global
    """
    D = z_tsne.shape[1]

    if D == 1:
        fig, ax = plt.subplots(figsize=(12, 4))
        sc = ax.scatter(indices, z_tsne[:, 0], c=indices, cmap=cmap, s=5, alpha=0.7)
        ax.set_xlabel('Index temporel global')
        ax.set_ylabel('z_tsne[0]')
        ax.set_title('TSNE 1D : z vs temps')
        plt.colorbar(sc, ax=ax, label='index')

    elif D == 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=indices, cmap=cmap, s=5, alpha=0.7)
        ax.set_xlabel('z_tsne[0]')
        ax.set_ylabel('z_tsne[1]')
        ax.set_title('TSNE 2D')
        plt.colorbar(sc, ax=ax, label='index temporel')

    elif D == 3:
        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection='3d')
        sc  = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], z_tsne[:, 2],
                         c=indices, cmap=cmap, s=5, alpha=0.7)
        ax.set_xlabel('z[0]'); ax.set_ylabel('z[1]'); ax.set_zlabel('z[2]')
        ax.set_title('TSNE 3D')
        plt.colorbar(sc, ax=ax, label='index temporel')

    else:
        raise ValueError(f'TSNE dim {D} non supporté (1, 2 ou 3)')

    plt.tight_layout()
    return fig


# ── Encodeur ──────────────────────────────────────────────────────────────


def plot_tsne_vs_time(z_tsne: np.ndarray, video_lengths: list,
                      video_idx: int = 0, cmap='viridis') -> plt.Figure:
    """
    Trace les coordonnées TSNE en fonction du temps pour une vidéo donnée.

    z_tsne        : (N, D) avec D = 2 ou 3
    video_lengths : liste des longueurs de chaque vidéo
    video_idx     : index de la vidéo à visualiser
    """
    # ── Extraire les frames de la vidéo choisie ───────────────────────────
    offset = sum(video_lengths[:video_idx])
    length = video_lengths[video_idx]
    z      = z_tsne[offset:offset + length]   # (T, D)
    t      = np.arange(length)
    D      = z.shape[1]

    if D == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle(f'TSNE vs temps — vidéo {video_idx}  (T={length})')

        for i, ax in enumerate(axes):
            ax.plot(t, z[:, i], lw=0.8, color=f'C{i}')
            ax.scatter(t, z[:, i], c=t, cmap=cmap, s=8, zorder=3)
            ax.set_xlabel('Frame')
            ax.set_ylabel(f'z[{i}]')
            ax.set_title(f'Coordonnée {i}')
            ax.grid(True, alpha=0.3)

    elif D == 3:
        fig = plt.figure(figsize=(16, 4))
        fig.suptitle(f'TSNE vs temps — vidéo {video_idx}  (T={length})')

        for i in range(3):
            ax = fig.add_subplot(1, 3, i + 1)
            ax.plot(t, z[:, i], lw=0.8, color=f'C{i}')
            ax.scatter(t, z[:, i], c=t, cmap=cmap, s=8, zorder=3)
            ax.set_xlabel('Frame')
            ax.set_ylabel(f'z[{i}]')
            ax.set_title(f'Coordonnée {i}')
            ax.grid(True, alpha=0.3)

    else:
        raise ValueError(f'D={D} non supporté (attendu 2 ou 3)')

    plt.tight_layout()
    return fig

def plot_tsne_per_video(z: np.ndarray, video_lengths, video_names=None,
                        dt=None, title='z_tsne par vidéo') -> plt.Figure:
    """Grille : un sous-graphe par vidéo, chaque dimension latente vs temps.

    z             : (N, D) coordonnées latentes concaténées (ordre du dataset).
    video_lengths : longueurs (frames) par vidéo, MÊME ordre que la concaténation.
    video_names   : noms optionnels (sinon 'vidéo i').
    dt            : pas de temps (s) → axe en secondes ; None → index de frame.
    """
    z = np.asarray(z)
    D = z.shape[1]
    splits = np.cumsum([0] + list(video_lengths))
    n = len(video_lengths)
    cols = min(3, max(1, n))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)
    axes = axes.ravel()
    xlabel = 't (s)' if dt else 'index de frame'
    for i, (s, e) in enumerate(zip(splits[:-1], splits[1:])):
        zi = z[s:e]
        x = np.arange(len(zi)) * (dt if dt else 1)
        ax = axes[i]
        for d in range(D):
            ax.plot(x, zi[:, d], lw=0.8, label=f'z[{d}]')
        name = video_names[i] if video_names is not None else f'vidéo {i}'
        ax.set_title(f'{name}  ({len(zi)} f)', fontsize=9)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        if D <= 4:
            ax.legend(fontsize=7)
    for j in range(n, len(axes)):
        axes[j].axis('off')
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_encoder_check(z_tsne: np.ndarray, z_enc: np.ndarray,
                       indices: np.ndarray, cmap='viridis') -> plt.Figure:
    """Compare z_tsne et z_enc, dimension par dimension (général, tout D ≥ 1).

    Une ligne par dimension latente d :
      - colonne gauche : z_tsne[:, d] et z_enc[:, d] superposés vs index de frame
        (vérifie que l'encodeur suit la cible le long de la trajectoire) ;
      - colonne droite : nuage de corrélation z_tsne[:, d] vs z_enc[:, d]
        avec coefficient de corrélation et droite y = x (ajustement par dim).
    """
    D = z_tsne.shape[1]
    order = np.argsort(indices)   # trie par index de frame pour des courbes lisibles
    idx_s = indices[order]

    fig, axes = plt.subplots(D, 2, figsize=(13, 3.2 * D), squeeze=False)

    for d in range(D):
        zt = z_tsne[:, d]
        ze = z_enc[:, d]

        # ── Trajectoire vs index ────────────────────────────────────────────
        ax = axes[d, 0]
        ax.plot(idx_s, zt[order], lw=1.0, alpha=0.8, label='z_tsne')
        ax.plot(idx_s, ze[order], lw=1.0, alpha=0.8, label='z_enc')
        ax.set_ylabel(f'dim {d}')
        if d == 0:
            ax.set_title('z_tsne vs z_enc le long des frames')
            ax.legend(fontsize=8)
        if d == D - 1:
            ax.set_xlabel('index de frame')
        ax.grid(True, alpha=0.3)

        # ── Corrélation ────────────────────────────────────────────────────
        ax = axes[d, 1]
        ax.scatter(zt, ze, s=3, alpha=0.3, c=indices, cmap=cmap)
        r = np.corrcoef(zt, ze)[0, 1] if zt.std() > 0 and ze.std() > 0 else float('nan')
        lo = min(zt.min(), ze.min()); hi = max(zt.max(), ze.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.6)
        ax.set_xlabel('z_tsne'); ax.set_ylabel('z_enc')
        ax.set_title(f'dim {d} — r = {r:.3f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_encoder_latent_geometry(z_tsne: np.ndarray, z_enc: np.ndarray,
                                 indices: np.ndarray, cmap='viridis') -> plt.Figure:
    """Diagnostics de l'encodeur **spécifiques à d > 1** (géométrie latente).

    `plot_encoder_check` compare chaque dimension *séparément* : il ne dit rien sur
    la façon dont l'encodeur reproduit la **géométrie conjointe** de l'espace latent
    (couplages entre axes, forme du nuage, fuite d'une cible vers une autre dim).
    Cette figure couvre ce que la vue par-dimension manque, en trois panneaux :

      1. **Plongement 2D** (gauche) : nuages z_tsne et z_enc superposés, colorés par
         index de frame. Si d = 2 on trace directement (z[:,0], z[:,1]) ; si d > 2 on
         projette les deux jeux sur les 2 axes principaux **de z_tsne** (même base
         pour les deux → comparaison honnête). Vérifie que le nuage encodé recouvre
         la variété cible.
      2. **Matrice de corrélation croisée** (milieu) : |corr(z_tsne[:,i], z_enc[:,j])|
         (d × d). Une régression MSE bien apprise ⟹ diagonale ≈ 1, hors-diagonale ≈ 0.
         Des valeurs hors-diagonale fortes = fuite / rotation de base entre axes.
      3. **Corrélation par dimension** (droite) : r de Pearson par axe (barres), avec
         la moyenne. Résumé scalaire de la qualité d'alignement axe par axe.

    Pour d = 1, la matrice croisée et le plongement 2D dégénèrent : on se rabat sur un
    message — utilise `plot_encoder_check` à la place.
    """
    D = z_tsne.shape[1]

    if D < 2:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, 'Géométrie latente non définie pour d = 1\n'
                          '→ voir encoder_check.png',
                ha='center', va='center', fontsize=11)
        ax.axis('off')
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # ── 1. Plongement 2D ────────────────────────────────────────────────────
    ax = axes[0]
    if D == 2:
        P_t = z_tsne
        P_e = z_enc
        xlab, ylab = 'dim 0', 'dim 1'
    else:
        # PCA sur z_tsne → base commune (centrée sur la moyenne de z_tsne)
        mu = z_tsne.mean(0)
        _, _, Vt = np.linalg.svd(z_tsne - mu, full_matrices=False)
        basis = Vt[:2].T  # (D, 2)
        P_t = (z_tsne - mu) @ basis
        P_e = (z_enc - mu) @ basis
        xlab, ylab = 'PC1 (z_tsne)', 'PC2 (z_tsne)'
    # Sous-échantillonnage du nuage : au-delà de ~10k points, le rendu scatter
    # gonfle la RAM (path collections) sans rien apporter visuellement.
    N = P_t.shape[0]
    MAX_PTS = 10_000
    sub = (np.linspace(0, N - 1, MAX_PTS).astype(int) if N > MAX_PTS
           else np.arange(N))
    ax.scatter(P_t[sub, 0], P_t[sub, 1], s=6, alpha=0.35, c=indices[sub], cmap=cmap,
               marker='o', label='z_tsne')
    ax.scatter(P_e[sub, 0], P_e[sub, 1], s=6, alpha=0.35, c=indices[sub], cmap=cmap,
               marker='x', label='z_enc')
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title('Plongement latent 2D (○ z_tsne, × z_enc — couleur = frame)')
    ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.3); ax.set_aspect('equal', 'datalim')

    # ── 2. Matrice de corrélation croisée ───────────────────────────────────
    ax = axes[1]
    C = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            zi, zj = z_tsne[:, i], z_enc[:, j]
            C[i, j] = abs(np.corrcoef(zi, zj)[0, 1]) if zi.std() > 0 and zj.std() > 0 else 0.0
    im = ax.imshow(C, vmin=0, vmax=1, cmap='magma', aspect='equal')
    ax.set_xticks(range(D)); ax.set_yticks(range(D))
    ax.set_xlabel('z_enc dim'); ax.set_ylabel('z_tsne dim')
    ax.set_title('|corr| croisée (idéal : diagonale)')
    for i in range(D):
        for j in range(D):
            ax.text(j, i, f'{C[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color='w' if C[i, j] < 0.6 else 'k')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── 3. Corrélation par dimension ────────────────────────────────────────
    ax = axes[2]
    rs = np.array([
        np.corrcoef(z_tsne[:, d], z_enc[:, d])[0, 1]
        if z_tsne[:, d].std() > 0 and z_enc[:, d].std() > 0 else np.nan
        for d in range(D)
    ])
    ax.bar(range(D), rs, color='steelblue', alpha=0.8)
    ax.axhline(np.nanmean(rs), color='crimson', ls='--', lw=1,
               label=f'moyenne = {np.nanmean(rs):.3f}')
    ax.set_ylim(min(0, np.nanmin(rs)) - 0.05, 1.02)
    ax.set_xticks(range(D)); ax.set_xlabel('dimension latente')
    ax.set_ylabel('r de Pearson (z_tsne vs z_enc)')
    ax.set_title('Alignement par dimension')
    ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_encoder_loss(losses) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.semilogy(losses)
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE loss')
    ax.set_title("Encodeur : perte d'entraînement")
    ax.grid(True)
    plt.tight_layout()
    return fig


# ── LNN ───────────────────────────────────────────────────────────────────

def plot_lnn_training(lnn_losses, gamma_history=None, beta_history=None) -> plt.Figure:
    n_plots = 1 + bool(gamma_history) + bool(beta_history)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots + 1, 4))
    if n_plots == 1:
        axes = [axes]
    ax_idx = 0
    axes[ax_idx].semilogy(lnn_losses)
    axes[ax_idx].set_title('LNN : résidu Euler-Lagrange')
    axes[ax_idx].set_xlabel('Epoch'); axes[ax_idx].set_ylabel('Loss')
    axes[ax_idx].grid(True)
    ax_idx += 1
    if gamma_history:
        gamma_arr = np.array(gamma_history)   # (epochs, D) ou (epochs,) si scalaire
        if gamma_arr.ndim == 1:
            gamma_arr = gamma_arr[:, np.newaxis]
        for d in range(gamma_arr.shape[1]):
            axes[ax_idx].plot(gamma_arr[:, d], label=f'γ[{d}]')
        final = gamma_arr[-1]
        axes[ax_idx].set_title(f'Frottement visqueux γ (final = {np.round(final, 4)})')
        axes[ax_idx].set_xlabel('Epoch'); axes[ax_idx].set_ylabel('γ')
        axes[ax_idx].legend(fontsize=7); axes[ax_idx].grid(True)
        ax_idx += 1
    if beta_history:
        beta_arr = np.array(beta_history)   # (epochs, D) ou (epochs,) si scalaire
        if beta_arr.ndim == 1:
            beta_arr = beta_arr[:, np.newaxis]
        for d in range(beta_arr.shape[1]):
            axes[ax_idx].plot(beta_arr[:, d], label=f'β[{d}]')
        final = beta_arr[-1]
        axes[ax_idx].set_title(f'Frottement de Coulomb β (final = {np.round(final, 4)})')
        axes[ax_idx].set_xlabel('Epoch'); axes[ax_idx].set_ylabel('β')
        axes[ax_idx].legend(fontsize=7); axes[ax_idx].grid(True)
    plt.tight_layout()
    return fig


def plot_trajectories(z_enc: np.ndarray, video_lengths: list,
                      indices: np.ndarray, cmap='viridis') -> plt.Figure:
    """Trajectoires latentes par vidéo."""
    D = z_enc.shape[1]
    splits = np.cumsum([0] + video_lengths)

    if D == 1:
        fig, ax = plt.subplots(figsize=(13, 4))
        for i, (s, e) in enumerate(zip(splits[:-1], splits[1:])):
            ax.plot(np.arange(e - s), z_enc[s:e, 0], alpha=0.6, label=f'vidéo {i}')
        ax.set_xlabel('Frame'); ax.set_ylabel('z[0]')
        ax.set_title('Trajectoires latentes 1D')
        ax.legend(fontsize=7)

    elif D == 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        for i, (s, e) in enumerate(zip(splits[:-1], splits[1:])):
            ax.plot(z_enc[s:e, 0], z_enc[s:e, 1], alpha=0.5, linewidth=0.8)
            ax.scatter(z_enc[s, 0], z_enc[s, 1], marker='o', s=40, zorder=5)
        ax.set_xlabel('z[0]'); ax.set_ylabel('z[1]')
        ax.set_title('Trajectoires latentes 2D')

    else:
        # D >= 3 : grille de petits multiples z[d](frame), toutes vidéos
        # superposées. Lisible pour n'importe quelle dimension latente
        # (3D scatter ne passe pas l'échelle au-delà de d=3).
        ncols = min(D, 4)
        nrows = int(np.ceil(D / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                 squeeze=False)
        for d in range(D):
            ax = axes[d // ncols][d % ncols]
            for i, (s, e) in enumerate(zip(splits[:-1], splits[1:])):
                ax.plot(np.arange(e - s), z_enc[s:e, d], alpha=0.6, linewidth=0.8,
                        label=f'vidéo {i}' if d == 0 else None)
            ax.set_xlabel('Frame'); ax.set_ylabel(f'z[{d}]')
            ax.set_title(f'z[{d}]'); ax.grid(True, alpha=0.3)
        for k in range(D, nrows * ncols):              # masquer cases vides
            axes[k // ncols][k % ncols].axis('off')
        axes[0][0].legend(fontsize=7)
        fig.suptitle(f'Trajectoires latentes {D}D (par composante)')

    plt.tight_layout()
    return fig


def _energy_background(ax, lnn, z0_range, z1_range, device):
    """Trace le colormap d'énergie sur ax. Retourne (Z0, Z1)."""
    Z0, Z1 = np.meshgrid(z0_range, z1_range)
    grid = torch.tensor(
        np.stack([Z0.ravel(), Z1.ravel()], axis=1), dtype=torch.float32
    ).to(device)
    lnn.eval()
    with torch.no_grad():
        E_grid = lnn.energy(grid).cpu().numpy().reshape(len(z1_range), len(z0_range))
    cf = ax.contourf(Z0, Z1, E_grid, levels=50, cmap='RdBu_r')
    plt.colorbar(cf, ax=ax, label='E(z)')
    return Z0, Z1


def _pca_2d_basis(z_pts: np.ndarray, z_rest: np.ndarray) -> np.ndarray:
    """
    Deux axes principaux (PCA) des points z_pts centrés sur z_rest.

    Retourne PC de forme (D, 2) à colonnes orthonormées. Sert à projeter un
    espace latent de dimension D ≥ 3 sur un plan pour les cartes d'énergie et
    les plans de phase. Robuste au rang < 2 (complète par des colonnes nulles).
    """
    Zc = np.asarray(z_pts, dtype=np.float64) - np.asarray(z_rest, dtype=np.float64)[None]
    D  = Zc.shape[1]
    try:
        _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
        PC = Vt[:2].T                                   # (D, k), k = min(D, N, 2)
    except np.linalg.LinAlgError:
        PC = np.eye(D)[:, :2]
    if PC.shape[1] < 2:                                 # rang déficient
        PC = np.concatenate([PC, np.zeros((D, 2 - PC.shape[1]))], axis=1)
    return PC


def _energy_pca_slice(ax, lnn, z_pts: np.ndarray, z_rest: np.ndarray,
                      PC: np.ndarray, device, margin=0.5, grid_n=80):
    """
    contourf de E(z) sur le plan PCA-2D ancré en z_rest (cas D ≥ 3).

    Le plan est { z_rest + a·PC[:,0] + b·PC[:,1] } ; les autres directions
    latentes sont gelées à z_rest. L'étendue (a, b) couvre la projection de
    z_pts. Marque z_rest (origine du plan) d'une étoile verte.
    """
    z_rest = np.asarray(z_rest, dtype=np.float64)
    proj   = (np.asarray(z_pts, dtype=np.float64) - z_rest[None]) @ PC   # (N, 2)
    a_rng  = np.linspace(proj[:, 0].min() - margin, proj[:, 0].max() + margin, grid_n)
    b_rng  = np.linspace(proj[:, 1].min() - margin, proj[:, 1].max() + margin, grid_n)
    A, B   = np.meshgrid(a_rng, b_rng)
    flat   = np.stack([A.ravel(), B.ravel()], axis=1)                    # (G, 2)
    z_full = z_rest[None] + flat @ PC.T                                  # (G, D)
    lnn.eval()
    with torch.no_grad():
        E = lnn.energy(
            torch.tensor(z_full, dtype=torch.float32).to(device)
        ).cpu().numpy().reshape(grid_n, grid_n)
    cf = ax.contourf(A, B, E, levels=40, cmap='RdBu_r')
    plt.colorbar(cf, ax=ax, label='E(z)')
    ax.scatter([0], [0], color='green', marker='*', s=80, zorder=6, label='z_rest')
    ax.set_xlabel('PC0'); ax.set_ylabel('PC1')


def _draw_contact_plane(lnn, ax, mode, z0_range=None, z1_range=None):
    """
    Trace l'hyperplan de contact appris sur un axe matplotlib.

    mode='energy_1d' : axvline à z=d/n[0]  (plan en 1D sur courbe E(z))
    mode='zt_1d'     : axhline à z=d/n[0]  (plan en 1D sur plot z(t))
    mode='energy_2d' : droite nᵀ z = d sur le contourf 2D
                       (z0_range et z1_range requis)
    """
    if getattr(lnn, 'contact_n_raw', None) is None:
        return

    import torch
    with torch.no_grad():
        n   = lnn.contact_n.cpu().numpy()   # (D,)
        d   = lnn.contact_d.cpu().item()    # scalaire
        e   = lnn.contact_e.cpu().item()    # pour le label

    label = f'contact  d={d:.2f}  e={e:.2f}'
    kw    = dict(color='crimson', lw=1.5, ls='--', alpha=0.85, zorder=10)

    if mode == 'energy_1d':
        # En 1D : n=[n0], plan → n0·z = d → z = d/n0
        z_contact = d / n[0] if abs(n[0]) > 1e-6 else None
        if z_contact is not None:
            ax.axvline(z_contact, label=label, **kw)

    elif mode == 'zt_1d':
        # Droite horizontale sur z(t) au niveau de l'offset projeté
        z_contact = d / n[0] if abs(n[0]) > 1e-6 else None
        if z_contact is not None:
            ax.axhline(z_contact, label=label, **kw)

    elif mode == 'energy_2d':
        # Droite nᵀ z = d dans le plan (z0, z1)
        # n[0]·z0 + n[1]·z1 = d
        z0_min, z0_max = z0_range[0], z0_range[-1]
        z1_min, z1_max = z1_range[0], z1_range[-1]
        pts = []
        if abs(n[1]) > 1e-6:
            # z1 = (d - n[0]*z0) / n[1]
            for z0v in [z0_min, z0_max]:
                z1v = (d - n[0] * z0v) / n[1]
                if z1_min <= z1v <= z1_max:
                    pts.append((z0v, z1v))
        if abs(n[0]) > 1e-6:
            # z0 = (d - n[1]*z1) / n[0]
            for z1v in [z1_min, z1_max]:
                z0v = (d - n[1] * z1v) / n[0]
                if z0_min <= z0v <= z0_max:
                    pts.append((z0v, z1v))
        # Dédoublonner et trier
        pts = list({(round(p[0], 8), round(p[1], 8)) for p in pts})
        if len(pts) >= 2:
            pts.sort()
            xs, ys = zip(*pts[:2])
            ax.plot(xs, ys, label=label, **kw)


def plot_energy_map(lnn, z_enc: np.ndarray, video_lengths: list,
                    device, margin=0.5,
                    z_sim: np.ndarray = None) -> plt.Figure:
    """
    Carte d'énergie E(z) + trajectoires enc(x) et RK4.

    D=1 : courbe E(z) vs z sur une grille 1D.
          Axe gauche : E(z) (bleu). Axe droit : index de frame (noir/rouge).
          Les trajectoires enc(x) et RK4 sont tracées horizontalement
          (z en abscisse, frame en ordonnée) pour révéler où le système passe.
    D=2 : colormap contourf E(z), trajectoires superposées.

    z_enc et z_sim doivent être dans l'espace du LNN (blanchi ou brut selon config).
    """
    D      = z_enc.shape[1]
    splits = np.cumsum([0] + video_lengths)
    z_norm = z_enc   # déjà dans l'espace LNN

    n_plots = 2 if z_sim is not None else 1

    if D == 1:
        # ── Cas 1D : courbe E(z) + trajectoires projetées sur E(z) ──────
        # Pour chaque frame t, on plot le point (z_t, E(z_t)) sur la courbe.
        # Subplot gauche : enc(x). Subplot droit : RK4 (si fourni).
        all_z   = z_norm if z_sim is None else np.concatenate([z_norm, z_sim], axis=0)
        z_range = np.linspace(all_z[:, 0].min() - margin,
                              all_z[:, 0].max() + margin, 300)
        z_grid  = torch.tensor(z_range[:, None], dtype=torch.float32).to(device)

        lnn.eval()
        with torch.no_grad():
            E_grid = lnn.energy(z_grid).cpu().numpy()   # (300,)

        # Évaluation de E sur les trajectoires encodées et simulées
        def _eval_E(z_np):
            """z_np : (T, 1) → E_np : (T,)"""
            zt = torch.tensor(z_np, dtype=torch.float32).to(device)
            with torch.no_grad():
                return lnn.energy(zt).cpu().numpy()

        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        def _plot_1d(ax, z_traj_list, colors, labels, title):
            """Trace E(z) puis les trajectoires projetées dessus."""
            ax.plot(z_range, E_grid, color='steelblue', lw=1.8,
                    label='E(z)', zorder=2)
            ax.set_xlabel('z[0]')
            ax.set_ylabel('E(z)')
            ax.grid(True, alpha=0.25)
            # Marquer z_rest (minimum théorique de E)
            z_rest_val = lnn.energy.z_rest.detach().cpu().numpy()  # (D,)
            with torch.no_grad():
                E_rest = lnn.energy(
                    torch.tensor(z_rest_val[None], dtype=torch.float32).to(device)
                ).cpu().item()
            ax.axvline(z_rest_val[0], color='green', lw=1.0, ls='--',
                       alpha=0.7, label=f'z_rest={z_rest_val[0]:.2f}')
            ax.scatter([z_rest_val[0]], [E_rest], color='green', s=80,
                       marker='*', zorder=6)
            for z_tr, col, lab in zip(z_traj_list, colors, labels):
                E_tr = _eval_E(z_tr)                      # (T,)
                T    = len(z_tr)
                # Colorer par index temporel pour voir la dynamique
                ax.scatter(z_tr[:, 0], E_tr,
                           c=np.arange(T), cmap='viridis',
                           s=8, alpha=0.7, zorder=3, label=lab)
                # Point de départ
                ax.scatter(z_tr[0, 0], E_tr[0], s=60, color=col,
                           marker='o', zorder=5, edgecolors='k', linewidths=0.5)
            ax.legend(fontsize=7, loc='upper right')
            ax.set_title(title)
            _draw_contact_plane(lnn, ax, mode='energy_1d')

        # Subplot gauche : enc(x)
        enc_trajs  = [z_norm[s:e] for s, e in zip(splits[:-1], splits[1:])]
        enc_colors = [f'C{i}' for i in range(len(enc_trajs))]
        enc_labels = [f'enc vidéo {i}' for i in range(len(enc_trajs))]
        _plot_1d(axes[0], enc_trajs, enc_colors, enc_labels,
                 'E(z) + trajectoires enc(x)')

        if z_sim is not None:
            _plot_1d(axes[1], [z_sim], ['red'], ['RK4'],
                     'E(z) + trajectoire RK4 simulée')

    elif D == 2:
        # ── Cas 2D : contourf E(z) + trajectoires ─────────────────────
        all_z    = z_norm if z_sim is None else np.concatenate([z_norm, z_sim], axis=0)
        z0_range = np.linspace(all_z[:, 0].min() - margin,
                               all_z[:, 0].max() + margin, 100)
        z1_range = np.linspace(all_z[:, 1].min() - margin,
                               all_z[:, 1].max() + margin, 100)

        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
        if n_plots == 1:
            axes = [axes]

        _energy_background(axes[0], lnn, z0_range, z1_range, device)
        for s, e in zip(splits[:-1], splits[1:]):
            axes[0].plot(z_norm[s:e, 0], z_norm[s:e, 1], 'k-', alpha=0.5, lw=0.8)
            axes[0].scatter(z_norm[s, 0], z_norm[s, 1], s=40, color='k', zorder=5)
        axes[0].set_xlabel('z[0]'); axes[0].set_ylabel('z[1]')
        axes[0].set_title('E(z) + trajectoires enc(x)')
        _draw_contact_plane(lnn, axes[0], mode='energy_2d',
                            z0_range=z0_range, z1_range=z1_range)
        if axes[0].get_legend_handles_labels()[0]:
            axes[0].legend(fontsize=7)

        if z_sim is not None:
            _energy_background(axes[1], lnn, z0_range, z1_range, device)
            axes[1].plot(z_sim[:, 0], z_sim[:, 1], 'r-', alpha=0.7, lw=1.2, label='RK4')
            axes[1].scatter(z_sim[0, 0], z_sim[0, 1], s=60, color='red', zorder=5)
            axes[1].set_xlabel('z[0]'); axes[1].set_ylabel('z[1]')
            axes[1].set_title('E(z) + trajectoire RK4 simulée')
            _draw_contact_plane(lnn, axes[1], mode='energy_2d',
                                z0_range=z0_range, z1_range=z1_range)
            if axes[1].get_legend_handles_labels()[0]:
                axes[1].legend(fontsize=7)

    else:
        # ── Cas D ≥ 3 : tranche d'énergie sur le plan PCA-2D ───────────
        # Le contourf 2D brut n'existe plus (espace latent > 2). On projette
        # sur les 2 axes principaux de enc(x) (ancrés en z_rest) et on gèle
        # les autres directions. Mêmes axes/étendue pour les deux subplots.
        all_z  = z_norm if z_sim is None else np.concatenate([z_norm, z_sim], axis=0)
        z_rest = lnn.energy.z_rest.detach().cpu().numpy()
        PC     = _pca_2d_basis(all_z, z_rest)

        def _proj(z):
            return (z - z_rest[None]) @ PC

        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
        if n_plots == 1:
            axes = [axes]

        _energy_pca_slice(axes[0], lnn, all_z, z_rest, PC, device, margin)
        for s, e in zip(splits[:-1], splits[1:]):
            p = _proj(z_norm[s:e])
            axes[0].plot(p[:, 0], p[:, 1], 'k-', alpha=0.5, lw=0.8)
            axes[0].scatter(p[0, 0], p[0, 1], s=40, color='k', zorder=5)
        axes[0].set_title('E(z) PCA-2D + trajectoires enc(x)')
        axes[0].legend(fontsize=7)

        if z_sim is not None:
            _energy_pca_slice(axes[1], lnn, all_z, z_rest, PC, device, margin)
            ps = _proj(z_sim)
            axes[1].plot(ps[:, 0], ps[:, 1], 'r-', alpha=0.7, lw=1.2, label='RK4')
            axes[1].scatter(ps[0, 0], ps[0, 1], s=60, color='red', zorder=5)
            axes[1].set_title('E(z) PCA-2D + trajectoire RK4 simulée')
            axes[1].legend(fontsize=7)

    plt.tight_layout()
    return fig


# ── Simulation RK4 + comparaison trajectoire réelle ───────────────────────

def get_sim_pressure(lnn, frame_dataset, s: int, n_steps: int, device):
    """
    Construit la pression alignée (n_steps, n_c) pour simulate_rk4, ou None.

    Renvoie None si le forçage de pression n'est pas actif (lnn.use_pressure)
    ou si le dataset ne porte pas de pressions. Au-delà des frames disponibles
    (extrapolation n_frames > T_orig), maintient la dernière valeur (ordre zéro).
    """
    if not getattr(lnn, 'use_pressure', False):
        return None
    P = getattr(frame_dataset, 'pressures', None)
    if P is None:
        return None
    n_c   = P.shape[1]
    out   = np.empty((n_steps, n_c), dtype=np.float32)
    avail = max(0, min(n_steps, len(P) - s))
    if avail > 0:
        out[:avail] = P[s:s + avail]
        out[avail:] = P[s + avail - 1]      # maintien de la dernière pression
    else:
        out[:] = 0.0
    return torch.from_numpy(out).to(device)


def _poly_edge_weights(w: int, deg: int = 2):
    """Poids w tels que `weights @ z[s:s+w]` = dérivée en t=s d'un fit polynomial.

    Régression polynomiale UNILATÉRALE (on n'a rien avant s) de degré `deg` sur
    `w` points ; la dérivée en t=0 est le coefficient de t, donc la ligne 1 de la
    pseudo-inverse de la matrice de Vandermonde. `w=3, deg=2` redonne EXACTEMENT
    la différence avant d'ordre 2 de `initial_velocity` (vérifié).
    """
    t = np.arange(w, dtype=float)
    V = np.vander(t, deg + 1, increasing=True)          # colonnes 1, t, t², …
    return np.linalg.pinv(V)[1]


def initial_velocity_poly(z_seq, w: int = 7, deg: int = 2):
    """v0 par régression polynomiale unilatérale sur `w` frames (torch ou numpy).

    Même quantité que `initial_velocity` (dérivée en t=0, unités par frame) mais
    estimée sur `w` points au lieu de 3 : l'écart-type de l'estimateur à 3 points
    vaut 2.55·σ_encodeur, ce qui atteint 25 % de la vitesse sur nos cas test.
    Élargir la fenêtre échange cette variance contre un biais de troncature, qui
    croît en (ω·w·h)² — d'où `select_v0_estimator`, qui arbitre sur les données.
    """
    z = z_seq[:w]
    if z.shape[0] < w:
        return initial_velocity(z_seq)
    ww = _poly_edge_weights(z.shape[0], deg)
    if isinstance(z, np.ndarray):
        return ww @ z
    t = torch.as_tensor(ww, dtype=z.dtype, device=z.device)
    return t @ z


def select_v0_estimator(z_np: np.ndarray, candidates=None, ref_window: int = 11,
                        ref_poly: int = 3, verbose: bool = True):
    """Choisit l'estimateur de v0 le mieux adapté à CETTE trajectoire, sans modèle.

    Protocole : sur les frames intérieures, on dispose d'une dérivée de RÉFÉRENCE
    centrée (Savitzky-Golay, `ref_window`/`ref_poly`), qui utilise les deux côtés
    et n'est donc pas disponible en t=0. Chaque estimateur UNILATÉRAL candidat est
    appliqué aux mêmes frames et comparé à cette référence ; on garde celui de plus
    faible RMS. Aucune dynamique apprise n'intervient : la sélection ne regarde que
    q(t), donc elle ne peut pas « ajuster la CI pour flatter le modèle ».

    Retourne (nom, w, deg, rms_relatif).
    """
    from scipy.signal import savgol_filter
    T = len(z_np)
    if candidates is None:
        candidates = [(3, 2)] + [(w, d) for w in (5, 7, 9, 11, 15) for d in (2, 3)]
    candidates = [(w, d) for (w, d) in candidates if w + 2 <= T and d + 1 <= w]
    if not candidates or T < ref_window + 6:
        return ('fd2', 3, 2, float('nan'))
    ref = savgol_filter(z_np, ref_window, ref_poly, deriv=1, axis=0)
    lo, hi = ref_window // 2, T - max(w for w, _ in candidates) - 1
    if hi <= lo:
        return ('fd2', 3, 2, float('nan'))
    ks = np.arange(lo, hi)
    tru = ref[ks]
    scale = float(np.sqrt((tru ** 2).mean())) or 1.0
    best = None
    for (w, deg) in candidates:
        ww = _poly_edge_weights(w, deg)
        est = np.stack([ww @ z_np[k:k + w] for k in ks])
        rms = float(np.sqrt(((est - tru) ** 2).mean()))
        if best is None or rms < best[3]:
            best = (('fd2' if (w, deg) == (3, 2) else f'poly{deg}_w{w}'), w, deg, rms)
    name, w, deg, rms = best
    if verbose:
        print(f'v0 : estimateur retenu {name} (RMS {rms:.4f} contre une dérivée '
              f'centrée de référence, soit {100 * rms / scale:.1f} % de la vitesse)')
    return (name, w, deg, rms / scale)


def initial_velocity(z_seq):
    """
    Vitesse initiale v0 = dz/dt à t=0, en unités « par frame » (dt=1).

    Une différence avant d'ordre 1 (z[1]-z[0]) vaut, par le théorème des
    accroissements finis, la dérivée au milieu t=½ et non en t=0 : son biais
    O(h)·z̈(0) rabote systématiquement |v0| (sous-estimation à l'amorce d'une
    oscillation/relaxation). On utilise donc une différence avant d'ordre 2
    (3 points) qui annule ce terme — cohérente avec le résidu d'entraînement
    (différences centrées, cf. LNN.residual). Repli sur l'ordre 1 si <3 frames.

    z_seq : (T, D) tensor — trajectoire encodée (T >= 2).
    Retourne (D,).
    """
    if z_seq.shape[0] >= 3:
        return (-3.0 * z_seq[0] + 4.0 * z_seq[1] - z_seq[2]) / 2.0
    return z_seq[1] - z_seq[0]


def simulate_rk4(lnn, z0: torch.Tensor, v0: torch.Tensor,
                 n_steps: int, dt: float,
                 pressure: torch.Tensor = None) -> torch.Tensor:
    """
    Intègre l'équation de Lagrange dans l'espace normalisé.

        dz/dt = v
        dv/dt = M⁻¹ ( -γ·v - dE/dz(z) + b(q)ᵀ P )

    Schéma sélectionné par config.LNN_INTEGRATOR :
      'verlet' (défaut) — velocity-Verlet semi-implicite (symplectique, ordre 2),
                          1 éval de force/pas (hors pression/contact). Conserve
                          l'énergie en long horizon, ~4× moins de backward que RK4.
      'rk4'             — Runge-Kutta 4 (ordre 4, 4 évals/pas), legacy.
    (Le nom historique `simulate_rk4` est conservé pour compatibilité d'appel.)

    Le terme de pression b(q)ᵀ P (forçage pneumatique) n'est ajouté que si
    `pressure` est fourni ET lnn.use_pressure=True. La pression est tenue
    constante sur les 4 sous-pas RK4 d'un même step (maintien d'ordre zéro),
    cohérent avec des step inputs.

    Si config.LNN_CONTACT est actif, le contact est géré par bisection :
    quand une traversée du plan est détectée dans [t, t+dt], on localise
    le temps de contact t* par bisection linéaire sur φ(z(t)), puis on
    intègre jusqu'à t*, applique le rebond, et continue avec dt - t*.

    z0, v0   : (D,)  conditions initiales dans l'espace normalisé
    pressure : (n_steps, n_c) ou None — pression par step (cf. get_sim_pressure)
    Retourne z_traj : (n_steps, D)
    """
    import config
    device = z0.device
    _contact = getattr(lnn, 'contact_n_raw', None) is not None
    _use_p   = pressure is not None and getattr(lnn, 'use_pressure', False)
    _integrator = getattr(config, 'LNN_INTEGRATOR', 'verlet')

    # generalized-α (Chung & Hulbert 1993) : paramètres dérivés du rayon spectral
    # à fréquence infinie ρ∞ ∈ [0, 1]. ρ∞=1 ⟹ Newmark trapézoïdal (AUCUNE dissipation,
    # = comportement conservatif) ; ρ∞<1 ⟹ hautes fréquences amorties, basses fréquences
    # préservées (ordre 2 maintenu) ; ρ∞=0 ⟹ annihilation asymptotique des HF.
    _ga_rho   = min(max(float(getattr(config, 'LNN_RHO_INF', 1.0)), 0.0), 1.0)
    _ga_am    = (2.0 * _ga_rho - 1.0) / (_ga_rho + 1.0)
    _ga_af    = _ga_rho / (_ga_rho + 1.0)
    _ga_beta  = 0.25 * (1.0 - _ga_am + _ga_af) ** 2
    _ga_gamma = 0.5 - _ga_am + _ga_af
    _ga_iters = max(1, int(getattr(config, 'LNN_GENALPHA_ITERS', 8)))
    _ga_tol   = float(getattr(config, 'LNN_GENALPHA_TOL', 1e-8))
    _ga_jac_refresh = max(1, int(getattr(config, 'LNN_GENALPHA_JAC_REFRESH', 3)))
    _ga_D     = z0.numel()

    def dzdt(z, v):
        return v

    def dvdt(z, v, p):
        # enable_grad : nécessaire pour le repli autograd de dE_dz et pour
        # pressure_force('potential'), même si l'appelant est sous torch.no_grad().
        # On détache a en sortie → la trajectoire n'accumule aucun graphe sur les
        # milliers de pas (comportement identique à l'ancien autograd.grad sans
        # create_graph), évitant une fuite mémoire en rollout long.
        with torch.enable_grad():
            # Métrique courbe dérivée du décodeur : physique = lnn.accel (Coriolis +
            # Rayleigh α·M̃ + pression inclus, source unique partagée avec le résidu).
            if getattr(lnn, 'metric', None) is not None:
                pin = None if p is None else p.unsqueeze(0)
                return lnn.accel(z.unsqueeze(0), v.unsqueeze(0), pin).squeeze(0).detach()
            grad = lnn.dE_dz(z.unsqueeze(0)).squeeze(0)
            friction = torch.zeros_like(v)
            if getattr(lnn, 'Gamma', None) is not None:
                friction = friction + lnn.Gamma @ v
            elif lnn.gamma is not None:
                friction = friction + lnn.gamma * v
            if getattr(lnn, 'Beta', None) is not None:
                v_norm = v.norm().clamp(min=1e-6)
                friction = friction + lnn.Beta @ (v / v_norm)
            elif lnn.beta is not None:
                v_norm = v.norm().clamp(min=1e-6)
                friction = friction + lnn.beta * v / v_norm
            a = -friction - grad
            if p is not None:
                # F_P = b(q)ᵀ P (même signe qu'au second membre EL d'entraînement)
                a = a + lnn.pressure_force(z.unsqueeze(0), p.unsqueeze(0)).squeeze(0)
            Minv = getattr(lnn, 'Minv', None)
            if Minv is not None:
                a = Minv @ a
        return a.detach()

    def _rk4_step(z, v, h, p=None):
        """Un step RK4 de durée h. Retourne (z_new, v_new). p tenu constant."""
        k1z = dzdt(z, v);               k1v = dvdt(z, v, p)
        k2z = dzdt(z+.5*h*k1z, v+.5*h*k1v); k2v = dvdt(z+.5*h*k1z, v+.5*h*k1v, p)
        k3z = dzdt(z+.5*h*k2z, v+.5*h*k2v); k3v = dvdt(z+.5*h*k2z, v+.5*h*k2v, p)
        k4z = dzdt(z+h*k3z,    v+h*k3v);    k4v = dvdt(z+h*k3z,    v+h*k3v, p)
        return (z + (h/6)*(k1z+2*k2z+2*k3z+k4z),
                v + (h/6)*(k1v+2*k2v+2*k3v+k4v))

    def _verlet_step(z, v, h, p=None, a_cur=None):
        """
        Velocity-Verlet semi-implicite (symplectique, ordre 2) de durée h.

        a_cur = a(z, v) déjà connue (réutilisée → 1 éval de force/pas) ou None
        (recalculée). Retourne (z_new, v_new, a_new) où a_new = a(z_new, v_half)
        ≈ a(z_new, v_new) — réutilisable au pas suivant (le terme de frottement,
        seule dépendance en v, est petit). Le demi-kick visqueux v_half rend le
        frottement semi-implicite.
        """
        a1 = dvdt(z, v, p) if a_cur is None else a_cur
        v_half = v + 0.5 * h * a1
        z_new  = z + h * v_half
        a2     = dvdt(z_new, v_half, p)
        v_new  = v_half + 0.5 * h * a2
        return z_new, v_new, a2

    def _genalpha_step(z, v, h, p=None, a_prev=None):
        """
        Un pas generalized-α (implicite, dissipation numérique HF réglée par ρ∞).

        Schéma : mises à jour Newmark (β, γ) + équation du mouvement évaluée au
        point milieu généralisé (α_m sur l'accélération, α_f sur q/q̇). Ici dvdt
        renvoie déjà a = M⁻¹·force ⟹ l'équilibre s'écrit
            (1−α_m) a_{n+1} + α_m a_n = dvdt(q_{n+1−α_f}, v_{n+1−α_f}).
        a_{n+1} (implicite) est résolu par Newton COMPLET : à chaque itération le
        Jacobien de dvdt (A=∂a/∂q, C=∂a/∂v) est ré-estimé par différences finies AU
        POINT MILIEU courant (black-box vis-à-vis des variantes LNN) → convergence
        quadratique, robuste sur le système raide à métrique courbe (un Jacobien gelé
        y convergeait trop lentement et injectait de l'énergie). Arrêt anticipé sur
        ‖R‖ < tol·(‖a_n‖+1) ; au plus LNN_GENALPHA_ITERS itérations. Exact en 1 solve
        pour un système linéaire ⟹ stable/dissipatif jusqu'aux modes proches de Nyquist.
        a_prev = accélération ALGORITHMIQUE a_n du pas précédent (état du schéma,
        toujours reportable) ou None (recalculée = a(z, v)). Retourne (z, v, a_{n+1}).
        """
        a_n = dvdt(z, v, p) if a_prev is None else a_prev
        eps = 1e-4
        I   = torch.eye(_ga_D, dtype=z.dtype, device=z.device)
        cq  = (1.0 - _ga_af) * h * h * _ga_beta     # ∂q_mid/∂a_{n+1}
        cv  = (1.0 - _ga_af) * h * _ga_gamma        # ∂v_mid/∂a_{n+1}
        tol = _ga_tol * (a_n.norm().item() + 1.0)

        def _mids(a1):
            z_n = z + h * v + h * h * ((0.5 - _ga_beta) * a_n + _ga_beta * a1)
            v_n = v + h * ((1.0 - _ga_gamma) * a_n + _ga_gamma * a1)
            return z_n, v_n, (1.0 - _ga_af) * z_n + _ga_af * z, \
                             (1.0 - _ga_af) * v_n + _ga_af * v

        a1 = a_n.clone()                            # prédicteur : a_{n+1} ← a_n
        z_new, v_new, q_mid, v_mid = _mids(a1)
        a_ext = dvdt(q_mid, v_mid, p)
        # Jacobien FD au point milieu PRÉDICTEUR, gelé sur le pas (a_n est un bon
        # prédicteur pour une dynamique lisse ⟹ Newton modifié converge en ~2–3 iters,
        # ~3× moins d'évals qu'un rafraîchissement à chaque itération). Rafraîchi tous
        # les LNN_GENALPHA_JAC_REFRESH iters en repli si un pas raide traîne.
        def _jac(qm, vm, a_base):
            A = torch.empty(_ga_D, _ga_D, dtype=z.dtype, device=z.device)
            C = torch.empty(_ga_D, _ga_D, dtype=z.dtype, device=z.device)
            for j in range(_ga_D):
                e = torch.zeros(_ga_D, dtype=z.dtype, device=z.device); e[j] = eps
                A[:, j] = (dvdt(qm + e, vm, p) - a_base) / eps
                C[:, j] = (dvdt(qm, vm + e, p) - a_base) / eps
            return (1.0 - _ga_am) * I - cq * A - cv * C   # ∂R/∂a_{n+1}

        J = _jac(q_mid, v_mid, a_ext)
        for _it in range(_ga_iters):
            R = (1.0 - _ga_am) * a1 + _ga_am * a_n - a_ext
            if R.norm().item() < tol:
                break
            if _it > 0 and _it % _ga_jac_refresh == 0:   # repli : rafraîchir sur pas raide
                J = _jac(q_mid, v_mid, a_ext)
            a1 = a1 - torch.linalg.solve(J, R)
            z_new, v_new, q_mid, v_mid = _mids(a1)
            a_ext = dvdt(q_mid, v_mid, p)
        return z_new, v_new, a1

    def _step(z, v, h, p=None):
        """Pas auto-contenu (sans réutilisation de force) — utilisé par le contact.
        gen_alpha retombe sur Verlet ici : les sous-pas de contact (événement
        discontinu, bisection) n'ont pas besoin de la dissipation HF réglée."""
        if _integrator == 'rk4':
            return _rk4_step(z, v, h, p)
        z_new, v_new, _ = _verlet_step(z, v, h, p)
        return z_new, v_new

    def _apply_contact(z, v, n_vec, e_val):
        """Réfléchit v par rapport à n_vec avec restitution e_val."""
        v_n = (n_vec @ v) * n_vec   # composante normale
        v_t = v - v_n               # composante tangentielle
        return v_t - e_val * v_n    # rebond

    z = z0.clone().float()
    v = v0.clone().float()
    traj = [z.detach()]
    a_cur = None                       # accélération réutilisable (verlet)
    # Réutilisation valable seulement si la force ne change pas entre les pas :
    # désactivée sous pression (ZOH par pas) et sur les pas de contact.
    _reuse = (_integrator != 'rk4') and not _use_p

    for i in range(n_steps - 1):
        # Arrêt propre si l'état a divergé : sinon q=NaN/inf est réinjecté dans
        # Mhat(q) → torch.linalg.solve « singular » (crash). On stoppe et on
        # complète la trajectoire (longueur n_steps préservée pour les plots).
        if not (torch.isfinite(z).all() and torch.isfinite(v).all()):
            break
        p_i = pressure[i] if _use_p else None
        # M̃(q) peut devenir SINGULIÈRE à un q extrapolé (LNN à peine entraîné), même
        # avec q FINI : torch.linalg.solve lève alors _LinAlgError AVANT que l'état ne
        # devienne NaN, donc le garde-fou isfinite ci-dessus ne l'attrape pas. On traite
        # cette exception comme une divergence : on stoppe et on complète la trajectoire
        # (plage valide du début préservée) → le rollout ne crashe jamais et le plot de
        # debug se fait toujours, sur la portion intègre uniquement.
        try:
            if not _contact:
                if _integrator == 'rk4':
                    z, v = _rk4_step(z, v, dt, p_i)
                elif _integrator == 'gen_alpha':
                    # a_cur = accélération algorithmique a_n (toujours reportée, même
                    # sous pression ZOH ; None au 1er pas et après un contact).
                    z, v, a_cur = _genalpha_step(z, v, dt, p_i, a_prev=a_cur)
                else:
                    z, v, a_cur = _verlet_step(z, v, dt, p_i,
                                               a_cur=a_cur if _reuse else None)
            else:
                a_cur = None               # le contact invalide la réutilisation
                with torch.no_grad():
                    n_vec = lnn.contact_n   # (D,) normalisé
                    d_val = lnn.contact_d   # scalaire
                    e_val = lnn.contact_e   # ∈ (0, 1]

                phi_start = (n_vec @ z) - d_val
                z_try, v_try = _step(z, v, dt, p_i)

                with torch.no_grad():
                    phi_end = (n_vec @ z_try) - d_val

                # Pas de traversée → step normal
                if phi_start * phi_end >= 0:
                    z, v = z_try, v_try
                else:
                    # ── Bisection pour localiser t* ───────────────────────
                    t_lo, t_hi = 0.0, dt
                    z_lo = z.detach().clone()
                    v_lo = v.detach().clone()
                    phi_lo = phi_start.clone() if torch.is_tensor(phi_start) \
                             else torch.tensor(phi_start)

                    for _ in range(8):
                        t_mid = (t_lo + t_hi) * 0.5
                        h_sub = t_mid - t_lo
                        z_mid, v_mid = _step(z_lo, v_lo, h_sub, p_i)
                        with torch.no_grad():
                            phi_mid = (n_vec @ z_mid) - d_val
                        if phi_lo * phi_mid >= 0:
                            # contact pas encore atteint — avancer t_lo
                            t_lo   = t_mid
                            z_lo   = z_mid.detach().clone()
                            v_lo   = v_mid.detach().clone()
                            phi_lo = phi_mid.clone()
                        else:
                            t_hi = t_mid

                    # z_lo est juste avant le contact (du bon côté)
                    # Projeter z_lo exactement sur le plan par sécurité
                    with torch.no_grad():
                        phi_lo = (n_vec @ z_lo) - d_val
                        z_contact = z_lo - phi_lo * n_vec
                        v_bounced = _apply_contact(z_lo, v_lo, n_vec, e_val)

                    # Intégrer le temps restant après le rebond
                    t_remaining = dt - t_lo
                    z, v = _step(z_contact, v_bounced, t_remaining, p_i)
        except torch.linalg.LinAlgError:
            break

        traj.append(z.detach())

    # Complète si arrêt anticipé (divergence) : répète le dernier état → (n_steps, D).
    if len(traj) < n_steps:
        traj.extend([traj[-1]] * (n_steps - len(traj)))
    return torch.stack(traj)   # (n_steps, D)


def plot_trajectory_validation(
    lnn, encoder, frame_dataset, device, dt: float,
    video_idx: int = 0, cmap: str = 'viridis', max_frames: int = None
) -> plt.Figure:
    """
    Grille 2×2 :
        [0,0] z(t) vidéo 0  enc(x) + RK4   [0,1] z(t) vidéo 1  enc(x) + RK4
        [1,0] E(z) vidéo 0  enc(x) + RK4   [1,1] E(z) vidéo 1  enc(x) + RK4

    En 1D : z(t) = courbe scalaire. E(z) = courbe énergie avec points projetés.
    En 2D : z(t) = espace latent z0 vs z1.  E(z) = contourf.
    En D≥3 (p. ex. Krauss d=4) : figure 2×D pour la 1ʳᵉ vidéo —
        ligne 0 = z[d](t) enc vs RK4 pour TOUTES les composantes ;
        ligne 1 = plans de phase consécutifs z[d] vs z[d+1] + dernière
        colonne = tranche d'énergie E(z) sur le plan PCA-2D.
    Contact plane tracé sur chaque subplot si LNN_CONTACT actif.
    """
    lnn.eval()
    encoder.eval()

    splits   = np.cumsum([0] + frame_dataset.video_lengths)
    n_videos = len(frame_dataset.video_lengths)
    idx_list = [video_idx, min(video_idx + 1, n_videos - 1)]

    def _to_float(x):
        """Frames uint8 [0,255] (store_uint8) → float [0,1] ; float inchangé."""
        return x.float().div(255.0) if x.dtype == torch.uint8 else x

    # ── running stats : encodage par batch (évite de charger toutes les frames
    #    d'un coup sur le GPU — 55k frames RGB 256 ≈ 41 GB VRAM — et gère uint8) ──
    _bs = 256
    with torch.no_grad():
        for i in range(0, len(frame_dataset.frames), _bs):
            xb = torch.from_numpy(frame_dataset.frames[i:i + _bs]).to(device)
            encoder(_to_float(xb))

    # ── Encodage + simulation pour chaque vidéo ───────────────────────────
    results = []
    for idx in idx_list:
        s, e = splits[idx], splits[idx + 1]
        T    = e - s
        if max_frames is not None and T > max_frames:
            T = max_frames
        frames_v = _to_float(torch.from_numpy(frame_dataset.frames[s:s + T]).to(device))
        with torch.no_grad():
            z_enc = encoder(frames_v)   # (T, D)
        z0  = z_enc[0]
        v0  = initial_velocity(z_enc)
        p_sim = get_sim_pressure(lnn, frame_dataset, s, T, device)
        z_sim = simulate_rk4(lnn, z0, v0, n_steps=T, dt=1.0, pressure=p_sim)
        results.append({
            'idx':   idx,
            'T':     T,
            'z_enc': z_enc.cpu().numpy(),
            'z_sim': z_sim.cpu().numpy(),
            't':     np.arange(T) * dt,
        })

    D   = results[0]['z_enc'].shape[1]
    if D >= 3:
        # 2 lignes × D colonnes : une colonne par composante latente
        # (couvre TOUTES les dimensions, p. ex. d=4 montre bien z[3]).
        fig, axes = plt.subplots(2, D, figsize=(5 * D, 9))
    else:
        n_rows = 3 if D == 2 else 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))

    # ── Helper énergie ────────────────────────────────────────────────────
    def _eval_E(z_np):
        zt = torch.tensor(z_np, dtype=torch.float32).to(device)
        with torch.no_grad():
            return lnn.energy(zt).cpu().numpy()

    # ── Ligne 0 : z(t) — D=1 ou D=2 seulement ───────────────────────────
    if D <= 2:
        for col, r in enumerate(results):
            ax  = axes[0, col]
            z_e = r['z_enc']
            z_s = r['z_sim']
            t   = r['t']

            if D == 1:
                ax.plot(t, z_e[:, 0], label='enc(x)', color='steelblue', linewidth=1.4)
                ax.plot(t, z_s[:, 0], label='RK4',    color='tomato',
                        linewidth=1.4, linestyle='--')
                ax.set_xlabel('Temps (s)'); ax.set_ylabel('z[0]')
                ax.grid(True, alpha=0.3)
                _draw_contact_plane(lnn, ax, mode='zt_1d')

            elif D == 2:
                colors_t = np.linspace(0, 1, r['T'])
                ax.scatter(z_e[:, 0], z_e[:, 1], c=colors_t, cmap=cmap,
                           s=6, label='enc(x)', alpha=0.8)
                ax.scatter(z_s[:, 0], z_s[:, 1], c=colors_t, cmap=cmap,
                           s=6, marker='x', label='RK4', alpha=0.8)
                ax.plot(z_e[:, 0], z_e[:, 1], 'b-', alpha=0.2, linewidth=0.6)
                ax.plot(z_s[:, 0], z_s[:, 1], 'r--', alpha=0.2, linewidth=0.6)
                ax.scatter(z_e[0, 0], z_e[0, 1], s=50, color='steelblue', zorder=6)
                ax.scatter(z_s[0, 0], z_s[0, 1], s=50, color='tomato',    zorder=6)
                ax.set_xlabel('z[0]'); ax.set_ylabel('z[1]')
                z0r = np.array([min(z_e[:,0].min(), z_s[:,0].min()) - 0.2,
                                 max(z_e[:,0].max(), z_s[:,0].max()) + 0.2])
                z1r = np.array([min(z_e[:,1].min(), z_s[:,1].min()) - 0.2,
                                 max(z_e[:,1].max(), z_s[:,1].max()) + 0.2])
                _draw_contact_plane(lnn, ax, mode='energy_2d',
                                    z0_range=z0r, z1_range=z1r)

            ax.set_title(f'vidéo {r["idx"]} — z(t)')
            ax.legend(fontsize=7)

    # ── Ligne 1 : E(z) — D=1 ou D=2 seulement ───────────────────────────
    if D == 1:
        all_z   = np.concatenate([r['z_enc'] for r in results] +
                                  [r['z_sim'] for r in results])
        margin  = 0.5
        z_range = np.linspace(all_z[:, 0].min() - margin,
                              all_z[:, 0].max() + margin, 300)
        z_grid  = torch.tensor(z_range[:, None], dtype=torch.float32).to(device)
        with torch.no_grad():
            E_grid = lnn.energy(z_grid).cpu().numpy()

        z_rest_val = lnn.energy.z_rest.detach().cpu().numpy()
        with torch.no_grad():
            E_rest = lnn.energy(
                torch.tensor(z_rest_val[None], dtype=torch.float32).to(device)
            ).cpu().item()

        for col, r in enumerate(results):
            ax = axes[1, col]
            ax.plot(z_range, E_grid, color='steelblue', lw=1.6, label='E(z)', zorder=2)
            ax.axvline(z_rest_val[0], color='green', lw=1.0, ls='--', alpha=0.7,
                       label=f'z_rest={z_rest_val[0]:.2f}')
            ax.scatter([z_rest_val[0]], [E_rest], color='green', s=60, marker='*', zorder=6)

            for z_np, col_pt, lab in [(r['z_enc'], 'steelblue', 'enc(x)'),
                                       (r['z_sim'], 'tomato',    'RK4')]:
                E_pt = _eval_E(z_np)
                ax.scatter(z_np[:, 0], E_pt, c=np.arange(r['T']), cmap=cmap,
                           s=6, alpha=0.7, zorder=3, label=lab)
                ax.scatter(z_np[0, 0], E_pt[0], s=50, color=col_pt,
                           marker='o', zorder=5, edgecolors='k', linewidths=0.5)

            ax.set_xlabel('z[0]'); ax.set_ylabel('E(z)')
            ax.set_title(f'vidéo {r["idx"]} — E(z)')
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7)
            _draw_contact_plane(lnn, ax, mode='energy_1d')

    elif D == 2:
        all_z    = np.concatenate([r['z_enc'] for r in results] +
                                   [r['z_sim'] for r in results])
        margin   = 0.5
        z0_range = np.linspace(all_z[:, 0].min() - margin,
                               all_z[:, 0].max() + margin, 80)
        z1_range = np.linspace(all_z[:, 1].min() - margin,
                               all_z[:, 1].max() + margin, 80)

        for col, r in enumerate(results):
            ax = axes[1, col]
            _energy_background(ax, lnn, z0_range, z1_range, device)
            z_e, z_s = r['z_enc'], r['z_sim']
            ax.plot(z_e[:, 0], z_e[:, 1], 'b-',  alpha=0.6, lw=0.9, label='enc(x)')
            ax.plot(z_s[:, 0], z_s[:, 1], 'r--', alpha=0.6, lw=0.9, label='RK4')
            ax.scatter(z_e[0, 0], z_e[0, 1], s=50, color='steelblue', zorder=6)
            ax.scatter(z_s[0, 0], z_s[0, 1], s=50, color='tomato',    zorder=6)
            ax.set_xlabel('z[0]'); ax.set_ylabel('z[1]')
            ax.set_title(f'vidéo {r["idx"]} — E(z)')
            ax.legend(fontsize=7)
            _draw_contact_plane(lnn, ax, mode='energy_2d',
                                z0_range=z0_range, z1_range=z1_range)

        # ── Ligne 2 : z[0](t) et z[1](t) ─────────────────────────────────
        for col, r in enumerate(results):
            ax  = axes[2, col]
            z_e = r['z_enc']
            z_s = r['z_sim']
            t   = r['t']
            ax.plot(t, z_e[:, 0], color='steelblue', lw=1.4, label='enc z[0]')
            ax.plot(t, z_s[:, 0], color='steelblue', lw=1.4, ls='--', label='RK4 z[0]')
            ax.plot(t, z_e[:, 1], color='tomato',    lw=1.4, label='enc z[1]')
            ax.plot(t, z_s[:, 1], color='tomato',    lw=1.4, ls='--', label='RK4 z[1]')
            ax.set_xlabel('Temps (s)'); ax.set_ylabel('z[d]')
            ax.set_title(f'vidéo {r["idx"]} — z[d](t)')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    else:
        # D >= 3 : figure 2×D dédiée pour la première vidéo, COUVRE TOUTES
        # les composantes (généralise l'ancien 2×3 limité à z[0..2]).
        #   Ligne 0 : z[d](t) enc vs RK4, une colonne par composante d.
        #   Ligne 1 : plans de phase consécutifs z[d] vs z[d+1] (d=0..D-2)
        #             + dernière colonne = tranche d'énergie PCA-2D.
        r   = results[0]
        z_e = r['z_enc']
        z_s = r['z_sim']
        t   = r['t']

        for d in range(D):
            ax = axes[0, d]
            ax.plot(t, z_e[:, d], color='steelblue', lw=1.4, label='enc(x)')
            ax.plot(t, z_s[:, d], color='tomato',    lw=1.4, ls='--', label='RK4')
            ax.set_xlabel('Temps (s)'); ax.set_ylabel(f'z[{d}]')
            ax.set_title(f'vidéo {r["idx"]} — z[{d}](t)')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        colors_t = np.linspace(0, 1, r['T'])
        for d in range(D - 1):
            da, db = d, d + 1
            ax = axes[1, d]
            ax.scatter(z_e[:, da], z_e[:, db], c=colors_t, cmap=cmap,
                       s=6, alpha=0.8, label='enc(x)')
            ax.scatter(z_s[:, da], z_s[:, db], c=colors_t, cmap=cmap,
                       s=6, alpha=0.8, marker='x', label='RK4')
            ax.plot(z_e[:, da], z_e[:, db], 'b-',  alpha=0.2, lw=0.6)
            ax.plot(z_s[:, da], z_s[:, db], 'r--', alpha=0.2, lw=0.6)
            ax.scatter(z_e[0, da], z_e[0, db], s=50, color='steelblue', zorder=6)
            ax.scatter(z_s[0, da], z_s[0, db], s=50, color='tomato',    zorder=6)
            ax.set_xlabel(f'z[{da}]'); ax.set_ylabel(f'z[{db}]')
            ax.set_title(f'vidéo {r["idx"]} — z[{da}] vs z[{db}]')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Dernière colonne ligne 1 : carte d'énergie sur le plan PCA-2D
        ax = axes[1, D - 1]
        try:
            z_rest = lnn.energy.z_rest.detach().cpu().numpy()
            all_z  = np.concatenate([z_e, z_s], axis=0)
            PC     = _pca_2d_basis(all_z, z_rest)
            _energy_pca_slice(ax, lnn, all_z, z_rest, PC, device)
            pe = (z_e - z_rest[None]) @ PC
            ps = (z_s - z_rest[None]) @ PC
            ax.plot(pe[:, 0], pe[:, 1], 'b-',  alpha=0.6, lw=0.9, label='enc(x)')
            ax.plot(ps[:, 0], ps[:, 1], 'r--', alpha=0.6, lw=0.9, label='RK4')
            ax.set_title(f'vidéo {r["idx"]} — E(z) PCA-2D')
            ax.legend(fontsize=7)
        except Exception as exc:
            ax.text(0.5, 0.5, f'E(z) PCA indispo\n{exc}',
                    ha='center', va='center', fontsize=8)
            ax.set_axis_off()

    fig.suptitle('Validation LNN', fontsize=11, y=1.01)
    plt.tight_layout()
    return fig



# ── Décodeur GS + export vidéo ────────────────────────────────────────────

def generate_gs_video(
    lnn,
    encoder,
    decoder_gs,
    frame_dataset,
    device,
    output_path: str,
    video_idx: int = 0,
    fps: float = 30.0,
    n_frames: int = 0,
    query_dataset=None,
    display_frames_np: np.ndarray = None,
    z_white_mean=None,   # (D,) tensor -- si fourni, whitening au lieu de diag norm
    z_white_W=None,      # (D, D) tensor -- matrice W = V Lambda^{-1/2}
) -> None:
    """
    Genere une video MP4 de la trajectoire RK4 decodee par le decodeur GS.

    Normalisation :
        - Si z_white_mean + z_white_W fournis (train_all) :
              z_norm = (z - z_white_mean) @ z_white_W
              z_dec  = z_norm @ W^{-1} + z_white_mean  (espace decodeur = espace blanchi)
        - Sinon (pipeline separe) : normalisation diagonale mean/std,
              z_dec = z_sim_norm @ W_inv + z_mean  (espace decodeur = blanchi)
    """
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lnn.eval()
    encoder.eval()
    decoder_gs.eval()

    # ── 1. Stats de normalisation ─────────────────────────────────────────
    enc_frames_t = torch.from_numpy(frame_dataset.frames).to(device)
    with torch.no_grad():
        z_all = encoder(enc_frames_t)                  # (N, D) — blanchi si normalize=True

    # encoder.eval() → forward() utilise running_mean/running_W si normalize=True
    # z_all et z_enc_raw sont déjà dans le bon espace (blanchi ou brut)
    def _norm(z):    return z   # encoder.forward() a déjà appliqué le whitening
    def _denorm(zn): return zn

    disp_frames_np = display_frames_np if display_frames_np is not None \
                     else frame_dataset.frames

    # ── 2. Trajectoire RK4 ────────────────────────────────────────────────
    src_dataset  = query_dataset if query_dataset is not None else frame_dataset
    src_frames_t = torch.from_numpy(src_dataset.frames).to(device)

    splits  = np.cumsum([0] + src_dataset.video_lengths)
    s, e    = splits[video_idx], splits[video_idx + 1]
    T_orig  = e - s
    T_total = n_frames if n_frames > 0 else T_orig

    with torch.no_grad():
        z_enc_raw = encoder(src_frames_t[s:e])         # (T_orig, D)
    z_enc_norm = _norm(z_enc_raw)                       # (T_orig, D)

    z0 = z_enc_norm[0]
    v0 = initial_velocity(z_enc_norm)
    p_sim = get_sim_pressure(lnn, src_dataset, s, T_total, device)
    z_sim_norm = simulate_rk4(lnn, z0, v0, n_steps=T_total, dt=1.0,
                              pressure=p_sim)            # (T_total, D)

    # z pour le decodeur : meme espace que pendant l'entrainement
    # Le decodeur attend z blanchi = z_sim_norm directement
    z_sim_dec = z_sim_norm                              # (T_total, D)

    # ── 3. Decoder toutes les frames simulees en batch ────────────────────
    print('Decodage GS de la trajectoire simulee...')
    decoded_frames = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, T_total, batch_size):
            z_batch = z_sim_dec[i:i + batch_size].to(device)
            imgs = decoder_gs(z_batch).cpu().numpy()   # (B, C, H, W)
            decoded_frames.append(imgs)
    decoded_frames = np.concatenate(decoded_frames, axis=0)  # (T_total, C, H, W)

    # ── 4. Préparer le writer vidéo ───────────────────────────────────────
    H, W = src_dataset.img_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (int(W * 3), int(H)))
    assert writer.isOpened(), f"Impossible d'ouvrir VideoWriter sur {output_path}"

    z_enc_np = z_enc_norm.cpu().numpy()    # (T_orig, D) normalisés pour le plot
    z_sim_np = z_sim_norm.cpu().numpy()    # (T_total, D)
    D        = z_enc_np.shape[1]

    def _frame_to_bgr(frame_np):
        """(C, H, W) float32 → BGR uint8 (H, W, 3)."""
        img = np.clip(frame_np, 0.0, 1.0)
        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        img = (img * 255).astype(np.uint8).transpose(1, 2, 0)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _make_latent_panel(t):
        t_orig = min(t, T_orig - 1)
        fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)

        if D >= 2:
            ax.plot(z_enc_np[:, 0], z_enc_np[:, 1],
                    'b-', alpha=0.2, linewidth=0.7)
            ax.plot(z_sim_np[:, 0], z_sim_np[:, 1],
                    'r--', alpha=0.15, linewidth=0.7)
            ax.plot(z_sim_np[:t+1, 0], z_sim_np[:t+1, 1],
                    'r-', alpha=0.6, linewidth=1.0)
            if T_total > T_orig and t >= T_orig:
                ax.scatter(z_enc_np[-1, 0], z_enc_np[-1, 1],
                           s=80, color='blue', marker='X', zorder=8,
                           label='fin orig.')
            ax.scatter(z_enc_np[t_orig, 0], z_enc_np[t_orig, 1],
                       s=50, color='blue', zorder=6,
                       label='enc(x)' + (' [figé]' if t >= T_orig else ''))
            ax.scatter(z_sim_np[t, 0], z_sim_np[t, 1],
                       s=50, color='red', zorder=6, label='RK4')
            ax.set_xlabel('z[0]', fontsize=6)
            ax.set_ylabel('z[1]', fontsize=6)
        else:
            ax.plot(z_enc_np[:, 0], 'b-', alpha=0.4, linewidth=0.7, label='enc(x)')
            ax.plot(z_sim_np[:, 0], 'r--', alpha=0.4, linewidth=0.7, label='RK4')
            if T_total > T_orig:
                ax.axvline(T_orig - 1, color='gray', linewidth=0.8,
                           linestyle=':', label='fin orig.')
            ax.scatter(t_orig, z_enc_np[t_orig, 0], s=40, color='blue', zorder=5)
            ax.scatter(t,      z_sim_np[t, 0],      s=40, color='red',  zorder=5)

        title = f't={t}'
        if T_total > T_orig and t >= T_orig:
            title += f'  [+{t - T_orig + 1} extrap.]'
        ax.set_title(title, fontsize=7)
        ax.legend(fontsize=5, loc='upper right', markerscale=0.8)
        ax.tick_params(labelsize=5)
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()

        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, :3].copy()
        plt.close(fig)
        buf = cv2.resize(buf, (W, H), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

    # ── 5. Générer frame par frame ────────────────────────────────────────
    extra = max(0, T_total - T_orig)
    print(f'Génération vidéo GS ({T_total} frames'
          + (f', dont {extra} extrapolées' if extra else '')
          + f') → {output_path}')

    for t in range(T_total):
        orig_t     = min(t, T_orig - 1)
        panel_orig = _frame_to_bgr(disp_frames_np[s + orig_t])
        if t >= T_orig:
            panel_orig = panel_orig.copy()
            cv2.putText(panel_orig, 'EXTRAPOLE', (4, H - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 80, 255), 1,
                        cv2.LINE_AA)

        panel_gs     = _frame_to_bgr(decoded_frames[t])
        panel_latent = _make_latent_panel(t)

        combined = np.concatenate([panel_orig, panel_gs, panel_latent], axis=1)
        writer.write(combined)

        if (t + 1) % 50 == 0:
            print(f'  {t+1}/{T_total}')

    writer.release()
    print(f'Vidéo GS sauvegardée : {output_path}')
