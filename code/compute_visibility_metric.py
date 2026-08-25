"""
compute_visibility_metric.py — précalcule la MÉTRIQUE DE VISIBILITÉ du décodeur GS

    Ḡ = E_u[ (∂I/∂u)ᵀ (∂I/∂u) ]        (d×d, SPD, espace latent BLANCHI)

et la sauve dans `<SAVE_DIR>/visibility_metric.pt` pour servir de métrique à la loss
de rollout (`finetune_lnn_fixedae.py`, clé `LNN_ROLLOUT_METRIC`).

Motivation
----------
Toutes les losses du pipeline (résidu FD, rollout) mesurent l'écart latent en norme
EUCLIDIENNE : `‖Δu‖²`. En espace blanchi, chaque direction latente y pèse donc
exactement pareil, puisque `LatentWhiten` impose une covariance ≈ I. Or « variance
unité » n'est pas « visibilité unité » : une direction peut être de variance 1 dans
les données tout en ne déplaçant quasiment aucun pixel.

Le terme de reconstruction décodée de Krauss et al. 2026 (VON) corrige exactement
ça : au premier ordre,

    ‖I(û) − I(u)‖²  ≈  Δuᵀ G(u) Δu ,      G(u) = (∂I/∂u)ᵀ (∂I/∂u)

c'est-à-dire la MÊME MSE latente, mais en métrique `G` (le pull-back de la métrique
pixel) au lieu de l'identité. Les modes peu visibles ont un petit `λ(G)` et sont donc
naturellement dépondérés. Deux justifications, pas une :
  • la métrique d'évaluation est une MSE image, donc train et éval s'alignent ;
  • un mode peu visible est peu OBSERVABLE, donc son q(t) encodé est dominé par le
    bruit ; pondérer par `G` revient à pondérer par l'inverse de la variance de bruit,
    ce que ferait un estimateur du maximum de vraisemblance.

Précalculer `Ḡ` évite de rasteriser dans le graphe BPTT (`H` rendus gsplat par fenêtre
serait prohibitif) : la loss reste une forme quadratique `d×d`, de coût nul.

Loi d'échantillonnage de `u`
----------------------------
  • `--source data`   : `u = whiten(enc(frames))` sur un sous-échantillon de frames.
    Le plus fidèle, demande la vidéo (ou son framecache).
  • `--source normal` : `u ~ N(0, I)`. Justifié PAR CONSTRUCTION, `LatentWhiten` étant
    ajustée sur les stats globales du dataset (covariance ≈ I). Utile sans les images.
    ⚠ à `d` élevé la loi jointe n'est pas gaussienne : des tirages tombent hors-variété,
    là où le décodeur n'est pas fini. Ils sont REJETÉS et comptés.
  • `--source auto` (défaut) : `data` si les frames se chargent, sinon `normal`.

Note : la résolution de rendu ne change que l'échelle globale de `Ḡ` (∝ nb de pixels),
pas son spectre RELATIF, seul utilisé en aval (la métrique est normalisée à trace = d).

Lancer :  py compute_visibility_metric.py --config ../cases/krauss2026_2seg_npz/config.py
Prérequis : decoder2dpt_ae.pt (+ latent_whiten.pt) dans SAVE_DIR.
"""
import argparse

from _bootstrap import load_config
config = load_config()

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--res', type=int, default=None,
                 help='résolution de rendu pour la jacobienne (déf. VIS_METRIC_RES=64)')
_ap.add_argument('--samples', type=int, default=None,
                 help='nb de points u (déf. VIS_METRIC_SAMPLES=64)')
_ap.add_argument('--source', choices=['auto', 'data', 'normal'], default=None,
                 help='loi de u (déf. VIS_METRIC_SOURCE=auto)')
_ap.add_argument('--decoder', default=None,
                 help='checkpoint décodeur (déf. decoder2dpt_ae.pt, repli decoder2dpt.pt)')
_ap.add_argument('--seed', type=int, default=0)
_ap.add_argument('--torch-decoder', action='store_true',
                 help='force le décodeur torch pur (GaussianSplatDecoder2pt) au lieu de '
                      'gsplat : REQUIS pour la jacobienne jvp (gsplat ne supporte pas les '
                      'transforms functorch). Spectre RELATIF quasi identique (compositing '
                      'à peine différent).')
_args, _ = _ap.parse_known_args()

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import build_encoder, WhitenedEncoder, load_latent_whiten
from models_2pt import GaussianSplatDecoder2pt


def _cfg(name, default):
    return getattr(config, name, default)


def _build_decoder(device, res, n_ch, ckpt_name):
    """Décodeur GS GELÉ, rendu à `res`. gsplat si disponible (= rendu d'entraînement),
    sinon la classe parente torch pur (mêmes paramètres, compositing légèrement
    différent ; sans effet notable sur le spectre RELATIF)."""
    path = config.SAVE_DIR / ckpt_name
    if not path.exists():
        path = config.SAVE_DIR / 'decoder2dpt.pt'
    assert path.exists(), f'décodeur introuvable dans {config.SAVE_DIR}'
    state = torch.load(path, map_location=device)
    K = state['mu_raw'].shape[0]

    # Repli torch pur géré par la fabrique : elle avertit bruyamment (le repli
    # est lent ET les poids ne sont pas interchangeables entre compositings).
    # `--torch-decoder` force le repli sans avertissement.
    from models_2pt import decoder2pt_class
    backend = 'torch' if _args.torch_decoder else getattr(
        config, 'DEC2PT_BACKEND', 'auto')
    cls = decoder2pt_class(backend)
    tag = 'gsplat' if cls is not GaussianSplatDecoder2pt else 'torch'

    dec = cls(latent_dim=config.LATENT_DIM, n_gaussians=K,
              img_size=config.IMG_SIZE, n_channels=n_ch).to(device)
    dec.load_state_dict(state)
    # Rebascule la grille de pixels à `res` : les paramètres GS sont en coordonnées
    # normalisées [0,1], donc indépendants de la résolution.
    ys = torch.linspace(0, 1, res, device=device)
    xs = torch.linspace(0, 1, res, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    dec.grid = torch.stack([gx, gy], dim=-1)
    dec.img_size = (res, res)
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    print(f'Décodeur GS GELÉ : {path.name} ({K} gaussiennes), rendu {res}×{res} [{tag}]')
    return dec


def _sample_u_from_data(device, n, whiten):
    """u = whiten(enc(frames)) sur `n` frames tirées au hasard. Lève si indisponible."""
    from dataset import VideoFrameDataset
    n_ch = 3 if _cfg('ENC_COLOR', True) else 1
    fds = VideoFrameDataset(
        video_dir=config.VIDEO_DIR, img_size=config.IMG_SIZE, n_channels=n_ch,
        rest_video=config.REST_VIDEO, rest_n_frames=config.REST_N_FRAMES,
        rest_first_n_frames=_cfg('REST_FIRST_N_FRAMES', 0), crop=_cfg('CROP', None),
        exclude_videos=[config.VAL_VIDEO] if config.VAL_VIDEO else None,
        load_pressure=False, store_uint8=True)
    enc = build_encoder(config.IMG_SIZE, config.ENC_HIDDEN, config.LATENT_DIM,
                        n_channels=n_ch, normalize=_cfg('ENC_NORMALIZE', False)).to(device)
    p_enc = config.SAVE_DIR / 'encoder_ae.pt'
    if not p_enc.exists():
        p_enc = config.SAVE_DIR / 'encoder.pt'
    enc.load_state_dict(torch.load(p_enc, map_location=device))
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    if whiten is not None:
        enc = WhitenedEncoder(enc, whiten).to(device).eval()

    idx = torch.randperm(len(fds.frames))[:n]
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 32):
            x = torch.from_numpy(np.ascontiguousarray(fds.frames[idx[i:i + 32].numpy()]))
            x = x.to(device)
            if x.dtype == torch.uint8:
                x = x.float().div_(255.0)
            out.append(enc(x))
    print(f'Source u : {n} frames encodées ({p_enc.name})')
    return torch.cat(out)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(_args.seed)
    np.random.seed(_args.seed)

    res = _args.res if _args.res is not None else int(_cfg('VIS_METRIC_RES', 64))
    ns = _args.samples if _args.samples is not None else int(_cfg('VIS_METRIC_SAMPLES', 64))
    source = _args.source or _cfg('VIS_METRIC_SOURCE', 'auto')
    ckpt = _args.decoder or 'decoder2dpt_ae.pt'
    d = config.LATENT_DIM
    n_ch = 3 if _cfg('DEC_COLOR', True) else 1

    print(f'=== métrique de visibilité — d={d}, rendu {res}×{res}×{n_ch}, '
          f'{ns} points, device {device} ===')

    dec = _build_decoder(device, res, n_ch, ckpt)
    whiten = load_latent_whiten(config.SAVE_DIR, device, d)
    print(f'LatentWhiten : {"chargé" if whiten is not None else "ABSENT (latent brut)"}')

    # ── Points d'évaluation u ────────────────────────────────────────────────
    U = None
    if source in ('auto', 'data'):
        try:
            U = _sample_u_from_data(device, ns, whiten)
        except Exception as e:
            if source == 'data':
                raise
            print(f'Source "data" indisponible ({type(e).__name__}: {e}) → repli N(0,I).')
    if U is None:
        print(f'Source u : N(0, I) — justifié par le blanchiment (cov ≈ I).')
        U = torch.randn(4 * ns, d, device=device)   # marge pour les rejets
        source = 'normal'
    else:
        source = 'data'

    def f(u):
        """u : (d,) latent blanchi → image aplatie (C·res·res,), comme le décodage
        de la loss (whiten⁻¹ puis décodeur)."""
        z = whiten.inverse(u.unsqueeze(0)) if whiten is not None else u.unsqueeze(0)
        return dec(z).reshape(-1)

    def jacobian_fwd(u):
        """J = ∂f/∂u : (C·res·res, d), en mode DIRECT (d jvp). Le mode inverse
        demanderait C·res·res backward, soit des téraoctets."""
        cols = []
        for j in range(d):
            e = torch.zeros(d, device=u.device)
            e[j] = 1.0
            with torch.no_grad():
                _, col = torch.func.jvp(f, (u,), (e,))
            cols.append(col)
        return torch.stack(cols, dim=1)

    # ── Ḡ = E[JᵀJ] ───────────────────────────────────────────────────────────
    G = torch.zeros(d, d, dtype=torch.float64, device=device)
    per_sample, n_ok, n_rej = [], 0, 0
    for u in U:
        if n_ok >= ns:
            break
        J = jacobian_fwd(u).double()
        if not torch.isfinite(J).all():
            n_rej += 1                     # hors-variété : décodeur non fini
            continue
        G += J.T @ J
        # λ(JᵀJ) via les valeurs singulières de J : stable même quand une direction
        # est quasi invisible (eigh échoue en float32 sur ces matrices).
        per_sample.append(np.sort(torch.linalg.svdvals(J).cpu().numpy() ** 2))
        n_ok += 1
        if n_ok % 16 == 0:
            print(f'  {n_ok}/{ns} points ({n_rej} rejetés)')
    assert n_ok > 0, 'aucun point exploitable (décodeur non fini partout).'
    G /= n_ok
    G = G.cpu()

    evals, evecs = torch.linalg.eigh(G)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    ev = evals.numpy()
    CHW = n_ch * res * res

    # ── Rapport ──────────────────────────────────────────────────────────────
    print(f'\n--- spectre de Ḡ ({n_ok} points, {n_rej} rejetés) ---')
    print(f'{"mode":>5} {"λ":>13} {"λ/λmax":>11} {"RMS pixel / 1σ":>16}')
    for i, lam in enumerate(ev):
        print(f'{i:>5} {lam:13.5g} {lam / ev[0]:11.4g} '
              f'{np.sqrt(max(lam, 0.0) / CHW):16.5g}')
    cond = ev[0] / max(ev[-1], 1e-30)
    print(f'\nconditionnement λmax/λmin : {cond:.4g}')
    ridge = float(_cfg('LNN_ROLLOUT_METRIC_RIDGE', 0.01))
    print(f'avec la ridge par défaut ({ridge:g}·λmax) le rapport de pondération '
          f'retombe à {(ev[0] * (1 + ridge)) / (ev[-1] + ridge * ev[0]):.4g}')

    # ── Sauvegarde + plot ────────────────────────────────────────────────────
    out = config.SAVE_DIR / 'visibility_metric.pt'
    torch.save({'G': G.float(), 'evals': evals.float(), 'evecs': evecs.float(),
                'latent_dim': d, 'res': res, 'n_channels': n_ch,
                'n_samples': n_ok, 'n_rejected': n_rej, 'source': source,
                'decoder': ckpt}, out)
    print(f'\nMétrique sauvegardée : {out}')

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(len(ev)), np.maximum(ev, 1e-30), 'o-', color='tab:blue',
                label='λ(Ḡ) — visibilité')
    ax.axhline(ridge * ev[0], color='tab:red', ls='--',
               label=f'ridge {ridge:g}·λmax')
    ax.set_xlabel('mode latent'); ax.set_ylabel('λ (échelle log)')
    ax.set_title(f'Métrique de visibilité — {config.SAVE_DIR.parent.name} '
                 f'(cond {cond:.0f})')
    ax.grid(True, alpha=0.3, which='both'); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.SAVE_DIR / 'visibility_metric.png', dpi=130)
    plt.close(fig)
    print(f'Plot du spectre : {config.SAVE_DIR / "visibility_metric.png"}')


if __name__ == '__main__':
    main()
