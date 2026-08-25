"""
compute_latent_whiten.py — ajuste le blanchiment latent POST-HOC (figé) après l'AE.

À partir de l'encodeur GELÉ de l'autoencodeur conjoint (`encoder_ae.pt`, repli
`encoder.pt`), encode TOUTES les frames du dataset une fois → z_all, puis ajuste une
`LatentWhiten` (cf. models.py) sur les stats globales : u = (z − μ) W décorrèle et
égalise les variances latentes (mode 'pca' ou 'zca'). La transformée est ENSUITE GELÉE
et sauvée → tout l'aval (LNN, métrique, génération) travaille en espace u équilibré,
SANS réentraîner encodeur ni décodeur.

C'est l'étape qui « rajoute un blanchiment une fois l'encodeur entraîné » :
    train_ae  →  compute_latent_whiten  →  precompute_metric_geom  →  train_lnn_fixedae

Opt-in : ne fait quelque chose que si config.LATENT_WHITEN=True (sinon avertit et sort).

Lancer  :  py compute_latent_whiten.py [--config ../<cas>/config.py]
Produit :  <SAVE_DIR>/latent_whiten.pt   (state_dict de LatentWhiten : mean, W, W_inv)
"""
import argparse
import numpy as np
import torch

from _bootstrap import load_config
config = load_config()

from dataset import VideoFrameDataset
from models import build_encoder, LatentWhiten


@torch.no_grad()
def encode_all(encoder, frames_np, device, bs=128):
    """Encode toutes les frames une fois → (N, d) sur CPU (encodeur en eval)."""
    encoder.eval()
    zs = []
    for i in range(0, len(frames_np), bs):
        x = torch.from_numpy(frames_np[i:i + bs]).to(device)
        if x.dtype == torch.uint8:        # frames stockées en uint8 [0,255] (store_uint8)
            x = x.float().div_(255.0)
        zs.append(encoder(x).cpu())
    return torch.cat(zs, 0)


def main():
    if not getattr(config, 'LATENT_WHITEN', False):
        print('LATENT_WHITEN=False → rien à faire (active le flag pour ajuster le '
              'blanchiment latent post-hoc). Sortie.')
        return

    config.SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # ── Données (mêmes réglages que train_ae / train_lnn_fixedae) ────────────
    n_ch_enc = 3 if config.ENC_COLOR else 1
    # store_uint8=True : frames uint8 [0,255] (~4× moins de RAM ; conversion float
    # par batch dans encode_all). Indispensable en 256×256 (float32 ≈ 41 GB → OOM).
    fds = VideoFrameDataset(
        video_dir=config.VIDEO_DIR, img_size=config.IMG_SIZE, n_channels=n_ch_enc,
        crop=getattr(config, 'CROP', None),
        rest_video=config.REST_VIDEO, rest_n_frames=config.REST_N_FRAMES,
        rest_first_n_frames=getattr(config, 'REST_FIRST_N_FRAMES', 0),
        exclude_videos=[config.VAL_VIDEO] if config.VAL_VIDEO else None,
        store_uint8=True)

    # ── Encodeur figé (AE conjoint en priorité) ──────────────────────────────
    enc = build_encoder(config.IMG_SIZE, config.ENC_HIDDEN, config.LATENT_DIM,
                        n_channels=n_ch_enc,
                        normalize=getattr(config, 'ENC_NORMALIZE', False)).to(device)
    enc_path = config.SAVE_DIR / 'encoder_ae.pt'
    if not enc_path.exists():
        enc_path = config.SAVE_DIR / 'encoder.pt'
    assert enc_path.exists(), \
        f'encodeur introuvable : ni encoder_ae.pt ni encoder.pt dans {config.SAVE_DIR}'
    enc.load_state_dict(torch.load(enc_path, map_location=device))
    enc.eval()
    print(f'Encodeur chargé et figé : {enc_path}')

    # ── Encodage complet + ajustement de la transformée ──────────────────────
    z_all = encode_all(enc, fds.frames, device)              # (N, d) CPU
    mode = getattr(config, 'LATENT_WHITEN_MODE', 'pca')
    eps  = getattr(config, 'LATENT_WHITEN_EPS', 1e-6)
    whiten = LatentWhiten(config.LATENT_DIM).fit(z_all, mode=mode, eps=eps)

    u_all = whiten(z_all)                                    # (N, d) blanchi
    std_z = z_all.std(0).numpy()
    std_u = u_all.std(0).numpy()
    print(f'{len(z_all)} frames encodées, d={config.LATENT_DIM}, mode={mode!r}, eps={eps:g}')
    print(f'  std(z) brut   = {np.array2string(std_z, precision=4)}'
          f'   (min/max = {std_z.min():.4f}/{std_z.max():.4f}, '
          f'ratio = {std_z.max()/max(std_z.min(),1e-12):.1f})')
    print(f'  std(u) blanchi= {np.array2string(std_u, precision=4)}'
          f'   (min/max = {std_u.min():.4f}/{std_u.max():.4f}, '
          f'ratio = {std_u.max()/max(std_u.min(),1e-12):.1f})')

    out = config.SAVE_DIR / 'latent_whiten.pt'
    torch.save(whiten.state_dict(), out)
    print(f'Blanchiment latent sauvegardé : {out}')
    print('Étapes suivantes : compute_visibility_metric.py puis train_lnn_krauss.py '
          '(cf. scripts/run_2seg_npz.py).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='Config propre au cas test (cf. _bootstrap.load_config)')
    parser.parse_args()
    main()
