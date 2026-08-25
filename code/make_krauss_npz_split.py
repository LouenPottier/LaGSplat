"""
make_krauss_npz_split.py — découpe les NPZ prétraités de Krauss et al. 2026 en
`train` / `val` avec EXACTEMENT leur split, pour un head-to-head rigoureux.

Pourquoi
--------
Nos cas `krauss2026_{1,2}seg` s'entraînent sur NOS MP4 (re-encodés depuis les
Full HD) avec NOTRE alignement pression (`PRESSURE_SYNC_OFFSETS`) et NOTRE coupe
held-out. Krauss, lui, entraîne sur ses NPZ `scr_dataset_raw_<seg>_32x32_59fps.npz`
(images 32×32 RGB à 59.94 fps, pressions déjà alignées par l'événement de
dépressurisation) et sépare **les premiers 80 % en train, les derniers 20 % en
val** (`train_val_ratio: 0.80`, cellule 6 de `Latent_dynamics_learning.ipynb` :
`n_train = int(0.8 * n_total)`, découpe CONTIGUË).

Ce script matérialise ces deux moitiés en deux fichiers, que le pipeline traite
comme des vidéos (cf. `dataset.load_npz_frames` / `load_pressure_npz`) :
    <case>/data/scr_<seg>_32x32_train80.npz     (VIDEO_DIR)
    <case>/data/scr_<seg>_32x32_val20.npz       (VAL_VIDEO)

Les images sont stockées en **uint8** (les NPZ sources sont des float32 dont les
valeurs sont des multiples exacts de 1/255 — vérifié : conversion sans perte,
4× moins de disque). Les pressions sont recopiées telles quelles (déjà
normalisées par 101 325 Pa ⟹ `PRESSURE_NORM = 1.0` côté config).

Lancer :
    py make_krauss_npz_split.py --seg 2seg
    py make_krauss_npz_split.py --seg 1seg
"""
import argparse
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
SRC_DIR = _HERE.parent / 'data' / 'raw'
CASE_DIR = {seg: _HERE.parent / 'cases' / f'krauss2026_{seg}_npz'
            for seg in ('1seg', '2seg')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg', choices=['1seg', '2seg'], required=True)
    ap.add_argument('--res', type=int, default=32)
    ap.add_argument('--ratio', type=float, default=0.80,
                    help='train_val_ratio de Krauss (configs/base.yaml : 0.80)')
    ap.add_argument('--out-dir', type=str, default=None)
    ap.add_argument('--src-dir', type=str, default=None,
                    help='dossier des NPZ Krauss bruts (defaut : celui '
                         'que pose scripts/fetch_data.py).')
    args = ap.parse_args()

    src_dir = Path(args.src_dir) if args.src_dir else SRC_DIR
    src = src_dir / f'scr_dataset_raw_{args.seg}_{args.res}x{args.res}_59fps.npz'
    assert src.exists(), f'NPZ source introuvable : {src}'
    out_dir = Path(args.out_dir) if args.out_dir else CASE_DIR[args.seg] / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(src)
    o = d['images']
    p_keys = sorted(k for k in d.files if k.startswith('p') and k != 'images')
    T = o.shape[0]
    n_train = int(args.ratio * T)        # int() comme eux (troncature)
    print(f'{src.name} : {T} frames, pressions {p_keys}')
    print(f'  train = [0, {n_train})   val = [{n_train}, {T})   (ratio {args.ratio})')

    # float32 [0,1] → uint8 sans perte (valeurs = k/255 dans les NPZ Krauss)
    o = np.asarray(o, dtype=np.float32)
    if o.max() <= 1.0:
        o255 = o * 255.0
        err = np.abs(o255 - np.rint(o255)).max()
        assert err < 1e-3, f'images non issues d\'uint8 (écart max {err:.2e}) — garder le float32'
        o8 = np.rint(o255).astype(np.uint8)
    else:
        o8 = o.astype(np.uint8)

    for tag, sl in (('train80', slice(0, n_train)), ('val20', slice(n_train, T))):
        payload = {'images': o8[sl]}
        for k in p_keys:
            payload[k] = np.asarray(d[k][sl], dtype=np.float32)
        out = out_dir / f'scr_{args.seg}_{args.res}x{args.res}_{tag}.npz'
        np.savez(out, **payload)
        print(f'  -> {out}  ({payload["images"].shape}, '
              f'{out.stat().st_size / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
