"""
fetch_data.py — recupere les NPZ prétraités de Krauss et al. 2026 (Zenodo),
verifie leur empreinte, et reconstruit le split 80/20 officiel.

Licence des donnees : CC BY-ND 4.0. « ND » interdit la redistribution d'oeuvres
derivees : le split produit ici n'est donc NI versionne NI publie, il est
reconstruit localement de facon deterministe (les donnees sont CC BY-ND).

Usage :
    py scripts/fetch_data.py --seg 2seg                 # telecharge depuis Zenodo
    py scripts/fetch_data.py --seg 2seg --zip <chemin>  # zip deja telecharge
    py scripts/fetch_data.py --seg 2seg --no-split      # extraction seule
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ZENODO_RECORD = '17812071'                  # DOI 10.5281/zenodo.17812071
ARCHIVE = 'processed_data.zip'              # ~4.5 GB

# Empreintes des fichiers effectivement consommes (mesurees localement).
MEMBERS = {
    '1seg': ('processed_data/scr_dataset_raw_1seg_32x32_59fps.npz',
             '3e57695054b5025bb2bfebd83ccbf46546d28857eb40ada2918bbf060b692164'),
    '2seg': ('processed_data/scr_dataset_raw_2seg_32x32_59fps.npz',
             'dd2d8508cffb30284e46a375fc662d168c1f50b759445a4eaf601734293e82bc'),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def zenodo_url() -> str:
    api = f'https://zenodo.org/api/records/{ZENODO_RECORD}'
    with urllib.request.urlopen(api) as r:
        rec = json.load(r)
    for f in rec.get('files', []):
        key = f.get('key') or f.get('filename')
        if key == ARCHIVE:
            link = f.get('links', {})
            return link.get('self') or link.get('download')
    raise SystemExit(f'{ARCHIVE} introuvable dans le record Zenodo {ZENODO_RECORD}.')


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    print(f'telechargement de {ARCHIVE} (~4.5 GB) depuis Zenodo...')
    with urllib.request.urlopen(url) as r, tmp.open('wb') as out:
        total = int(r.headers.get('Content-Length') or 0)
        done = 0
        while chunk := r.read(1 << 22):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f'\r  {done / total:6.1%}', end='', flush=True)
    print()
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg', choices=['1seg', '2seg'], default='2seg')
    ap.add_argument('--zip', type=Path, default=None,
                    help='processed_data.zip deja present localement')
    ap.add_argument('--no-split', action='store_true')
    args = ap.parse_args()

    member, digest = MEMBERS[args.seg]
    raw_dir = REPO / 'data' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    npz = raw_dir / Path(member).name

    if npz.is_file() and sha256(npz) == digest:
        print(f'{npz.name} deja present et conforme.')
    else:
        zip_path = args.zip or (REPO / 'downloads' / ARCHIVE)
        if not zip_path.is_file():
            download(zenodo_url(), zip_path)
        print(f'extraction de {member}')
        with zipfile.ZipFile(zip_path) as z, z.open(member) as src, \
                npz.open('wb') as out:
            shutil.copyfileobj(src, out, 1 << 22)

        got = sha256(npz)
        if got != digest:
            print(f'ERREUR d\'empreinte.\n  attendu : {digest}\n  obtenu  : {got}')
            print('Le contenu Zenodo a change, ou le telechargement est corrompu.')
            return 1
        print(f'{npz.name} extrait, SHA256 conforme.')

    if args.no_split:
        return 0

    # Le split est une oeuvre derivee (CC BY-ND) : reconstruit, jamais publie.
    case = REPO / 'cases' / f'krauss2026_{args.seg}_npz'
    out_dir = case / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'\nreconstruction du split 80/20 officiel dans {out_dir}')
    cmd = [sys.executable, str(REPO / 'code' / 'make_krauss_npz_split.py'),
           '--seg', args.seg, '--src-dir', str(raw_dir), '--out-dir', str(out_dir)]
    print('+', ' '.join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc:
        print('\nATTENTION : make_krauss_npz_split.py a echoue. Si les options --src-dir '
              '/ --out-dir ne sont pas reconnues, lancer le script a la main : '
              'il a ses propres chemins par defaut.')
    return rc


if __name__ == '__main__':
    sys.exit(main())
