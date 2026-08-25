"""
run_2seg_npz.py — protocole complet du cas Krauss 2 segments, sur les NPZ et le
split officiels des auteurs.

Chaine :
    0. donnees      scripts/fetch_data.py --seg 2seg      (a lancer une fois)
    1. AE           train_ae.py                           (decodeur GS latent)
    2. blanchiment  compute_latent_whiten.py              (PCA figee post-AE)
    3. visibilite   compute_visibility_metric.py          (metrique G du decodeur)
    4. LNN          train_lnn_krauss.py  500 ep, lr 1e-3, c0 = 1, sigma 1,
                    graine 0, entraine DE ZERO a AE gele
    5. evaluation   eval_multistep_mse.py       MSE a 0.5 s
    6. evaluation   eval_freq_rest_cycles.py    erreur de frequence

Usage :
    py scripts/run_2seg_npz.py                 # tout
    py scripts/run_2seg_npz.py --from 4        # reprendre a l'etape 4
    py scripts/run_2seg_npz.py --only 5        # une seule etape
    py scripts/run_2seg_npz.py --dry-run       # afficher les commandes

Les etapes 1 et 4 sont longues (plusieurs heures GPU). Interpreteur : celui qui
dispose de torch + gsplat (`py` sous Windows).
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CODE = REPO / 'code'
CFG = '../cases/krauss2026_2seg_npz/config.py'
VAL = 'scr_2seg_32x32_val20.npz'

# Point de fonctionnement du modele rapporte. Les reglages d'ENTRAINEMENT du LNN
# (sigma, c0) sont repasses tels quels aux deux evaluations : ce sont des
# composantes de la dynamique simulee, pas des reglages d'affichage, et un
# checkpoint evalue a un autre c0 ne decrit plus le meme systeme.
LNN = 'lnn_lr1e-3_c1_s1_500ep_seed0.pt'
LR, C0, SIGMA, EPOCHS, SEED = '1e-3', '1', '1', '500', '0'

# Periode de reference du critere de frequence, IMPOSEE (--period-s) et non
# deduite de q_enc, pour que l'horizon soit identique d'un modele a l'autre.
PERIOD_S, N_PERIODS = '0.6870', '2'

# --frames 20000 : le defaut (6000) tronque le held-out a 55 % de ses 10 968
# frames, et la seconde moitie est plus facile (facteur 1.74 mesure sur cette
# chaine). 50 fenetres minimum : a 20, une seule fenetre dominante deplace le
# plancher AE de 21 %.
FRAMES, N_WIN = '20000', '50'

# (numero, libelle, argv) ; --config est consomme par code/_bootstrap.py
STEPS = [
    (1, 'autoencodeur GS latent (long)',
     ['train_ae.py', '--config', CFG]),

    (2, 'blanchiment latent PCA fige',
     ['compute_latent_whiten.py', '--config', CFG]),

    (3, 'metrique de visibilite G du decodeur',
     ['compute_visibility_metric.py', '--config', CFG, '--source', 'data']),

    (4, 'LNN, 500 epoques, lr 1e-3, c0 = 1, sigma 1, graine 0 (long)',
     ['train_lnn_krauss.py', '--config', CFG,
      '--metric-latent', '--w-dec', '0',
      '--sigma', SIGMA, '--sigma-pressure', SIGMA,
      '--lr', LR, '--cq-eps', C0, '--epochs', EPOCHS,
      '--plot-every', '25', '--seed', SEED, '--out', LNN]),

    (5, 'evaluation : MSE multi-step 0.5 s',
     ['eval_multistep_mse.py', '--config', CFG, '--lnn', LNN,
      '--video', VAL, '--sigma', SIGMA, '--sigma-pressure', SIGMA,
      '--cq-eps', C0, '--frames', FRAMES, '--n-traj', N_WIN, '--seed', '0',
      '--out', 'multistep_seed0.png']),

    # Le montage NPZ etant le `smooth_input`, a pression continument variable, il
    # ne contient quasiment pas d'intervalle a pression constante : le critere de
    # frequence porte ici sur UNE fenetre libre. C'est un point, pas une mesure
    # (voir README).
    (6, 'evaluation : erreur de frequence en oscillation libre',
     ['eval_freq_rest_cycles.py', '--config', CFG, '--lnn', LNN,
      '--video', VAL, '--sigma', SIGMA, '--sigma-pressure', SIGMA,
      '--cq-eps', C0, '--frames', FRAMES,
      '--n-periods', N_PERIODS, '--period-s', PERIOD_S,
      '--n-windows', N_WIN, '--seed', '0', '--json', 'freq2T_seed0.json']),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='start', type=int, default=1)
    ap.add_argument('--only', type=int, default=None)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    data = REPO / 'cases' / 'krauss2026_2seg_npz' / 'data' / VAL
    if not data.is_file() and not args.dry_run:
        print(f'Donnees absentes : {data}\n'
              f'Lancer d\'abord :  py scripts/fetch_data.py --seg 2seg')
        return 2

    for num, label, argv in STEPS:
        if args.only is not None and num != args.only:
            continue
        if args.only is None and num < args.start:
            continue
        cmd = [sys.executable, '-u', *argv]
        print(f'\n=== etape {num} : {label}')
        print('+ (cd code)', ' '.join(cmd[2:]))
        if args.dry_run:
            continue
        rc = subprocess.run(cmd, cwd=CODE).returncode
        if rc:
            print(f'etape {num} en echec (code {rc}), arret.')
            return rc

    print('\nTermine. Chiffres attendus : README.md, section Results.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
