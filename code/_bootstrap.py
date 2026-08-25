"""
Pipeline 2 — Chargement de la configuration par cas test (base + overrides).

``code/config.py`` est la **config de base** : elle contient TOUTES les
clés, avec les valeurs par défaut (cas Krauss). Chaque cas test n'a, dans son dossier,
qu'un ``config.py`` d'**overrides** : il ne (re)définit QUE les clés qui diffèrent de la
base (vidéo, fps, DDL, profondeur de scène…). On le sélectionne par invocation via
``--config ../<cas>/config.py``, ce qui permet de lancer plusieurs cas en parallèle sans
collision (sorties isolées dans ``<cas>/checkpoints/``).

Usage dans un script d'entrée — AVANT tout import de projet (``models``, ``dataset``,
``viz``, ``generate_video``…), car ``models.py`` fait ``import config`` au chargement :

    from _bootstrap import load_config
    config = load_config()

Sans ``--config``, on retombe sur la base seule (comportement par défaut = Krauss).

Mécanique des overrides : le fichier de cas est exécuté dans un namespace où ``_HERE``
(son propre dossier) et ``Path`` sont déjà injectés, puis chaque nom qu'il définit est
posé sur le module de base. Les valeurs **dérivées** (``LATENT_DIM``, ``SEQ_LEN``) sont
recalculées APRÈS application des overrides, et ``SAVE_DIR`` est ancré sur le dossier du
cas (sauf si le fichier de cas le fixe explicitement). Le module résultant est enregistré
sous ``config`` dans ``sys.modules`` : tout ``import config`` ultérieur résout vers lui.
"""
import sys
import argparse
from pathlib import Path

# ── Encodage console UTF-8 ───────────────────────────────────────────────────
# Sous Windows la console est en cp1252 : tout print contenant un caractère non
# ASCII (p. ex. « λ_J » dans train_encoder) lève UnicodeEncodeError et tue le
# script. _bootstrap étant importé en tête de TOUS les scripts d'entrée, on
# force ici l'UTF-8 une fois pour toutes (équivaut à PYTHONIOENCODING=utf-8).
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, 'reconfigure', None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding='utf-8')
        except Exception:
            pass


def _apply_overrides(mod, path):
    """Exécute le fichier d'overrides ``path`` et pose ses clés sur le module ``mod``."""
    ns = {'__file__': str(path), '__name__': 'config_override',
          '_HERE': path.parent, 'Path': Path}
    code = compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    exec(code, ns)
    for key, val in ns.items():
        if key.startswith('__') or key in ('Path', '_HERE'):
            continue
        setattr(mod, key, val)

    # ── Valeurs dérivées : à recalculer après les overrides (cf. config.py) ──────
    mod.LATENT_DIM = mod.TSNE_DIM
    mod.SEQ_LEN = {2: 3, 4: 5}[mod.LNN_FD_ORDER]

    # ── Isolation des sorties : ancrage sur le dossier du cas test ──────────────
    if 'SAVE_DIR' not in ns:
        mod.SAVE_DIR = path.parent / 'checkpoints'


def load_config():
    """Parse ``--config`` depuis sys.argv, charge la base + overrides, enregistre le module.

    Idempotent : si une config a déjà été chargée dans ce process, on la réutilise
    (un seul objet module → les attributs posés après coup, p. ex.
    ``config._enc_checkpoint_override``, survivent et sont partagés).
    """
    existing = sys.modules.get('config')
    if existing is not None and getattr(existing, '_config_loaded', False):
        return existing

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default=None)
    args, _ = parser.parse_known_args()   # tolère les autres flags propres à chaque script

    import config as mod                   # base : code/config.py (défauts Krauss)

    if args.config:
        path = Path(args.config).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--config : fichier introuvable : {path}")
        _apply_overrides(mod, path)
        print(f"[config] base + overrides : {path}")
    else:
        print("[config] base (code/config.py)")

    mod._config_loaded = True
    sys.modules['config'] = mod
    return mod
