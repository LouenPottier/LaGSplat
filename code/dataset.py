"""
Pipeline 2 -- Datasets video.

VideoFrameDataset  : frames individuelles, pour encodeur / LNN / décodeur.
VideoSeqDataset    : sequences consecutives par video, pour LNN.

La rest_frame est calculée selon le mode choisi dans config :
  - REST_VIDEO fourni  + REST_FIRST_N_FRAMES > 0  → médiane des N premières frames de REST_VIDEO
  - REST_VIDEO fourni  + REST_FIRST_N_FRAMES = 0  → médiane des REST_N_FRAMES dernières frames de REST_VIDEO
  - REST_VIDEO = None  + REST_FIRST_N_FRAMES > 0  → médiane des N premières frames du dataset
  - REST_VIDEO = None  + REST_FIRST_N_FRAMES = 0  → médiane globale de toutes les frames

La rest_frame n'est jamais soustraite aux frames : elle sert uniquement à initialiser
z_rest dans le LNN (via train_lnn / train_all) et comme cible pour train_fluxopt.
"""
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


# -- Utilitaire ---------------------------------------------------------------

def _frame_cache_path(video_path: Path, img_size, n_channels, crop,
                      as_uint8=False) -> Path:
    """Chemin du cache .npy déduit de la vidéo + paramètres de décodage.

    `as_uint8` ajoute un suffixe ``_u8`` : le cache uint8 (frames brutes [0,255],
    ~4× plus léger) ne doit PAS entrer en collision avec le cache float32 [0,1]
    historique (même vidéo/img_size/n_channels/crop).
    """
    cropstr = 'none' if crop is None else 'x'.join(str(int(v)) for v in crop)
    dtypestr = '_u8' if as_uint8 else ''
    name = (f'{video_path.stem}.framecache_'
            f'{int(img_size[0])}x{int(img_size[1])}_c{n_channels}_{cropstr}{dtypestr}.npy')
    return video_path.with_name(name)


def _decode_video_frames(video_path: Path, img_size, n_channels, crop,
                         max_frames=None, as_uint8=False) -> np.ndarray:
    """Décode les frames, redimensionnées et normalisées [0, 1].

    Retourne directement le tableau final : (T, 1, H, W) gris ou (T, 3, H, W) RGB.
    `max_frames` (>0) : s'arrête après ce nombre de frames (évite de décoder toute
    une longue vidéo quand seul un préfixe est requis, p. ex. génération bornée).
    `as_uint8` : conserve les frames brutes uint8 [0,255] (pas de /255) — ~4× moins
    de RAM/disque ; l'appelant convertit en float [0,1] au moment voulu (par batch).
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        if max_frames is not None and len(frames) >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        # Crop fixe avant resize
        if crop is not None:
            x, y, w, h = crop
            frame = frame[y:y+h, x:x+w]

        if n_channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (img_size[1], img_size[0]))
            frames.append(frame if as_uint8 else frame.astype(np.float32) / 255.0)   # (H, W)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size[1], img_size[0]))
            frames.append(frame if as_uint8 else frame.astype(np.float32) / 255.0)   # (H, W, 3)
    cap.release()
    assert len(frames) > 0, f'Aucune frame chargee depuis {video_path}'
    arr = np.stack(frames)   # (T, H, W) ou (T, H, W, 3)
    if n_channels == 1:
        return np.ascontiguousarray(arr[:, np.newaxis, :, :])  # (T, 1, H, W)
    else:
        return np.ascontiguousarray(arr.transpose(0, 3, 1, 2)) # (T, 3, H, W)


def load_npz_frames(npz_path: Path, img_size, n_channels, max_frames=None,
                    as_uint8=False) -> np.ndarray:
    """Frames d'un dataset NPZ prétraité (clé `images`), au même format que
    `load_video_frames` : (T, C, H, W).

    Format Krauss (`processed_data.zip` → `scr_dataset_raw_<seg>_32x32_59fps.npz`) :
    `images` (T, 3, H, W) float32 déjà dans [0, 1], images et pressions DÉJÀ
    alignées temporellement par les auteurs (aucun `PRESSURE_SYNC_OFFSETS` à
    appliquer). Tolère aussi (T, H, W), (T, H, W, 3) et l'uint8 [0, 255].

    Pas de framecache : un `.npz` se relit déjà par `np.load` (mmap sur la clé),
    le cache `.npy` n'apporterait rien.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, mmap_mode='r') as d:
        assert 'images' in d.files, f'clé "images" absente de {npz_path.name}'
        o = d['images']
        o = o[:max_frames] if max_frames is not None else o[:]
    o = np.asarray(o)
    if o.ndim == 3:                                  # (T, H, W) → (T, 1, H, W)
        o = o[:, None]
    elif o.shape[-1] == 3 and o.shape[1] != 3:       # (T, H, W, 3) → (T, 3, H, W)
        o = np.transpose(o, (0, 3, 1, 2))
    o = o.astype(np.float32)
    if o.max() > 1.0:
        o = o / 255.0

    # Canaux : gris ← RGB par luminance (mêmes coefficients que cv2.COLOR_RGB2GRAY)
    if n_channels == 1 and o.shape[1] == 3:
        w = np.array([0.299, 0.587, 0.114], dtype=np.float32).reshape(1, 3, 1, 1)
        o = (o * w).sum(axis=1, keepdims=True)
    elif n_channels == 3 and o.shape[1] == 1:
        o = np.repeat(o, 3, axis=1)
    assert o.shape[1] == n_channels, \
        f'{npz_path.name} : {o.shape[1]} canaux, n_channels={n_channels} demandé'

    # Résolution : les NPZ sont déjà à la résolution cible dans le cas nominal.
    if (o.shape[2], o.shape[3]) != tuple(img_size):
        out = np.empty((o.shape[0], o.shape[1], img_size[0], img_size[1]),
                       dtype=np.float32)
        for t in range(o.shape[0]):
            f = cv2.resize(o[t].transpose(1, 2, 0), (img_size[1], img_size[0]))
            out[t] = f[:, :, None].transpose(2, 0, 1) if f.ndim == 2 else f.transpose(2, 0, 1)
        o = out

    if as_uint8:
        o = np.rint(o * 255.0).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(o)


def load_pressure_npz(npz_path: Path, n_frames: int, cols, norm: float = 1.0) -> np.ndarray:
    """Pression d'un dataset NPZ prétraité, DÉJÀ alignée sur les frames.

    `cols` = clés du NPZ (Krauss : `['p1','p2']` en 1-segment, `['p1',…,'p4']` en
    2-segments), chacune un vecteur (T,). Aucune interpolation ni `sync_offset` :
    l'alignement image↔pression est fait en amont par les auteurs (événement de
    dépressurisation, cf. `SCR_data_processing.ipynb`). `norm` vaut 1.0 pour les
    NPZ Krauss (pressions déjà divisées par 101325 Pa).

    Retourne (n_frames, n_c) float32.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, mmap_mode='r') as d:
        missing = [c for c in cols if c not in d.files]
        assert not missing, (f'Colonnes de pression absentes de {npz_path.name} : '
                             f'{missing} (disponibles : {list(d.files)})')
        p = np.stack([np.asarray(d[c][:n_frames], dtype=np.float32) for c in cols], axis=-1)
    assert p.shape[0] == n_frames, \
        f'{npz_path.name} : {p.shape[0]} échantillons de pression pour {n_frames} frames'
    return (p / float(norm)).astype(np.float32)


def load_video_frames(
    video_path: Path,
    img_size=(64, 64),
    n_channels=1,
    crop=None,          # (x, y, w, h) en pixels avant resize, ou None
    use_cache=True,     # cache .npy à côté de la vidéo (décodage ~50× plus lent)
    max_frames=None,    # >0 : ne charge/décode que les `max_frames` premières frames
    as_uint8=False,     # True : frames brutes uint8 [0,255] (~4× moins de RAM/disque)
) -> np.ndarray:
    """
    Charge toutes les frames d'une video normalisees [0, 1].
    n_channels=1 : niveaux de gris -> (T, 1, H, W)
    n_channels=3 : RGB             -> (T, 3, H, W)
    crop          : (x, y, w, h) crop fixe applique avant le resize, ou None

    Décoder une vidéo H.264 est ~50× plus lent que charger un .npy. On met donc en
    cache le tableau final *exact* (T, C, H, W) float32 : un cache-hit n'est alors
    qu'un `np.load` sans aucun post-traitement (le plus rapide ; privilégie la vitesse
    sur l'espace disque). Le cache est invalidé si la vidéo est plus récente que lui,
    et porté par les paramètres (img_size, n_channels, crop).
    """
    video_path = Path(video_path)
    if video_path.suffix.lower() == '.npz':
        # Dataset prétraité (frames déjà extraites) : pas de décodage, pas de cache.
        assert crop is None, 'crop non supporté sur une source .npz (frames déjà prêtes)'
        return load_npz_frames(video_path, img_size, n_channels,
                               max_frames=max_frames, as_uint8=as_uint8)
    cache_path = _frame_cache_path(video_path, img_size, n_channels, crop, as_uint8)

    if use_cache and cache_path.exists() and \
            cache_path.stat().st_mtime >= video_path.stat().st_mtime:
        try:
            if max_frames is not None:
                # mmap + slice : ne charge en RAM que le préfixe requis (le cache
                # complet peut faire plusieurs Go), sans matérialiser tout le tableau.
                return np.array(np.load(cache_path, mmap_mode='r')[:max_frames])
            return np.load(cache_path)   # (T, C, H, W) float32 — prêt à l'emploi
        except Exception:
            pass                         # cache corrompu → re-décodage

    arr = _decode_video_frames(video_path, img_size, n_channels, crop,
                               max_frames=max_frames, as_uint8=as_uint8)
    # Ne PAS écrire le cache sur un décodage tronqué : il écraserait le cache complet
    # par un préfixe (qui serait ensuite relu comme « toute la vidéo »).
    if use_cache and max_frames is None:
        try:
            np.save(cache_path, arr)
        except Exception:
            pass                         # cache best-effort (disque plein/RO, etc.)
    return arr


def _video_base_stem(video_path: Path) -> str:
    """
    Stem de la vidéo *originale* : retire les suffixes de prétraitement (résolution,
    skip, demi-vidéo, debug…) pour retrouver le nom de la vidéo source. Toutes les
    variantes d'une même originale (256, 128_half, 256_skip4…) partagent ce stem.

    Ex. step_input_random_1_segment_15_min_90kPa_max_128_half  (.mp4)
        → step_input_random_1_segment_15_min_90kPa_max
    """
    stem = video_path.stem
    # _debug300 peut s'ajouter après un suffixe de variante → l'enlever d'abord.
    if stem.endswith('_debug300'):
        stem = stem[: -len('_debug300')]
    # _train / _val : découpes d'une même originale (début / fin). Le chiffre optionnel
    # donne le pourcentage gardé (_train80 / _val20 = coupe 80/20 ; _train / _val nus =
    # coupe 50/50 historique). Toutes partagent le MÊME CSV de pression que l'originale
    # → on retire ce suffixe pour la résolution du CSV et le repli d'offset. La
    # distinction temporelle train↔val (la part val ne démarre PAS à la frame 0) est
    # portée par le stem COMPLET côté PRESSURE_SYNC_OFFSETS (lookup du stem complet
    # AVANT base_stem, cf. boucle dataset).
    stem = re.sub(r'_(?:train|val)\d*$', '', stem)
    for suf in ('_128_half', '_256_skip4', '_256', '_skip4', '_half', '_128'):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem


def _resolve_pressure_csv(video_path: Path, pressure_dir: Path) -> Path:
    """
    Déduit le CSV de pression associé à une vidéo en retirant les suffixes
    de prétraitement du nom de fichier, puis en ajoutant '_pressure_data.csv'.

    Ex. step_input_random_1_segment_15_min_90kPa_max_128_half.mp4
        → step_input_random_1_segment_15_min_90kPa_max_pressure_data.csv
    """
    return Path(pressure_dir) / f'{_video_base_stem(video_path)}_pressure_data.csv'


def load_pressure_frames(
    video_path: Path,
    n_frames: int,
    dt: float,
    pressure_dir: Path,
    cols,                       # liste des colonnes-chambres, ex. ['p_is_4', 'p_is_6']
    norm: float = 101325.0,
    sync_offset: float = 0.0,
) -> np.ndarray:
    """
    Charge la pression interne et l'aligne temporellement sur les frames.

    Le CSV est échantillonné à haute fréquence (≈1000 Hz, colonne `time_s`) ;
    on l'interpole sur les instants des frames `t_i = sync_offset + i·dt`
    (i = 0…n_frames-1). Chaque colonne de `cols` est interpolée indépendamment.

    `sync_offset` (s) recale l'axe pression sur l'axe vidéo : le capteur de pression
    et la caméra ne démarrent PAS au même instant (cf. dataset Krauss — la pression
    est loggée avant la caméra). `sync_offset` = temps RÉEL du CSV correspondant à la
    frame 0 de la vidéo. Comme la frame 0 est la même pour toutes les variantes d'une
    originale (256, 128_half, skip4…), l'offset se PROPAGE à toutes : seul `dt` change
    (période de frame effective). Voir `PRESSURE_SYNC_OFFSETS` dans config.

    Retourne : (n_frames, n_c) float32, normalisé par `norm`.
    """
    csv_path = _resolve_pressure_csv(video_path, pressure_dir)
    assert csv_path.exists(), (
        f'CSV de pression introuvable pour "{video_path.name}" : {csv_path}. '
        f'Vérifier PRESSURE_DIR / PRESSURE_COLS dans config.'
    )

    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')
    missing = [c for c in (['time_s'] + list(cols)) if c not in header]
    assert not missing, f'Colonnes absentes de {csv_path.name} : {missing}'

    use_idx = [header.index('time_s')] + [header.index(c) for c in cols]
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=use_idx)
    if data.ndim == 1:                       # CSV à une seule ligne de données
        data = data[np.newaxis, :]

    t_csv  = data[:, 0]
    t_frame = float(sync_offset) + np.arange(n_frames, dtype=np.float64) * dt
    out = np.stack(
        [np.interp(t_frame, t_csv, data[:, k + 1]) for k in range(len(cols))],
        axis=1,
    )                                        # (n_frames, n_c)
    return (out / float(norm)).astype(np.float32)


def compute_rest_frame(
    frames_all: np.ndarray,          # (N, C, H, W) — toutes les frames chargées
    video_paths: list,                # liste des Path des vidéos chargées
    img_size: tuple,
    n_channels: int,
    crop,
    rest_video=None,                  # None ou nom/chemin de la vidéo de repos
    rest_n_frames: int = 60,          # nb de dernières frames si REST_FIRST_N_FRAMES=0
    rest_first_n_frames: int = 0,     # > 0 : médiane des N premières frames
) -> np.ndarray:
    """
    Calcule la rest_frame selon les paramètres de config.

    Priorité :
      1. rest_video fourni et trouvable → médiane des N premières ou dernières frames
      2. rest_video=None               → médiane globale de toutes les frames

    Retourne : (C, H, W) float32
    """
    if rest_video is not None:
        rest_video = Path(rest_video)
        if rest_video.is_absolute() or rest_video.exists():
            rest_path = rest_video
        else:
            rest_path = video_paths[0].parent / rest_video

        if rest_path.exists():
            rest_all = load_video_frames(
                rest_path, img_size, n_channels=n_channels, crop=crop
            )  # (T, C, H, W)

            if rest_first_n_frames > 0:
                n = min(rest_first_n_frames, len(rest_all))
                rest_frame = np.median(rest_all[:n], axis=0).astype(np.float32)
                print(f'[rest_frame] médiane des {n} premières frames de "{rest_path.name}"')
            else:
                n = min(rest_n_frames, len(rest_all))
                rest_frame = np.median(rest_all[-n:], axis=0).astype(np.float32)
                print(f'[rest_frame] médiane des {n} dernières frames de "{rest_path.name}"')
            return rest_frame
        else:
            print(f'[rest_frame] rest_video "{rest_video}" introuvable — fallback médiane globale.')

    # Fallback : première(s) frame(s) ou médiane globale.
    # frames_all peut être uint8 [0,255] (store_uint8) → on ramène la rest_frame en
    # [0,1] float32 (contrat LNN) en divisant par 255 dans ce cas.
    _scale = 255.0 if frames_all.dtype == np.uint8 else 1.0
    if rest_first_n_frames > 0:
        n = min(rest_first_n_frames, len(frames_all))
        rest_frame = (np.median(frames_all[:n], axis=0) / _scale).astype(np.float32)
        print(f'[rest_frame] médiane des {n} premières frames du dataset')
    else:
        rest_frame = (np.median(frames_all, axis=0) / _scale).astype(np.float32)
        print(f'[rest_frame] médiane globale de {len(frames_all)} frames')
    return rest_frame


# -- Dataset principal --------------------------------------------------------

class VideoFrameDataset(Dataset):
    """
    Frames independantes de toutes les videos.

    `video_dir` peut etre :
      - un dossier  -> toutes les .mp4 sont chargees
      - un fichier  -> une seule video

    crop : None ou (x, y, w, h) appliqué avant resize.

    rest_frame :
      Toujours calculée (nécessaire pour initialiser z_rest dans le LNN).
      N'est jamais soustraite aux frames d'entraînement.
      Mode de calcul contrôlé par rest_video / rest_n_frames / rest_first_n_frames.

    Attributs publics :
        frames          : (N, C, H, W) float32  — frames brutes
        indices         : (N,)  int             — index temporel global continu
        video_lengths   : list[int]             — nb de frames par video
        rest_frame      : (C, H, W) float32     — frame de repos (jamais soustraite)
        pressures       : (N, n_c) float32 ou None — pression alignée sur les frames
                          (None si load_pressure=False)
    """

    def __init__(
        self,
        video_dir: Path,
        img_size=(64, 64),
        n_channels=1,
        crop=None,
        rest_video=None,
        rest_n_frames=60,
        rest_first_n_frames=0,
        exclude_videos=None,
        # Paramètre legacy ignoré — conservé pour compatibilité des appels existants
        subtract_rest=False,
        # Pression (opt-in) — None/False laisse self.pressures = None
        load_pressure=False,
        pressure_dir=None,
        pressure_cols=None,
        pressure_norm=101325.0,
        pressure_dt=None,
        pressure_sync_offsets=None,   # dict {stem_vidéo_originale: offset_s} ou None
        max_frames=None,              # >0 : ne charge que les `max_frames` 1ʳᵉˢ frames PAR vidéo
        store_uint8=False,            # True : self.frames en uint8 [0,255] (~4× moins de RAM)
    ):
        self.img_size   = img_size
        self.n_channels = n_channels
        self.crop       = crop
        self.store_uint8 = store_uint8
        video_dir = Path(video_dir)

        if subtract_rest:
            import warnings
            warnings.warn(
                'subtract_rest=True est ignoré : la soustraction du fond ne doit plus '
                'être appliquée aux frames. Utiliser Z_REST_MODE dans config pour '
                'contrôler l\'initialisation de z_rest.',
                DeprecationWarning, stacklevel=2,
            )

        # -- Liste des videos a charger ---------------------------------------
        if video_dir.is_file():
            video_paths = [video_dir]
        else:
            # .npz = dataset de frames prétraitées (cf. load_npz_frames), traité
            # exactement comme une vidéo par le reste du pipeline.
            video_paths = sorted(list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.npz')))
            assert len(video_paths) > 0, f'Aucune video .mp4/.npz dans {video_dir}'
            if exclude_videos:
                exclude_names = {Path(v).name for v in exclude_videos}
                video_paths = [p for p in video_paths if p.name not in exclude_names]
                assert len(video_paths) > 0, 'Toutes les videos ont ete exclues'

        # Noms des vidéos dans l'ordre de chargement (= ordre des segments de
        # video_lengths) — utile pour étiqueter les plots par vidéo.
        self.video_names = [Path(vp).name for vp in video_paths]

        # -- Chargement de toutes les videos ----------------------------------
        all_frames  = []
        all_indices = []
        all_press   = []
        self.video_lengths = []
        offset = 0

        for vp in video_paths:
            frames = load_video_frames(
                vp, img_size, n_channels=n_channels, crop=crop,
                max_frames=max_frames, as_uint8=store_uint8
            )  # (T, C, H, W) — float32 [0,1] ou uint8 [0,255] si store_uint8
            T = len(frames)
            self.video_lengths.append(T)
            all_frames.append(frames)
            all_indices.append(np.arange(offset, offset + T))
            offset += T

            if load_pressure and vp.suffix.lower() == '.npz':
                # Source NPZ : la pression est DANS le fichier, déjà échantillonnée
                # par frame et alignée par les auteurs → ni CSV, ni interpolation,
                # ni sync_offset (cf. load_pressure_npz).
                all_press.append(load_pressure_npz(
                    vp, n_frames=T, cols=pressure_cols, norm=pressure_norm))
            elif load_pressure:
                assert pressure_dt is not None, 'pressure_dt requis si load_pressure=True'
                base_stem = _video_base_stem(vp)
                full_stem = vp.stem
                sync_offset = 0.0
                if pressure_sync_offsets:
                    # Stem COMPLET prioritaire : permet à une demi-vidéo _val (qui ne
                    # démarre pas à la frame 0 de l'originale) de porter son propre
                    # offset (= offset_originale + n_frames_train·dt). Repli sur base_stem
                    # (originale) sinon → _train et les variantes pleines héritent de l'offset.
                    if full_stem in pressure_sync_offsets:
                        sync_offset = float(pressure_sync_offsets[full_stem])
                    elif base_stem in pressure_sync_offsets:
                        sync_offset = float(pressure_sync_offsets[base_stem])
                    else:
                        import warnings
                        warnings.warn(
                            f'PRESSURE_SYNC_OFFSETS ne contient pas "{base_stem}" : '
                            f'offset=0 supposé (pression/vidéo alignées à t=0). '
                            f'Si la pression a été loggée avant la caméra, l\'alignement '
                            f'sera faux — ajouter l\'offset pour cette vidéo.'
                        )
                all_press.append(load_pressure_frames(
                    vp, n_frames=T, dt=pressure_dt,
                    pressure_dir=pressure_dir, cols=pressure_cols,
                    norm=pressure_norm, sync_offset=sync_offset,
                ))  # (T, n_c)

        # Vidéo unique (cas Krauss) : éviter la copie de np.concatenate (un gros
        # tableau uint8/float peut faire plusieurs Go → pic RAM ×2 inutile).
        self.frames  = (all_frames[0] if len(all_frames) == 1
                        else np.concatenate(all_frames, axis=0))  # (N, C, H, W)
        self.indices = np.concatenate(all_indices, axis=0)  # (N,)
        self.pressures = (np.concatenate(all_press, axis=0)  # (N, n_c)
                          if load_pressure else None)

        # -- Rest frame (jamais soustraite) -----------------------------------
        self.rest_frame = compute_rest_frame(
            frames_all        = self.frames,
            video_paths       = video_paths,
            img_size          = img_size,
            n_channels        = n_channels,
            crop              = crop,
            rest_video        = rest_video,
            rest_n_frames     = rest_n_frames,
            rest_first_n_frames = rest_first_n_frames,
        )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.frames[idx])  # (C, H, W)
        return x, int(self.indices[idx])


# -- Dataset sequences --------------------------------------------------------

class VideoSeqDataset(Dataset):
    """
    Sequences de frames consecutives issues d'une meme video.
    Garantit qu'aucune sequence ne chevauche deux videos differentes.

    Retourne (frames_seq, indices_seq) avec :
        frames_seq  : (seq_len, C, H, W) float32
        indices_seq : (seq_len,) int

    Si return_pressure=True (et que frame_dataset.pressures existe), retourne en
    plus la pression de la frame CENTRALE de la séquence (indice s + seq_len//2),
    qui est exactement le point où le résidu d'Euler-Lagrange est évalué (z_mid
    pour l'ordre 2, z2 pour l'ordre 4) :
        (frames_seq, indices_seq, pressure_center)  avec pressure_center : (n_c,)
    """

    def __init__(self, frame_dataset: VideoFrameDataset, seq_len: int,
                 return_pressure: bool = False, seq_stride: int = 1):
        self.frame_dataset = frame_dataset
        self.seq_len = seq_len
        self.seq_stride = max(int(seq_stride), 1)   # espacement des frames de la fenêtre
        self.return_pressure = return_pressure and frame_dataset.pressures is not None

        # Une fenêtre couvre (seq_len-1)*stride + 1 frames ; borne le start pour que la
        # dernière frame (s+(seq_len-1)*stride) reste dans la vidéo (jamais à cheval).
        span = (seq_len - 1) * self.seq_stride
        self.seq_starts = []
        offset = 0
        for vlen in frame_dataset.video_lengths:
            for start in range(vlen - span):
                self.seq_starts.append(offset + start)
            offset += vlen

    def __len__(self):
        return len(self.seq_starts)

    def __getitem__(self, idx):
        s = self.seq_starts[idx]
        T = self.seq_len
        k = self.seq_stride
        end = s + (T - 1) * k + 1
        frames = torch.from_numpy(
            self.frame_dataset.frames[s:end:k]
        )                                                  # (T, C, H, W)
        indices = self.frame_dataset.indices[s:end:k]     # (T,) int array
        if self.return_pressure:
            # centre de la fenêtre (point d'évaluation du résidu) = s + (T//2)*stride
            p_center = torch.from_numpy(
                self.frame_dataset.pressures[s + (T // 2) * k]
            )                                              # (n_c,)
            return frames, indices, p_center
        return frames, indices


class VideoChunkDataset(Dataset):
    """
    Segments contigus de frames issus d'une même vidéo, pour le fine-tuning de
    l'encodeur dans le LNN.

    Le `VideoSeqDataset` glissant (stride 1) re-fournit chaque frame dans ~seq_len
    fenêtres distinctes ; les encoder une à une revient à passer chaque frame
    ~seq_len fois dans l'encodeur par époque. Avec un CNN lourd (CpAE) c'est le coût
    dominant. Ici on émet des **segments contigus** : on les encode UNE fois, puis on
    forme toutes les fenêtres `[j:j+seq_len]` dans l'espace latent (cf. train_lnn).
    Chaque frame n'est alors encodée qu'une fois → ~seq_len× moins de passages.

    Les segments partitionnent exactement les fenêtres (stride = chunk_len-seq_len+1,
    plus un dernier segment ancré sur la fin) et ne franchissent jamais une frontière
    de vidéo. Les longueurs peuvent varier (queue de vidéo) → utiliser `chunk_collate`.

    __getitem__ → (frames_chunk, pressure_chunk | None) avec
        frames_chunk   : (ℓ, C, H, W) float32,  seq_len ≤ ℓ ≤ chunk_len
        pressure_chunk : (ℓ, n_c) float32  (si return_pressure)
    """

    def __init__(self, frame_dataset: VideoFrameDataset, seq_len: int,
                 chunk_len: int, return_pressure: bool = False):
        assert chunk_len >= seq_len, 'chunk_len doit être ≥ seq_len'
        self.frame_dataset = frame_dataset
        self.seq_len = seq_len
        self.chunk_len = chunk_len
        self.return_pressure = return_pressure and frame_dataset.pressures is not None

        stride = chunk_len - seq_len + 1   # tuile les fenêtres sans trou ni doublon
        self.chunk_bounds = []             # (start, end) absolus, internes à une vidéo
        offset = 0
        for vlen in frame_dataset.video_lengths:
            if vlen >= seq_len:
                for start in range(0, vlen - seq_len + 1, stride):
                    end = min(start + chunk_len, vlen)
                    self.chunk_bounds.append((offset + start, offset + end))
            offset += vlen

    def __len__(self):
        return len(self.chunk_bounds)

    def __getitem__(self, idx):
        s, e = self.chunk_bounds[idx]
        frames = torch.from_numpy(self.frame_dataset.frames[s:e])   # (ℓ, C, H, W)
        if self.return_pressure:
            p = torch.from_numpy(self.frame_dataset.pressures[s:e])  # (ℓ, n_c)
        else:
            p = None
        return frames, p


def chunk_collate(batch):
    """Collate pour VideoChunkDataset : concatène les segments de longueurs variables.

    Retourne (frames_cat, lengths, pressures_cat | None) :
        frames_cat   : (Σℓ, C, H, W)  — à encoder en un seul forward
        lengths      : list[int]       — longueur de chaque segment (pour re-découper z)
        pressures_cat: (Σℓ, n_c) | None
    """
    frames_cat = torch.cat([b[0] for b in batch], dim=0)
    lengths    = [b[0].shape[0] for b in batch]
    if batch[0][1] is None:
        pressures_cat = None
    else:
        pressures_cat = torch.cat([b[1] for b in batch], dim=0)
    return frames_cat, lengths, pressures_cat
