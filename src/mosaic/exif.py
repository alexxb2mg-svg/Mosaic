"""Date de prise de vue d'une photo, lue dans son EXIF — sans aucune dépendance.

Pourquoi un parseur maison plutôt que Pillow : le cœur de Mosaic ne dépend que de
numpy, et une date ne justifie pas de tirer une bibliothèque d'imagerie complète
dans l'arbre de dépendances d'un index. Le segment APP1 d'un JPEG se lit en une
centaine de lignes déterministes et testables — même parti pris que les lecteurs
binaires de `store.py`.

Ce que ça résout : une photo de chantier s'appelle `IMG_2043.jpg`, donc
`temporel.date_du_chemin` ne trouve rien et la facette date tombe à
« 0000-00-00 » — le document est hors de portée de `--recence` et de
`mosaic actuel`. L'appareil, lui, a écrit la date de prise de vue dans le fichier.

PORTÉE ASSUMÉE : **JPEG uniquement** (1 009 des 1 158 photos du corpus chantiers
de référence, soit 87 %). PNG n'a pas d'EXIF standardisé et HEIC demanderait un
parseur ISOBMFF complet ; les 149 fichiers concernés gardent la date du chemin.
Un format non couvert, un fichier tronqué ou un EXIF absent rendent None — jamais
une exception : l'appelant retombe simplement sur la date du chemin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

Ordre = Literal[
    "little", "big"
]  # boutisme du bloc TIFF, décidé par ses 2 premiers octets

EXTS = {".jpg", ".jpeg"}  # les seuls formats dont l'EXIF est lu ici

_TAG_DATE_ORIGINALE = 0x9003  # DateTimeOriginal, dans le sous-IFD Exif
_TAG_DATE_FICHIER = 0x0132  # DateTime d'IFD0 : dernière modification, repli seulement
_TAG_POINTEUR_EXIF = 0x8769  # offset du sous-IFD Exif, depuis IFD0

# Marqueurs JPEG sans charge utile : ils ne sont pas suivis d'un champ de longueur,
# les sauter d'un octet au lieu de lire une taille inexistante.
_SANS_CHARGE = {0x01} | set(range(0xD0, 0xD8))
_LONGUEUR_DATE = 19  # « 2026:05:12 14:33:07 » (le NUL final n'est pas lu)

# Garde-fou de plausibilité : un appareil dont la pile d'horloge est morte écrit des
# dates aberrantes (1970, 2000:00:00). Mieux vaut aucune date qu'une fausse — elle
# fausserait le classement par récence sans que personne ne s'en aperçoive.
_ANNEE_MIN, _ANNEE_MAX = 1990, 2100


def _extraire_app1(donnees: bytes) -> bytes | None:
    """Le bloc TIFF du segment APP1 « Exif\\x00\\x00 », ou None s'il n'y en a pas."""
    if donnees[:2] != b"\xff\xd8":  # SOI : pas un JPEG
        return None
    i = 2
    n = len(donnees)
    while i + 1 < n:
        if donnees[i] != 0xFF:
            return None  # flux désynchronisé : on renonce plutôt que de deviner
        marqueur = donnees[i + 1]
        i += 2
        if marqueur in _SANS_CHARGE:
            continue
        if marqueur in (
            0xDA,
            0xD9,
        ):  # début des données image / fin : plus d'EXIF après
            return None
        if i + 2 > n:
            return None
        taille = int.from_bytes(donnees[i : i + 2], "big")
        if taille < 2 or i + taille > n:
            return None
        if marqueur == 0xE1 and donnees[i + 2 : i + 8] == b"Exif\x00\x00":
            return donnees[i + 8 : i + taille]
        i += taille
    return None


def _entrees_ifd(tiff: bytes, offset: int, ordre: Ordre) -> dict[int, bytes]:
    """tag -> les 4 octets de valeur/offset de l'entrée. {} si l'offset est hors bornes."""
    if offset <= 0 or offset + 2 > len(tiff):
        return {}
    nombre = int.from_bytes(tiff[offset : offset + 2], ordre)
    entrees: dict[int, bytes] = {}
    for rang in range(nombre):
        debut = offset + 2 + rang * 12
        if debut + 12 > len(tiff):
            break
        tag = int.from_bytes(tiff[debut : debut + 2], ordre)
        entrees[tag] = tiff[debut + 8 : debut + 12]
    return entrees


def _ascii_pointe(tiff: bytes, champ: bytes, ordre: Ordre) -> str | None:
    """Une date EXIF fait 20 octets, donc toujours stockée par OFFSET (au-delà des 4
    octets inscriptibles dans l'entrée elle-même)."""
    offset = int.from_bytes(champ, ordre)
    if offset <= 0 or offset + _LONGUEUR_DATE > len(tiff):
        return None
    return tiff[offset : offset + _LONGUEUR_DATE].decode("ascii", "ignore")


def _normaliser(brut: str | None) -> str | None:
    """« 2026:05:12 14:33:07 » -> « 2026-05-12 ». None si la date est absurde."""
    if not brut or len(brut) < 10:
        return None
    annee, mois, jour = brut[0:4], brut[5:7], brut[8:10]
    if not (annee.isdigit() and mois.isdigit() and jour.isdigit()):
        return None
    if not (
        _ANNEE_MIN <= int(annee) <= _ANNEE_MAX
        and 1 <= int(mois) <= 12
        and 1 <= int(jour) <= 31
    ):
        return None
    return f"{annee}-{mois}-{jour}"


def _date_du_tiff(tiff: bytes) -> str | None:
    if len(tiff) < 8:
        return None
    ordre: Ordre
    if tiff[:2] == b"II":
        ordre = "little"
    elif tiff[:2] == b"MM":
        ordre = "big"
    else:
        return None
    ifd0 = _entrees_ifd(tiff, int.from_bytes(tiff[4:8], ordre), ordre)
    # La PRISE DE VUE d'abord (sous-IFD Exif) : c'est la date que l'utilisateur a en
    # tête. DateTime d'IFD0 n'est qu'une date de dernière écriture — une retouche ou
    # une copie mal faite la décale, elle ne sert que de repli.
    if _TAG_POINTEUR_EXIF in ifd0:
        sous = _entrees_ifd(
            tiff, int.from_bytes(ifd0[_TAG_POINTEUR_EXIF], ordre), ordre
        )
        if _TAG_DATE_ORIGINALE in sous:
            date = _normaliser(_ascii_pointe(tiff, sous[_TAG_DATE_ORIGINALE], ordre))
            if date:
                return date
    if _TAG_DATE_FICHIER in ifd0:
        return _normaliser(_ascii_pointe(tiff, ifd0[_TAG_DATE_FICHIER], ordre))
    return None


def date_de_prise_de_vue(path: Path) -> str | None:
    """« AAAA-MM-JJ » lue dans l'EXIF, ou None (format non couvert, EXIF absent,
    fichier illisible ou date implausible). Ne lève jamais."""
    p = Path(path)
    if p.suffix.lower() not in EXTS:
        return None
    try:
        # Lecture PARTIELLE : l'EXIF vit en tête de fichier, inutile de charger 4 Mo
        # de pixels pour dix octets de date — 128 Ko couvrent largement APP1, même
        # avec une vignette embarquée.
        with p.open("rb") as f:
            donnees = f.read(131072)
    except OSError:
        return None
    tiff = _extraire_app1(donnees)
    return _date_du_tiff(tiff) if tiff else None
