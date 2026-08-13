"""Le parseur EXIF maison : il doit rendre la bonne date, ou rien, jamais une erreur.

Les JPEG de test sont FABRIQUÉS ici, octet par octet — pas de fixture binaire opaque
dans le dépôt : le lecteur doit pouvoir vérifier de tête ce que le fichier contient.
"""

from pathlib import Path

import pytest

from mosaic import exif
from mosaic.facettes import extraire
from mosaic.temporel import SANS_DATE


def _jpeg_exif(
    date: bytes | None, *, ordre: str = "II", original: bool = True
) -> bytes:
    """Un JPEG minimal (SOI + APP1 Exif + EOI). `date` au format EXIF « AAAA:MM:JJ hh:mm:ss ».

    Structure du bloc TIFF produit, tout offset compté depuis son premier octet :
      0   boutisme (II/MM) + 42 + offset d'IFD0 (8)
      8   IFD0 : 1 entrée -> pointeur ExifIFD (0x8769) vers 26
      26  sous-IFD : 1 entrée -> DateTimeOriginal (0x9003), ASCII[20] pointant vers 44
      44  la date, 20 octets NUL-terminés
    """
    boutisme = "little" if ordre == "II" else "big"

    def n2(v: int) -> bytes:
        return v.to_bytes(2, boutisme)

    def n4(v: int) -> bytes:
        return v.to_bytes(4, boutisme)

    if date is None:
        tiff = ordre.encode() + n2(42) + n4(8) + n2(0) + n4(0)
    else:
        tag = 0x9003 if original else 0x0132
        valeur = date + b"\x00"
        if original:
            ifd0 = n2(1) + n2(0x8769) + n2(4) + n4(1) + n4(26) + n4(0)
            sous = n2(1) + n2(tag) + n2(2) + n4(len(valeur)) + n4(44) + n4(0)
            tiff = ordre.encode() + n2(42) + n4(8) + ifd0 + sous + valeur
        else:
            # DateTime directement dans IFD0 : la valeur suit l'IFD (offset 26)
            ifd0 = n2(1) + n2(tag) + n2(2) + n4(len(valeur)) + n4(26) + n4(0)
            tiff = ordre.encode() + n2(42) + n4(8) + ifd0 + valeur

    charge = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + (len(charge) + 2).to_bytes(2, "big") + charge
    return b"\xff\xd8" + app1 + b"\xff\xd9"


def _ecrire(tmp_path: Path, nom: str, contenu: bytes) -> Path:
    f = tmp_path / nom
    f.write_bytes(contenu)
    return f


def test_date_de_prise_de_vue_lit_datetimeoriginal(tmp_path):
    f = _ecrire(tmp_path, "IMG_2043.jpg", _jpeg_exif(b"2026:05:12 14:33:07"))
    assert exif.date_de_prise_de_vue(f) == "2026-05-12"


def test_date_de_prise_de_vue_gros_boutisme(tmp_path):
    """Les deux boutismes existent dans la nature (Motorola/Intel) — les deux se lisent."""
    f = _ecrire(tmp_path, "mm.jpg", _jpeg_exif(b"2024:11:03 08:00:00", ordre="MM"))
    assert exif.date_de_prise_de_vue(f) == "2024-11-03"


def test_datetime_ifd0_sert_de_repli(tmp_path):
    f = _ecrire(
        tmp_path, "repli.jpg", _jpeg_exif(b"2025:01:09 10:00:00", original=False)
    )
    assert exif.date_de_prise_de_vue(f) == "2025-01-09"


def test_sans_exif_rend_none(tmp_path):
    f = _ecrire(tmp_path, "nu.jpg", _jpeg_exif(None))
    assert exif.date_de_prise_de_vue(f) is None


@pytest.mark.parametrize(
    "date",
    [
        b"0000:00:00 00:00:00",  # appareil sans horloge réglée
        b"1899:12:31 23:59:59",  # avant la borne de plausibilité
        b"2200:01:01 00:00:00",  # après
        b"20xx:01:01 00:00:00",  # non numérique
        b"2026:13:45 00:00:00",  # mois et jour hors bornes
    ],
)
def test_dates_implausibles_rejetees(tmp_path, date):
    """Mieux vaut aucune date qu'une fausse : une date aberrante fausserait le
    classement par récence sans que personne ne s'en aperçoive."""
    f = _ecrire(tmp_path, "casse.jpg", _jpeg_exif(date))
    assert exif.date_de_prise_de_vue(f) is None


@pytest.mark.parametrize(
    "nom,contenu",
    [
        ("pas_un_jpeg.jpg", b"\x89PNG\r\n\x1a\n" + b"\x00" * 40),
        ("tronque.jpg", b"\xff\xd8\xff\xe1\x00"),
        ("vide.jpg", b""),
        ("desynchronise.jpg", b"\xff\xd8" + b"\x41" * 60),
    ],
)
def test_fichiers_malformes_ne_levent_jamais(tmp_path, nom, contenu):
    assert exif.date_de_prise_de_vue(_ecrire(tmp_path, nom, contenu)) is None


def test_formats_hors_portee_rendent_none(tmp_path):
    """PNG et HEIC ne sont pas couverts — assumé et documenté, pas un oubli."""
    for nom in ("photo.png", "photo.heic", "note.md"):
        f = _ecrire(tmp_path, nom, _jpeg_exif(b"2026:05:12 14:33:07"))
        assert exif.date_de_prise_de_vue(f) is None


def test_fichier_absent_rend_none(tmp_path):
    assert exif.date_de_prise_de_vue(tmp_path / "jamais_ecrit.jpg") is None


# -- Branchement dans les facettes ------------------------------------------------------


def test_facette_date_dune_photo_vient_de_lexif(tmp_path):
    """Le cas réel : `IMG_2043.jpg` n'a aucune date dans son nom, elle tombait donc à
    SANS_DATE et sortait du champ de `--recence` et de `mosaic actuel`."""
    f = _ecrire(tmp_path, "IMG_2043.jpg", _jpeg_exif(b"2026:06:14 09:12:00"))
    assert extraire(f, "chantier/photos/IMG_2043.jpg", "")["date"] == "2026-06-14"


def test_le_chemin_reste_prioritaire_sur_lexif(tmp_path):
    """Une date dans le chemin est un classement VOULU : elle fait foi. L'EXIF ne prend
    le relais que sur les documents autrement sans date."""
    f = _ecrire(tmp_path, "IMG_2043.jpg", _jpeg_exif(b"2026:06:14 09:12:00"))
    fac = extraire(f, "2020-01-31_reception/IMG_2043.jpg", "")
    assert fac["date"] == "2020-01-31"


def test_photo_sans_exif_garde_la_sentinelle(tmp_path):
    f = _ecrire(tmp_path, "IMG_9999.jpg", _jpeg_exif(None))
    assert extraire(f, "chantier/IMG_9999.jpg", "")["date"] == SANS_DATE
