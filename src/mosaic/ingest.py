"""Prisme de lecture multi-formats (markitdown).

Principe : markitdown est un PRISME DE LECTURE, pas un transcripteur. La
conversion vit en mémoire le temps d'encoder la grille puis disparaît — ce qui
persiste, c'est la mosaïque, jamais la transcription. Aucun fichier converti
n'est déposé à côté des documents.

Ce module est le SEUL du cœur Mosaic à importer markitdown, sous garde
try/except : sans l'extra `ingest`, Mosaic continue de fonctionner comme avant
(les fichiers convertibles sont simplement ignorés, jamais un crash).
"""

import hashlib
from pathlib import Path

CONVERTIBLE_EXTS = {".pdf", ".docx", ".xlsx", ".html", ".pptx"}

# Photos et images (v1.6 §B) : ensemble VOLONTAIREMENT SÉPARÉ de CONVERTIBLE_EXTS —
# markitdown n'a aucune couche texte à offrir pour une image (contrairement à un PDF/
# docx muet), donc chaque consommateur (Index.build/add, run_bench...) doit décider
# explicitement de les prendre en compte plutôt que d'hériter silencieusement du
# comportement markitdown via une fusion aveugle dans CONVERTIBLE_EXTS.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff"}

# Type de document en toutes lettres — une FACETTE de la carte d'identité, indépendante du
# contenu : un tableur, un PDF scanné, une photo ne portent pas la même nature d'information.
# Universel (tout corpus a des types), déterministe, gratuit (connu à l'ingestion). Injecté au
# build comme signal supplémentaire, complémentaire du sens.
SEUIL_PDF_SCAN = (
    200  # sous ce nb de caractères extraits, un PDF est probablement un scan
)


def type_semantique(
    path: Path, texte: str | None, types_extra: dict[str, str] | None = None
) -> str:
    """Type SÉMANTIQUE du document (pas l'extension brute) : la nature d'information qu'il porte.
    Le PDF est ventilé scanné/numérique selon la quantité de texte réellement extraite.
    `types_extra` (profil d'index) : extensions supplémentaires -> types custom (.dwg -> plan,
    .py -> code) — prioritaires sur les défauts, qui restent pour tout le reste."""
    ext = path.suffix.lower()
    if types_extra and ext in types_extra:
        return types_extra[ext]
    if ext in {".xlsx", ".xls", ".csv", ".ods"}:
        return "tableur"
    if ext in IMAGE_EXTS:
        return "photo"
    if ext in {".docx", ".odt", ".rtf"}:
        return "document rédigé"
    if ext in {".pptx", ".odp"}:
        return "présentation"
    if ext in {".html", ".htm"}:
        return "page web"
    if ext == ".pdf":
        return (
            "pdf scanné"
            if len((texte or "").strip()) < SEUIL_PDF_SCAN
            else "pdf numérique"
        )
    return "note texte"


# Import PARESSEUX (v2.0), même motif que rapidocr ci-dessous : markitdown tire magika
# (détection de type de fichier), qui charge onnxruntime — et son avertissement stderr sous
# Linux. Chargé au démarrage, il corromprait le contrat « erreur = JSON sur stderr » du CLI
# pour une simple recherche. On ne résout donc l'import qu'à la première conversion réelle.
_MARKITDOWN_NON_TENTE = object()
MarkItDown = _MARKITDOWN_NON_TENTE  # ty: ignore[invalid-assignment]  # résolu paresseusement


def _resoudre_markitdown() -> None:
    """Tente l'import MarkItDown UNE seule fois, en mettant à jour le global `MarkItDown`
    (classe ou None). No-op si déjà résolu ou monkeypatché (None/objet par les tests)."""
    global MarkItDown
    if MarkItDown is _MARKITDOWN_NON_TENTE:
        try:
            from markitdown import MarkItDown as _MI  # ty: ignore[unresolved-import]
        except ImportError:
            _MI = None  # ty: ignore[invalid-assignment]  # repli garde dépendance optionnelle
        MarkItDown = _MI  # ty: ignore[invalid-assignment]


# Crochet OCR (v1.5) : documents muets (convertible dont la conversion markitdown
# rend < 200 caractères). Provider PAR DÉFAUT = rapidocr (paquet unifié RapidAI,
# validé par Alex le 09/08) avec repli sur l'ancien rapidocr_onnxruntime ; les PDF
# sont rastérisés par pypdfium2. Tout est sous garde — sans les paquets, la
# détection dégrade proprement (available_ocr() False), jamais lever à l'import.
# Import PARESSEUX (v2.0) : rapidocr tire onnxruntime, qui au chargement sous Linux émet
# un avertissement `device_discovery` sur stderr. Importé au démarrage, il corromprait le
# contrat « erreur = JSON sur stderr » du CLI pour une SIMPLE recherche (qui n'a aucun besoin
# d'OCR). On ne résout donc l'import qu'à la première consultation OCR réelle. La sentinelle
# `_OCR_NON_TENTE` distingue « pas encore importé » de `None` (« importé, absent ») — pour que
# les tests qui monkeypatchent `ingest.RapidOCR` (None/objet) court-circuitent la résolution.
_OCR_NON_TENTE = object()
RapidOCR = _OCR_NON_TENTE  # ty: ignore[invalid-assignment]  # résolu paresseusement


def _resoudre_rapidocr() -> None:
    """Tente l'import RapidOCR (paquet unifié `rapidocr`, repli `rapidocr_onnxruntime`)
    UNE seule fois, en mettant à jour le global `RapidOCR` (classe ou None). Ne fait rien
    si `RapidOCR` a déjà été résolu ou monkeypatché."""
    global RapidOCR
    if RapidOCR is _OCR_NON_TENTE:
        try:
            from rapidocr import RapidOCR as _RO  # ty: ignore[unresolved-import]
        except ImportError:
            try:
                from rapidocr_onnxruntime import RapidOCR as _RO  # ty: ignore[unresolved-import]
            except ImportError:
                _RO = None  # ty: ignore[conflicting-declarations]  # repli garde dépendance optionnelle
        RapidOCR = _RO  # ty: ignore[invalid-assignment]


try:
    import pypdfium2 as _pdfium  # ty: ignore[unresolved-import]
except ImportError:
    _pdfium = None  # ty: ignore[invalid-assignment]  # garde dépendance optionnelle

_OCR_MAX_PAGES = 10  # borne de temps sur les gros scans : premières pages seulement

_OCR_MIN_CHARS = 200

# Garde de volume (v1.6 §B) : une photo > 12 Mo est un scan aberrant (résolution
# délirante, fichier corrompu...) — ignorée+comptée plutôt que de lancer l'OCR dessus
# (temps de traitement disproportionné pour un document chantier).
_OCR_MAX_IMAGE_BYTES = 12 * 1024 * 1024

_converter = None  # singleton paresseux, construit une seule fois par process
_ocr_engine = None  # idem, pour le moteur OCR par défaut


def available() -> bool:
    """True si markitdown est importable (extra `ingest` installé).
    Déclenche l'import paresseux à la première demande."""
    _resoudre_markitdown()
    return MarkItDown is not None


def _get_converter():
    global _converter
    if _converter is None:
        _resoudre_markitdown()
        converter_cls = MarkItDown
        if converter_cls is None or converter_cls is _MARKITDOWN_NON_TENTE:
            raise RuntimeError(
                "markitdown non installé — appeler available() avant _get_converter()"
            )
        _converter = converter_cls()  # ty: ignore[call-non-callable]  # sentinelle écartée juste au-dessus
    return _converter


def available_ocr() -> bool:
    """True si un moteur RapidOCR est importable (rapidocr unifié ou rapidocr_onnxruntime).
    Déclenche l'import paresseux à la première demande."""
    _resoudre_rapidocr()
    return RapidOCR is not None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _resoudre_rapidocr()
        moteur_cls = RapidOCR
        if moteur_cls is None or moteur_cls is _OCR_NON_TENTE:
            raise RuntimeError(
                "rapidocr non installé — appeler available_ocr() avant _get_ocr_engine()"
            )
        _ocr_engine = moteur_cls()  # ty: ignore[call-non-callable]  # sentinelle écartée juste au-dessus
    return _ocr_engine


def _ocr_result_texts(sortie) -> list[str]:
    """Normalise la sortie des deux API RapidOCR : objet .txts (paquet unifié 3.x)
    ou liste de [boîte, texte, score] (ancien rapidocr_onnxruntime)."""
    txts = getattr(sortie, "txts", None)
    if txts is not None:
        return [str(t) for t in txts]
    if isinstance(sortie, (list, tuple)):
        return [str(ligne[1]) for ligne in sortie if len(ligne) > 1]
    return []


def ocr_provider(path: Path) -> str | None:
    """Fournisseur OCR PAR DÉFAUT (rapidocr, sous garde) — interface enfichable :
    `ocr_provider(path) -> str | None`, remplaçable par tout autre callable en
    monkeypatchant `ingest.ocr_provider`. Les PDF sont rastérisés page à page
    (pypdfium2, bornés à _OCR_MAX_PAGES) ; les images passent directement.
    None si le moteur est indisponible, ne reconnaît aucun texte, ou échoue —
    jamais de crash."""
    if not available_ocr():
        return None
    try:
        engine = _get_ocr_engine()
        morceaux: list[str] = []
        if path.suffix.lower() == ".pdf":
            if _pdfium is None:
                return None
            doc = _pdfium.PdfDocument(str(path))
            try:
                for i in range(min(len(doc), _OCR_MAX_PAGES)):
                    image = doc[i].render(scale=2.0).to_pil()
                    import numpy as _np

                    sortie = engine(_np.array(image))
                    morceaux.extend(_ocr_result_texts(sortie))
            finally:
                doc.close()
        else:
            sortie = engine(str(path))
            morceaux.extend(_ocr_result_texts(sortie))
    except Exception:
        return None
    if not morceaux:
        return None
    return "\n".join(morceaux)


def _cache_key(path: Path, ocr: bool = False) -> str:
    """sha256(chemin absolu + mtime_ns + taille + drapeau ocr) — invalidé si le fichier
    change OU si le drapeau OCR diffère.

    Résout en absolu avant de hasher : deux chemins relatifs identiques mais
    pointant vers des fichiers différents selon le cwd (deux corpus distincts
    ouverts avec un `corpus_dir` relatif) ne doivent jamais partager une clé.

    `ocr` (revue finale v1.5, important, reproduit) : sans ce drapeau dans la clé, un
    document muet mis en cache SANS --ocr était servi tel quel à un run AVEC --ocr (aucun
    OCR déclenché, aucune ValueError promise) — et inversement. build/add avec ocr=True et
    ocr=False sur le même fichier doivent aboutir à deux entrées de cache distinctes.
    """
    path = path.resolve()
    st = path.stat()
    raw = f"{path}|{st.st_mtime_ns}|{st.st_size}"
    if ocr:
        raw += "-ocr"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def to_text(path: Path, cache_dir: Path | None = None, ocr: bool = False) -> str | None:
    """Convertit `path` en texte via markitdown, en mémoire. None si illisible.

    Cache strictement OPT-IN : fourni `cache_dir` uniquement, jamais activé par
    défaut. Toute exception (fichier corrompu, format non supporté, etc.) est
    absorbée et retourne None — jamais de crash du build.

    `ocr` (v1.5) : si la conversion markitdown rend un texte muet (None ou
    < `_OCR_MIN_CHARS` caractères), relaie vers `ocr_provider(path)`. Sans
    moteur disponible (`available_ocr()` False) : ValueError claire — jamais un
    document muet qui passe silencieusement pour un succès. `ocr=False`
    (défaut) : comportement v1.4 exact, jamais de crochet OCR consulté.

    Photos et images (v1.6 §B, `path.suffix` dans `IMAGE_EXTS`) : branche dédiée,
    markitdown jamais consulté (aucune couche texte possible pour une image) —
    directement l'OCR si `ocr=True`, sinon None (ignorée+comptée par l'appelant,
    comme un convertible sans markitdown). Pas de seuil `_OCR_MIN_CHARS` : une
    image passe TOUJOURS par l'OCR, il n'y a rien à mesurer avant. Garde de
    volume : une image > `_OCR_MAX_IMAGE_BYTES` est ignorée+comptée sans même
    consulter le provider (scan aberrant).
    """
    path = Path(path)
    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        try:
            cache_file = cache_dir / f"{_cache_key(path, ocr=ocr)}.txt"
        except OSError:
            cache_file = None
        if cache_file is not None and cache_file.is_file():
            try:
                return cache_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                pass  # cache illisible/corrompu : on retombe sur une conversion fraîche

    text: str | None = None
    if path.suffix.lower() in IMAGE_EXTS:
        try:
            trop_lourde = path.stat().st_size > _OCR_MAX_IMAGE_BYTES
        except OSError:
            trop_lourde = False
        if not trop_lourde and ocr:
            if not available_ocr():
                raise ValueError(
                    "--ocr demandé mais aucun moteur OCR disponible — "
                    'pip install "rapidocr_onnxruntime" (provider par défaut, non installé)'
                )
            text = ocr_provider(path)
        # trop_lourde, ou ocr=False : text reste None (ignorée+comptée par l'appelant)
    else:
        if available():
            try:
                text = _get_converter().convert(path).text_content
            except Exception:
                text = None

        if ocr and (text is None or len(text) < _OCR_MIN_CHARS):
            if not available_ocr():
                raise ValueError(
                    "--ocr demandé mais aucun moteur OCR disponible — "
                    'pip install "rapidocr_onnxruntime" (provider par défaut, non installé)'
                )
            ocr_text = ocr_provider(path)
            if ocr_text is not None:
                text = ocr_text

    if text is None:
        return None

    if cache_dir is not None and cache_file is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(text, encoding="utf-8")
        except OSError:
            pass  # cache non-écrivable : la conversion reste valide, juste pas mémorisée

    return text
