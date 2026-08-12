"""Carte d'identité de dossier — `mosaic carte`.

Compose les briques existantes (`Index.build` avec ingestion, `render.to_png`,
`Index.explain`) en une page HTML autonome : une carte par document — sa
mosaïque, ses concepts dominants, son nombre de tokens. Rien n'est
réimplémenté ici : ce module assemble et met en forme.
"""

import base64
import html
import re
from datetime import date
from pathlib import Path

from mosaic import GRID_DEFAULT, ingest
from mosaic.index import Index
from mosaic.render import to_png
from mosaic.tokenize import tokenize

MOSAIC_DIRNAME = "_MOSAIC"

# Curation d'AFFICHAGE uniquement (n'altère jamais explain()/l'index) : un devis
# est dense en tokens numériques/administratifs (montants, taux, quantités) qui
# dominent souvent le classement par poids sans rien distinguer visuellement au
# lecteur. On demande donc un bassin plus large que k_concepts à explain(), puis
# on filtre pour l'affichage seulement.
_POOL_MULTIPLIER = 5
_BRUIT_NUMERIQUE_RE = re.compile(r"[\d.,%€\-]+")
_LONGUEUR_MIN_TOKEN = 3
# Un concept partagé par une trop grande proportion des documents du DOSSIER
# COURANT ne distingue rien (ex. "tva"/"total_ht" sur un lot de devis) — calculé
# à chaque génération à partir des concepts réels du dossier, jamais une liste
# figée (le bruit "administratif" diffère selon le type de dossier indexé).
_SEUIL_BOILERPLATE = 0.8
# Plancher par carte : si le filtre boilerplate fait tomber une carte sous ce
# nombre de concepts (ex. 2 documents au vocabulaire quasi-identique où TOUT est
# "partagé"), on revient pour CETTE carte au top-k filtré du bruit numérique
# SEULEMENT — mieux vaut des concepts partagés visibles qu'une carte vide.
_PLANCHER_CONCEPTS = 3


def index_dir(dossier: Path) -> Path:
    """Emplacement de l'index `_MOSAIC/index` pour `dossier` — partagé avec la CLI."""
    return Path(dossier) / MOSAIC_DIRNAME / "index"


def html_path(dossier: Path) -> Path:
    """Emplacement de `_MOSAIC/cartes.html` pour `dossier` — partagé avec la CLI."""
    return Path(dossier) / MOSAIC_DIRNAME / "cartes.html"


def _token_count(file: Path, ingest_cache_dir: Path | None) -> int:
    """Nombre de tokens du document original — calcul d'affichage indépendant de
    la vectorisation (l'index ne garde pas ce compte par document). Réutilise le
    cache d'ingestion de `Index.build` s'il a été fourni : évite de reconvertir
    un PDF/docx déjà converti une première fois pour l'encodage."""
    if file.suffix.lower() in ingest.CONVERTIBLE_EXTS:
        text = ingest.to_text(file, cache_dir=ingest_cache_dir)
        return len(tokenize(text)) if text is not None else 0
    return len(tokenize(file.read_text(encoding="utf-8", errors="replace")))


def _est_bruit(token: str) -> bool:
    """Token trop court ou purement numérique/ponctuation pour être un concept
    lisible d'un coup d'œil (ex. "00", "20", "1", "-", "12,5%")."""
    return (
        len(token) < _LONGUEUR_MIN_TOKEN
        or _BRUIT_NUMERIQUE_RE.fullmatch(token) is not None
    )


def _sans_bruit(concepts: list[dict]) -> list[dict]:
    return [c for c in concepts if not _est_bruit(c["token"])]


def _stoplist_boilerplate(
    pools_sans_bruit: dict[str, list[dict]], k_concepts: int
) -> set[str]:
    """Tokens présents dans le top-k_concepts (déjà nettoyé du bruit numérique) de
    plus de `_SEUIL_BOILERPLATE` des documents du dossier. Nécessite au moins 2
    documents : à un seul document, aucune notion de « partagé par le dossier »
    n'est calculable, et le seuil viderait à tort la carte unique."""
    n_docs = len(pools_sans_bruit)
    if n_docs < 2:
        return set()
    doc_freq: dict[str, int] = {}
    for concepts in pools_sans_bruit.values():
        for token in {c["token"] for c in concepts[:k_concepts]}:
            doc_freq[token] = doc_freq.get(token, 0) + 1
    return {token for token, n in doc_freq.items() if n / n_docs > _SEUIL_BOILERPLATE}


def _concepts_pour_affichage(
    idx: Index, doc_ids: list[str], k_concepts: int
) -> dict[str, list[dict]]:
    """Concepts dominants par document, curés pour la lecture d'un coup d'œil :
    bassin `explain(k=k_concepts * _POOL_MULTIPLIER)` -> filtre bruit numérique ->
    filtre boilerplate propre au dossier -> les k_concepts premiers survivants.
    `explain()` lui-même n'est jamais modifié."""
    pool_size = max(k_concepts * _POOL_MULTIPLIER, k_concepts)
    pools = {
        doc_id: _sans_bruit(idx.explain(doc_id, k=pool_size)) for doc_id in doc_ids
    }
    stoplist = _stoplist_boilerplate(pools, k_concepts)
    plancher = min(_PLANCHER_CONCEPTS, k_concepts)
    resultat = {}
    for doc_id, concepts in pools.items():
        sans_boilerplate = [c for c in concepts if c["token"] not in stoplist][
            :k_concepts
        ]
        # Sous le plancher : le stoplist viderait (ou quasi-viderait) cette carte
        # précise -> on retombe sur le top-k numérique-filtré SEULEMENT (jamais le
        # bruit numérique, mais le boilerplate n'est plus appliqué pour cette carte).
        resultat[doc_id] = (
            sans_boilerplate
            if len(sans_boilerplate) >= plancher
            else concepts[:k_concepts]
        )
    return resultat


def _avec_largeurs(concepts: list[dict]) -> list[dict]:
    """Ajoute `pct` (largeur de barre 0-100, proportionnelle au poids max du doc)."""
    top = max((c["poids"] for c in concepts), default=0.0)
    for c in concepts:
        c["pct"] = round(max(c["poids"], 0.0) / top * 100, 1) if top > 0 else 0.0
    return concepts


_STYLE = """
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2.5rem 3rem 4rem;
  font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
  background: #f4f5f7; color: #1c1e21;
}
header {
  margin-bottom: 2rem; padding-bottom: 1.25rem; border-bottom: 1px solid #d8dade;
}
header h1 { margin: 0 0 .35rem; font-size: 1.6rem; font-weight: 600; }
header p { margin: 0; color: #5a5f68; font-size: .9rem; }
main.grid {
  display: grid; gap: 1.25rem;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
}
.card {
  background: #fff; border: 1px solid #e1e3e6; border-radius: 10px;
  padding: 1rem 1.1rem 1.15rem; box-shadow: 0 1px 3px rgba(0,0,0,.06);
  display: flex; flex-direction: column; gap: .7rem;
}
.card h2 { margin: 0; font-size: .95rem; font-weight: 600; overflow-wrap: anywhere; }
.card h2 a { color: #1c1e21; text-decoration: none; }
.card h2 a:hover { color: #b5651d; text-decoration: underline; }
.mosaique {
  width: 128px; height: 128px; align-self: center;
  image-rendering: pixelated; border-radius: 4px; border: 1px solid #e1e3e6;
}
.concepts { display: flex; flex-direction: column; gap: .3rem; }
.bar-row { display: grid; grid-template-columns: 6.5rem 1fr 3.4rem; align-items: center; gap: .5rem; }
.bar-label { font-size: .78rem; color: #33363b; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-track { background: #eceef1; border-radius: 3px; height: .55rem; overflow: hidden; }
.bar-fill { background: #b5651d; height: 100%; border-radius: 3px; }
.bar-poids { font-size: .72rem; color: #7a7f87; text-align: right; font-variant-numeric: tabular-nums; }
.tokens { margin: 0; font-size: .75rem; color: #8a8f97; border-top: 1px solid #eceef1; padding-top: .55rem; }
"""


def _bars_html(concepts: list[dict]) -> str:
    return "\n".join(
        '<div class="bar-row">'
        f'<span class="bar-label" title="{html.escape(c["token"])}">{html.escape(c["token"])}</span>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{c["pct"]}%"></div></div>'
        f'<span class="bar-poids">{c["poids"]:.4f}</span>'
        "</div>"
        for c in concepts
    )


def _card_html(
    idx: Index,
    dossier: Path,
    doc_id: str,
    concepts: list[dict],
    ingest_cache_dir: Path | None,
) -> str:
    row = idx.ids.index(doc_id)
    png_b64 = base64.b64encode(to_png(idx.mat[row], idx.grid)).decode("ascii")
    concepts = _avec_largeurs(concepts)
    original_uri = (dossier / doc_id).resolve().as_uri()
    tokens = _token_count(dossier / doc_id, ingest_cache_dir)
    name = Path(doc_id).name
    return f"""<article class="card">
  <h2><a href="{original_uri}" title="{html.escape(str((dossier / doc_id).resolve()))}">{html.escape(name)}</a></h2>
  <img class="mosaique" src="data:image/png;base64,{png_b64}" alt="mosaïque de {html.escape(name)}">
  <div class="concepts">
{_bars_html(concepts)}
  </div>
  <p class="tokens">{tokens} tokens</p>
</article>"""


def generer(
    dossier: Path,
    k_concepts: int = 8,
    grid: tuple[int, int, int] = GRID_DEFAULT,
    date_str: str | None = None,
    **build_kwargs,
) -> tuple[Path, int]:
    """Construit `dossier/_MOSAIC/index/` (via `Index.build`, ingestion incluse)
    puis écrit `dossier/_MOSAIC/cartes.html` : une carte par document, triées par
    nom. Retourne (chemin du HTML, nombre de documents) — le compte vient de
    l'index DÉJÀ construit, jamais d'une réouverture (sur un gros dossier, rouvrir
    l'index juste pour le compter doublait le coût de chargement).
    """
    dossier = Path(dossier)
    if date_str is None:
        date_str = date.today().strftime("%d/%m/%Y")
    idx = Index.build(dossier, index_dir(dossier), grid=grid, **build_kwargs)
    ingest_cache_dir = build_kwargs.get("ingest_cache_dir")
    doc_ids = sorted(idx.ids)
    concepts_par_doc = _concepts_pour_affichage(idx, doc_ids, k_concepts)
    cards = "\n".join(
        _card_html(idx, dossier, doc_id, concepts_par_doc[doc_id], ingest_cache_dir)
        for doc_id in doc_ids
    )
    n = len(doc_ids)
    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Carte d'identité — {html.escape(dossier.name)}</title>
<style>{_STYLE}</style>
</head>
<body>
<header>
  <h1>{html.escape(dossier.name)}</h1>
  <p>{n} document{"s" if n != 1 else ""} · généré le {html.escape(date_str)} · généré par Mosaic</p>
</header>
<main class="grid">
{cards}
</main>
</body>
</html>
"""
    out = html_path(dossier)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out, n
