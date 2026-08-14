"""Garde-fou d'hygiène : les fixtures et les descriptions restent synthétiques.

Un dépôt de moteur de recherche accumule vite des données d'exemple — fixtures de
tests, extraits de corpus, exemples de requêtes dans les descriptions d'outils.
Ce contrôle garantit qu'elles restent entièrement synthétiques : `gitleaks`
attrape les secrets à signature (clés, jetons, IBAN), ce garde-fou attrape ce qui
n'en a pas — un nom propre, un chemin de poste de travail, une adresse mail.

PRINCIPE : la liste de termes à refuser est propre à chaque site et vit HORS du
dépôt (`~/.mosaic/noms_interdits.txt`, un terme par ligne, `#` pour un
commentaire, ou le chemin donné par MOSAIC_NOMS_INTERDITS). Le script est publié
sans elle et reste inoffensif : liste absente, seuls les contrôles structurels
tournent — un contributeur extérieur n'est jamais bloqué par une liste qu'il n'a
pas.

DEUX PASSAGES LIBRES, voulus :
- **dépôt non public** : le contrôle ne mord que si `origin` correspond au motif
  MOSAIC_REMOTE_PUBLIC — un dépôt de travail local garde ses libertés.
- **liste absente** : avertissement sur stderr, contrôles structurels seulement.

Usage :
    python scripts/verifier_hygiene.py               # fichiers suivis par git
    python scripts/verifier_hygiene.py f1 f2         # fichiers précis (hook)
    python scripts/verifier_hygiene.py --historique  # historique + tags + auteurs
    MOSAIC_HYGIENE_FORCE=1 python scripts/…         # contrôler quel que soit le remote

Le mode --historique regarde ce que l'arbre courant ne peut pas montrer : chaque
commit atteignable (branches ET tags), chaque arbre, et l'identité de chaque
auteur/committer — l'arbre de tête ne dit rien de ce qu'un vieux commit ou un tag
continue de servir.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

DEFAUT_LISTE = Path.home() / ".mosaic" / "noms_interdits.txt"
# Binaires, données de banc et caches : rien à contrôler, et les lire coûte cher.
EXTENSIONS_IGNOREES = {
    ".msee",
    ".msei",
    ".msbm",
    ".msat",
    ".msrel",
    ".msrv",
    ".msev",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".gguf",
    ".zip",
    ".gz",
    ".pyc",
}
DOSSIERS_IGNORES = {".git", "__pycache__", "data_externes"}
# Dictionnaires de LANGUE : un terme de la liste peut y être un mot courant d'une
# autre langue. Les exclure évite un faux positif permanent et illisible.
FICHIERS_IGNORES = {"lexicon_wikdict_fr_en.json"}

# Identité attendue de TOUT commit de ce dépôt public.
EMAIL_PUBLIC = "alexxb2mg@users.noreply.github.com"

# Motifs STRUCTURELS — pas besoin de liste, toujours contrôlés : un chemin de
# poste de travail ou un email personnel n'a rien à faire dans un dépôt public.
# Le chemin Windows tolère 1 à 2 antislashs : dans un JSON il est stocké échappé
# (antislashs doublés), et la forme simple ne le matche pas.
MOTIFS_STRUCTURELS: list[tuple[str, re.Pattern[str]]] = [
    ("chemin de poste", re.compile(r"C:[/\\]+(?:Users|LOCAL)", re.IGNORECASE)),
    (
        "email hors noreply",
        re.compile(
            r"[\w.+-]+@(?!users\.noreply\.github\.com)[\w-]+\.\w{2,}", re.IGNORECASE
        ),
    ),
]


def depot_est_public() -> bool:
    """Le dépôt courant est-il le dépôt public ? (motif cherché dans l'URL d'origin)

    Le motif par défaut est l'URL publique elle-même — la publier ici n'est pas un
    problème, c'est un dépôt public. Un remote absent (clone local, CI détachée)
    est traité comme NON public : on ne bloque pas ce qu'on ne sait pas
    identifier."""
    if os.environ.get("MOSAIC_HYGIENE_FORCE"):
        return True
    motif = os.environ.get("MOSAIC_REMOTE_PUBLIC", "alexxb2mg-svg/Mosaic")
    try:
        url = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return False
    return motif.lower() in url.lower()


def charger_liste() -> list[str]:
    chemin = Path(os.environ.get("MOSAIC_NOMS_INTERDITS", DEFAUT_LISTE))
    if not chemin.exists():
        print(
            f"[hygiene] liste absente ({chemin}) — contrôles structurels seulement.",
            file=sys.stderr,
        )
        return []
    return [
        li.strip()
        for li in chemin.read_text(encoding="utf-8").splitlines()
        if li.strip() and not li.startswith("#")
    ]


def fichiers_a_verifier(args: list[str]) -> list[Path]:
    if args:
        return [Path(a) for a in args]
    sortie = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [Path(li) for li in sortie.splitlines() if li.strip()]


def compiler_noms(noms: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    # Bornes par LETTRES uniquement, pas \b : un underscore ou un chiffre collé au
    # terme est un caractère de mot, `\bterme\b` ne matcherait pas TERME_2026.
    # « Roger » ne matche toujours pas « Rogerien » : la borne reste une lettre.
    # CONVENTION DE CASSE : une entrée de la liste TOUT EN MAJUSCULES est cherchée
    # en respectant la casse — un sigle sans interdire le nom commun homographe.
    return [
        (
            n,
            re.compile(
                rf"(?<![A-Za-zÀ-ÿ]){re.escape(n)}(?![A-Za-zÀ-ÿ])",
                0 if n.isupper() else re.IGNORECASE,
            ),
        )
        for n in noms
    ]


def verifier_historique(motifs: list[tuple[str, re.Pattern[str]]]) -> int:
    """Contrôle l'HISTORIQUE complet : tous les commits atteignables (branches ET
    tags), leurs arbres, et l'identité de chaque auteur/committer.

    `git ls-files` ne voit que l'arbre courant ; un vieux commit resté atteignable
    par un tag, ou un commit signé d'une identité inattendue, ne se voit qu'ici."""
    problemes: list[str] = []

    identites = subprocess.run(
        ["git", "log", "--all", "--format=%an <%ae>%n%cn <%ce>"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    for ident in sorted(set(identites)):
        if ident and EMAIL_PUBLIC not in ident:
            problemes.append(f"identité inattendue : {ident}")

    revs = subprocess.run(
        ["git", "rev-list", "--all"], capture_output=True, text=True, check=True
    ).stdout.split()
    # La sensibilité à la casse de chaque motif compilé est reportée en PCRE
    # inline `(?i)` — un `-i` global écraserait la convention de casse de la liste.
    a_chercher = [
        (nom, ("(?i)" if m.flags & re.IGNORECASE else "") + m.pattern)
        for nom, m in motifs + MOTIFS_STRUCTURELS
    ]
    for nom, pattern in a_chercher:
        # -P (PCRE) obligatoire : les motifs portent des lookarounds, que le -E
        # POSIX rejette — et un motif rejeté sortirait en code 2, silencieusement
        # confondu avec « aucun résultat » si on ne distinguait pas les codes.
        grep = subprocess.run(
            ["git", "grep", "-l", "-I", "-P", pattern, *revs],
            capture_output=True,
            text=True,
        )
        if grep.returncode > 1:
            problemes.append(
                f"échec git grep sur « {nom} » : {grep.stderr.strip()[:120]}"
            )
            continue
        for ligne in grep.stdout.splitlines():
            if any(f in ligne for f in FICHIERS_IGNORES):
                continue
            problemes.append(f"« {nom} » dans {ligne}")

    if problemes:
        print(
            f"[hygiene] HISTORIQUE REFUSÉ — {len(problemes)} problème(s) :",
            file=sys.stderr,
        )
        for p in problemes[:30]:
            print(f"  ! {p}", file=sys.stderr)
        return 1
    print(f"[hygiene] historique OK — {len(revs)} commits contrôlés (branches + tags).")
    return 0


def main(argv: list[str]) -> int:
    if not depot_est_public():
        return 0  # dépôt de travail : ses données ne regardent que lui
    noms = charger_liste()
    motifs = compiler_noms(noms)

    if "--historique" in argv:
        return verifier_historique(motifs)

    trouvailles: list[tuple[Path, int, str]] = []
    fichiers = fichiers_a_verifier([a for a in argv if not a.startswith("--")])
    for f in fichiers:
        if (
            f.suffix.lower() in EXTENSIONS_IGNOREES
            or f.name in FICHIERS_IGNORES
            or set(f.parts) & DOSSIERS_IGNORES
            or not f.is_file()
        ):
            continue
        try:
            texte = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for num, ligne in enumerate(texte.splitlines(), 1):
            for nom, motif in motifs + MOTIFS_STRUCTURELS:
                if motif.search(ligne):
                    trouvailles.append((f, num, nom))

    if trouvailles:
        print(
            f"[hygiene] REFUSÉ — {len(trouvailles)} occurrence(s) :",
            file=sys.stderr,
        )
        for f, num, nom in trouvailles[:20]:
            print(f"  {f}:{num} -> « {nom} »", file=sys.stderr)
        print(
            "\nRemplacer par une valeur synthétique (FOURNISSEUR, client, AFFAIRE). "
            "Les données d'exemple restent inventées.",
            file=sys.stderr,
        )
        return 1
    print(f"[hygiene] OK — {len(noms)} termes + motifs structurels, rien trouvé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
