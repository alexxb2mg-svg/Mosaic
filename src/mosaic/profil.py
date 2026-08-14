"""Profil d'index — le paramétrage déclaratif qui adapte Mosaic à UN environnement métier.

Ce que le profil décrit (niveau « décrire ton monde », jamais la géométrie du moteur) :
  - `roles`   : comment lire l'arborescence — quels motifs de segments de chemin deviennent
                quelles entités du graphe de relations (client/affaire, projet/version…) ;
  - `types`   : extensions supplémentaires -> types de documents (.dwg -> plan, .py -> code) ;
  - `refs`    : à quoi ressemble une référence/code dans ce métier (longueurs minimales,
                motif regex optionnel).

Trois principes (revue de paramétrage, 11/08) :
  1. PERSISTÉ dans l'index (meta) — build/add/search relisent le même profil, jamais de
     divergence silencieuse ; même corpus + même profil = même index (déterminisme intact).
  2. Un profil invalide CASSE FORT ET TÔT (validation stricte au chargement) — jamais une
     dégradation muette.
  3. DÉCOUVRABLE : `mosaic profil <index>` le montre ; `--explique` le raconte en français
     (mode humain) ; `mosaic profil <corpus> --suggere` le CALIBRE depuis le corpus réel
     (mode agent : scanner l'environnement, proposer, ajuster, construire).

Sans profil, rien ne change : les défauts sont le comportement historique (le « profil
LOCAL » implicite).
"""

import json
import re
from pathlib import Path

# Rôles par défaut = la logique historique entities_from_path (dossier/annee/mois avec
# contexte d'année). Un profil `roles` la REMPLACE entièrement (déclaratif pur, tout visible).
ROLES_RESERVES_DEFAUT = None  # sentinel : « utiliser la logique historique »

_CLES_VALIDES = {"nom", "description", "roles", "types", "refs", "grilles", "grammaire"}
# grilles typées (v4) : surcharge des types de base (dim/poids/lissage/embeddings)
# ou déclaration de types custom (motif de routage OBLIGATOIRE pour un type custom)
_CLES_GRILLE = {"dim", "poids", "lissage", "embeddings", "motif"}
_TYPES_BASE = {"sens", "ref", "chemin"}
_CLES_ROLE = {"role", "motif", "valeur"}
_CLES_REFS = {"min_mixte", "min_chiffres", "motif"}
# canal grammatical : seules les listes verbales OUVERTES AU MÉTIER sont extensibles —
# les classes fermées du français (négateurs, copules, portées) ne le sont pas
_CLES_GRAMMAIRE = {"verbes_actifs", "participes_passifs", "saut_gauche"}


def valider(profil: dict) -> dict:
    """Validation STRICTE — toute anomalie lève ValueError avec le chemin exact du problème.
    Rend le profil normalisé (regex compilables vérifiées, types en minuscules)."""
    if not isinstance(profil, dict):
        raise ValueError(f"profil : objet JSON attendu, reçu {type(profil).__name__}")
    inconnues = set(profil) - _CLES_VALIDES
    if inconnues:
        raise ValueError(
            f"profil : clés inconnues {sorted(inconnues)} (valides : {sorted(_CLES_VALIDES)})"
        )
    if "roles" in profil:
        if not isinstance(profil["roles"], list) or not profil["roles"]:
            raise ValueError("profil.roles : liste non vide de règles attendue")
        for i, regle_brute in enumerate(profil["roles"]):
            if not isinstance(regle_brute, dict):
                raise ValueError(
                    f"profil.roles[{i}] : objet attendu, reçu {regle_brute!r}"
                )
            regle: dict = regle_brute
            if set(regle) - _CLES_ROLE:
                raise ValueError(
                    f"profil.roles[{i}] : clés valides {sorted(_CLES_ROLE)}, reçu {regle!r}"
                )
            role_val = regle.get("role")
            if not role_val or not isinstance(role_val, str):
                raise ValueError(f"profil.roles[{i}].role : nom de rôle (str) requis")
            motif_val = regle.get("motif")
            if not motif_val or not isinstance(motif_val, str):
                raise ValueError(f"profil.roles[{i}].motif : regex (str) requise")
            try:
                re.compile(motif_val)
            except re.error as exc:
                raise ValueError(
                    f"profil.roles[{i}].motif : regex invalide {motif_val!r} ({exc})"
                ) from None
    if "types" in profil:
        if not isinstance(profil["types"], dict):
            raise ValueError("profil.types : objet {extension: type} attendu")
        for ext, t in profil["types"].items():
            if not ext.startswith(".") or not isinstance(t, str) or not t:
                raise ValueError(
                    f"profil.types : {ext!r}: {t!r} — extension '.xxx' -> libellé non vide"
                )
    if "refs" in profil:
        r_brut = profil["refs"]
        if not isinstance(r_brut, dict):
            raise ValueError(f"profil.refs : objet attendu, reçu {r_brut!r}")
        r: dict = r_brut
        if set(r) - _CLES_REFS:
            raise ValueError(
                f"profil.refs : clés valides {sorted(_CLES_REFS)}, reçu {r!r}"
            )
        for cle in ("min_mixte", "min_chiffres"):
            if cle in r and (not isinstance(r[cle], int) or r[cle] < 1):
                raise ValueError(
                    f"profil.refs.{cle} : entier >= 1 attendu, reçu {r[cle]!r}"
                )
        motif_refs = r.get("motif")
        if motif_refs is not None:
            if not isinstance(motif_refs, str):
                raise ValueError(
                    f"profil.refs.motif : regex (str) attendue, reçu {motif_refs!r}"
                )
            try:
                re.compile(motif_refs)
            except re.error as exc:
                raise ValueError(
                    f"profil.refs.motif : regex invalide {motif_refs!r} ({exc})"
                ) from None
    if "grammaire" in profil:
        gram_brut = profil["grammaire"]
        if not isinstance(gram_brut, dict) or not gram_brut:
            raise ValueError(
                f"profil.grammaire : objet non vide attendu, reçu {gram_brut!r}"
            )
        gram: dict = gram_brut
        if set(gram) - _CLES_GRAMMAIRE:
            raise ValueError(
                f"profil.grammaire : clés valides {sorted(_CLES_GRAMMAIRE)}, "
                f"reçu {sorted(gram)}"
            )
        for cle_g, formes in gram.items():
            if (
                not isinstance(formes, list)
                or not formes
                or not all(isinstance(f, str) and f for f in formes)
            ):
                raise ValueError(
                    f"profil.grammaire.{cle_g} : liste non vide de formes (str) attendue"
                )
            en_faute = [f for f in formes if f != f.lower() or " " in f]
            if en_faute:
                raise ValueError(
                    f"profil.grammaire.{cle_g} : formes en minuscules, un seul mot "
                    f"chacune (l'analyseur compare des tokens) — reçu {en_faute!r}"
                )
    if "grilles" in profil:
        g_brut = profil["grilles"]
        if not isinstance(g_brut, dict) or not g_brut:
            raise ValueError(
                f"profil.grilles : objet non vide attendu, reçu {g_brut!r}"
            )
        for nom_g, cfg_brut in g_brut.items():
            if not isinstance(cfg_brut, dict):
                raise ValueError(
                    f"profil.grilles.{nom_g} : objet attendu, reçu {cfg_brut!r}"
                )
            cfg: dict = cfg_brut
            if set(cfg) - _CLES_GRILLE:
                raise ValueError(
                    f"profil.grilles.{nom_g} : clés valides {sorted(_CLES_GRILLE)}, "
                    f"reçu {cfg!r}"
                )
            if nom_g not in _TYPES_BASE and "motif" not in cfg:
                raise ValueError(
                    f"profil.grilles.{nom_g} : un type custom exige un `motif` "
                    "(sa règle de routage) — sans lui, la grille ne recevrait jamais rien"
                )
            if "motif" in cfg:
                if nom_g in _TYPES_BASE:
                    raise ValueError(
                        f"profil.grilles.{nom_g} : `motif` interdit sur un type de base "
                        "(leur routage est la règle du moteur, pas une regex)"
                    )
                try:
                    re.compile(str(cfg["motif"]))
                except re.error as exc:
                    raise ValueError(
                        f"profil.grilles.{nom_g}.motif : regex invalide ({exc})"
                    ) from None
            if "dim" in cfg and (not isinstance(cfg["dim"], int) or cfg["dim"] < 3):
                raise ValueError(
                    f"profil.grilles.{nom_g}.dim : entier >= 3 attendu, reçu {cfg['dim']!r}"
                )
            if "lissage" in cfg and (
                not isinstance(cfg["lissage"], int) or cfg["lissage"] < 0
            ):
                raise ValueError(
                    f"profil.grilles.{nom_g}.lissage : entier >= 0 attendu, "
                    f"reçu {cfg['lissage']!r}"
                )
            if "poids" in cfg and (
                not isinstance(cfg["poids"], (list, tuple))
                or len(cfg["poids"]) != 3
                or not all(isinstance(x, (int, float)) for x in cfg["poids"])
            ):
                raise ValueError(
                    f"profil.grilles.{nom_g}.poids : triplet numérique attendu, "
                    f"reçu {cfg['poids']!r}"
                )
    return {
        **profil,
        "types": {k.lower(): v for k, v in profil.get("types", {}).items()},
    }


def charger(chemin: Path) -> dict:
    """Charge et valide un profil JSON. Erreurs de syntaxe et de schéma : loud, avec contexte."""
    try:
        brut = json.loads(Path(chemin).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"profil {chemin} : JSON invalide ({exc})") from None
    return valider(brut)


def roles_du_profil(profil: dict | None):
    """Les règles de rôles compilées du profil, ou None (= logique historique)."""
    if not profil or "roles" not in profil:
        return None
    return [
        (re.compile(r["motif"]), r["role"], r.get("valeur")) for r in profil["roles"]
    ]


def expliquer(profil: dict | None, langue: str = "fr") -> str:
    """MODE HUMAIN : raconte ce que le profil fait faire au moteur — chaque règle, son effet
    concret, et ce qui reste au comportement par défaut. `langue` : "fr" (défaut) ou "en".
    NB : les types par défaut (« tableur », « pdf scanné »…) sont des identifiants canoniques
    persistés dans les index — l'anglais les GLOSE mais ne les renomme pas ; un profil peut
    définir ses propres types en anglais (clé `types`)."""
    if langue not in ("fr", "en"):
        raise ValueError(f"langue : 'fr' ou 'en' attendu, reçu {langue!r}")
    en = langue == "en"
    lignes = []
    if not profil:
        if en:
            return (
                "No profile: default behavior.\n"
                '- Folder tree: every folder in the path becomes a "dossier" (folder) '
                "entity; a 2026 segment becomes the year; 08-Août or 08.2026 becomes the "
                "month.\n"
                "- Document types (canonical ids): tableur (spreadsheet), pdf numérique "
                "(digital pdf), pdf scanné (scanned pdf), photo, document rédigé (written "
                "document), présentation, page web, note texte (text note).\n"
                "- References: a code of at least 5 mixed letters+digits, or at least "
                "6 digits."
            )
        return (
            "Aucun profil : comportement par défaut.\n"
            "- Arborescence : chaque dossier du chemin devient une entité « dossier » ; un "
            "segment 2026 devient l'année ; 08-Août ou 08.2026 devient le mois.\n"
            "- Types de documents : tableur, pdf numérique/scanné, photo, document rédigé, "
            "présentation, page web, note texte.\n"
            "- Références : un code d'au moins 5 caractères mêlant lettres et chiffres, ou "
            "d'au moins 6 chiffres."
        )
    nom = profil.get("nom", "(sans nom)" if not en else "(unnamed)")
    lignes.append(f'Profile "{nom}"' if en else f"Profil « {nom} »")
    if profil.get("description"):
        lignes.append(f"  {profil['description']}")
    if "roles" in profil:
        lignes.append(
            "\nFolder-tree reading (replaces the default) — for each folder in the path, "
            "the first matching rule wins:"
            if en
            else "\nLecture de l'arborescence (remplace la lecture par défaut) — pour chaque "
            "dossier du chemin, la première règle qui correspond gagne :"
        )
        for r in profil["roles"]:
            if en:
                valeur = (
                    f', kept value is "{r["valeur"]}" (pattern groups)'
                    if r.get("valeur")
                    else ""
                )
                lignes.append(
                    f"  - a folder matching /{r['motif']}/ becomes a "
                    f'"{r["role"]}" entity{valeur}'
                )
            else:
                valeur = (
                    f", la valeur retenue est « {r['valeur']} » (groupes du motif)"
                    if r.get("valeur")
                    else ""
                )
                lignes.append(
                    f"  - un dossier qui ressemble à /{r['motif']}/ devient une entité "
                    f"« {r['role']} »{valeur}"
                )
        lignes.append(
            '  Effect: "mosaic chemin" and "mosaic related" reason with THESE roles.'
            if en
            else "  Effet : « mosaic chemin » et « mosaic related » raisonnent avec CES rôles."
        )
    else:
        lignes.append(
            "\nFolder tree: default reading (dossier / annee / mois)."
            if en
            else "\nArborescence : lecture par défaut (dossier / année / mois)."
        )
    if profil.get("types"):
        lignes.append(
            "\nAdded document types (on top of the defaults):"
            if en
            else "\nTypes de documents ajoutés (en plus des types par défaut) :"
        )
        for ext, t in sorted(profil["types"].items()):
            lignes.append(
                f'  - {ext} files are "{t}"'
                if en
                else f"  - les fichiers {ext} sont des « {t} »"
            )
        lignes.append(
            '  Effect: searchable by type ("mosaic search --type ...").'
            if en
            else "  Effet : cherchables par type (« mosaic search --type ... »)."
        )
    if profil.get("refs"):
        r = profil["refs"]
        if "motif" in r:
            lignes.append(
                f"\nReferences: any code matching /{r['motif']}/ (replaces the default rule)."
                if en
                else f"\nRéférences : tout code correspondant au motif /{r['motif']}/ "
                "(remplace le critère par défaut)."
            )
        else:
            lignes.append(
                f"\nReferences: at least {r.get('min_mixte', 5)} mixed letters+digits, or "
                f"at least {r.get('min_chiffres', 6)} digits."
                if en
                else f"\nRéférences : au moins {r.get('min_mixte', 5)} caractères mêlant "
                f"lettres et chiffres, ou au moins {r.get('min_chiffres', 6)} chiffres."
            )
        lignes.append(
            "  Effect: a reference typed in a query pushes the documents that carry it "
            "exactly to the top."
            if en
            else "  Effet : une référence tapée dans une question fait remonter en tête les "
            "documents qui la portent exactement."
        )
    return "\n".join(lignes)


def suggerer(corpus_dir: Path, max_fichiers: int = 4000) -> dict:
    """MODE AGENT (calibration d'environnement) : scanne le corpus réel et propose un profil
    candidat — extensions inconnues à mapper, motifs de segments de chemins observés (années,
    mois, codes, libellés), avec commentaires. L'agent (ou l'humain) ajuste puis construit."""
    corpus_dir = Path(corpus_dir)
    exts_connues = {
        ".md",
        ".txt",
        ".pdf",
        ".docx",
        ".xlsx",
        ".html",
        ".pptx",
        ".jpg",
        ".jpeg",
        ".png",
        ".tiff",
    }
    exts_vues: dict[str, int] = {}
    segments: dict[str, int] = {}
    n = 0
    for p in corpus_dir.rglob("*"):
        if not p.is_file():
            continue
        n += 1
        if n > max_fichiers:
            break
        ext = p.suffix.lower()
        if ext:
            exts_vues[ext] = exts_vues.get(ext, 0) + 1
        for seg in p.relative_to(corpus_dir).parts[:-1]:
            segments[seg] = segments.get(seg, 0) + 1

    roles: list[dict] = []
    if any(re.fullmatch(r"20\d\d", s) for s in segments):
        roles.append({"role": "annee", "motif": r"^20\d\d$"})
    if any(re.fullmatch(r"(0[1-9]|1[0-2])\.\d{4}", s) for s in segments):
        roles.append(
            {
                "role": "mois",
                "motif": r"^(0[1-9]|1[0-2])\.(\d{4})$",
                "valeur": "{2}-{1}",
            }
        )
    if any(re.fullmatch(r"(0[1-9]|1[0-2])-.+", s) for s in segments):
        roles.append({"role": "mois_libelle", "motif": r"^(0[1-9]|1[0-2])-(.+)$"})
    roles.append({"role": "dossier", "motif": ".*"})  # attrape-tout final, explicite

    types_a_mapper = {
        ext: "?"
        for ext, _c in sorted(exts_vues.items(), key=lambda x: -x[1])
        if ext not in exts_connues
    }
    profil: dict = {
        "nom": corpus_dir.name,
        "description": f"Profil suggéré depuis {n} fichiers — à relire et ajuster.",
        "roles": roles,
    }
    if types_a_mapper:
        profil["types"] = (
            types_a_mapper  # « ? » à remplacer par un libellé, sinon retirer
        )
    return profil
