"""Magasin STRUCTUREL — répondre par un COMPTE, pas par une liste de documents.

Une recherche rend des documents classés par ressemblance. Elle ne peut pas dire
« combien de bons de livraison en juin », « lequel est le plus récent », « quels
documents portent cette référence » : ce sont des questions d'AGRÉGATION, et
aucune amélioration du classement ne les résoudra — un moteur sémantique ne
compte pas, il ordonne. C'est l'angle mort que ce module ferme, sans modèle et
sans coût par requête (arXiv 2606.01435 échoue précisément sur ces cas, ce qui
confirme de l'extérieur que le trou est réel).

Le magasin est DÉRIVÉ : il se reconstruit en mémoire depuis les `facettes.json`
des index (type, date, références, chemin), en une fraction de seconde pour
quelques milliers de documents. Il n'y a donc rien à maintenir, rien à
synchroniser, et aucun risque de dérive avec l'index — la source de vérité reste
l'index, comme les fichiers restent la source de vérité de l'index.

Limite ASSUMÉE et mesurable : un document dont la date est inconnue
(« 0000-00-00 ») sort des agrégations temporelles. `sans_date()` chiffre ce trou
plutôt que de le cacher — mieux vaut un compte accompagné de sa couverture qu'un
compte qui prétend tout savoir.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

FICHIER_FACETTES = "facettes.json"
SANS_DATE = "0000-00-00"
_SCHEMA = """
CREATE TABLE documents (
    idx TEXT NOT NULL, doc_id TEXT NOT NULL, type TEXT, date TEXT,
    PRIMARY KEY (idx, doc_id)
);
CREATE TABLE refs (idx TEXT, doc_id TEXT, ref TEXT);
CREATE INDEX i_doc_date ON documents(idx, date);
CREATE INDEX i_doc_type ON documents(idx, type);
CREATE INDEX i_refs ON refs(ref);
"""


def _echapper(motif: str) -> str:
    """Neutralise les jokers SQL (`%`, `_`) : un chemin est cherché LITTÉRALEMENT.

    Sans ça, `IMG_2043` matcherait `IMGx2043` et un `%` élargirait la recherche à
    tout le corpus — une agrégation fausse est pire qu'une agrégation refusée."""
    return motif.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class Magasin:
    """Magasin en mémoire, alimenté par `charger()` puis interrogé par méthodes
    TYPÉES. Pas de SQL en entrée : les questions possibles sont celles qui ont
    une réponse exacte, et elles sont énumérées ici."""

    def __init__(self) -> None:
        self.db = sqlite3.connect(":memory:")
        self.db.executescript(_SCHEMA)

    def charger(self, nom: str, index_dir: Path | str) -> int:
        """Dérive le magasin des facettes d'un index. Rend le nombre de documents.

        Idempotent : recharger le même index remplace son contenu au lieu de le
        dupliquer (un magasin est un état, pas un journal)."""
        chemin = Path(index_dir) / FICHIER_FACETTES
        if not chemin.exists():
            raise FileNotFoundError(
                f"{chemin} absent : index construit avant les facettes (v1.5) — "
                "le reconstruire pour utiliser le magasin structurel"
            )
        return self.charger_depuis(nom, json.loads(chemin.read_text(encoding="utf-8")))

    def charger_depuis(self, nom: str, facettes: dict[str, dict]) -> int:
        """Même chose depuis des facettes DÉJÀ lues — pour un appelant qui garde le
        JSON en cache (le serveur MCP) et ne veut pas repayer la lecture disque à
        chaque question. Rend le nombre de documents."""
        self.db.execute("DELETE FROM documents WHERE idx = ?", (nom,))
        self.db.execute("DELETE FROM refs WHERE idx = ?", (nom,))
        self.db.executemany(
            "INSERT INTO documents VALUES (?,?,?,?)",
            [
                (nom, doc_id, fac.get("type"), fac.get("date", SANS_DATE))
                for doc_id, fac in facettes.items()
            ],
        )
        self.db.executemany(
            "INSERT INTO refs VALUES (?,?,?)",
            [
                (nom, doc_id, ref)
                for doc_id, fac in facettes.items()
                for ref in fac.get("refs", ())
            ],
        )
        self.db.commit()
        return len(facettes)

    def _filtres(
        self, idx: str, chemin_contient: str, type_doc: str, date_prefixe: str
    ) -> tuple[str, list]:
        q, args = " WHERE idx = ?", [idx]
        if chemin_contient:
            q += " AND doc_id LIKE ? ESCAPE '\\'"
            args.append(f"%{_echapper(chemin_contient)}%")
        if type_doc:
            q += " AND type = ?"
            args.append(type_doc)
        if date_prefixe:
            q += " AND date LIKE ? ESCAPE '\\'"
            args.append(f"{_echapper(date_prefixe)}%")
        return q, args

    def compter(
        self,
        idx: str,
        chemin_contient: str = "",
        type_doc: str = "",
        date_prefixe: str = "",
    ) -> int:
        """« Combien de documents … ? » — filtres cumulatifs, tous optionnels.
        Un index inconnu rend 0, jamais une erreur : l'absence est une réponse."""
        q, args = self._filtres(idx, chemin_contient, type_doc, date_prefixe)
        return self.db.execute("SELECT COUNT(*) FROM documents" + q, args).fetchone()[0]

    def plus_recents(
        self,
        idx: str,
        k: int = 1,
        chemin_contient: str = "",
        type_doc: str = "",
    ) -> list[tuple[str, str]]:
        """« Le (k-ième) plus récent … ? » — rend [(doc_id, date)] du plus récent
        au plus ancien. Les documents SANS date sont exclus : on ne les classe
        pas au hasard, on ne les fait pas passer pour vieux."""
        q, args = self._filtres(idx, chemin_contient, type_doc, "")
        q += " AND date != ? ORDER BY date DESC, doc_id LIMIT ?"
        args += [SANS_DATE, k]
        return list(self.db.execute("SELECT doc_id, date FROM documents" + q, args))

    def repartition_par_mois(
        self, idx: str, chemin_contient: str = "", type_doc: str = ""
    ) -> dict[str, int]:
        """« Combien par mois ? » — d'où se lisent le mois le plus chargé et les
        trous. Les documents sans date sont exclus (voir `sans_date`)."""
        q, args = self._filtres(idx, chemin_contient, type_doc, "")
        q += " AND date != ? GROUP BY m ORDER BY m"
        args.append(SANS_DATE)
        return dict(
            self.db.execute(
                "SELECT substr(date, 1, 7) m, COUNT(*) FROM documents" + q, args
            )
        )

    def documents_portant_ref(self, ref: str) -> list[tuple[str, str, str]]:
        """« Quels documents portent cette référence ? » — TOUS index confondus,
        du plus récent au plus ancien : rend [(index, doc_id, date)].

        C'est la jointure qui manque à la recherche : une référence relie un devis,
        un bon de livraison et une facture sans qu'aucun mot ne se ressemble."""
        return list(
            self.db.execute(
                "SELECT r.idx, r.doc_id, d.date FROM refs r "
                "JOIN documents d ON d.idx = r.idx AND d.doc_id = r.doc_id "
                "WHERE r.ref = ? ORDER BY d.date DESC, r.doc_id",
                (ref,),
            )
        )

    def types_disponibles(self, idx: str) -> dict[str, int]:
        """Les types de documents RÉELLEMENT présents, avec leurs effectifs.

        Sert à corriger une qualification fausse : un appelant qui filtre sur un
        type absent reçoit zéro résultat et en conclut que le document n'existe
        pas — alors qu'il l'a lui-même exclu. Lui rendre le vocabulaire réel du
        domaine le laisse se corriger en un tour, sans deviner."""
        return dict(
            self.db.execute(
                "SELECT type, COUNT(*) c FROM documents WHERE idx = ? AND type IS NOT NULL "
                "GROUP BY type ORDER BY c DESC",
                (idx,),
            )
        )

    def sans_date(self, idx: str) -> int:
        """Le trou de couverture, chiffré : combien de documents échappent aux
        agrégations temporelles. À rendre AVEC tout compte daté."""
        return self.compter(idx, date_prefixe=SANS_DATE)
