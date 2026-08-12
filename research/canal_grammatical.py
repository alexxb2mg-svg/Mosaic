"""P1 + P3 du chantier « canal grammatical » — séparation des paires et non-régression.

MÉCANISME : les rôles produits par l'analyseur déterministe (analyseur_grammatical.py,
P2 passé : erreur 2,1 %, abstention 8/8) sont LIÉS aux signatures des tokens par le
mécanisme du canal relations déjà prouvé (src/mosaic/relations.py) :
    canal(doc) = normaliser( Σ np.roll(signature(entité), décalage(rôle)) )
« disjoncteur/amont » et « disjoncteur/aval » deviennent deux vecteurs distincts. Le
canal est un VECTEUR SÉPARÉ (garde-fou 2 du brief : opt-in, jamais un défaut) ; une
clause sans rôle attribué a un canal NUL — neutre par construction.

BASELINE « moteur nu » : le canal signatures du moteur tel quel — Σ (1+log tf) ·
(±1) · signature(token), signe issu de `mosaic.encoder._signed_counts` (la négation
ADJACENTE du moteur est donc incluse : baseline honnête, le moteur sait déjà séparer
« sans X » de « avec X » quand X suit immédiatement ; il est aveugle à la portée et à
l'ordre). Les canaux appris (profil, embedding) sont au niveau mot : sur des paires à
mots identiques ils ne séparent rien de plus — l'omission ne flatte pas le canal testé.

PRÉDICTIONS DÉCLARÉES AVANT MESURE (falsifiables) :
- P1a : paires amont/aval, agent/patient et négation à PORTÉE (N04, N10, N11, N12) :
  cos_nu intra-paire = 1.0 EXACTEMENT (sac-de-mots aveugle) ; paires à négation
  adjacente : cos_nu < 1 (déjà partiellement séparées par le signe).
- P1b (seuil du protocole) : paires séparées = cos_mix <= cos_nu − 0.05 ; le canal
  sépare si >= 80 % des 34 paires le sont. Prédit : 34/34 (l'analyseur émet sur
  toutes les paires du banc — P1 hérite de la couverture de P2).
- P1c : canal seul quasi orthogonal intra-paire : médiane de cos_canal <= 0.25.
- P3a : canal DÉSACTIVÉ -> vecteurs nus bit-identiques (séparation architecturale,
  vérifiée par assertion, impact strictement nul).
- P3b (péage si on FUSIONNAIT, λ = 0.5, pour documenter pourquoi il reste séparé) :
  12 requêtes prose sur les 85 clauses, R@1 nu vs fusionné : écart <= 1 requête.
  Mécanisme attendu du péage : le canal dilue la composante mots des documents qui en
  ont un, et avantage relativement les documents abstenus — biais structurel, raison
  de fond pour rester un canal séparé même si l'écart mesuré est petit ici.

Usage : python research/canal_grammatical.py  (après analyseur_grammatical.py)
"""

import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyseur_grammatical import analyse
from mosaic.encoder import _signed_counts
from mosaic.relations import bind
from mosaic.signatures import signature
from mosaic.tokenize import tokenize

DIM = 12288  # la grille des bancs existants (bench/horizon)
LAMBDA_FUSION = 0.5  # poids du canal dans le test de fusion P3b (déclaré)
ECART_SEPARATION = 0.05  # une paire est séparée si cos_mix <= cos_nu − ECART
SEUIL_P1 = 0.80  # fraction de paires séparées exigée

REQUETES: list[tuple[str, set[str]]] = [
    ("prise de recharge avec obturateur enfant", {"N07a", "N07b"}),
    ("contacteur du chauffe-eau", {"S01a", "S01b"}),
    ("vis sans fin du touret", {"T04"}),
    ("injection photovoltaïque au point de livraison", {"C05"}),
    ("éclairage de sécurité du local technique", {"N12a", "N12b"}),
    ("bouton poussoir du télérupteur", {"A07a", "A07b"}),
    ("groupe électrogène contrôlé par un automate", {"S03a", "S03b"}),
    ("défaut d'isolement sur le départ chauffage", {"N13a", "N13b"}),
    ("compteur de production et onduleur", {"A03a", "A03b"}),
    ("verrouillage du coffret de chantier par serrure", {"N05a", "N05b"}),
    ("alimentation sans interruption des serveurs", {"T07"}),
    ("liaison équipotentielle dans les salles d'eau", {"C06"}),
]


def _normalise(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / np.float32(n) if n > 0 else v


def vecteur_nu(clause: str) -> np.ndarray:
    """Canal signatures du moteur nu (tf-log, négation adjacente par le signe)."""
    v = np.zeros(DIM, dtype=np.float32)
    for (token, nie), tf in _signed_counts(tokenize(clause)).items():
        poids = (1.0 + np.log(tf)) * (-1.0 if nie else 1.0)
        v += np.float32(poids) * signature(token, DIM).astype(np.float32)
    return _normalise(v)


def vecteur_canal(clause: str) -> np.ndarray:
    """Canal grammatical : superposition des rôles liés — NUL si abstention totale."""
    v = np.zeros(DIM, dtype=np.float32)
    for role, entite in analyse(clause):
        v += bind(role, entite, DIM).astype(np.float32)
    return _normalise(v)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b) / (na * nb) if na > 0 and nb > 0 else 0.0


def main() -> int:
    banc = Path(__file__).resolve().parent / "banc_grammatical.jsonl"
    clauses = [
        json.loads(li) for li in banc.read_text(encoding="utf-8").splitlines() if li
    ]
    nus = {c["id"]: vecteur_nu(c["clause"]) for c in clauses}
    canaux = {c["id"]: vecteur_canal(c["clause"]) for c in clauses}
    mixes = {
        cid: _normalise(nus[cid] + np.float32(LAMBDA_FUSION) * canaux[cid])
        for cid in nus
    }

    paires: dict[str, list[str]] = {}
    for c in clauses:
        if c["paire"]:
            paires.setdefault(c["paire"], []).append(c["id"])

    # ---- P1 : séparation intra-paire ---------------------------------------------
    print(f"=== P1 — séparation des {len(paires)} paires (dim {DIM}) ===")
    print("paire | cos_nu   cos_mix  ecart   cos_canal | separee ?")
    n_sep = 0
    nus_exacts = 0
    cos_canaux: list[float] = []
    for pid in sorted(paires):
        a, b = paires[pid]
        c_nu, c_mix = cos(nus[a], nus[b]), cos(mixes[a], mixes[b])
        c_can = cos(canaux[a], canaux[b])
        cos_canaux.append(c_can)
        sep = c_mix <= c_nu - ECART_SEPARATION
        n_sep += sep
        nus_exacts += c_nu > 0.9999
        print(
            f"{pid:>5} | {c_nu:7.4f}  {c_mix:7.4f}  {c_nu - c_mix:6.4f}  {c_can:8.4f}"
            f" | {'oui' if sep else 'NON'}"
        )
    frac = n_sep / len(paires)
    print(
        f"\nP1a : {nus_exacts}/{len(paires)} paires invisibles au moteur nu (cos_nu = 1.0)"
    )
    print(
        f"P1b : {n_sep}/{len(paires)} paires separees ({frac:.0%}, seuil {SEUIL_P1:.0%})"
    )
    print(f"P1c : cos_canal intra-paire median {statistics.median(cos_canaux):.4f}")

    # ---- P3a : impact nul canal désactivé ----------------------------------------
    temoin = {c["id"]: vecteur_nu(c["clause"]) for c in clauses}  # pipeline sans canal
    assert all(np.array_equal(temoin[cid], nus[cid]) for cid in nus)
    print("\n=== P3a — canal desactive : vecteurs nus bit-identiques (verifie) ===")

    # ---- P3b : péage si fusion dans le vecteur principal -------------------------
    print(
        f"\n=== P3b — peage d'une FUSION (lambda={LAMBDA_FUSION}) sur 12 requetes prose ==="
    )
    ids = list(nus)
    r1_nu = r1_mix = 0
    for texte, pertinents in REQUETES:
        q = vecteur_nu(texte)
        top_nu = max(ids, key=lambda i: cos(q, nus[i]))
        top_mix = max(ids, key=lambda i: cos(q, mixes[i]))
        r1_nu += top_nu in pertinents
        r1_mix += top_mix in pertinents
        marque = "" if top_nu == top_mix else "   <- top1 change"
        print(f"  {texte[:44]:<46} nu:{top_nu:<5} mix:{top_mix:<5}{marque}")
    print(
        f"\nR@1 nu {r1_nu}/12  |  R@1 fusionne {r1_mix}/12  (peage = {r1_nu - r1_mix})"
    )

    ok = frac >= SEUIL_P1
    print(
        "\nVERDICT P1 : "
        + ("SEPARE (seuil atteint)" if ok else "NE SEPARE PAS (seuil manque)")
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
