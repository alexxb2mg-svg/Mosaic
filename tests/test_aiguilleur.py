"""Tests — aiguilleur de requête (src/mosaic/aiguilleur.py).

Le moteur a plusieurs circuits qui ne répondent PAS aux mêmes questions : la
recherche classe par ressemblance, le magasin structurel compte et ordonne, le
boost de référence retrouve un code exact. Aujourd'hui c'est à l'appelant de
choisir — et il choisit mal : mesuré le 14/08, la même question posée en trois
mots sort au rang 2, décorée de six mots plausibles au rang 231.

L'aiguilleur choisit le circuit à partir de la QUESTION SEULE, sans modèle et
sans jamais deviner : chaque décision nomme le motif qui l'a déclenchée, et
l'ambiguïté renvoie au circuit sémantique — le seul qui ne se trompe jamais de
nature (il rend des documents, ce que toute question accepte).
"""

import pytest

from mosaic.aiguilleur import Circuit, aiguiller


# --- comptage : la question attend un NOMBRE, pas des documents ------------------
@pytest.mark.parametrize(
    "question",
    [
        "combien de bons de livraison en juin ?",
        "combien y a-t-il de photos sur ce chantier",
        "nombre de devis émis en 2026",
        "quel est le nombre de factures impayées",
    ],
)
def test_comptage_reconnu(question):
    r = aiguiller(question)
    assert r.circuit is Circuit.COMPTAGE
    assert r.motif  # toute décision se justifie


# --- ordre : la question attend LE plus récent, pas les plus ressemblants --------
@pytest.mark.parametrize(
    "question",
    [
        "le dernier bon de livraison reçu",
        "la note de chantier la plus récente",
        "les 3 derniers devis",
        "quel est le devis le plus récent pour ce client",
    ],
)
def test_ordre_reconnu(question):
    assert aiguiller(question).circuit is Circuit.ORDRE


# --- référence : un code exact existe, il ne se cherche pas « à peu près » -------
@pytest.mark.parametrize(
    "question,ref",
    [
        ("devis DE26040008 relamping", "DE26040008"),
        ("le BL 9990001", "9990001"),
        ("produit 9990004 chez le fournisseur", "9990004"),
    ],
)
def test_reference_reconnue_et_extraite(question, ref):
    r = aiguiller(question)
    assert r.circuit is Circuit.REFERENCE
    assert ref in r.refs


def test_reference_prime_sur_le_reste():
    """Une question qui contient un code exact part au circuit référence même si
    elle ressemble à une question de sens : un code se cherche à l'identique."""
    r = aiguiller("quelle est la marque du luminaire de la commande 4524608")
    assert r.circuit is Circuit.REFERENCE


def test_une_annee_seule_n_est_pas_une_reference():
    """« 2026 » et « 240 mm » sont des nombres, pas des codes — les prendre pour
    des références enverrait la moitié des questions métier au mauvais circuit."""
    r = aiguiller("luminaires posés en 2026 dans le hall")
    assert r.circuit is not Circuit.REFERENCE
    r2 = aiguiller("encastré LED diamètre 240 mm 3000K")
    assert r2.circuit is not Circuit.REFERENCE


# --- sémantique : le circuit par défaut, et le refuge de l'ambiguïté -------------
@pytest.mark.parametrize(
    "question",
    [
        "photo de tableau électrique prise sur un chantier",
        "quelle marque de sèche-serviettes a été retenue",
        "comment calculer l'aire d'un triangle",
        "pourquoi mon disjoncteur saute quand j'allume le four",
    ],
)
def test_semantique_par_defaut(question):
    assert aiguiller(question).circuit is Circuit.SEMANTIQUE


def test_question_vide_ne_casse_pas():
    r = aiguiller("   ")
    assert r.circuit is Circuit.SEMANTIQUE


def test_le_type_de_document_est_suggere_sans_changer_de_circuit():
    """Reconnaître « photo » ne doit pas dérouter la question : c'est un FILTRE
    proposé au circuit sémantique, pas un circuit à part."""
    r = aiguiller("photo du tableau électrique avant travaux")
    assert r.circuit is Circuit.SEMANTIQUE
    assert r.type_doc == "photo"


def test_le_comptage_l_emporte_sur_le_type():
    r = aiguiller("combien de photos sur le chantier")
    assert r.circuit is Circuit.COMPTAGE
    assert r.type_doc == "photo"


# --- le garde-fou qui décide de tout : ne pas dérouter la prose ------------------
def test_les_questions_de_sens_ne_partent_jamais_en_comptage():
    """P-C3 déclarée le 14/08 : « le piège sera la détection ; un routage naïf
    enverrait tout à la lecture exacte et casserait la prose ». Ces questions
    CONTIENNENT des mots de comptage sans être des comptages."""
    for q in (
        "un grand nombre de disjoncteurs ont sauté",
        "combien coûte un tableau électrique",  # question de PRIX, pas de compte
        "le dernier étage du bâtiment",  # « dernier » spatial, pas temporel
    ):
        assert aiguiller(q).circuit is Circuit.SEMANTIQUE, q


def test_la_justification_nomme_le_motif_declencheur():
    r = aiguiller("combien de BL en juin")
    assert "combien" in r.motif.lower()
