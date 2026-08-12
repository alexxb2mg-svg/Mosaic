# Banc LSCD SemEval-2020 Task 1 — `mosaic diff` contre le champ académique (12/08/2026)

Pipeline : prepare.py (téléchargement IMS/Zenodo, POS `_nn`->`-nn`, homonymie latine
`#N`->`-N`, découpage en paquets de 20 phrases) · subsample.py (stride déterministe) ·
builds séquentiels `--no-path-tokens` + `mosaic diff` en mode INDEX · score.py
(Spearman numpy pur + contrôles).

| | ANGLAIS (2 500 docs/période, contrainte RAM) | LATIN (4 804 docs/période, plein) |
|---|---|---|
| Spearman derive_mots vs graded | **+0,026** | **+0,161** |
| Contrôle : derive_usage seul | −0,150 (≈ baseline fréquence du champ) | +0,029 |
| Contrôle : delta vs delta-df relatif | +0,165 | +0,111 |

Repères du champ : baseline fréquence −0,217 · count-vectors 0,022 · vainqueur
statique 2020 (EN) 0,422 · contextuel 0,757. Prédiction déclarée : 0,1–0,4.

VERDICT : latin VALIDE (bat les baselines naïves, dans la bande prédite) ; anglais
DÉFAITE expliquée par autopsie — (1) saturation du cosinus entre corpus indépendants
(dérive médiane de tout le vocabulaire : 0,947 — l'ordonnancement se joue dans le
bruit) ; (2) le bruit est piloté par la RARETÉ des cibles : Spearman(delta, min df)
= −0,386 alors que la vérité n'est pas corrélée au df (−0,009) — en latin, cibles
plus fréquentes, la corrélation parasite tombe à −0,135 et le signal émerge ;
(3) sous-échantillonnage FORCÉ à 20 % du corpus EN (le build plein à 13 M tokens
atteint 11,8 Go de RAM — les deux index de diff_corpus coexistent en mémoire ;
piste moteur : construire sur disque puis rouvrir en memmap paresseux, le mode
INDEX du diff l'a prouvé : pic divisé par deux, diff en 14 s).

ACQUIS STRUCTUREL : nos signaux séparés (dérive de mot ≠ dérive d'usage) évitent
par construction le piège n°1 du champ — le cosinus ne fuit pas de la fréquence
(contrôles modérés des deux côtés). Limite documentée : `mosaic diff` est conçu
pour l'ÉVOLUTION d'un même corpus ; deux corpus indépendants (genres différents
par période, cas SemEval EN) saturent le cosinus de profils.
