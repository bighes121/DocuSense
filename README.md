# DocuSense — Extraction d'entités et classification de documents (FR)

DocuSense lit un document professionnel en français (facture, contrat, rapport, article scientifique),
**extrait automatiquement les entités clés** (dates, organisations, personnes, lieux, montants, références,
pourcentages) puis **identifie le type et le secteur** du document. Une interface Streamlit affiche le texte
annoté, une fiche de synthèse et produit un rapport HTML/JSON.

## Résultats

| NER — F1 global (entité, correspondance exacte) | Test (gabarits et noms inédits) | 6 documents réels |
|---|---|---|
| BiLSTM-CRF codé de zéro | 0,849 | 0,783 |
| **CamemBERT fine-tuné** | **0,989** | **0,923** (0,942 avec post-traitement) |

Classification (entités prédites par CamemBERT) : type 79 % sur le test / 6 sur 6 documents réels,
secteur 71 % / 4 sur 6. Détails, ablation et analyse d'erreurs dans `results/Rapport_DocuSense.docx`.

## Ce qui a été construit

| Brique | Détail |
|---|---|
| Corpus annoté maison | générateur de documents (5 types × 4 secteurs), annotation exacte au caractère près, 7 types d'entités ; test sur **gabarits et valeurs jamais vus** + 6 documents réels rédigés et annotés à la main |
| NER n°1 — BiLSTM-CRF *from scratch* | embeddings de mots + CNN de caractères + BiLSTM + **CRF codé à la main** (forward / Viterbi), aucun poids pré-entraîné |
| NER n°2 — CamemBERT fine-tuné | modèle de langue brut `camembert-base` + tête de classification de tokens entraînée sur le corpus (fine-tuning partiel sur CPU : 4 couches hautes), boucle PyTorch écrite à la main |
| Classification | régression logistique sur TF-IDF + profil d'entités ; ablation entités / texte / hybride / texte masqué |
| Interface | Streamlit : texte surligné, fiche de synthèse, probabilités, filtres, import .txt/.pdf, export JSON & rapport HTML |

## Lancer le projet

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

python -m docusense.data_gen data         # 1. générer le corpus annoté
python train_bilstm.py                    # 2. NER from scratch (CPU, quelques minutes)
python train_camembert.py                 # 3. fine-tuning CamemBERT (CPU, ~27 min)
python evaluate.py                        # 4. métriques + entraînement des classifieurs
python make_figures.py                    # 5. figures
streamlit run app.py                      # 6. interface
python demo_screens.py                    # (optionnel) démonstration automatisée + captures
```

`train_camembert.py` attend le modèle de langue brut dans `models/camembert-base`
(copie de `almanach/camembert-base` : config, tokenizer et `model.safetensors`).
Les poids entraînés (`models/`) ne sont pas versionnés : relancer les étapes 2 à 4 pour les recréer.

## Limite connue

La segmentation en phrases coupe après une initiale ou une civilité (« S. Martin », « M. Rhazi »),
ce qui fragmente certains noms. La correction est décrite dans le rapport (section 8) ; elle demande de
réentraîner les modèles.

## Arborescence

```
docusense/        textproc (tokenisation, BIO) · data_gen (corpus) · corpus · bilstm_crf
                  classifier · pipeline (inférence + fiche de synthèse) · metrics
demo_docs/        documents réels annotés à la main + document d'import inédit
train_*.py        entraînement des deux modèles NER
evaluate.py       évaluation NER & classification -> results/metrics.json
app.py            interface Streamlit
results/          métriques, figures, captures, rapport Word
```
