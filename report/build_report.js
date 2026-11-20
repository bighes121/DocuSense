// Rapport Word du projet DocuSense — généré à partir de results/metrics.json, des figures et des captures.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Header, Footer, HeadingLevel, AlignmentType, BorderStyle,
  ShadingType, PageNumber, TabStopType, Table, TableRow, TableCell, WidthType, LevelFormat, TableOfContents,
  PageBreak, VerticalAlign,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const M = JSON.parse(fs.readFileSync(path.join(ROOT, "results/metrics.json"), "utf8"));
const T = JSON.parse(fs.readFileSync(path.join(__dirname, "texte_resultats.json"), "utf8"));
const FIG = (n) => path.join(ROOT, "results/figures", n);
const SHOT = (n) => path.join(ROOT, "results/screenshots", n);
const FONT = "Calibri", MONO = "Consolas", ACCENT = "3730A3", INK2 = "52514E";
const W = 9638; // largeur utile (A4, marges 2 cm) en DXA

const pct = (v, d = 1) => `${(v * 100).toFixed(d).replace(".", ",")} %`;
const f3 = (v) => v.toFixed(3).replace(".", ",");

// ---------------------------------------------------------------------------------------------- texte
function runs(text, base = {}) {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`|_[^_]+_)/).filter(Boolean).map((p) => {
    if (p.startsWith("**")) return new TextRun({ ...base, text: p.slice(2, -2), bold: true });
    if (p.startsWith("`")) return new TextRun({ ...base, text: p.slice(1, -1), font: MONO, size: 19 });
    if (p.startsWith("_") && p.length > 2) return new TextRun({ ...base, text: p.slice(1, -1), italics: true });
    return new TextRun({ ...base, text: p });
  });
}
const P = (text, o = {}) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 276 },
  ...o, children: runs(text) });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: "puces", level }, spacing: { after: 60, line: 264 },
  alignment: AlignmentType.LEFT, children: runs(text) });
const N = (text) => new Paragraph({ numbering: { reference: "numeros", level: 0 }, spacing: { after: 60, line: 264 },
  children: runs(text) });

let figNo = 0, tabNo = 0;
function figure(file, widthPx, caption) {
  const buf = fs.readFileSync(file);
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  figNo += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: widthPx, height: Math.round(h * widthPx / w) },
        altText: { title: `Figure ${figNo}`, description: caption, name: `figure${figNo}` } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Figure ${figNo} – `, bold: true, size: 19, color: INK2 }),
                 ...runs(caption, { size: 19, color: INK2, italics: true })] }),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "D1D5DB" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths, caption, o = {}) {
  tabNo += 1;
  const cell = (txt, i, head, shade) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    shading: head ? { fill: "E0E7FF", type: ShadingType.CLEAR } : shade ? { fill: "F7F7FB", type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: i > 0 && o.alignRight ? AlignmentType.RIGHT : AlignmentType.LEFT,
      children: runs(String(txt), { size: 19, bold: head || (o.boldLast && shade === "last") }) })],
  });
  return [
    new Paragraph({ spacing: { before: 160, after: 60 }, keepNext: true,
      children: [new TextRun({ text: `Tableau ${tabNo} – `, bold: true, size: 19, color: INK2 }),
                 ...runs(caption, { size: 19, color: INK2, italics: true })] }),
    new Table({ width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths,
      rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
             ...rows.map((r, k) => new TableRow({ children: r.map((c, i) => cell(c, i, false,
               o.boldLast && k === rows.length - 1 ? "last" : k % 2 === 1)) }))] }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

function callout(text) {
  return new Paragraph({ spacing: { before: 120, after: 160, line: 276 }, indent: { left: 200, right: 200 },
    shading: { fill: "EEF2FF", type: ShadingType.CLEAR },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT, space: 8 } },
    children: runs(text) });
}

// ---------------------------------------------------------------------------------------------- données
const TYPES = ["DATE", "ORG", "PER", "LIEU", "MONTANT", "REF", "POURCENT"];
const ner = (m, s, t = "micro avg") => M.ner[m][s][t];
const cls = (target, mode, src) => M.classification[target][mode][src];
const DOC = { facture: "Facture", contrat_licence: "Contrat de licence", contrat_travail: "Contrat de travail",
  rapport_etude: "Rapport d'études", article_scientifique: "Article scientifique" };

// ---------------------------------------------------------------------------------------------- page de garde
const logo = fs.readFileSync(path.join(__dirname, "logo.png"));
const cover = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 120 },
    children: [new ImageRun({ type: "png", data: logo, transformation: { width: 110, height: 110 },
      altText: { title: "Logo", description: "Logo de l'établissement", name: "logo" } })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text:
    "École Nationale Supérieure de l'Intelligence Artificielle et Sciences des Données", size: 22, color: INK2 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1100 }, children: [new TextRun({ text:
    "Université Ibn Zohr", size: 22, color: INK2 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text:
    "Rapport de projet – Traitement Automatique du Langage", size: 24, color: INK2, allCaps: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: "DocuSense",
    size: 64, bold: true, color: ACCENT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1300 }, children: [new TextRun({ text:
    "Extraction d'entités nommées et classification automatique de documents professionnels en français",
    size: 28, color: "111827" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [
    new TextRun({ text: "Réalisé par : ", bold: true, size: 23 }), new TextRun({ text: "……………………………………", size: 23 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [
    new TextRun({ text: "Encadré par : ", bold: true, size: 23 }), new TextRun({ text: "……………………………………", size: 23 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [
    new TextRun({ text: "Année universitaire 2026 – 2027", size: 23 })] }),
];

// ---------------------------------------------------------------------------------------------- contenu
const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));

// Résumé
add(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Résumé")] }));
add(T.resume.map((t) => P(t)));
add(P(`**Mots-clés :** reconnaissance d'entités nommées, CamemBERT, BiLSTM-CRF, fine-tuning, classification de documents, extraction d'information, Streamlit.`));
add(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun("Table des matières")] }));
add(new TableOfContents("Table des matières", { hyperlink: true, headingStyleRange: "1-2" }));

// 1. Introduction
add(H1("1. Introduction"));
add(H2("1.1 Contexte"));
add(P("Les entreprises et les administrations traitent chaque jour des volumes importants de documents : factures, contrats, rapports, publications. Une grande partie du travail consiste à **lire ces documents pour en extraire quelques informations clés** (qui sont les parties ? quelle date ? quel montant ? quelle référence ?) puis à **les ranger dans la bonne catégorie**. Ce travail manuel est lent, coûteux et source d'erreurs de saisie."));
add(P("L'extraction d'information (IE) automatise cette tâche. Son composant central est la **reconnaissance d'entités nommées** (NER, _Named Entity Recognition_), qui repère dans un texte les segments désignant des personnes, des organisations, des lieux, des dates ou des montants. Combinée à une classification du document, elle permet d'indexer, de router et de résumer automatiquement un flux documentaire."));
add(H2("1.2 Objectifs du projet"));
add(P("Le sujet demande de construire une chaîne complète « extraction d'information + classification rapide ». Nous l'avons décliné en cinq objectifs :"));
add(N("**Constituer un corpus annoté** en français couvrant cinq types de documents et sept types d'entités, avec un protocole d'évaluation qui mesure une vraie généralisation."));
add(N("**Entraîner nous-mêmes deux modèles NER** : un BiLSTM-CRF implémenté de zéro (aucun poids pré-entraîné) et un modèle CamemBERT affiné (_fine-tuning_) sur notre corpus."));
add(N("**Classifier le document** (type et secteur) à partir des entités extraites et du texte, et mesurer l'apport réel des entités."));
add(N("**Livrer une interface** qui montre les entités surlignées, une fiche de synthèse, la classification, et produit un rapport automatique."));
add(N("**Évaluer honnêtement** l'ensemble, y compris sur des documents rédigés à la main, et analyser les erreurs."));
add(callout("**Choix de conception.** Nous n'avons pas utilisé de modèle NER prêt à l'emploi (par exemple un modèle déjà entraîné à reconnaître des entités). Le seul composant pré-entraîné est le **modèle de langue brut** `camembert-base`, qui ne sait pas reconnaître d'entités : sa tête de classification est initialisée aléatoirement et tout l'apprentissage NER est réalisé dans ce projet, sur notre corpus."));

// 2. Architecture
add(H1("2. Architecture générale"));
add(P("DocuSense est organisé en une chaîne de traitement en cinq étapes (figure 1). Chaque étape est un module Python indépendant, ce qui permet de remplacer un composant (par exemple le modèle NER) sans toucher au reste."));
add(figure(FIG("architecture.png"), 620, "Chaîne de traitement de DocuSense."));
add(B("**Segmentation** : le document est découpé en lignes puis en phrases, et chaque phrase est tokenisée par une expression régulière adaptée au français (élisions `l'`, `d'`, nombres à séparateurs `12 500,00`, références composées `FAC-2025-0042`). Chaque token garde sa position exacte dans le document."));
add(B("**NER** : un des deux modèles attribue une étiquette BIO à chaque token ; les étiquettes sont regroupées en entités avec leur position et un score de confiance."));
add(B("**Classification** : un classifieur linéaire prédit le type et le secteur du document à partir du texte et du profil d'entités."));
add(B("**Restitution** : une fiche de synthèse est construite (parties, dates, montant le plus élevé, références…) et l'ensemble est affiché dans l'interface et exportable en JSON et en rapport HTML."));

// 3. Corpus
add(H1("3. Constitution du corpus annoté"));
add(H2("3.1 Types de documents et d'entités"));
add(P("Aucun corpus public ne couvre à la fois des documents professionnels français et les entités qui nous intéressent (les corpus NER classiques, comme WikiNER, n'annotent ni les montants, ni les références, ni les pourcentages). Nous avons donc **construit notre propre corpus annoté**, conformément à la seconde option du sujet."));
add(table(["Type", "Description", "Exemples"], [
  ["DATE", "Dates d'émission, d'échéance, de signature", "14 octobre 2026 · 11/08/2026 · 2026-03-12"],
  ["ORG", "Entreprises, institutions, organismes", "Solaris Maroc Énergie SARL · Université Ibn Zohr"],
  ["PER", "Personnes physiques", "Hicham Ouakrim · M. Rhazi"],
  ["LIEU", "Villes, régions, pays", "Agadir · Souss-Massa · Côte d'Ivoire"],
  ["MONTANT", "Montants avec devise", "141 540,00 DH · 4 800 euros · 12,5 millions de dirhams"],
  ["REF", "Numéros de facture, contrat, projet", "SME-2026-0318 · CNRST-PPR-2024-19"],
  ["POURCENT", "Taux, remises, TVA", "20 % · 6,4 %"],
], [1300, 3700, 4638], "Les sept types d'entités extraits."));
add(P("Les documents appartiennent à **cinq types** (facture, contrat de licence, contrat de travail, rapport d'études, article scientifique) et à **quatre secteurs** (santé, finance, technologie, énergie), soit vingt combinaisons."));

add(H2("3.2 Génération et annotation"));
add(P("Le générateur (`docusense/data_gen.py`) assemble chaque document à partir de **gabarits de phrases** propres à chaque type (16 par type, plus 8 phrases génériques communes) dans lesquels sont insérées des valeurs d'entités tirées aléatoirement : noms et prénoms, entreprises construites par combinaison (cœur + mot sectoriel + forme juridique), institutions, villes, dates sous cinq formats, montants sous plusieurs formats et devises, références et pourcentages. Le vocabulaire sectoriel (produits, domaines) est inséré de la même manière."));
add(P("Comme chaque valeur est insérée par le programme, **sa position exacte au caractère près est connue** : l'annotation est donc parfaite et ne demande aucun effort manuel. Les positions sont ensuite converties en étiquettes **BIO** (B = début d'entité, I = intérieur, O = hors entité) sur les tokens, comme le montre le tableau 2."));
add(table(["Token", "Facturé", "à", "Nova", "Pharma", "SA", ",", "le", "14", "octobre", "2026"],
  [["Étiquette", "O", "O", "B-ORG", "I-ORG", "I-ORG", "O", "O", "B-DATE", "I-DATE", "I-DATE"]],
  [1150, 900, 600, 900, 900, 700, 500, 600, 900, 1000, 1488], "Exemple d'annotation au format BIO."));
add(P("Pour rendre la classification plus réaliste, 15 % des documents d'entraînement contiennent une phrase « parasite » issue d'un autre type de document, et des phrases génériques (coordonnées, mentions de confidentialité) sont partagées par tous les types."));

add(H2("3.3 Protocole d'évaluation sans fuite"));
add(P("Un piège classique des corpus générés est d'évaluer le modèle sur des phrases qu'il a déjà vues avec d'autres valeurs : il suffit alors de mémoriser les gabarits pour obtenir des scores parfaits. Nous avons mis en place un protocole strict :"));
add(B("**Entraînement / validation** : 12 gabarits sur 16 par type, et 70 % des valeurs (noms, prénoms, cœurs d'entreprises, institutions, villes, préfixes de référence)."));
add(B("**Test** : uniquement les **4 gabarits réservés** par type et les **30 % de valeurs réservées**. Aucun nom de personne, aucune entreprise et aucune ville du test n'apparaît à l'entraînement, et la formulation des phrases est nouvelle."));
add(B("**Documents réels** : six documents rédigés à la main, dans un style libre (facture d'installation solaire, licence SaaS, contrat bancaire, rapport de télémédecine, article scientifique, facture de clinique), puis annotés manuellement (92 entités). Ils ne partagent aucun gabarit avec le générateur."));
add(callout("Le jeu de **validation** utilise les mêmes gabarits que l'entraînement : il sert uniquement à suivre l'apprentissage et ses scores sont naturellement très élevés. Les conclusions du rapport reposent sur le **test** et sur les **documents réels**."));

add(H2("3.4 Statistiques"));
add(table(["Jeu", "Documents", "Phrases", "Entités", "Rôle"], T.stats_rows, [1700, 1500, 1500, 1500, 3438],
  "Taille des différents jeux de données.", { alignRight: false }));
add(figure(FIG("repartition_entites.png"), 470, "Répartition des entités par type dans le jeu d'entraînement."));

// 4. Modèles NER
add(H1("4. Modèles de reconnaissance d'entités"));
add(P("Nous comparons deux approches représentatives : un modèle séquentiel classique entraîné entièrement à partir de zéro, et un transformeur pré-entraîné adapté à notre tâche."));
add(H2("4.1 BiLSTM-CRF implémenté de zéro"));
add(P("Le modèle (`docusense/bilstm_crf.py`) est écrit entièrement en PyTorch, y compris la couche CRF :"));
add(B("**Représentation des mots** : embedding du mot en minuscules (100 dimensions, appris), **CNN sur les caractères** (30 dimensions par caractère, 50 filtres de largeur 3, max-pooling) pour traiter les mots inconnus, et un indicateur de casse (majuscule initiale, tout en majuscules, chiffres…)."));
add(B("**Encodeur** : BiLSTM bidirectionnel à 128 unités par direction, dropout 0,5."));
add(B("**Décodage CRF** : une matrice de transitions entre étiquettes est apprise. La log-vraisemblance est calculée par l'**algorithme forward** et le décodage utilise l'**algorithme de Viterbi**, tous deux implémentés à la main. Le CRF impose la cohérence des séquences (par exemple, `I-ORG` ne peut pas suivre `B-DATE`)."));
add(B("**Word dropout** : pendant l'entraînement, 25 % des mots sont remplacés par le jeton inconnu, pour forcer le modèle à s'appuyer sur le contexte et les caractères plutôt que sur la mémorisation des noms."));
add(table(["Hyperparamètre", "Valeur"], [
  ["Paramètres", "491 758 (tous appris à partir de zéro)"], ["Vocabulaire", "1 854 mots"],
  ["Optimiseur", "Adam, taux d'apprentissage 2·10⁻³, écrêtage du gradient à 5"],
  ["Lots / époques", "32 phrases ; point de contrôle de l'époque 4 retenu (meilleur F1 de validation, voir section 8)"],
  ["Durée d'entraînement", "≈ 30 secondes par époque sur CPU (Intel Core i5-7300U)"],
], [3000, 6638], "Configuration du BiLSTM-CRF."));

add(H2("4.2 CamemBERT affiné"));
add(P("CamemBERT est un modèle de langue de type RoBERTa (12 couches, 110 millions de paramètres) pré-entraîné sur un grand corpus français. Il connaît la langue mais **ne sait pas reconnaître des entités** : nous lui ajoutons une tête de classification de tokens (15 sorties, une par étiquette BIO), initialisée aléatoirement, et l'entraînons sur notre corpus avec une boucle PyTorch écrite à la main (`train_camembert.py`)."));
add(B("**Alignement sous-mots / mots** : CamemBERT découpe les mots en sous-mots (« Ouakrim » → « ▁Ou », « ak », « rim »). L'étiquette d'un mot est portée par son premier sous-mot ; les autres sont ignorés dans la perte (étiquette −100)."));
add(B("**Contrainte matérielle** : l'entraînement complet coûtait 7 s par lot sur le CPU disponible (≈ 1 h 30). Nous avons mesuré plusieurs configurations et retenu un **fine-tuning partiel** : les embeddings et les 8 couches basses sont gelés, les 4 couches hautes et la tête sont entraînées (28 M paramètres). Le temps par lot tombe à environ 2,2 s. Les couches basses encodent des connaissances linguistiques générales ; les couches hautes, plus spécialisées, sont celles qui s'adaptent le plus à la tâche."));
add(B("**Regroupement par longueur** : les phrases de longueur voisine sont placées dans le même lot pour réduire le remplissage (_padding_)."));
add(table(["Hyperparamètre", "Valeur"], [
  ["Modèle de base", "camembert-base (modèle de langue brut, sans tête NER)"],
  ["Paramètres entraînés", "28 363 023 sur 110 042 895 (4 couches hautes + tête)"],
  ["Optimiseur", "AdamW, taux 5·10⁻⁵, décroissance des poids 0,01, warm-up linéaire 10 %"],
  ["Lots / époques", "16 phrases / 3 époques (762 pas d'optimisation)"],
  ["Durée d'entraînement", T.camembert_duree],
], [3000, 6638], "Configuration du fine-tuning de CamemBERT."));
add(figure(FIG("courbes_apprentissage.png"), 520, "F1 sur le jeu de validation au fil des époques (validation : mêmes gabarits que l'entraînement)."));
add(P("Les deux modèles atteignent un F1 de validation proche de 1 en deux à trois époques. Ce plafond est attendu, car la validation réutilise les gabarits d'entraînement : il montre que l'apprentissage converge, pas que le modèle généralise. C'est le rôle du jeu de test (section 6). Pour le BiLSTM-CRF, la courbe a été mesurée pendant l'entraînement avec la segmentation corrigée (section 8) ; évalué avec la segmentation d'origine, son F1 de validation est de 0,976 (tableau 7)."));

// 5. Classification
add(H1("5. Classification du document"));
add(P("Le classifieur (`docusense/classifier.py`) prédit deux étiquettes indépendantes : le **type** (5 classes) et le **secteur** (4 classes). Nous utilisons une **régression logistique**, rapide, interprétable et bien adaptée aux représentations creuses. Pour mesurer ce qu'apportent réellement les entités, quatre jeux de caractéristiques sont comparés :"));
add(B("**Entités seules** : profil d'entités du document (proportion de chaque type, présence) et mots contenus dans les entités (par exemple `ORG:pharma`, préfixe de référence `FAC`)."));
add(B("**Texte brut** : TF-IDF sur les unigrammes et bigrammes du texte."));
add(B("**Hybride** (modèle déployé) : concaténation des deux blocs, chacun normalisé, le bloc entités étant pondéré par 0,5. Ce poids est fixé a priori et n'a pas été optimisé sur le test."));
add(B("**Hybride, texte masqué** : comme l'hybride, mais chaque entité est remplacée dans le texte par son type (`__ORG__`, `__MONTANT__`…) avant le TF-IDF, pour empêcher le modèle de s'appuyer sur des noms précis."));
add(P("Les classifieurs sont entraînés sur les entités de référence du jeu d'entraînement, puis évalués **de bout en bout** : au test, ils reçoivent les entités **prédites** par CamemBERT, comme en conditions réelles."));

// 6. Résultats
add(H1("6. Résultats"));
add(H2("6.1 Reconnaissance d'entités"));
const nerRows = [...TYPES, "micro avg"].map((t) => [t === "micro avg" ? "**Global (micro)**" : t,
  f3(ner("bilstm_crf", "test", t)["f1-score"]), f3(ner("camembert", "test", t)["f1-score"]),
  f3(ner("bilstm_crf", "reel", t)["f1-score"]), f3(ner("camembert", "reel", t)["f1-score"]),
  String(ner("camembert", "test", t).support)]);
add(table(["Entité", "BiLSTM-CRF test", "CamemBERT test", "BiLSTM-CRF réels", "CamemBERT réels", "Support test"],
  nerRows, [1700, 1600, 1600, 1600, 1600, 1538],
  "F1 par type d'entité (niveau entité, correspondance exacte des bornes et du type).", { boldLast: true }));
add(table(["Modèle", "Jeu", "Précision", "Rappel", "F1"], ["bilstm_crf", "camembert"].flatMap((m) =>
  ["dev", "test", "reel"].map((s) => [m === "camembert" ? "CamemBERT" : "BiLSTM-CRF",
    { dev: "Validation", test: "Test", reel: "Documents réels" }[s],
    f3(ner(m, s).precision), f3(ner(m, s).recall), f3(ner(m, s)["f1-score"])])),
  [2200, 2200, 1746, 1746, 1746], "Précision, rappel et F1 globaux (micro) des deux modèles."));
add(figure(FIG("ner_f1_par_type.png"), 560, "F1 par type d'entité sur le jeu de test (gabarits et valeurs jamais vus)."));
add(T.ner_analyse.map((t) => P(t)));

add(H2("6.2 Analyse d'erreurs sur les documents réels"));
add(T.erreurs_intro.map((t) => P(t)));
add(table(["Document", "Segment", "Attendu", "Prédit / problème"], T.erreurs_rows, [2200, 3300, 1300, 2838],
  "Exemples d'erreurs de CamemBERT sur les documents rédigés à la main."));
add(T.erreurs_analyse.map((t) => P(t)));

add(H2("6.3 Classification des documents"));
const modeNames = { entites: "Entités seules", texte: "Texte brut (TF-IDF)", hybride: "**Hybride (déployé)**",
  hybride_masque: "Hybride, texte masqué" };
add(table(["Caractéristiques", "Type – test", "Type – réels", "Secteur – test", "Secteur – réels"],
  Object.keys(modeNames).map((k) => [modeNames[k], pct(cls("type", k, "test_pred").accuracy),
    pct(cls("type", k, "reel_pred").accuracy), pct(cls("sector", k, "test_pred").accuracy),
    pct(cls("sector", k, "reel_pred").accuracy)]),
  [2838, 1700, 1700, 1700, 1700], "Exactitude de la classification avec les entités prédites par CamemBERT (test : 150 documents ; réels : 6 documents)."));
add(figure(FIG("ablation_classification.png"), 620, "Ablation des caractéristiques de classification."));
add(T.cls_analyse.map((t) => P(t)));
add(figure(FIG("confusion_type.png"), 380, "Matrice de confusion du type de document (classifieur hybride, jeu de test, entités prédites)."));
add(T.confusion_analyse.map((t) => P(t)));

add(H2("6.4 Temps de traitement"));
add(P(T.vitesse));

// 7. Démonstration
add(H1("7. Démonstration de l'application"));
add(P("L'interface est développée avec **Streamlit** (`app.py`). La démonstration ci-dessous a été réalisée de façon automatisée (`demo_screens.py`) : un script lance l'application, la pilote dans le navigateur Edge avec Playwright (sélection d'exemples, import de fichier, filtres, changement de modèle, téléchargement) et prend les captures. Toutes les captures montrent donc l'application réelle en fonctionnement."));
add(H2("7.1 Analyse d'une facture"));
add(P("À l'ouverture, l'application analyse une facture d'installation solaire rédigée à la main. Le bandeau supérieur donne le type et le secteur détectés avec leur confiance, le nombre d'entités et le temps d'analyse. Le texte est affiché avec chaque entité surlignée dans la couleur de son type ; à droite, la **fiche de synthèse** regroupe les informations clés et les barres indiquent les probabilités de chaque classe."));
add(figure(SHOT("01_facture_vue_complete.png"), 470, "Vue complète de l'application après analyse d'une facture."));
add(figure(SHOT("02_facture_document_annote.png"), 600, "Détail : document annoté et fiche de synthèse."));
add(H2("7.2 Autres types de documents"));
add(P(T.demo_types));
for (const [file, cap] of T.demo_shots) add(figure(SHOT(file), 600, cap));
add(H2("7.3 Import d'un document inédit"));
add(P(T.demo_import));
add(figure(SHOT("07_import_fichier_inedit.png"), 470, "Analyse d'une note de conjoncture importée depuis un fichier .txt jamais vu."));
add(H2("7.4 Filtres, comparaison des modèles et rapport automatique"));
add(P("La barre latérale permet de n'afficher que certains types d'entités, par exemple les montants et les dates pour un contrôle comptable (figure suivante), et de basculer entre les deux modèles NER pour les comparer sur le même document."));
add(figure(SHOT("08_filtre_montants_dates.png"), 600, "Filtrage : seuls les montants et les dates sont affichés."));
add(P(T.demo_bilstm));
add(figure(SHOT("09_bilstm_meme_document.png"), 600, "Même document analysé par le BiLSTM-CRF entraîné de zéro."));
add(P("Le bouton « Rapport HTML » produit un **rapport automatique** autonome (type, secteur, fiche de synthèse, tableau des entités et document annoté), prêt à être archivé ou transmis. Un export JSON est également disponible pour l'intégration dans un autre système."));
add(figure(SHOT("10_rapport_html_auto.png"), 400, "Rapport HTML généré automatiquement par l'application."));
add(P("Enfin, l'onglet « Performances des modèles » présente les métriques d'évaluation directement dans l'application."));
add(figure(SHOT("11_onglet_performances.png"), 470, "Onglet de performances intégré à l'application."));

// 8. Limites
add(H1("8. Limites et perspectives"));
add(H2("8.1 Limites"));
add(T.limites.map((t) => B(t)));
add(H2("8.2 Perspectives"));
add(B("**Corriger la segmentation et réentraîner** : la correction (ne pas couper après une initiale ou une civilité) est prête et validée ; il reste à réentraîner les deux modèles avec elle pour supprimer les entités fragmentées et rétablir le score du BiLSTM-CRF sur les personnes."));
add(B("**Rendre le classifieur robuste à la formulation** : enrichir les gabarits de contrats de travail, ajouter une option de refus sous un seuil de confiance et calibrer les probabilités."));
add(B("**Annoter de vrais documents** (quelques centaines suffisent) et mélanger corpus réel et corpus généré : c'est le levier principal pour fiabiliser le système en production."));
add(B("**Fine-tuning complet de CamemBERT** sur GPU, et comparaison avec des modèles plus récents (CamemBERTa-v2, modèles multilingues pour les documents bilingues français-arabe)."));
add(B("**Documents scannés** : ajouter une étape d'OCR (Tesseract) et exploiter la mise en page avec un modèle de type LayoutLM."));
add(B("**Normalisation des entités** : convertir les dates au format ISO et les montants en valeurs numériques avec devise, pour alimenter directement une base de données."));
add(B("**Classification par l'encodeur** : utiliser la représentation de CamemBERT du document entier plutôt qu'un TF-IDF, et ajouter une option de refus lorsque la confiance est faible."));

// 9. Conclusion
add(H1("9. Conclusion"));
add(T.conclusion.map((t) => P(t)));

// Annexe
add(H1("Annexe – Organisation du code et exécution"));
add(table(["Fichier", "Rôle"], [
  ["docusense/textproc.py", "Tokenisation, schéma BIO, conversions positions ↔ étiquettes, segmentation"],
  ["docusense/data_gen.py", "Générateur du corpus annoté (gabarits, valeurs, découpage sans fuite)"],
  ["docusense/corpus.py", "Chargement des documents, lecture des documents annotés à la main"],
  ["docusense/bilstm_crf.py", "BiLSTM-CRF : CNN de caractères, BiLSTM, CRF (forward, Viterbi)"],
  ["docusense/classifier.py", "Caractéristiques d'entités, TF-IDF, régression logistique"],
  ["docusense/pipeline.py", "Inférence de bout en bout et fiche de synthèse"],
  ["train_bilstm.py / train_camembert.py", "Entraînement des deux modèles NER"],
  ["evaluate.py / make_figures.py", "Évaluation complète et figures du rapport"],
  ["app.py / demo_screens.py", "Interface Streamlit et démonstration automatisée"],
], [3600, 6038], "Organisation du code source."));
add(P("**Reproduire les résultats :**"));
for (const c of ["python -m docusense.data_gen data", "python train_bilstm.py", "python train_camembert.py",
  "python evaluate.py", "python make_figures.py", "streamlit run app.py"]) {
  add(new Paragraph({ spacing: { after: 20 }, indent: { left: 300 }, shading: { fill: "F3F4F6", type: ShadingType.CLEAR },
    children: [new TextRun({ text: c, font: MONO, size: 19 })] }));
}

// ---------------------------------------------------------------------------------------------- document
const header = new Header({ children: [new Paragraph({
  tabStops: [{ type: TabStopType.RIGHT, position: W }],
  border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "C7D2FE", space: 4 } },
  children: [new ImageRun({ type: "png", data: logo, transformation: { width: 30, height: 30 },
      altText: { title: "Logo", description: "Logo", name: "logo-entete" } }),
    new TextRun({ text: "   DocuSense – Extraction d'information et classification de documents", size: 17, color: INK2 }),
    new TextRun({ text: "\tENSIASD · Université Ibn Zohr", size: 17, color: INK2 })] })] });
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: INK2 })] })] });

const hstyle = (size, before, after, color = ACCENT) => ({ run: { font: FONT, size, bold: true, color },
  paragraph: { spacing: { before, after }, keepNext: true, keepLines: true } });

const doc = new Document({
  creator: "DocuSense", title: "DocuSense – Rapport de projet",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        ...hstyle(34, 0, 240), paragraph: { spacing: { before: 0, after: 240 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        ...hstyle(27, 300, 140), paragraph: { spacing: { before: 300, after: 140 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        ...hstyle(23, 200, 100, "111827"), paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "puces", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 500, hanging: 260 } } } }] },
    { reference: "numeros", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 500, hanging: 300 } } } }] },
  ] },
  features: { updateFields: true },
  sections: [
    { properties: { page: { margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } }, children: cover },
    { properties: { page: { margin: { top: 1134, bottom: 1134, left: 1134, right: 1134, header: 500 },
        pageNumbers: { start: 1 } } },
      headers: { default: header }, footers: { default: footer }, children: body },
  ],
});

Packer.toBuffer(doc).then((b) => {
  const out = process.argv[2] || path.join(ROOT, "results/Rapport_DocuSense.docx");
  fs.writeFileSync(out, b);
  console.log("écrit :", out);
});
