"""Tokenisation, schéma d'étiquettes BIO et conversions spans <-> étiquettes."""
import re

ENT_TYPES = ["DATE", "ORG", "PER", "LIEU", "MONTANT", "REF", "POURCENT"]
LABELS = ["O"] + [f"{p}-{t}" for t in ENT_TYPES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

ENT_DESCRIPTIONS = {
    "DATE": "Dates (émission, échéance, signature…)",
    "ORG": "Organisations, entreprises, institutions",
    "PER": "Personnes",
    "LIEU": "Villes, pays",
    "MONTANT": "Montants avec devise",
    "REF": "Références (n° de facture, contrat, projet)",
    "POURCENT": "Pourcentages (TVA, remises, taux)",
}

# élision (l', d', qu') | nombres (12 500,00 -> "12", "500,00") | mots composés (FAC-2025-0042) | ponctuation
TOKEN_RE = re.compile(r"\w+['’](?=\w)|\d+(?:[.,]\d+)*|\w+(?:-\w+)*|[^\w\s]")


def tokenize(text):
    """Retourne une liste de (token, début, fin)."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def spans_to_bio(tokens, spans):
    """tokens: [(tok, s, e)], spans: [(s, e, type)] -> étiquettes BIO alignées sur les tokens."""
    labels = ["O"] * len(tokens)
    for s, e, typ in spans:
        first = True
        for i, (_, ts, te) in enumerate(tokens):
            if ts >= s and te <= e:
                labels[i] = ("B-" if first else "I-") + typ
                first = False
    return labels


def bio_to_spans(tokens, labels, scores=None):
    """Regroupe les étiquettes BIO en entités [(début, fin, type, score)]."""
    spans, cur = [], None
    for i, ((_, ts, te), lab) in enumerate(zip(tokens, labels)):
        sc = scores[i] if scores is not None else 1.0
        if lab == "O":
            if cur: spans.append(cur); cur = None
            continue
        prefix, typ = lab.split("-", 1)
        if prefix == "B" or cur is None or cur[2] != typ:
            if cur: spans.append(cur)
            cur = [ts, te, typ, [sc]]
        else:
            cur[1] = te
            cur[3].append(sc)
    if cur: spans.append(cur)
    return [(s, e, t, sum(v) / len(v)) for s, e, t, v in spans]


# fin de phrase = . ! ? ; suivi d'une majuscule.
# Limite connue : coupe aussi après une initiale ou une civilité (« S. Martin », « M. Rhazi »), voir le rapport.
SENT_RE = re.compile(r".+?(?:[.!?;](?=\s+[A-ZÀ-Ý«])|$)")


def split_sentences(text):
    """Découpe un document en segments (lignes puis phrases) en gardant les positions."""
    out = []
    for m in re.finditer(r"[^\n]+", text):
        line, base = m.group(), m.start()
        for sm in SENT_RE.finditer(line):
            seg = sm.group()
            if seg.strip():
                out.append((seg, base + sm.start()))
    return out
