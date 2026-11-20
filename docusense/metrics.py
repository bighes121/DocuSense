"""Métriques NER au niveau entité (seqeval, schéma IOB2 strict)."""
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2


def ner_report(gold, pred):
    rep = classification_report(gold, pred, mode="strict", scheme=IOB2, output_dict=True, zero_division=0)
    return {k: {m: round(float(v[m]), 4) for m in ("precision", "recall", "f1-score", "support")}
            for k, v in rep.items()}


def ner_f1(gold, pred):
    return f1_score(gold, pred, mode="strict", scheme=IOB2, zero_division=0)
