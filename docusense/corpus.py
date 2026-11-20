"""Chargement du corpus et conversion document -> phrases étiquetées (BIO)."""
import json
from .textproc import tokenize, spans_to_bio, split_sentences


def load_docs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def doc_to_sentences(doc):
    """Retourne [(tokens_absolus, etiquettes)] ; tokens_absolus = [(tok, début, fin)] dans le document."""
    out = []
    for seg, base in split_sentences(doc["text"]):
        toks = [(t, s + base, e + base) for t, s, e in tokenize(seg)]
        if toks:
            out.append((toks, spans_to_bio(toks, doc.get("entities", []))))
    return out


def load_sentences(path):
    return [s for d in load_docs(path) for s in doc_to_sentences(d)]


def load_annotated_md(path):
    """Documents rédigés à la main : '=== id | type | secteur' puis le texte avec des entités '[texte](TYPE)'."""
    import re
    raw = open(path, encoding="utf-8").read()
    docs = []
    for block in re.split(r"^=== ", raw, flags=re.M)[1:]:
        head, body = block.split("\n", 1)
        did, dtype, sector = [x.strip() for x in head.split("|")]
        text, ents, pos = "", [], 0
        body = body.strip("\n")
        for m in re.finditer(r"\[([^\]]+)\]\(([A-Z]+)\)", body):
            text += body[pos:m.start()]
            ents.append((len(text), len(text) + len(m.group(1)), m.group(2)))
            text += m.group(1)
            pos = m.end()
        text += body[pos:]
        docs.append({"id": did, "type": dtype, "sector": sector, "text": text, "entities": ents})
    return docs
