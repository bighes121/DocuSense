"""Pipeline d'inférence : texte -> entités -> classification -> fiche de synthèse."""
import re, time
from pathlib import Path
import joblib
import torch
from .textproc import tokenize, split_sentences, bio_to_spans, ID2LABEL
from .data_gen import DOC_TYPES, SECTORS

ROOT = Path(__file__).resolve().parent.parent
torch.set_num_threads(4)


def segments(text):
    out = []
    for seg, base in split_sentences(text):
        toks = [(t, s + base, e + base) for t, s, e in tokenize(seg)]
        if toks:
            out.append(toks)
    return out


class CamembertNER:
    name = "CamemBERT fine-tuné"

    def __init__(self, path=ROOT / "models/camembert_ner"):
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path).eval()

    def predict_tokens(self, segs, bs=16):
        """Retourne, pour chaque segment, les étiquettes et les confiances par mot."""
        out = []
        for i in range(0, len(segs), bs):
            chunk = [[t for t, _, _ in s] for s in segs[i:i + bs]]
            enc = self.tok(chunk, is_split_into_words=True, truncation=True, max_length=256,
                           padding=True, return_tensors="pt")
            with torch.no_grad():
                probs = self.model(**enc).logits.softmax(-1)
            for b, words in enumerate(chunk):
                labs, scores = ["O"] * len(words), [1.0] * len(words)
                prev = None
                for t, w in enumerate(enc.word_ids(b)):
                    if w is not None and w != prev:
                        p, k = probs[b, t].max(-1)
                        labs[w], scores[w] = ID2LABEL[int(k)], float(p)
                    prev = w
                out.append((labs, scores))
        return out


class BiLSTMNER:
    name = "BiLSTM-CRF (from scratch)"

    def __init__(self, path=ROOT / "models/bilstm_crf/model.pt"):
        from .bilstm_crf import BiLSTMCRF, Vocab, collate
        from .textproc import LABELS
        ck = torch.load(path, map_location="cpu")
        self.vocab, self.collate = Vocab.from_state(ck["vocab"]), collate
        self.model = BiLSTMCRF(len(self.vocab.words), len(self.vocab.chars), len(LABELS))
        self.model.load_state_dict(ck["state"]); self.model.eval()

    def predict_tokens(self, segs, bs=64):
        out = []
        for i in range(0, len(segs), bs):
            batch = [([t for t, _, _ in s], None) for s in segs[i:i + bs]]
            W, C, K, _, M = self.collate(batch, self.vocab, with_labels=False)
            with torch.no_grad():
                paths = self.model.predict(W, C, K, M)
            out += [([ID2LABEL[k] for k in p], [1.0] * len(p)) for p in paths]
        return out


def postprocess(text, spans):
    """Règle de cohérence : une entité doit contenir au moins une lettre ou un chiffre
    (écarte les points de conduite « ....... » ou une apostrophe isolée étiquetés par erreur)."""
    return [sp for sp in spans if any(c.isalnum() for c in text[sp[0]:sp[1]])]


def extract(ner, text, clean=True):
    segs = segments(text)
    spans = []
    for toks, (labs, scores) in zip(segs, ner.predict_tokens(segs)):
        spans += bio_to_spans(toks, labs, scores)
    return postprocess(text, spans) if clean else spans


# ----------------------------------------------------------------------------- fiche de synthèse
def parse_amount(s):
    low = s.lower()
    cur = next((c for k, c in [("€", "EUR"), ("eur", "EUR"), ("usd", "USD"), ("dollar", "USD"),
                               ("mad", "MAD"), ("dh", "MAD"), ("dirham", "MAD")] if k in low), "")
    m = re.search(r"\d[\d\s.,]*", s)
    if not m:
        return None, cur
    num = re.sub(r"\s", "", m.group()).rstrip(".,")
    if "," in num:
        num = num.replace(".", "").replace(",", ".")
    elif num.count(".") >= 1 and len(num.split(".")[-1]) == 3:
        num = num.replace(".", "")
    try:
        v = float(num)
    except ValueError:
        return None, cur
    if "million" in low:
        v *= 1e6
    return v, cur


def summary(text, ents):
    by = {}
    for e in ents:
        by.setdefault(e["type"], [])
        if e["text"] not in by[e["type"]]:
            by[e["type"]].append(e["text"])
    amounts = [(parse_amount(a), a) for a in by.get("MONTANT", [])]
    amounts = [(v, c, a) for (v, c), a in amounts if v is not None]
    top = max(amounts, default=None)
    return {
        "Organisations": by.get("ORG", []),
        "Personnes": by.get("PER", []),
        "Lieux": by.get("LIEU", []),
        "Dates": by.get("DATE", []),
        "Montant le plus élevé": top[2] if top else None,
        "Références": by.get("REF", []),
        "Taux / pourcentages": by.get("POURCENT", []),
    }


class DocuSense:
    def __init__(self, ner="camembert"):
        self.ners = {}
        self.clfs = joblib.load(ROOT / "models/classifiers.joblib")
        self.set_ner(ner)

    def set_ner(self, key):
        if key not in self.ners:
            self.ners[key] = CamembertNER() if key == "camembert" else BiLSTMNER()
        self.ner = self.ners[key]

    def analyze(self, text):
        t0 = time.time()
        spans = extract(self.ner, text)
        t_ner = time.time() - t0
        ents = [{"text": text[s:e], "type": t, "start": s, "end": e, "score": round(sc, 3)} for s, e, t, sc in spans]
        res = {"entities": ents, "model": self.ner.name}
        for target, names in (("type", DOC_TYPES), ("sector", SECTORS)):
            clf = self.clfs[target]
            p = clf.predict_proba([text], [spans])[0]
            order = p.argsort()[::-1]
            res[target] = {"label": names[clf.classes_[order[0]]], "confidence": float(p[order[0]]),
                           "probas": {names[clf.classes_[i]]: float(p[i]) for i in order}}
        res["summary"] = summary(text, ents)
        res["timing"] = {"ner_s": round(t_ner, 3), "total_s": round(time.time() - t0, 3)}
        return res
