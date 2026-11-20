"""Classification du document (type / secteur) à partir des entités extraites et du texte.

Quatre jeux de caractéristiques sont comparés (ablation) :
  - "entites"        : profil d'entités (proportion de chaque type, présence) + mots contenus dans les entités ;
  - "texte"          : TF-IDF (uni- et bigrammes) sur le texte brut ;
  - "hybride"        : texte + entités, chaque bloc normalisé (L2), bloc entités pondéré par 0,5 ;
  - "hybride_masque" : idem, mais chaque entité du texte est remplacée par son type (__ORG__…) avant TF-IDF.
Le poids 0,5 est fixé a priori (bloc auxiliaire à mi-poids) et non optimisé sur le jeu de test.
"""
import re
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from .textproc import ENT_TYPES

MODES = ("entites", "texte", "hybride", "hybride_masque")
W_ENT = 0.5


def entity_features(text, spans):
    n = max(len(spans), 1)
    f = {}
    for t in ENT_TYPES:
        k = sum(1 for sp in spans if sp[2] == t)
        f[f"part_{t}"] = k / n
        f[f"has_{t}"] = float(k > 0)
    for s, e, typ, *_ in spans:
        for w in re.findall(r"[^\W\d_]{2,}", text[s:e].lower()):
            f[f"{typ}:{w}"] = 1.0
        if typ == "REF":
            f[f"REF_prefixe:{re.split(r'[-/0-9]', text[s:e])[0].upper()}"] = 1.0
    return f


def mask_entities(text, spans):
    for s, e, typ, *_ in sorted(spans, key=lambda x: -x[0]):
        text = text[:s] + f" __{typ}__ " + text[e:]
    return text


class DocClassifier:
    def __init__(self, mode="hybride", C=5.0):
        assert mode in MODES
        self.mode, self.C = mode, C

    def _features(self, texts, spans_list, fit):
        parts = []
        if self.mode != "entites":
            docs = [mask_entities(t, s) for t, s in zip(texts, spans_list)] if self.mode == "hybride_masque" else texts
            parts.append(self.tf.fit_transform(docs) if fit else self.tf.transform(docs))
        if self.mode != "texte":
            d = [entity_features(t, s) for t, s in zip(texts, spans_list)]
            block = normalize(self.dv.fit_transform(d) if fit else self.dv.transform(d))
            parts.append(block * (W_ENT if self.mode.startswith("hybride") else 1.0))
        return hstack(parts).tocsr()

    def fit(self, texts, spans_list, y):
        self.dv = DictVectorizer()
        self.tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.clf = LogisticRegression(C=self.C, max_iter=3000)
        self.clf.fit(self._features(texts, spans_list, fit=True), y)
        return self

    def predict_proba(self, texts, spans_list):
        return self.clf.predict_proba(self._features(texts, spans_list, fit=False))

    def predict(self, texts, spans_list):
        return self.clf.classes_[self.predict_proba(texts, spans_list).argmax(1)]

    @property
    def classes_(self):
        return self.clf.classes_
