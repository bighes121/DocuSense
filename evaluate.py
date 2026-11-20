"""Évaluation complète : NER (2 modèles x 3 jeux) puis classification (3 variantes x 2 cibles).

Les classifieurs sont entraînés sur les entités de référence du jeu d'entraînement et évalués de bout en
bout sur les entités PRÉDITES par CamemBERT (et aussi sur les entités de référence, pour isoler l'effet
des erreurs du NER). Le classifieur hybride est sauvegardé pour l'application.
"""
import json, time
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from docusense.corpus import load_docs, load_annotated_md
from docusense.textproc import tokenize, spans_to_bio
from docusense.pipeline import CamembertNER, BiLSTMNER, segments, postprocess
from docusense.textproc import bio_to_spans
from docusense.metrics import ner_report
from docusense.classifier import DocClassifier, MODES

OUT = Path("results"); OUT.mkdir(exist_ok=True)
sets = {"dev": load_docs("data/docs_dev.jsonl"), "test": load_docs("data/docs_test.jsonl"),
        "reel": load_annotated_md("demo_docs/annotated.md")}
train = load_docs("data/docs_train.jsonl")
results = {"ner": {}, "classification": {}, "speed": {}}


def run_ner(ner, docs):
    """Évalue les entités telles que la chaîne les produit (BIO -> entités -> BIO bien formé).
    Une séquence I- sans B- devient donc une entité, comme dans l'application."""
    gold, pred, pred_clean, spans_per_doc, n_tok, t0 = [], [], [], [], 0, time.time()
    for d in docs:
        segs = segments(d["text"])
        outs = ner.predict_tokens(segs)
        doc_spans = []
        for toks, (labs, scores) in zip(segs, outs):
            sp = bio_to_spans(toks, labs, scores)
            gold.append(spans_to_bio(toks, d["entities"]))
            pred.append(spans_to_bio(toks, [(s, e, t) for s, e, t, _ in sp]))
            pred_clean.append(spans_to_bio(toks, [(s, e, t) for s, e, t, _ in postprocess(d["text"], sp)]))
            doc_spans += sp
            n_tok += len(toks)
        spans_per_doc.append(doc_spans)
    return gold, pred, pred_clean, spans_per_doc, n_tok / (time.time() - t0)


predicted = {}
results["ner_postprocess"] = {}
for key, ner in (("bilstm_crf", BiLSTMNER()), ("camembert", CamembertNER())):
    results["ner"][key], results["ner_postprocess"][key] = {}, {}
    for name, docs in sets.items():
        gold, pred, pred_clean, spans, speed = run_ner(ner, docs)
        results["ner"][key][name] = ner_report(gold, pred)
        results["ner_postprocess"][key][name] = ner_report(gold, pred_clean)["micro avg"]
        if name == "test":
            results["speed"][key] = round(speed, 1)
        if key == "camembert":
            predicted[name] = spans
        r, rc = results["ner"][key][name]["micro avg"], results["ner_postprocess"][key][name]
        print(f"NER {key:11s} {name:5s}  P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1-score']:.3f}"
              f"   (post-traité F1 {rc['f1-score']:.3f})", flush=True)

# exemples d'erreurs sur les documents réels (pour l'analyse qualitative du rapport)
errors = []
for d, spans in zip(sets["reel"], predicted["reel"]):
    g = {(s, e, t) for s, e, t in d["entities"]}
    p = {(s, e, t) for s, e, t, _ in spans}
    for s, e, t in sorted(g - p):
        errors.append({"doc": d["id"], "texte": d["text"][s:e], "attendu": t, "type": "manqué/mal typé"})
    for s, e, t in sorted(p - g):
        errors.append({"doc": d["id"], "texte": d["text"][s:e], "prédit": t, "type": "en trop/mal délimité"})
results["erreurs_reel"] = errors

texts_tr = [d["text"] for d in train]
gold_tr = [[(s, e, t) for s, e, t in d["entities"]] for d in train]
for target in ("type", "sector"):
    y_tr = [d[target] for d in train]
    results["classification"][target] = {}
    for mode in MODES:
        clf = DocClassifier(mode).fit(texts_tr, gold_tr, y_tr)
        res = {}
        for name in ("test", "reel"):
            docs = sets[name]
            y = [d[target] for d in docs]
            for ent_src, spans in (("gold", [d["entities"] for d in docs]), ("pred", predicted[name])):
                yp = clf.predict([d["text"] for d in docs], spans)
                res[f"{name}_{ent_src}"] = {"accuracy": round(accuracy_score(y, yp), 4),
                                            "macro_f1": round(f1_score(y, yp, average="macro"), 4)}
                if mode == "hybride" and ent_src == "pred":
                    res[f"{name}_pred"]["confusion"] = confusion_matrix(y, yp, labels=clf.classes_).tolist()
                    res[f"{name}_pred"]["labels"] = list(clf.classes_)
                    res[f"{name}_pred"]["predictions"] = list(zip([d["id"] for d in docs], y, list(yp)))
        results["classification"][target][mode] = res
        print(f"CLS {target:6s} {mode:8s}  test(pred) acc {res['test_pred']['accuracy']:.3f}  "
              f"réel(pred) acc {res['reel_pred']['accuracy']:.3f}", flush=True)
        if mode == "hybride":
            joblib.dump(clf, OUT / f"_clf_{target}.joblib")

joblib.dump({t: joblib.load(OUT / f"_clf_{t}.joblib") for t in ("type", "sector")}, "models/classifiers.joblib")
for t in ("type", "sector"):
    (OUT / f"_clf_{t}.joblib").unlink()
for key in ("bilstm_crf", "camembert"):
    results["history_" + key] = json.load(open(f"models/{key if key == 'bilstm_crf' else 'camembert_ner'}/history.json"))
json.dump(results, open(OUT / "metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1,
          default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("résultats écrits dans results/metrics.json")
