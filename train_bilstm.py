"""Entraînement du BiLSTM-CRF (from scratch) sur le corpus généré."""
import json, random, time
from pathlib import Path
import torch
from docusense.corpus import load_sentences
from docusense.bilstm_crf import BiLSTMCRF, Vocab, collate
from docusense.textproc import LABEL2ID, ID2LABEL, LABELS
from docusense.metrics import ner_f1

random.seed(0); torch.manual_seed(0)
torch.set_num_threads(4)
OUT = Path("models/bilstm_crf"); OUT.mkdir(parents=True, exist_ok=True)


def prep(sents):
    return [([t for t, _, _ in toks], [LABEL2ID[l] for l in labs]) for toks, labs in sents]


train, dev = prep(load_sentences("data/docs_train.jsonl")), prep(load_sentences("data/docs_dev.jsonl"))
vocab = Vocab([( [(w, 0, 0) for w in ws], None) for ws, _ in train])
model = BiLSTMCRF(len(vocab.words), len(vocab.chars), len(LABELS))
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
print(f"{len(train)} phrases d'entraînement, vocabulaire {len(vocab.words)} mots, "
      f"{sum(p.numel() for p in model.parameters()):,} paramètres")


def predict(data, bs=64):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(data), bs):
            W, C, K, _, M = collate(data[i:i + bs], vocab)
            out += model.predict(W, C, K, M)
    return out


def evaluate(data):
    pred = predict(data)
    return ner_f1([[ID2LABEL[i] for i in y] for _, y in data], [[ID2LABEL[i] for i in p] for p in pred])


history, best, EPOCHS, BS = [], -1, 15, 32
for ep in range(1, EPOCHS + 1):
    model.train(); random.shuffle(train); tot, t0 = 0.0, time.time()
    for i in range(0, len(train), BS):
        W, C, K, Y, M = collate(train[i:i + BS], vocab, word_dropout=0.25)
        loss = model.loss(W, C, K, Y, M)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); tot += loss.item() * len(W)
    f1 = evaluate(dev)
    history.append({"epoch": ep, "loss": tot / len(train), "dev_f1": f1, "sec": time.time() - t0})
    print(f"époque {ep:2d}  perte {tot / len(train):.4f}  F1 dev {f1:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    if f1 > best:
        best = f1
        torch.save({"state": model.state_dict(), "vocab": vocab.state()}, OUT / "model.pt")

json.dump(history, open(OUT / "history.json", "w"), indent=1)
print("meilleur F1 dev :", round(best, 4))
