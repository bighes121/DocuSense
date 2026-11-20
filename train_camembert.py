"""Fine-tuning de CamemBERT (modèle de langue brut, sans tête NER) en classification de tokens.

Boucle d'entraînement PyTorch écrite à la main : AdamW, warm-up linéaire, écrêtage du gradient.
L'étiquette d'un mot est portée par son premier sous-mot ; les autres sous-mots sont ignorés (-100).
"""
import json, random, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from docusense.corpus import load_sentences
from docusense.textproc import LABEL2ID, ID2LABEL, LABELS
from docusense.metrics import ner_f1

BASE = "models/camembert-base"      # copie locale de almanach/camembert-base (modèle de langue brut)
OUT = Path("models/camembert_ner"); OUT.mkdir(parents=True, exist_ok=True)
EPOCHS, BS, LR, MAXLEN = 3, 16, 5e-5, 128
FREEZE = 8          # contrainte CPU : embeddings + 8 couches basses gelées, 4 couches hautes + tête entraînées
random.seed(0); torch.manual_seed(0); torch.set_num_threads(4)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForTokenClassification.from_pretrained(
    BASE, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
for p in model.roberta.embeddings.parameters():
    p.requires_grad = False
for layer in model.roberta.encoder.layer[:FREEZE]:
    for p in layer.parameters():
        p.requires_grad = False


def encode(sents):
    out = []
    for toks, labs in sents:
        words = [t for t, _, _ in toks]
        enc = tok(words, is_split_into_words=True, truncation=True, max_length=MAXLEN)
        y, prev = [], None
        for w in enc.word_ids():
            y.append(-100 if w is None or w == prev else LABEL2ID[labs[w]])
            prev = w
        out.append((enc["input_ids"], y, labs))
    return out


def batches(data, bs, shuffle=False):
    # regroupement par longueur (moins de padding), ordre des lots mélangé à chaque époque
    idx = sorted(range(len(data)), key=lambda j: len(data[j][0]))
    groups = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if shuffle: random.shuffle(groups)
    for g in groups:
        chunk = [data[j] for j in g]
        T = max(len(c[0]) for c in chunk)
        ids = torch.full((len(chunk), T), tok.pad_token_id)
        att = torch.zeros((len(chunk), T), dtype=torch.long)
        lab = torch.full((len(chunk), T), -100)
        for b, (x, y, _) in enumerate(chunk):
            ids[b, :len(x)], att[b, :len(x)], lab[b, :len(y)] = torch.tensor(x), 1, torch.tensor(y)
        yield ids, att, lab, chunk


def evaluate(data):
    model.eval(); gold, pred = [], []
    with torch.no_grad():
        for ids, att, lab, chunk in batches(data, 32):
            logits = model(input_ids=ids, attention_mask=att).logits.argmax(-1)
            for b, (_, _, labs) in enumerate(chunk):
                p = [ID2LABEL[int(logits[b, t])] for t in range(lab.size(1)) if lab[b, t] != -100]
                gold.append(labs[:len(p)]); pred.append(p)
    return ner_f1(gold, pred)


train = encode(load_sentences("data/docs_train.jsonl"))
dev = encode(load_sentences("data/docs_dev.jsonl"))
steps = EPOCHS * ((len(train) + BS - 1) // BS)
no_decay = ("bias", "LayerNorm.weight")
trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
opt = torch.optim.AdamW([
    {"params": [p for n, p in trainable if not any(k in n for k in no_decay)], "weight_decay": 0.01},
    {"params": [p for n, p in trainable if any(k in n for k in no_decay)], "weight_decay": 0.0},
], lr=LR)
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
print(f"{len(train)} phrases, {steps} pas d'optimisation, {sum(p.numel() for p in model.parameters()):,} paramètres "
      f"dont {sum(p.numel() for _, p in trainable):,} entraînés", flush=True)

history, step, t0 = [], 0, time.time()
for ep in range(1, EPOCHS + 1):
    model.train(); tot, n = 0.0, 0
    for ids, att, lab, _ in batches(train, BS, shuffle=True):
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        tot += loss.item(); n += 1; step += 1
        if step % 25 == 0:
            print(f"  pas {step}/{steps}  perte {tot / n:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    f1 = evaluate(dev)
    history.append({"epoch": ep, "loss": tot / n, "dev_f1": f1, "sec": time.time() - t0})
    print(f"époque {ep}  perte {tot / n:.4f}  F1 dev {f1:.4f}", flush=True)

model.save_pretrained(OUT); tok.save_pretrained(OUT)
json.dump(history, open(OUT / "history.json", "w"), indent=1)
print("terminé en", round(time.time() - t0), "s")
