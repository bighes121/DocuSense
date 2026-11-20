"""Figures du rapport (PNG, 200 dpi) à partir de results/metrics.json et du corpus."""
import json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from docusense.corpus import load_docs
from docusense.data_gen import DOC_TYPES
from docusense.textproc import ENT_TYPES

OUT = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)
M = json.load(open("results/metrics.json", encoding="utf-8"))

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"               # créneaux catégoriels 1-3
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUES = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": INK2, "axes.edgecolor": AXIS, "figure.dpi": 200})


def clean(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. F1 par type d'entité, deux modèles (test)
fig, ax = plt.subplots(figsize=(7.2, 3.9))
types = ENT_TYPES + ["micro avg"]
names = ENT_TYPES + ["Global"]
y = np.arange(len(types))[::-1]
h = 0.36
for k, (key, lab, col, off) in enumerate([("bilstm_crf", "BiLSTM-CRF (from scratch)", S1, h / 2 + 0.02),
                                         ("camembert", "CamemBERT fine-tuné", S2, -h / 2 - 0.02)]):
    v = [M["ner"][key]["test"][t]["f1-score"] for t in types]
    ax.barh(y + off, v, height=h, color=col, label=lab, edgecolor="white", linewidth=1)
    for yi, vi in zip(y + off, v):
        ax.text(vi + 0.008, yi, f"{vi:.2f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(y, names)
ax.set_xlim(0, 1.09)
ax.set_xlabel("F1 (niveau entité, jeu de test)")
ax.legend(loc="lower left", frameon=False, ncol=2, bbox_to_anchor=(0, 1.0))
clean(ax)
save(fig, "ner_f1_par_type.png")

# 2. F1 dev par époque
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for key, lab, col, mk in (("bilstm_crf", "BiLSTM-CRF", S1, "o"), ("camembert", "CamemBERT", S2, "s")):
    hist = M["history_" + key]
    ax.plot([e["epoch"] for e in hist], [e["dev_f1"] for e in hist], color=col, lw=2, marker=mk, ms=5,
            label=lab, markeredgecolor="white", markeredgewidth=1)
ax.set_xlabel("Époque"); ax.set_ylabel("F1 (validation)")
ax.set_xticks(range(1, max(len(M["history_" + k]) for k in ("bilstm_crf", "camembert")) + 1))
ax.legend(frameon=False, loc="lower right")
clean(ax, "y")
save(fig, "courbes_apprentissage.png")

# 3. Matrice de confusion (type, test, entités prédites, hybride)
C = M["classification"]["type"]["hybride"]["test_pred"]
cm, labs = np.array(C["confusion"]), [DOC_TYPES[l] for l in C["labels"]]
fig, ax = plt.subplots(figsize=(5.6, 4.4))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("blues", BLUES)
ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
for i in range(len(labs)):
    for j in range(len(labs)):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                color="white" if cm[i, j] > cm.max() * 0.55 else INK)
ax.set_xticks(range(len(labs)), labs, rotation=30, ha="right")
ax.set_yticks(range(len(labs)), labs)
ax.set_xlabel("Classe prédite"); ax.set_ylabel("Classe réelle")
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
save(fig, "confusion_type.png")

# 4. Ablation des caractéristiques de classification
modes = [("entites", "Entités\nseules"), ("texte", "Texte brut\n(TF-IDF)"), ("hybride", "Hybride"),
         ("hybride_masque", "Hybride\n(texte masqué)")]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.1), sharey=True)
x = np.arange(len(modes))
w = 0.38
for ax, target, title in zip(axes, ("type", "sector"), ("Type de document", "Secteur")):
    for k, (src, lab, col) in enumerate([("test_pred", "Test (gabarits inédits)", S1),
                                         ("reel_pred", "Documents réels (6)", S2)]):
        v = [M["classification"][target][m][src]["accuracy"] for m, _ in modes]
        bars = ax.bar(x + (k - 0.5) * (w + 0.03), v, width=w, color=col, label=lab, edgecolor="white")
        for b, vi in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, vi + 0.015, f"{vi:.0%}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [l for _, l in modes], fontsize=8.5)
    ax.set_title(title, fontsize=10.5, loc="left", color=INK)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    clean(ax, "y")
axes[0].set_ylabel("Exactitude")
axes[0].legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.25))
save(fig, "ablation_classification.png")

# 5. Répartition des entités dans le corpus d'entraînement
cnt = Counter(t for d in load_docs("data/docs_train.jsonl") for _, _, t in d["entities"])
order = sorted(ENT_TYPES, key=lambda t: cnt[t])
fig, ax = plt.subplots(figsize=(6.0, 2.8))
ax.barh(order, [cnt[t] for t in order], color=S1, height=0.6)
for i, t in enumerate(order):
    ax.text(cnt[t] + 10, i, str(cnt[t]), va="center", fontsize=8.5, color=INK2)
ax.set_xlabel("Nombre d'entités (entraînement)")
clean(ax)
save(fig, "repartition_entites.png")

# 6. Schéma de l'architecture
fig, ax = plt.subplots(figsize=(9.6, 2.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
boxes = [(1, "Document", ".txt / .pdf\nou texte collé", "#f3f4f6"),
         (21, "Segmentation", "phrases +\ntokenisation", "#f3f4f6"),
         (41, "NER", "CamemBERT fine-tuné\nou BiLSTM-CRF", "#fde7dd"),
         (61, "Classification", "texte + profil d'entités\n→ type & secteur", "#dbe9fa"),
         (81, "Restitution", "interface, fiche,\nJSON, rapport HTML", "#f3f4f6")]
for x0, title, sub, fill in boxes:
    ax.add_patch(FancyBboxPatch((x0, 6), 17, 18, boxstyle="round,pad=0.4,rounding_size=2", fc=fill, ec=AXIS, lw=1))
    ax.text(x0 + 8.5, 19, title, ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(x0 + 8.5, 11.5, sub, ha="center", va="center", fontsize=8.3, color=INK2, linespacing=1.4)
for x0 in (18.6, 38.6, 58.6, 78.6):
    ax.annotate("", xy=(x0 + 2.2, 15), xytext=(x0, 15), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
save(fig, "architecture.png")
print("figures :", sorted(p.name for p in OUT.iterdir()))
