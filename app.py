"""DocuSense — interface Streamlit : extraction d'entités + classification de documents."""
import html, json
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
from docusense.pipeline import DocuSense
from docusense.textproc import ENT_TYPES, ENT_DESCRIPTIONS

ROOT = Path(__file__).parent
COLORS = {"DATE": "#2563EB", "ORG": "#7C3AED", "PER": "#DB2777", "LIEU": "#059669",
          "MONTANT": "#D97706", "REF": "#475569", "POURCENT": "#DC2626"}
EXAMPLES = {
    "Facture – station de pompage solaire": "facture_solaire",
    "Contrat de licence SaaS": "licence_logiciel",
    "Contrat de travail – banque": "cdi_banque",
    "Rapport d'études – télémédecine": "rapport_telemedecine",
    "Article scientifique – NER": "article_ner",
    "Facture – clinique": "facture_clinique",
}

st.set_page_config(page_title="DocuSense", page_icon="📄", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1350px;}
.hero {background: linear-gradient(120deg, #1E1B4B 0%, #3730A3 55%, #6D28D9 100%); color: #fff;
       padding: 22px 28px; border-radius: 14px; margin-bottom: 18px;}
.hero h1 {color: #fff; font-size: 30px; margin: 0 0 4px 0; padding: 0;}
.hero p {color: #E0E7FF; margin: 0; font-size: 15px;}
.card {border: 1px solid #E5E7EB; border-radius: 12px; padding: 14px 16px; background: #fff; height: 100%;}
.card .k {font-size: 12px; text-transform: uppercase; letter-spacing: .05em; color: #6B7280;}
.card .v {font-size: 21px; font-weight: 700; color: #111827; margin-top: 2px;}
.card .s {font-size: 13px; color: #6B7280;}
.doc {line-height: 2.05; font-size: 15px; border: 1px solid #E5E7EB;
      border-radius: 12px; padding: 18px 20px; background: #FCFCFD; max-height: 900px; overflow-y: auto;}
.ent {padding: 2px 4px; border-radius: 5px; border-bottom: 2px solid; font-weight: 500;}
.lbl {font-size: 10px; font-weight: 700; color: #fff; border-radius: 4px; padding: 1px 4px; margin-left: 4px;
      vertical-align: 1px; letter-spacing: .03em;}
.bar {display: flex; align-items: center; gap: 8px; margin: 5px 0; font-size: 13.5px;}
.bar .name {width: 150px; color: #374151;}
.bar .track {flex: 1; background: #EEF2FF; border-radius: 6px; height: 12px; overflow: hidden;}
.bar .fill {height: 100%; background: #4F46E5; border-radius: 6px;}
.bar .pct {width: 52px; text-align: right; color: #111827; font-variant-numeric: tabular-nums;}
.field {margin: 0 0 10px 0;} .field .k {font-size: 12px; color: #6B7280; text-transform: uppercase;
        letter-spacing: .04em;} .field .v {font-size: 14.5px; color: #111827;}
.chip {display: inline-block; border-radius: 999px; padding: 2px 10px; margin: 2px 4px 2px 0; font-size: 12.5px;
       font-weight: 600;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Chargement des modèles…")
def engine():
    eng = DocuSense("camembert")
    eng.set_ner("bilstm")
    eng.analyze("Facture émise à Agadir le 3 mars 2026 pour 1 200 DH.")      # préchauffage des deux modèles
    eng.set_ner("camembert")
    eng.analyze("Facture émise à Agadir le 3 mars 2026 pour 1 200 DH.")
    return eng


def example_text(key):
    return (ROOT / "demo_docs/txt" / f"{key}.txt").read_text(encoding="utf-8")


def highlight(text, ents, shown):
    out, pos = [], 0
    for e in sorted(ents, key=lambda e: e["start"]):
        if e["type"] not in shown or e["start"] < pos:
            continue
        c = COLORS[e["type"]]
        out.append(html.escape(text[pos:e["start"]]))
        out.append(f'<span class="ent" style="background:{c}1F;border-color:{c}" title="{e["type"]} – '
                   f'confiance {e["score"]:.0%}">{html.escape(e["text"])}<span class="lbl" '
                   f'style="background:{c}">{e["type"]}</span></span>')
        pos = e["end"]
    out.append(html.escape(text[pos:]))
    # sauts de ligne explicites : empêche Markdown d'interpréter « 1. Contexte » comme une liste
    return "".join(out).replace("\n", "<br>")


def bars(probas):
    return "".join(f'<div class="bar"><div class="name">{html.escape(k)}</div><div class="track">'
                   f'<div class="fill" style="width:{v * 100:.1f}%"></div></div><div class="pct">{v:.0%}</div></div>'
                   for k, v in probas.items())


def report_html(text, res):
    rows = "".join(f"<tr><td>{e['type']}</td><td>{html.escape(e['text'])}</td><td>{e['score']:.0%}</td></tr>"
                   for e in res["entities"])
    fields = "".join(f"<tr><th>{k}</th><td>{html.escape(', '.join(v) if isinstance(v, list) else (v or '—'))}</td></tr>"
                     for k, v in res["summary"].items())
    return f"""<!doctype html><html lang="fr"><meta charset="utf-8"><title>Rapport DocuSense</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:30px auto;color:#111827}}
h1{{color:#3730A3}} table{{border-collapse:collapse;width:100%;margin:10px 0 24px}}
td,th{{border:1px solid #E5E7EB;padding:6px 10px;text-align:left;font-size:14px}} th{{background:#F3F4F6;width:220px}}
.doc{{white-space:pre-wrap;line-height:2;border:1px solid #E5E7EB;border-radius:8px;padding:14px}}
.ent{{padding:1px 3px;border-radius:4px;border-bottom:2px solid}}
.lbl{{font-size:10px;color:#fff;border-radius:3px;padding:0 3px;margin-left:3px}}</style>
<h1>Rapport d'analyse DocuSense</h1>
<p>Généré le {datetime.now():%d/%m/%Y à %H:%M} — modèle NER : {res['model']}</p>
<table><tr><th>Type de document</th><td>{res['type']['label']} ({res['type']['confidence']:.0%})</td></tr>
<tr><th>Secteur</th><td>{res['sector']['label']} ({res['sector']['confidence']:.0%})</td></tr>{fields}</table>
<h2>Entités extraites ({len(res['entities'])})</h2><table><tr><th>Type</th><th>Texte</th><th>Confiance</th></tr>{rows}</table>
<h2>Document annoté</h2><div class="doc">{highlight(text, res['entities'], set(ENT_TYPES))}</div></html>"""


# ----------------------------------------------------------------------------- barre latérale
with st.sidebar:
    st.markdown("## 📄 DocuSense")
    st.caption("Extraction d'entités nommées et classification automatique de documents professionnels en français.")
    model_label = st.radio("Modèle NER", ["CamemBERT fine-tuné", "BiLSTM-CRF (from scratch)"])
    st.markdown("**Entités affichées**")
    shown = {t for t in ENT_TYPES if st.checkbox(t, True, key=f"show_{t}", help=ENT_DESCRIPTIONS[t])}
    st.markdown("".join(f'<span class="chip" style="background:{COLORS[t]}1F;color:{COLORS[t]}">{t}</span>'
                        for t in ENT_TYPES), unsafe_allow_html=True)
    st.divider()
    st.caption("Corpus annoté maison · 2 modèles NER entraînés · classifieur hybride entités + TF-IDF.")

eng = engine()
eng.set_ner("camembert" if model_label.startswith("CamemBERT") else "bilstm")

st.markdown('<div class="hero"><h1>DocuSense</h1><p>Déposez une facture, un contrat, un rapport ou un article : '
            'DocuSense extrait les dates, organisations, montants, personnes, lieux et références, puis identifie '
            'le type et le secteur du document.</p></div>', unsafe_allow_html=True)

tab_an, tab_perf = st.tabs(["🔍 Analyse d'un document", "📊 Performances des modèles"])

with tab_an:
    c1, c2 = st.columns([3, 2])
    with c1:
        ex = st.selectbox("Exemple", list(EXAMPLES), key="example")
    with c2:
        up = st.file_uploader("…ou importer un fichier (.txt, .pdf)", type=["txt", "pdf"])
    if up is not None:
        if up.name.lower().endswith(".pdf"):
            from pypdf import PdfReader
            default = "\n".join(p.extract_text() or "" for p in PdfReader(up).pages)
        else:
            default = up.read().decode("utf-8", errors="replace")
    else:
        default = example_text(EXAMPLES[ex])
    text = st.text_area("Texte du document", default, height=210, key=f"txt_{ex}_{up.name if up else ''}")

    if st.button("Analyser le document", type="primary", use_container_width=True) or "res" not in st.session_state:
        with st.spinner("Analyse en cours…"):
            st.session_state.res = eng.analyze(text)
            st.session_state.res_text = text
    res, rtext = st.session_state.res, st.session_state.res_text

    m = st.columns(4)
    cards = [("Type de document", res["type"]["label"], f"confiance {res['type']['confidence']:.0%}"),
             ("Secteur", res["sector"]["label"], f"confiance {res['sector']['confidence']:.0%}"),
             ("Entités détectées", str(len(res["entities"])),
              " · ".join(f"{sum(e['type'] == t for e in res['entities'])} {t}" for t in ("ORG", "DATE", "MONTANT"))),
             ("Temps d'analyse", f"{res['timing']['total_s'] * 1000:.0f} ms", res["model"])]
    for col, (k, v, s) in zip(m, cards):
        col.markdown(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>',
                     unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Document annoté")
        st.markdown(f'<div class="doc">{highlight(rtext, res["entities"], shown)}</div>', unsafe_allow_html=True)
    with right:
        st.markdown("#### Fiche de synthèse")
        for k, v in res["summary"].items():
            val = ", ".join(v) if isinstance(v, list) else v
            st.markdown(f'<div class="field"><div class="k">{k}</div><div class="v">{html.escape(val or "—")}</div></div>',
                        unsafe_allow_html=True)
        st.markdown("#### Type de document")
        st.markdown(bars(res["type"]["probas"]), unsafe_allow_html=True)
        st.markdown("#### Secteur")
        st.markdown(bars(res["sector"]["probas"]), unsafe_allow_html=True)

    st.markdown("#### Entités extraites")
    df = pd.DataFrame([{"Type": e["type"], "Texte": e["text"], "Confiance": e["score"], "Position": f"{e['start']}–{e['end']}"}
                       for e in res["entities"] if e["type"] in shown])
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={"Confiance": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("⬇ Résultat JSON", json.dumps(res, ensure_ascii=False, indent=2), "docusense.json",
                       "application/json", use_container_width=True)
    d2.download_button("⬇ Rapport HTML", report_html(rtext, res), "rapport_docusense.html", "text/html",
                       use_container_width=True)

with tab_perf:
    mpath = ROOT / "results/metrics.json"
    if not mpath.exists():
        st.info("Lancez evaluate.py pour générer les métriques.")
    else:
        M = json.loads(mpath.read_text(encoding="utf-8"))
        st.markdown("#### Reconnaissance d'entités — F1 par type (jeu de test : gabarits et valeurs jamais vus)")
        rows = []
        for t in ENT_TYPES + ["micro avg"]:
            rows.append({"Entité": "Global (micro)" if t == "micro avg" else t,
                         "BiLSTM-CRF": M["ner"]["bilstm_crf"]["test"][t]["f1-score"],
                         "CamemBERT": M["ner"]["camembert"]["test"][t]["f1-score"],
                         "Support": M["ner"]["camembert"]["test"][t]["support"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)
                                    for c in ("BiLSTM-CRF", "CamemBERT")})
        f = ROOT / "results/figures"
        cols = st.columns(2)
        for col, name in zip(cols, ["ner_f1_par_type.png", "confusion_type.png"]):
            if (f / name).exists():
                col.image((f / name).read_bytes(), use_container_width=True)
        st.markdown("#### Classification (entités prédites par CamemBERT) — ablation des caractéristiques")
        names = {"entites": "Entités seules", "texte": "Texte brut (TF-IDF)", "hybride": "Hybride ✓ (déployé)",
                 "hybride_masque": "Hybride, texte masqué"}
        st.dataframe(pd.DataFrame([{"Caractéristiques": names[k],
                                    "Type – test": M["classification"]["type"][k]["test_pred"]["accuracy"],
                                    "Type – docs réels": M["classification"]["type"][k]["reel_pred"]["accuracy"],
                                    "Secteur – test": M["classification"]["sector"][k]["test_pred"]["accuracy"],
                                    "Secteur – docs réels": M["classification"]["sector"][k]["reel_pred"]["accuracy"]}
                                   for k in names]),
                     hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="percent")
                                    for c in ("Type – test", "Type – docs réels", "Secteur – test",
                                              "Secteur – docs réels")})
        if (f / "ablation_classification.png").exists():
            st.image((f / "ablation_classification.png").read_bytes(), use_container_width=True)
