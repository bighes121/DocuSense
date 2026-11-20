"""Démonstration automatisée : lance l'application Streamlit, la pilote dans Edge et prend les captures."""
import socket, subprocess, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent
SHOTS = ROOT / "results/screenshots"; SHOTS.mkdir(parents=True, exist_ok=True)
PORT = 8599
URL = f"http://localhost:{PORT}"

server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT)],
                          cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(120):
    try:
        socket.create_connection(("localhost", PORT), 1).close(); break
    except OSError:
        time.sleep(1)


IDLE_JS = """() => {
  const busy = document.querySelector('[data-testid="stSpinner"], [data-stale="true"], [data-testid="stSkeleton"]');
  const running = [...document.querySelectorAll('[data-testid="stStatusWidget"]')].some(e => /Running|Exécution/i.test(e.innerText));
  return !busy && !running;
}"""


def settle(page, timeout=240_000):
    """Attend que Streamlit ait fini : pas de spinner, pas d'éléments périmés (grisés), pas de tableau en chargement.
    La condition doit rester vraie deux fois de suite (évite de capturer entre deux rafraîchissements)."""
    page.wait_for_timeout(1000)
    for _ in range(2):
        page.wait_for_function(IDLE_JS, timeout=timeout, polling=250)
        page.wait_for_timeout(1500)


def shot(page, name, el=None):
    path = SHOTS / f"{name}.png"
    (page.locator(el).first if el else page).screenshot(path=str(path), **({} if el else {"full_page": True}))
    print("capture", path.name, flush=True)


def analyze(page):
    page.get_by_role("button", name="Analyser le document").click()
    settle(page)


def choose_example(page, label):
    page.locator('[data-testid="stSelectbox"]').first.click()
    page.get_by_role("option", name=label).click()
    settle(page)


try:
    with sync_playwright() as p:
        exe = next(x for x in [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                               r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                               r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"] if Path(x).exists())
        browser = p.chromium.launch(executable_path=exe, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 2000}, device_scale_factor=1.4,
                                accept_downloads=True)
        page.goto(URL)
        page.get_by_text("Document annoté").wait_for(timeout=300_000)    # premier chargement des modèles
        settle(page)

        # 1. Facture (exemple par défaut, analysé au chargement)
        shot(page, "01_facture_vue_complete")
        shot(page, "02_facture_document_annote", '[data-testid="stHorizontalBlock"]:has-text("Document annoté")')

        # 2. Autres types de documents
        for i, label in enumerate(["Contrat de licence SaaS", "Contrat de travail – banque",
                                   "Rapport d'études – télémédecine", "Article scientifique – NER"], start=3):
            choose_example(page, label); analyze(page)
            name = f"{i:02d}_{label.split(' –')[0].split(' SaaS')[0].replace(' ', '_').replace(chr(39), '').lower()}"
            shot(page, name)
            shot(page, name + "_detail", '[data-testid="stHorizontalBlock"]:has-text("Document annoté")')

        # 3. Import d'un fichier jamais vu
        choose_example(page, "Facture – station de pompage solaire")
        page.locator('input[type="file"]').set_input_files(str(ROOT / "demo_docs/upload_note_conjoncture.txt"))
        settle(page); analyze(page)
        shot(page, "07_import_fichier_inedit")

        # 4. Filtre d'entités (montants et dates uniquement)
        for t in ("ORG", "PER", "LIEU", "REF", "POURCENT"):
            page.get_by_text(t, exact=True).first.click(); page.wait_for_timeout(300)
        settle(page)
        shot(page, "08_filtre_montants_dates", '[data-testid="stHorizontalBlock"]:has-text("Document annoté")')
        for t in ("ORG", "PER", "LIEU", "REF", "POURCENT"):
            page.get_by_text(t, exact=True).first.click(); page.wait_for_timeout(300)
        settle(page)

        # 5. Comparaison avec le BiLSTM-CRF sur le même document
        page.get_by_text("BiLSTM-CRF (from scratch)").click(); settle(page); analyze(page)
        shot(page, "09_bilstm_meme_document", '[data-testid="stHorizontalBlock"]:has-text("Document annoté")')
        page.get_by_text("CamemBERT fine-tuné").first.click(); settle(page); analyze(page)

        # 6. Rapport HTML téléchargé
        with page.expect_download() as dl:
            page.get_by_role("button", name="⬇ Rapport HTML").click()
        report = SHOTS / "rapport_auto.html"
        dl.value.save_as(report)
        rp = browser.new_page(viewport={"width": 1100, "height": 1200}, device_scale_factor=1.4)
        rp.goto(report.as_uri()); rp.wait_for_timeout(800)
        rp.screenshot(path=str(SHOTS / "10_rapport_html_auto.png"), full_page=True)
        print("capture 10_rapport_html_auto.png", flush=True)

        # 7. Onglet performances
        page.get_by_role("tab", name="📊 Performances des modèles").click(); settle(page)
        shot(page, "11_onglet_performances")
        browser.close()
finally:
    server.terminate()
