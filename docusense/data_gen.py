"""Générateur de corpus annoté : documents professionnels français (5 types x 4 secteurs).

Chaque entité est insérée à partir d'un gabarit : sa position exacte est connue, l'annotation est donc
parfaite. Pour mesurer une vraie généralisation, le jeu de test utilise des gabarits de phrases ET des
valeurs d'entités (noms, entreprises, villes, préfixes de référence) jamais vus à l'entraînement.
"""
import json, random, re
from pathlib import Path

DOC_TYPES = {
    "facture": "Facture",
    "contrat_licence": "Contrat de licence",
    "contrat_travail": "Contrat de travail",
    "rapport_etude": "Rapport d'études",
    "article_scientifique": "Article scientifique",
}
SECTORS = {"sante": "Santé", "finance": "Finance", "technologie": "Technologie", "energie": "Énergie"}

# ----------------------------------------------------------------------------- valeurs d'entités
FIRST = """Ahmed Youssef Salma Karim Nadia Mehdi Imane Omar Hajar Thomas Julie Nicolas Camille Pierre Sophie
Hamza Khadija Rachid Leïla Anas Meryem Lucas Emma Hugo Chloé Ilyas Sara Yasmine Mathieu Claire Adil Zineb
Reda Nora Samir Inès Antoine Laura Bilal Amine Soukaina Othmane Manon Julien Marwa Ayoub Élodie Driss Rim
Kenza Vincent Hiba Badr Céline Taha Asmae Paul Loubna Walid Aya Simon""".split()
LAST = """El Amrani|Benali|Alaoui|Bennani|Tazi|Idrissi|Chraibi|Berrada|Lahlou|Ouazzani|Martin|Bernard|Dubois|
Moreau|Laurent|Lefebvre|Girard|Fontaine|Rousseau|Mercier|Haddad|Ait Lahcen|Amzil|Boukili|Fassi|Kettani|
Sefrioui|Durand|Petit|Garnier|Benjelloun|Zahidi|Ouali|Naciri|Bouzid|Faure|Chevalier|Roux|Benkirane|Mansouri|
El Fassi|Lambert|Bonnet|Rami|Skalli|Hamdaoui|Belghiti|Guerin|Morel|Essaidi""".replace("\n", "").split("|")

CORES = """Atlas Argan Saphir Nova Horizon Orion Sahara Zenith Delta Medina Oasis Cèdre Aurora Vertex Kairos Altis
Lumen Nexa Sigma Tamra Amane Ifrane Corail Boréal Helios Quantis Optima Mistral Polaris Azur Toubkal Anfa Ksar
Noria Sirius Talia Sorel Majorelle Vega Alto Riviera Opale Tafraout Melia Krono Ondine Brava Lyra""".split()
LEGAL = ["SARL", "SA", "SAS", "Group", "Holding", ""]
SECTOR_ORG_WORDS = {
    "sante": ["Santé", "Pharma", "Médical", "Biotech", "Diagnostics"],
    "finance": ["Capital", "Finance", "Invest", "Assurances", "Crédit"],
    "technologie": ["Digital", "Technologies", "Software", "Data", "Systems"],
    "energie": ["Énergie", "Solar", "Power", "Électricité", "Renewables"],
}
SECTOR_ORG_HEADS = {
    "sante": ["Laboratoire", "Clinique", "Centre Médical", "Groupe Hospitalier"],
    "finance": ["Banque", "Cabinet", "Caisse", "Fonds"],
    "technologie": ["Studio", "Agence", "Institut", "Lab"],
    "energie": ["Compagnie", "Société", "Agence", "Centrale"],
}
INSTITUTIONS = """Université Ibn Zohr|Université Mohammed V|Université Cadi Ayyad|Université Hassan II|Université Paris-Saclay|
Université de Lyon|Université Ibn Tofail|Université Laval|CNRS|INSERM|CNRST|INRIA|ANRT|OMPIC|CHU Ibn Rochd|
École Centrale de Lyon|Institut Pasteur|Agence Nationale de la Recherche|Bank Al-Maghrib|Ministère de la Santé|
Office National de l'Électricité|Université Euro-Méditerranéenne|Université Sorbonne Nouvelle|Mines ParisTech|
Institut National de Statistique|Banque Mondiale|Université de Genève|Université Abdelmalek Essaadi""".replace("\n", "").split("|")

CITIES = """Agadir Casablanca Rabat Marrakech Tanger Fès Taroudant Oujda Kénitra Tétouan Meknès El-Jadida Laâyoune
Paris Lyon Marseille Toulouse Bordeaux Lille Nantes Montréal Genève Bruxelles Dakar Tunis Madrid Essaouira
Safi Nador Béni-Mellal Ouarzazate Dakhla Strasbourg Nice Rennes Grenoble Abidjan Alger Lisbonne Barcelone""".split()
COUNTRIES = ["Maroc", "France", "Sénégal", "Tunisie", "Belgique", "Canada", "Espagne", "Côte d'Ivoire", "Suisse"]

REF_PREFIXES = ["FAC", "INV", "CMD", "LIC", "CT", "RH", "ET", "RAP", "PRJ", "ANR", "DOS", "DEV",
                "FA", "LGL", "CDI", "MKT", "CNRST", "BC", "AV", "REF"]

MONTHS = "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split()
CURRENCIES = ["MAD", "DH", "dirhams", "€", "euros", "EUR", "USD", "dollars"]

# ----------------------------------------------------------------------------- vocabulaire sectoriel (non-entités)
DOM = {
    "sante": ["dispositifs médicaux", "services hospitaliers", "produits pharmaceutiques", "analyses biologiques",
              "solutions de télémédecine"],
    "finance": ["services bancaires", "crédits immobiliers", "paiements mobiles", "produits d'assurance",
                "marchés financiers"],
    "technologie": ["solutions cloud", "logiciels de gestion", "systèmes d'information", "applications mobiles",
                    "objets connectés"],
    "energie": ["énergies renouvelables", "centrales solaires", "réseaux électriques", "parcs éoliens",
                "solutions de stockage d'énergie"],
}
PRODUIT = {
    "sante": ["logiciel de gestion des dossiers patients", "système d'imagerie portable", "kit de diagnostic rapide",
              "module de téléconsultation"],
    "finance": ["logiciel de scoring de crédit", "terminal de paiement", "module de détection de fraude",
                "progiciel de gestion de portefeuille"],
    "technologie": ["logiciel de gestion de projet", "serveur de stockage", "progiciel d'analyse de données",
                    "module d'authentification"],
    "energie": ["système de supervision du réseau", "compteur intelligent", "module de prévision de production",
                "convertisseur solaire"],
}
TASK = {
    "sante": ["la détection précoce des maladies chroniques", "l'analyse automatique d'images médicales"],
    "finance": ["la prédiction du risque de crédit", "la détection de transactions frauduleuses"],
    "technologie": ["la classification automatique de textes", "l'optimisation des infrastructures cloud"],
    "energie": ["la prévision de la production solaire", "l'optimisation des réseaux électriques"],
}

# ----------------------------------------------------------------------------- gabarits
# Les titres et les phrases d'indice 3, 7, 11, 15 sont réservés au test.
TITLES = {
    "facture": ["FACTURE N° {REF}", "Facture {REF}", "Facture commerciale", "FACTURE – Réf. {REF}"],
    "contrat_licence": ["CONTRAT DE LICENCE D'UTILISATION N° {REF}", "Contrat de licence logicielle",
                        "Accord de concession de licence", "CONTRAT DE LICENCE – {ORG}"],
    "contrat_travail": ["CONTRAT DE TRAVAIL À DURÉE INDÉTERMINÉE", "Contrat de travail", "Contrat d'engagement",
                        "CONTRAT DE TRAVAIL N° {REF}"],
    "rapport_etude": ["RAPPORT D'ÉTUDE – Marché des {DOM}", "Rapport d'études sectoriel",
                      "Étude de marché : {DOM}", "RAPPORT FINAL D'ÉTUDE"],
    "article_scientifique": ["Une approche par apprentissage profond pour {TASK}",
                             "Vers une meilleure modélisation pour {TASK}",
                             "Étude expérimentale sur {TASK}", "Apport des transformeurs à {TASK}"],
}
TITLE_TEST_IDX = {3}

TEMPLATES = {
    "facture": [
        "Date d'émission : {DATE}",
        "Émetteur : {ORG}, {LIEU}",
        "Client : {ORG}, à l'attention de {PER}",
        "Facturé à {ORG}, service achats, {LIEU}.",
        "Désignation : {PRODUIT}, quantité 3, prix unitaire {MONTANT}",
        "Montant total HT : {MONTANT}",
        "TVA ({POURCENT}) : {MONTANT}",
        "Net à payer TTC : {MONTANT}",
        "Date d'échéance : {DATE}",
        "Paiement par virement avant le {DATE} sur le compte de {ORG}.",
        "Tout retard de paiement entraînera une pénalité de {POURCENT} du montant dû.",
        "Remise commerciale de {POURCENT} appliquée sur la commande {REF}.",
        "Facture établie à {LIEU} le {DATE} par {PER}, responsable comptable.",
        "Merci de rappeler la référence {REF} lors de votre règlement.",
        "Acompte versé le {DATE} : {MONTANT}.",
        "Prestations de maintenance réalisées pour {ORG} en {ANNEE}.",
    ],
    "contrat_licence": [
        "Entre {ORG}, société dont le siège est situé à {LIEU}, représentée par {PER}, ci-après « le Concédant »,",
        "Et {ORG}, ci-après « le Licencié », représentée par {PER}.",
        "Le Concédant accorde au Licencié une licence non exclusive d'utilisation du {PRODUIT}.",
        "La présente licence est concédée à {ORG} pour le territoire du {PAYS}.",
        "La licence est consentie pour une durée de trois ans à compter du {DATE}.",
        "En contrepartie, le Licencié versera une redevance annuelle de {MONTANT}.",
        "Une redevance proportionnelle de {POURCENT} du chiffre d'affaires sera due chaque trimestre.",
        "Le Licencié règle des frais d'intégration de {MONTANT} à la signature.",
        "Le présent contrat prend effet le {DATE} et expire le {DATE}.",
        "Toute sous-licence est interdite sans l'accord écrit de {ORG}.",
        "Les droits de propriété intellectuelle restent la propriété exclusive du Concédant.",
        "En cas de litige, les tribunaux de {LIEU} seront seuls compétents.",
        "Fait à {LIEU}, le {DATE}, en deux exemplaires originaux.",
        "Le Licencié s'engage à ne pas décompiler le logiciel ni à le céder à des tiers.",
        "Les mises à jour du {PRODUIT} sont incluses pour un montant forfaitaire de {MONTANT}.",
        "Le contrat pourra être résilié par {ORG} avec un préavis de trois mois.",
    ],
    "contrat_travail": [
        "Entre la société {ORG}, sise à {LIEU}, représentée par {PER}, directeur des ressources humaines,",
        "Et {PER}, demeurant à {LIEU}, ci-après « le Salarié ».",
        "Le Salarié est engagé au sein du pôle {DOM} à compter du {DATE}.",
        "{PER} rejoindra l'équipe de {ORG} le {DATE} en qualité de cheffe de projet.",
        "La période d'essai est fixée à trois mois, renouvelable une fois.",
        "La rémunération brute mensuelle est fixée à {MONTANT}.",
        "Une prime annuelle pouvant atteindre {POURCENT} du salaire de base sera versée.",
        "Le salaire sera revalorisé de {POURCENT} après la première année.",
        "Le lieu de travail est fixé à {LIEU}.",
        "Le Salarié bénéficie de 22 jours ouvrables de congés payés par an.",
        "Le présent contrat est enregistré sous la référence {REF}.",
        "Fait à {LIEU}, le {DATE}.",
        "Le Salarié s'engage à respecter la clause de confidentialité au profit de {ORG}.",
        "Toute modification du contrat fera l'objet d'un avenant signé par {PER}.",
        "Une indemnité de transport de {MONTANT} est accordée mensuellement.",
        "Le contrat prend fin de plein droit en cas de départ à la retraite.",
    ],
    "rapport_etude": [
        "Étude réalisée par {ORG} pour le compte de {ORG}.",
        "Période couverte : du {DATE} au {DATE}.",
        "Le marché des {DOM} a atteint {MONTANT} en {ANNEE}.",
        "Les auteurs remercient {ORG} pour l'accès aux données du secteur.",
        "Il enregistre une croissance annuelle de {POURCENT} sur la période étudiée.",
        "L'enquête a été menée auprès de 450 entreprises situées à {LIEU} et à {LIEU}.",
        "Selon {PER}, chef de projet chez {ORG}, la demande devrait doubler d'ici {ANNEE}.",
        "Le chiffre d'affaires cumulé des dix premiers acteurs dépasse {MONTANT}.",
        "Les investissements prévus s'élèvent à {MONTANT}.",
        "Recommandation 1 : renforcer les partenariats entre {ORG} et les acteurs régionaux.",
        "La part de marché de {ORG} est estimée à {POURCENT}.",
        "Le taux d'adoption a progressé de {POURCENT} en un an selon notre panel.",
        "Le rapport final a été remis le {DATE} au comité de pilotage.",
        "Méthodologie : entretiens qualitatifs et analyse de données secondaires.",
        "Les principaux freins identifiés concernent le financement et la réglementation.",
        "Synthèse rédigée par {PER} et {PER}.",
    ],
    "article_scientifique": [
        "{PER}, {PER} et {PER}",
        "Affiliations : {ORG}, {LIEU} ; {ORG}, {LIEU}.",
        "Résumé — Cet article propose une nouvelle approche pour {TASK}.",
        "Ce travail s'inscrit dans le cadre d'une collaboration entre {ORG} et {ORG}.",
        "Les expériences montrent une amélioration de {POURCENT} du score F1 par rapport à l'état de l'art.",
        "Mots-clés : apprentissage profond, {DOM}, analyse de données.",
        "Nous utilisons un corpus de 12 000 documents collectés entre {DATE} et {DATE}.",
        "Le code et les données sont disponibles sur demande auprès de {PER}.",
        "Ce travail a été financé par {ORG} (projet {REF}).",
        "Reçu le {DATE} ; accepté le {DATE}.",
        "Les travaux antérieurs de {PER} ont montré les limites des méthodes classiques.",
        "Le modèle atteint une exactitude de {POURCENT} sur le jeu de test.",
        "Les données ont été fournies par {ORG}.",
        "Correspondance : {PER}, {ORG}, {LIEU}.",
        "La section 4 présente les résultats expérimentaux et la discussion.",
        "Une version préliminaire a été présentée à {LIEU} en {ANNEE}.",
    ],
}
GENERIC = [
    "Pour toute question, contactez {PER} au service client de {ORG}.",
    "Document confidentiel – ne pas diffuser.",
    "Siège social : {LIEU}, {PAYS}.",
    "Version mise à jour le {DATE}.",
    "Page 1 sur 2.",
    "{ORG} est un acteur reconnu dans le domaine des {DOM}.",
    "Les montants sont exprimés hors taxes.",
    "Ce document fait référence au dossier {REF}.",
]
TEST_IDX = {3, 7, 11, 15}

SLOT_TYPES = {"ORG": "ORG", "PER": "PER", "LIEU": "LIEU", "PAYS": "LIEU", "DATE": "DATE", "ANNEE": "DATE",
              "MONTANT": "MONTANT", "REF": "REF", "POURCENT": "POURCENT"}
SLOT_RE = re.compile(r"\{(\w+)\}")


def _split(seq, rng, frac=0.7):
    seq = list(seq)
    rng.shuffle(seq)
    k = int(len(seq) * frac)
    return seq[:k], seq[k:]


class Generator:
    def __init__(self, split, seed):
        self.rng = random.Random(seed)
        pool_rng = random.Random(1234)          # découpage des valeurs identique pour toutes les graines
        parts = {name: _split(vals, pool_rng) for name, vals in [
            ("first", FIRST), ("last", LAST), ("cores", CORES), ("inst", INSTITUTIONS),
            ("cities", CITIES), ("prefix", REF_PREFIXES)]}
        pick = 0 if split == "train" else 1
        self.p = {k: v[pick] for k, v in parts.items()}
        self.split = split

    # ------------------------------------------------------------ générateurs de valeurs
    def per(self):
        r = self.rng
        f, l = r.choice(self.p["first"]), r.choice(self.p["last"])
        return r.choices([f"{f} {l}", f"{f[0]}. {l}", f"{l.upper()} {f}"], [0.7, 0.15, 0.15])[0]

    def org(self, sector):
        r, c = self.rng, self.rng.choice(self.p["cores"])
        k = r.random()
        if k < 0.35:
            return f"{c} {r.choice(SECTOR_ORG_WORDS[sector])} {r.choice(LEGAL)}".strip()
        if k < 0.55:
            return f"{r.choice(SECTOR_ORG_HEADS[sector])} {c}"
        if k < 0.7:
            return f"Groupe {c}"
        if k < 0.9:
            return r.choice(self.p["inst"])
        return "".join(r.choice("ABCDEFGHIJKLMNOPRSTV") for _ in range(r.choice([3, 4])))

    def lieu(self):
        return self.rng.choice(self.p["cities"])

    def date(self):
        r = self.rng
        d, m, y = r.randint(1, 28), r.randint(1, 12), r.randint(2018, 2027)
        day = "1er" if d == 1 else str(d)
        return r.choices([f"{day} {MONTHS[m-1]} {y}", f"{d:02d}/{m:02d}/{y}", f"{y}-{m:02d}-{d:02d}",
                          f"{day} {MONTHS[m-1]}", f"{d:02d}.{m:02d}.{y}"], [0.45, 0.3, 0.1, 0.1, 0.05])[0]

    def montant(self):
        r = self.rng
        if r.random() < 0.1:
            n = f"{r.randint(1, 9)},{r.randint(1, 9)}"
            return f"{n} millions de {r.choice(['dirhams', 'dollars'])}" if r.random() < 0.5 else f"{n} millions d'euros"
        v = r.choice([r.randint(50, 999), r.randint(1000, 99999), r.randint(100000, 9999999)])
        s = f"{v:,}".replace(",", r.choice([" ", " ", "."]))
        if r.random() < 0.4:
            s += f",{r.randint(0, 99):02d}"
        cur = r.choice(CURRENCIES)
        return f"{cur} {s}" if cur in ("€", "USD") and r.random() < 0.2 else f"{s} {cur}"

    def ref(self):
        r = self.rng
        p, y = r.choice(self.p["prefix"]), r.randint(2018, 2027)
        n = r.randint(1, 9999)
        return r.choice([f"{p}-{y}-{n:04d}", f"{p}/{y}/{n:03d}", f"{p}-{n:05d}", f"{p}{y}-{n:03d}"])

    def pourcent(self):
        r = self.rng
        v = r.choice([str(r.randint(1, 60)), f"{r.randint(1, 30)},{r.randint(1, 9)}"])
        return f"{v} %" if r.random() < 0.6 else f"{v}%"

    def value(self, slot, sector):
        r = self.rng
        return {
            "ORG": lambda: self.org(sector), "PER": self.per, "LIEU": self.lieu,
            "PAYS": lambda: r.choice(COUNTRIES), "DATE": self.date, "ANNEE": lambda: str(r.randint(2015, 2030)),
            "MONTANT": self.montant, "REF": self.ref, "POURCENT": self.pourcent,
            "DOM": lambda: r.choice(DOM[sector]), "PRODUIT": lambda: r.choice(PRODUIT[sector]),
            "TASK": lambda: r.choice(TASK[sector]),
        }[slot]()

    # ------------------------------------------------------------ remplissage
    def fill(self, template, sector):
        text, spans, pos = "", [], 0
        for m in SLOT_RE.finditer(template):
            text += template[pos:m.start()]
            val, slot = self.value(m.group(1), sector), m.group(1)
            if slot in SLOT_TYPES:
                spans.append((len(text), len(text) + len(val), SLOT_TYPES[slot]))
            text += val
            pos = m.end()
        text += template[pos:]
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text, spans

    def _pool(self, templates, idx_test):
        keep = (lambda i: i not in idx_test) if self.split == "train" else (lambda i: i in idx_test)
        return [t for i, t in enumerate(templates) if keep(i)]

    def document(self, dtype, sector):
        r = self.rng
        body_pool = self._pool(TEMPLATES[dtype], TEST_IDX)
        # entraînement : 4 à 9 phrases parmi 12 gabarits ; test : 3 ou 4 des 4 gabarits réservés
        k = r.randint(4, 9) if self.split == "train" else r.randint(3, 4)
        body = r.sample(body_pool, k)
        body += r.sample(self._pool(GENERIC, TEST_IDX), r.randint(0, 2))
        if r.random() < 0.15:           # bruit : une phrase d'un autre type de document
            other = r.choice([t for t in TEMPLATES if t != dtype])
            body.append(r.choice(self._pool(TEMPLATES[other], TEST_IDX)))
        r.shuffle(body)
        title = r.choice(self._pool(TITLES[dtype], TITLE_TEST_IDX))
        text, spans = "", []
        for tpl in [title] + body:
            line, sp = self.fill(tpl, sector)
            spans += [(s + len(text), e + len(text), t) for s, e, t in sp]
            text += line + "\n"
        return {"type": dtype, "sector": sector, "text": text.rstrip("\n"), "entities": spans}


def build(out_dir, n_train=450, n_dev=75, n_test=150):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    combos = [(t, s) for t in DOC_TYPES for s in SECTORS]
    stats = {}
    for split, n, seed in [("train", n_train, 1), ("dev", n_dev, 2), ("test", n_test, 3)]:
        g = Generator("train" if split == "dev" else split, seed)
        docs = []
        for i in range(n):
            t, s = combos[i % len(combos)]
            d = g.document(t, s)
            d["id"] = f"{split}-{i:04d}"
            docs.append(d)
        g.rng.shuffle(docs)
        with open(out / f"docs_{split}.jsonl", "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        stats[split] = {"documents": len(docs), "entites": sum(len(d["entities"]) for d in docs)}
    return stats


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1] if len(sys.argv) > 1 else "data"))
