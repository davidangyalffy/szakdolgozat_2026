# AZ INDEX.HU NYELVEZETÉNEK VIZSGÁLATA A 2020-AS TULAJDONOSVÁLTÁST KÖVETŐEN GÉPI TANULÁSI MÓDSZEREKKEL

Szakdolgozat kódbázisa — ELTE Tátk, 2026

---

## Absztrakt

A magyar online médiatér polarizációja és a szerkesztéspolitikai változások kvantitatív
kimutatása a digitális újságírás kutatásának aktuális kérdése. Jelen tanulmány azt elemzi, hogy
az Index.hu 2020-as tulajdonos- és szerkesztőségváltását követően kimutatható-e változás a
portál nyelvezetében. A vizsgálat proxy-alapú megközelítéssel történik. Ennek során gépi
tanulási modellek kerülnek betanításra két eltérő szerkesztéspolitikát képviselő hírportál, a
HVG és az Origo cikkein, majd ezek a modellek illesztésre kerülnek az Index cikkein. Az
elemzésben három modellarchitektúra, a logisztikus regresszió, az XGBoost és a BiLSTM
magyar fastText-beágyazásokkal, teljesítménye kerül összehasonlításra, SHAP- és
LIME-alapú interpretálhatósági vizsgálattal kiegészítve. A statisztikai validáció
Mann-Whitney U-próbával, és Cohen-féle d hatásmérettel történt. Az eredmények arra
utalnak, hogy az Index 2021-es cikkeinek átlagos P(Origo)-értéke konzisztensen magasabb a
2019-es cikkekénél mindhárom modell és mindkét tanítási év esetében, ami az Origo
stilisztikai pólusa felé történő szisztematikus elmozdulásra utal.

---

## Repó struktúra

```
.
├── scripts/                    # Python szkriptek
│   ├── prepare_tfidf_lemmatized.py     # Előfeldolgozás (LogReg/XGBoost)
│   ├── prepare_lstm_lemmatized.py      # Előfeldolgozás (LSTM)
│   ├── logreg_portal_classifier.py     # Logisztikus regresszió tréning
│   ├── xgboost_portal_classifier.py    # XGBoost tréning
│   ├── lstm_portal_classifier.py       # Kétirányú LSTM tréning
│   ├── retrain_lstm_2019.py            # LSTM újratanítás (2019, több futtatás)
│   ├── apply_logreg_to_index.py        # Előrejelzés Index cikkeken (LogReg)
│   ├── apply_xgboost_to_index.py       # Előrejelzés Index cikkeken (XGBoost)
│   ├── apply_lstm_to_index.py          # Előrejelzés Index cikkeken (LSTM)
│   ├── test_index_logreg_significance.py   # Szignifikanciateszt (LogReg)
│   ├── test_index_xgboost_significance.py  # Szignifikanciateszt (XGBoost)
│   ├── test_index_lstm_significance.py     # Szignifikanciateszt (LSTM)
│   ├── lstm_article_examples_lime.py   # LIME magyarázatok
│   ├── descriptive_statistics.py       # Leíró statisztika (3 portál)
│   ├── analyze_articles.py             # Szógyakoriság-elemzés
│   ├── tfidf_analysis.py               # TF-IDF mátrix elemzés
│   ├── find_max_sequence_length.py     # LSTM szekvencia hossz meghatározása
│   ├── regenerate_logreg_plots.py      # LogReg vizualizációk újragenerálása
│   └── cikkek01.py                     # Index.hu scraper
│
├── results/                    # Eredmények (vizualizációk, JSON összesítők)
│   ├── logreg_results/         # Logisztikus regresszió eredmények
│   ├── xgboost_results/        # XGBoost eredmények
│   ├── lstm_results/           # LSTM eredmények
│   ├── index_analysis/         # Index.hu előrejelzési eredmények
│   ├── tfidf_results/          # TF-IDF elemzési eredmények
│   ├── statistics/             # Leíró statisztikai riportok
│   └── visualizations/         # Általános vizualizációk
│
├── docs/
│   └── preprocessing.md        # Pipeline részletes dokumentációja
│
├── requirements.txt            # Python függőségek (rögzített verziók)
├── LICENSE                     # MIT licenc (csak a kódra vonatkozik)
└── .gitignore
```

> **Megjegyzés:** A `raw_data/`, `processed_data/`, `archive/` és `models/` mappák szerzői jogi és méretbeli okok miatt nem részei a repónak (lásd [Adatforrások](#adatforrások)).

---

## Telepítés

**Python verzió:** 3.11+

```bash
# 1. Virtuális környezet létrehozása
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 2. Függőségek telepítése
pip install -r requirements.txt

# 3. spaCy magyar modell letöltése
python -m spacy download hu_core_news_lg
```

### FastText modell letöltése

A modellek futtatásához szükséges a Facebook AI Research magyar FastText embedding (6,7 GB):

```bash
mkdir -p models
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hu.300.bin.gz
gunzip cc.hu.300.bin.gz
mv cc.hu.300.bin models/
```

---

## Pipeline futtatása

A teljes pipeline az alábbi sorrendben futtatható:

### 1. Előfeldolgozás

```bash
# LogReg és XGBoost adatkészlet (lemmatizálás, stopszó-szűrés)
python scripts/prepare_tfidf_lemmatized.py

# LSTM adatkészlet (lemmatizálás, portálnév stopszó-szűrés)
python scripts/prepare_lstm_lemmatized.py
```

### 2. Modell tréning

```bash
# Logisztikus regresszió (GridSearchCV, 25 kombináció)
python scripts/logreg_portal_classifier.py --year 2019
python scripts/logreg_portal_classifier.py --year 2021

# XGBoost (GridSearchCV, 72 kombináció)
python scripts/xgboost_portal_classifier.py --year 2019
python scripts/xgboost_portal_classifier.py --year 2021

# Kétirányú LSTM (hiperparaméter-rács, 12 kombináció, 5 futtatás)
python scripts/lstm_portal_classifier.py --year 2019
python scripts/lstm_portal_classifier.py --year 2021
```

### 3. Előrejelzés Index cikkeken

```bash
python scripts/apply_logreg_to_index.py
python scripts/apply_xgboost_to_index.py
python scripts/apply_lstm_to_index.py
```

### 4. Statisztikai tesztek (2019 vs. 2021 összehasonlítás)

```bash
python scripts/test_index_logreg_significance.py
python scripts/test_index_xgboost_significance.py
python scripts/test_index_lstm_significance.py
```

### 5. Elemzések és vizualizációk

```bash
python scripts/descriptive_statistics.py
python scripts/tfidf_analysis.py
python scripts/lstm_article_examples_lime.py
```

---

## Adatforrások

### Felhasznált portálok

| Portál | Rovat | Gyűjtési időszak | Cikkszám |
|---|---|---|---|
| HVG | itthon | 2019, 2021 | ~17 600 |
| Origo | belföld | 2019, 2021 | ~21 700 |
| Index | 24óra | 2019, 2021 | ~12 000 |

### Szerzői jogi nyilatkozat

**A HVG, Origo és Index cikkek szerzői jogvédett tartalmak.** Az eredeti cikkek és a belőlük készített feldolgozott adathalmazok (JSON fájlok) nem részei ennek a repónak, és nem kerültek közzétételre.

Ez a repó kizárólag a kutatáshoz használt Python kódot, a módszertant dokumentáló leírásokat és az aggregált/vizualizált eredményeket tartalmazza. A kutatás az adatokat kizárólag nem kereskedelmi, tudományos célra használta fel.

Az adatokhoz való hozzáférésre vonatkozó kéréseket az egyes portálok szerkesztőségéhez kell intézni.

### FastText embedding

A magyar FastText modell (`cc.hu.300.bin`) a Facebook AI Research által publikált, nyilvánosan elérhető modell:

> Grave, E., Bojanowski, P., Gupta, P., Joulin, A., & Mikolov, T. (2018). Learning Word Vectors for 157 Languages. *Proceedings of LREC 2018*.

---

## Licenc

A **kód** MIT licenc alatt áll (lásd [LICENSE](LICENSE)). Ez a licenc kizárólag a `scripts/` mappában található Python szkriptekre vonatkozik.

Az adatokra, a cikkekre és azok feldolgozott változataira az eredeti portálok szerzői jogi feltételei érvényesek.

---

## Hivatkozás

```bibtex
@mastersthesis{angyalffy2026portal,
  author  = {Angyalffy, Dávid},
  title   = {AZ INDEX.HU NYELVEZETÉNEK VIZSGÁLATA A 2020-AS TULAJDONOSVÁLTÁST KÖVETŐEN GÉPI TANULÁSI MÓDSZEREKKEL},
  school  = {Eötvös Lóránt Tudomány Egyetem},
  year    = {2026},
  url     = {https://github.com/davidangyalffy/szakdolgozat_2026}
}
```
