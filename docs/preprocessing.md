# Text Preprocessing Pipeline

This document describes how raw news articles are prepared for each of the three classification models: Logistic Regression (LogReg), XGBoost, and Bidirectional LSTM.

---

## 1. Raw Input Data

All pipelines begin from the same raw source files:

| File | Contents |
|---|---|
| `processed_data/final/hvg_itthon_combined.json` | HVG and Origo articles (training data) |
| `processed_data/final/index_itthon_combined.json` | Index articles (inference data) |

Each JSON record has four fields: `title`, `content`, `portal`, `year`.
Before lemmatization, `title` and `content` are concatenated into a single text field.

---

## 2. Text Cleaning (shared by all models)

Both preprocessing scripts (`prepare_tfidf_lemmatized.py` and `prepare_lstm_lemmatized.py`) apply identical cleaning steps:

| Step | Operation | Regex / rule |
|---|---|---|
| 1 | Remove URLs | `http\S+\|www\.\S+` |
| 2 | Remove email addresses | `\S+@\S+` |
| 3 | Lowercase | `str.lower()` |
| 4 | Keep only Hungarian alphabet | Replace `[^a-záéíóöőúüű\s]` with space |
| 5 | Normalize whitespace | Collapse multiple spaces to one |

---

## 3. Lemmatization (shared by all models)

Both pipelines use the same spaCy model and settings:

- **Model**: `hu_core_news_lg` (Hungarian)
- **Enabled components**: `tok2vec`, `tagger`, `morphologizer`, `lookup_lemmatizer`
- **Disabled components**: `trainable_lemmatizer`, `parser`, `ner`, `senter` — disabled for speed
- **Max text length**: 2,000,000 characters
- **Batch size**: 500 documents (`nlp.pipe()`)

After lemmatization, each token's base form (lemma) is lowercased. Tokens are then filtered by shared rules (see below) and model-specific stopword rules (Section 4).

**Shared post-lemmatization token filters (all models):**
- Drop tokens where `token.is_space` or `token.is_punct` is True
- Drop lemmas that do not match `^[a-záéíóöőúüű]+$` (non-Hungarian characters removed)
- Drop articles where the resulting text is shorter than **50 characters**

---

## 4. Stopword Strategy

This is the **key difference** between the TF-IDF and LSTM pipelines.

### LogReg & XGBoost — `prepare_tfidf_lemmatized.py`

Removes two layers of stopwords:

**Layer 1 — spaCy Hungarian stopwords** (~186 terms)
Standard function words: articles, pronouns, conjunctions, prepositions, common verbs.
Examples: `a`, `az`, `és`, `is`, `de`, `van`, `volt`, `ez`, `hogy`, ...

**Layer 2 — Portal-specific stopwords** (40 custom terms)
Prevent data leakage from portal self-references:

| Category | Examples |
|---|---|
| Portal names | `hvg`, `origo`, `index` |
| Domain forms | `hvghu`, `origohu`, `indexhu` |
| Inflected HVG forms | `hvgnak`, `hvgről`, `hvgtől`, `hvgnél`, `hvgn`, `hvgs` |
| Inflected Origo forms | `origonak`, `origoről`, `origotól`, `origon`, `origoval` |
| Inflected Index forms | `indexnek`, `indexről`, `indextől`, `indexnél`, `indexen` |
| Journalistic terms | `írja`, `közölte`, `portál`, `cikk`, `hír`, `hírek`, `online` |
| Authorship terms | `szerkesztőség`, `újságíró`, `riporter`, `tudósító` |
| Media/meta terms | `fotó`, `kép`, `illusztráció`, `videó`, `frissítés` |

**Minimum token length**: > 2 characters (drops single-char and two-char tokens)

**Rationale**: TF-IDF is a bag-of-words model — word order is discarded. Removing high-frequency function words reduces noise and improves discrimination between portal styles.

---

### LSTM — `prepare_lstm_lemmatized.py`

Removes only **portal-specific stopwords** (30 terms — portal names and inflected forms only):

| Category | Examples |
|---|---|
| Portal names | `hvg`, `origo`, `index` |
| Domain forms | `hvghu`, `origohu`, `indexhu` |
| Inflected HVG forms | `hvgnak`, `hvgről`, `hvgtől`, `hvgnél`, `hvgn`, `hvgs` |
| Inflected Origo forms | `origonak`, `origoről`, `origotól`, `origon`, `origoval` |
| Inflected Index forms | `indexnek`, `indexről`, `indextől`, `indexnél`, `indexen` |

General Hungarian stopwords are **intentionally kept**.

**Minimum token length**: > 1 character

**Rationale**: The LSTM processes text as an ordered sequence. Removing function words (articles, conjunctions, prepositions) would destroy the grammatical structure that the recurrent layers rely on to model context. Portal names are still removed to prevent data leakage.

---

## 5. Feature Extraction

After lemmatization, each model converts the cleaned text to numerical features in its own way.

### LogReg — TF-IDF + Logistic Regression

Script: `scripts/logreg_portal_classifier.py`

The TF-IDF vectorizer and classifier are wrapped in a single scikit-learn Pipeline:

| Parameter | Value |
|---|---|
| `max_features` | 3,000–7,000 (tuned via GridSearchCV) |
| `min_df` | 5 (term must appear in ≥ 5 documents) |
| `max_df` | 0.8 (term must appear in ≤ 80% of documents) |
| `ngram_range` | (1, 2) — unigrams and bigrams |
| `sublinear_tf` | True — replaces TF with 1 + log(TF) |

Hyperparameter search: 5-fold GridSearchCV over `max_features` (5 values) × `C` (5 values) = 25 combinations.
Saved output: `analysis/logreg_results/{year}/logreg_pipeline_{year}.pkl`

---

### XGBoost — TF-IDF + XGBClassifier

Script: `scripts/xgboost_portal_classifier.py`

TF-IDF vectorizer and classifier are saved separately.

| Parameter | Value |
|---|---|
| `max_features` | 5,000 (fixed) |
| `min_df` | 5 |
| `max_df` | 0.8 |
| `ngram_range` | (1, 2) |
| `sublinear_tf` | True |

Hyperparameter search: 5-fold GridSearchCV over 72 XGBoost combinations (`max_depth` × `learning_rate` × `n_estimators` × `colsample_bytree` × `subsample`).
Saved outputs: `analysis/xgboost_results/{year}/xgboost_model_{year}.pkl` + `vectorizer_{year}.pkl`

---

### LSTM — Keras Tokenizer + FastText Embeddings

Script: `scripts/lstm_portal_classifier.py`

**Tokenization**

| Parameter | Value |
|---|---|
| `num_words` | 30,000 (vocabulary size) |
| `oov_token` | `<OOV>` (out-of-vocabulary token) |
| Fit on | Training split only |

**Sequence padding**

| Parameter | Value |
|---|---|
| `maxlen` | 330 (95th percentile of article lengths, from `find_max_sequence_length.py`) |
| `padding` | `'post'` |
| `truncating` | `'post'` |

**Word embeddings**

| Parameter | Value |
|---|---|
| Source | `models/cc.hu.300.bin` — Hungarian FastText (6.7 GB) |
| Dimension | 300 |
| Coverage | All vocabulary words looked up via `ft.get_word_vector()` |
| Trainable | False (frozen during training) |

The FastText model is loaded once, used to build the embedding matrix, then immediately deleted (`del ft`) to free ~6.7 GB of RAM before training begins.

Hyperparameter search: 12 combinations (`lstm_units` × `dropout_rate` × `learning_rate`) with EarlyStopping (patience=4) per run.
Saved outputs: `analysis/lstm_results/{year}/lstm_model_{year}.keras` + `tokenizer_{year}.pkl`

---

## 6. Output Files

| File | Used by | Description |
|---|---|---|
| `processed_data/final/articles_tfidf_lemmatized.json` | LogReg, XGBoost | HVG + Origo, full stopword removal |
| `processed_data/final/articles_lstm_lemmatized.json` | LSTM | HVG + Origo, portal-names-only stopwords |
| `processed_data/final/index_tfidf_lemmatized.json` | LogReg/XGBoost apply scripts | Index articles, same as TF-IDF pipeline |
| `processed_data/final/index_lstm_lemmatized.json` | LSTM apply script | Index articles, same as LSTM pipeline |

Each output record contains: `id`, `portal`, `year`, `text` (space-separated lemmas).

---

## 7. Summary Comparison

| Aspect | LogReg | XGBoost | LSTM |
|---|---|---|---|
| **Preprocessing script** | `prepare_tfidf_lemmatized.py` | `prepare_tfidf_lemmatized.py` | `prepare_lstm_lemmatized.py` |
| **Input data file** | `articles_tfidf_lemmatized.json` | `articles_tfidf_lemmatized.json` | `articles_lstm_lemmatized.json` |
| **spaCy model** | `hu_core_news_lg` | `hu_core_news_lg` | `hu_core_news_lg` |
| **General stopwords removed** | Yes (~186) | Yes (~186) | No |
| **Portal stopwords removed** | Yes (40 terms) | Yes (40 terms) | Yes (30 terms) |
| **Min token length** | > 2 chars | > 2 chars | > 1 char |
| **Feature representation** | TF-IDF sparse matrix | TF-IDF sparse matrix | Dense 300-dim FastText embeddings |
| **Vocabulary size** | 3,000–7,000 (tuned) | 5,000 (fixed) | 30,000 tokens |
| **N-grams** | Unigrams + bigrams | Unigrams + bigrams | Unigrams only |
| **Sequence length** | N/A (bag-of-words) | N/A (bag-of-words) | 330 tokens (padded/truncated) |
| **Interpretability** | Coefficient weights | SHAP values | LIME scores |
