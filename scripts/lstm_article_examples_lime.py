import argparse
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer

sns.set_style('whitegrid')

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH    = PROJECT_ROOT / 'processed_data' / 'final' / 'articles_lstm_lemmatized.json'
MODEL_BASE   = PROJECT_ROOT / 'results' / 'lstm_results'

MAX_WORDS            = 150    # articles longer than this are excluded
CONFIDENCE_THRESHOLD = 0.85   # minimum max(P, 1-P) for "high confidence"
UNCERTAIN_LOW        = 0.40
UNCERTAIN_HIGH       = 0.60
MAX_SEQUENCE_LEN     = 330
BATCH_SIZE           = 64
LIME_NUM_FEATURES    = 20
LIME_NUM_SAMPLES     = 500
TEST_SIZE            = 0.2
RANDOM_STATE         = 67


def load_data(year: int):
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df[df['year'] == str(year)].reset_index(drop=True)

    texts = df['text'].tolist()
    y     = (df['portal'] == 'origo').astype(int).values

    _, t_test, _, y_test, idx_train, idx_test = train_test_split(
        texts, y, df.index.tolist(),
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    df_test = df.loc[idx_test].reset_index(drop=True)
    return t_test, np.array(y_test), df_test


def load_model_and_tokenizer(year: int):
    model_path     = MODEL_BASE / str(year) / f'lstm_model_{year}.keras'
    tokenizer_path = MODEL_BASE / str(year) / f'tokenizer_{year}.pkl'
    model = tf.keras.models.load_model(str(model_path))
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def predict_proba(model, tokenizer, texts):
    seqs   = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LEN,
                           padding='post', truncating='post')
    p = model.predict(padded, batch_size=BATCH_SIZE, verbose=0).flatten()
    return p


def make_predict_fn(model, tokenizer):
    def predict_fn(texts):
        p = predict_proba(model, tokenizer, texts)
        return np.column_stack([1 - p, p])
    return predict_fn


def select_articles(texts, y_true, y_proba, df_test):
    word_counts = np.array([len(t.split()) for t in texts])
    short_mask  = word_counts <= MAX_WORDS
    y_pred      = (y_proba >= 0.5).astype(int)
    confidence  = np.maximum(y_proba, 1 - y_proba)

    # 1. High-confidence, correctly classified, short
    cand = np.where(
        short_mask & (y_pred == y_true) & (confidence >= CONFIDENCE_THRESHOLD)
    )[0]
    if len(cand) == 0:
        cand = np.where(short_mask & (y_pred == y_true))[0]
    idx_confident = cand[np.argmax(confidence[cand])]

    # 2. Uncertain (closest to 0.5), short
    cand = np.where(
        short_mask & (y_proba >= UNCERTAIN_LOW) & (y_proba <= UNCERTAIN_HIGH)
    )[0]
    if len(cand) == 0:
        cand = np.where(short_mask)[0]
    idx_uncertain = cand[np.argmin(np.abs(y_proba[cand] - 0.5))]

    # 3. Falsely classified, short — prefer high-confidence mistakes
    cand = np.where(short_mask & (y_pred != y_true))[0]
    if len(cand) == 0:
        cand = np.where(y_pred != y_true)[0]
    idx_false = cand[np.argmax(confidence[cand])]

    return idx_confident, idx_uncertain, idx_false


def lime_bar(text, predict_fn, label, subtitle, color, out_path):
    explainer = LimeTextExplainer(class_names=['HVG', 'Origo'])
    exp = explainer.explain_instance(
        text, predict_fn,
        num_features=LIME_NUM_FEATURES,
        num_samples=LIME_NUM_SAMPLES,
    )
    items  = exp.as_list()
    words  = [w for w, _ in items]
    scores = [s for _, s in items]

    colors = ['coral' if s > 0 else 'steelblue' for s in scores]

    _, ax = plt.subplots(figsize=(9, 6))
    ax.barh(words, scores, color=colors, alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('LIME súly  (pozitív → Origo, negatív → HVG)', fontsize=10)
    ax.set_title(f'{label}\n{subtitle}', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_article_info(tag, text, true_label, proba, portal, word_count):
    true_str = 'Origo' if true_label == 1 else 'HVG'
    pred_str = 'Origo' if proba >= 0.5 else 'HVG'


def main():
    parser = argparse.ArgumentParser(
        description='LSTM article example LIME explanations')
    parser.add_argument('--year', type=int, default=2021,
                        help='Model and data year (2019 or 2021)')
    args = parser.parse_args()
    year = args.year

    out_dir = MODEL_BASE / str(year) / 'article_examples'
    out_dir.mkdir(parents=True, exist_ok=True)


    texts, y_true, df_test = load_data(year)
    model, tokenizer       = load_model_and_tokenizer(year)
    predict_fn             = make_predict_fn(model, tokenizer)

    y_proba = predict_proba(model, tokenizer, texts)

    idx_conf, idx_unc, idx_false = select_articles(texts, y_true, y_proba, df_test)

    examples = [
        (idx_conf,  'Magabiztos helyes osztályozás', 'confident'),
        (idx_unc,   'Bizonytalan osztályozás (0.4–0.6)',   'uncertain'),
        (idx_false, 'Téves osztályozás',              'misclassified'),
    ]


    info_records = []
    for idx, label, slug in examples:
        text      = texts[idx]
        proba     = float(y_proba[idx])
        true_lbl  = int(y_true[idx])
        portal    = df_test.iloc[idx]['portal']
        wc        = len(text.split())
        print_article_info(label, text, true_lbl, proba, portal, wc)
        info_records.append({
            'type':        slug,
            'portal':      portal,
            'true_label':  'origo' if true_lbl == 1 else 'hvg',
            'p_origo':     round(proba, 4),
            'predicted':   'origo' if proba >= 0.5 else 'hvg',
            'correct':     bool((proba >= 0.5) == bool(true_lbl)),
            'word_count':  wc,
            'text_preview': text[:200],
        })

    # Save article info JSON
    info_path = out_dir / f'article_examples_{year}.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info_records, f, ensure_ascii=False, indent=2)

    # LIME bar charts
    for idx, label, slug in examples:
        text     = texts[idx]
        proba    = float(y_proba[idx])
        true_lbl = int(y_true[idx])
        portal   = df_test.iloc[idx]['portal']
        true_str = 'Origo' if true_lbl == 1 else 'HVG'
        pred_str = 'Origo' if proba >= 0.5 else 'HVG'
        subtitle = (
            f'Valódi: {true_str}  |  Predikált: {pred_str}  |  '
            f'P(Origo) = {proba:.3f}  |  Portal: {portal}'
        )
        color = 'coral' if true_lbl == 1 else 'steelblue'
        lime_bar(
            text, predict_fn,
            label=label,
            subtitle=subtitle,
            color=color,
            out_path=out_dir / f'lime_{slug}_{year}.png',
        )


if __name__ == '__main__':
    main()
