import argparse
import itertools
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import fasttext

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, SpatialDropout1D, Bidirectional, LSTM, Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight

from lime.lime_text import LimeTextExplainer

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = Path(__file__).parent
PROJECT_ROOT    = SCRIPT_DIR.parent
DATA_PATH       = PROJECT_ROOT / 'processed_data' / 'final' / 'articles_lstm_lemmatized.json'
FASTTEXT_PATH   = PROJECT_ROOT / 'models' / 'cc.hu.300.bin'
OUTPUT_DIR_BASE = PROJECT_ROOT / 'results' / 'lstm_results'

# ── Hyperparameter grid (3 × 2 × 2 = 12 combinations) ────────────────────────
PARAM_GRID = {
    'lstm_units':    [32, 64, 128],
    'dropout_rate':  [0.25, 0.5],
    'learning_rate': [1e-3, 5e-5],
}

# ── Fixed model configuration ─────────────────────────────────────────────────
YEAR              = 2019
MAX_SEQUENCE_LEN  = 330        # 95th-percentile from find_max_sequence_length.py
MAX_VOCAB_SIZE    = 30_000
EMBEDDING_DIM     = 300
DENSE_UNITS       = 64
RECURRENT_DROPOUT = 0.2        # kept fixed during tuning

# ── Training configuration ────────────────────────────────────────────────────
BATCH_SIZE              = 64
EPOCHS                  = 20
EARLY_STOPPING_PATIENCE = 4
TEST_SIZE               = 0.2
RANDOM_STATE            = 67
LIME_SAMPLES            = 200   # test articles explained with LIME

# Set at runtime in main()
OUTPUT_DIR = None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Load & filter data
# ─────────────────────────────────────────────────────────────────────────────

def load_and_filter_data(year: int) -> pd.DataFrame:
    print(f'\n  Loading {DATA_PATH} …')
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    df = pd.DataFrame(articles)
    print(f'  Betöltött cikkek összesen: {len(df):,}')

    df = df[df['year'] == str(year)].dropna(subset=['text']).reset_index(drop=True)
    print(f'  Szűrés ({year}): {len(df):,} cikk')

    counts = df['portal'].value_counts()
    print(f'  HVG:   {counts.get("hvg", 0):,}')
    print(f'  Origo: {counts.get("origo", 0):,}')
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Tokenize & pad
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_and_pad(df: pd.DataFrame, test_size: float, random_state: int):
    y = (df['portal'] == 'origo').astype(int)
    texts = df['text'].tolist()

    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=random_state, stratify=y
    )

    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts_train)
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE + 1)
    print(f'  Szótár mérete: {vocab_size:,}')

    def encode(texts_list):
        seqs = tokenizer.texts_to_sequences(texts_list)
        return pad_sequences(seqs, maxlen=MAX_SEQUENCE_LEN,
                             padding='post', truncating='post')

    X_train = encode(texts_train)
    X_test  = encode(texts_test)
    print(f'  Tanító halmaz: {X_train.shape}  |  Teszt halmaz: {X_test.shape}')

    return X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True), tokenizer, vocab_size, texts_test


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Embedding matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_embedding_matrix(tokenizer: Tokenizer, ft_model, vocab_size: int) -> np.ndarray:
    word_index = tokenizer.word_index
    matrix = np.zeros((vocab_size, EMBEDDING_DIM), dtype=np.float32)
    found = 0
    for word, idx in word_index.items():
        if idx >= vocab_size:
            continue
        matrix[idx] = ft_model.get_word_vector(word)
        found += 1
    print(f'  Beágyazási mátrix: {matrix.shape}  |  Lefedettség: {found:,}/{min(len(word_index), vocab_size-1):,}')
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(vocab_size: int, embedding_matrix: np.ndarray,
                lstm_units: int, dropout_rate: float, learning_rate: float) -> Sequential:
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LEN,
            trainable=False,
            name='embedding',
        ),
        SpatialDropout1D(dropout_rate),
        Bidirectional(LSTM(
            lstm_units,
            dropout=dropout_rate,
            recurrent_dropout=RECURRENT_DROPOUT,
        )),
        Dense(DENSE_UNITS, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4b — Grid search
# ─────────────────────────────────────────────────────────────────────────────

def tune_hyperparameters(X_train, y_train, X_test, y_test,
                         vocab_size, embedding_matrix, class_weight_dict, patience):
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    n_combos = len(combos)
    total_fits = n_combos * EPOCHS  # upper bound

    print(f'\n  Rács keresés: {n_combos} kombináció  '
          f'(legfeljebb {n_combos} × {EPOCHS} = {total_fits} epoch)')
    print(f'  {"#":>3}  {"lstm_units":>10}  {"dropout":>7}  {"lr":>8}  {"val_loss":>9}  {"val_acc":>8}  {"epochs":>6}')
    print('  ' + '-' * 60)

    results = []
    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        model = build_model(vocab_size, embedding_matrix, **params)
        es = EarlyStopping(monitor='val_loss', patience=patience,
                           restore_best_weights=True, verbose=0)
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            class_weight=class_weight_dict,
            verbose=0,
        )
        best_epoch = int(np.argmin(hist.history['val_loss']))
        best_val_loss = hist.history['val_loss'][best_epoch]
        best_val_acc  = hist.history['val_accuracy'][best_epoch]
        epochs_run    = len(hist.history['val_loss'])

        print(f'  {i:>3}  {params["lstm_units"]:>10}  '
              f'{params["dropout_rate"]:>7.1f}  '
              f'{params["learning_rate"]:>8.5f}  '
              f'{best_val_loss:>9.4f}  {best_val_acc:>8.4f}  {epochs_run:>6}')

        results.append({**params, 'val_loss': best_val_loss,
                        'val_accuracy': best_val_acc, 'epochs': epochs_run})
        tf.keras.backend.clear_session()

    best = min(results, key=lambda r: r['val_loss'])
    print(f'\n  Legjobb kombináció: {best}')
    return {k: best[k] for k in keys}, results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4c — Final training
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(X_train, y_train, X_test, y_test,
                      vocab_size, embedding_matrix,
                      best_params, class_weight_dict, year, patience):
    print(f'\n  Végső modell betanítása: {best_params}')
    model = build_model(vocab_size, embedding_matrix, **best_params)
    model.summary()

    es = EarlyStopping(monitor='val_loss', patience=patience,
                       restore_best_weights=True, verbose=1)
    ckpt = ModelCheckpoint(
        filepath=str(OUTPUT_DIR / f'lstm_model_{year}.keras'),
        monitor='val_loss', save_best_only=True, verbose=1,
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es, ckpt],
        class_weight=class_weight_dict,
        verbose=1,
    )
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_proba = model.predict(X_train, batch_size=BATCH_SIZE).flatten()
    y_test_proba  = model.predict(X_test,  batch_size=BATCH_SIZE).flatten()

    y_train_pred = (y_train_proba >= 0.5).astype(int)
    y_test_pred  = (y_test_proba  >= 0.5).astype(int)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc  = accuracy_score(y_test,  y_test_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc  = roc_auc_score(y_test,  y_test_proba)

    print(f'\n  Pontosság  — tanító: {train_acc:.4f}  |  teszt: {test_acc:.4f}')
    print(f'  AUC-ROC    — tanító: {train_auc:.4f}  |  teszt: {test_auc:.4f}')
    print()
    print(classification_report(y_test, y_test_pred, target_names=['HVG', 'Origo'], digits=4))

    cm = confusion_matrix(y_test, y_test_pred)
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)

    return {
        'train_accuracy': train_acc, 'test_accuracy': test_acc,
        'train_auc': train_auc,     'test_auc': test_auc,
        'y_test': y_test, 'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba,
        'y_train_proba': y_train_proba, 'y_train': y_train,
        'confusion_matrix': cm, 'fpr': fpr, 'tpr': tpr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def create_visualizations(results: dict, history, year: int):
    # 1. Training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_x = range(1, len(history.history['loss']) + 1)

    axes[0].plot(epochs_x, history.history['loss'],     color='steelblue', linewidth=2, label='Tanítási veszteség')
    axes[0].plot(epochs_x, history.history['val_loss'], color='coral',     linewidth=2, label='Validációs veszteség')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Veszteség', fontsize=11)
    axes[0].set_title('Tanítási és validációs veszteség', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_x, history.history['accuracy'],     color='steelblue', linewidth=2, label='Tanítási pontosság')
    axes[1].plot(epochs_x, history.history['val_accuracy'], color='coral',     linewidth=2, label='Validációs pontosság')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Pontosság', fontsize=11)
    axes[1].set_title('Tanítási és validációs pontosság', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  ✓ training_history.png')

    # 2. Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['HVG', 'Origo'], yticklabels=['HVG', 'Origo'], ax=ax)
    ax.set_ylabel('Tényleges', fontsize=12)
    ax.set_xlabel('Predikált', fontsize=12)
    ax.set_title(f'Konfúziós Mátrix – Bidirectional LSTM ({year})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  ✓ confusion_matrix.png')

    # 3. ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(results['fpr'], results['tpr'], color='steelblue', linewidth=2,
            label=f'ROC Görbe (AUC = {results["test_auc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Véletlen osztályozó')
    ax.set_xlabel('Téves pozitív arány', fontsize=12)
    ax.set_ylabel('Valós pozitív arány', fontsize=12)
    ax.set_title(f'ROC Görbe – Bidirectional LSTM ({year})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  ✓ roc_curve.png')

    # 4. Probability distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    proba  = results['y_test_proba']
    labels = results['y_test'].values
    sns.kdeplot(proba[labels == 0], fill=True, alpha=0.5, color='steelblue', label='HVG',   ax=ax)
    sns.kdeplot(proba[labels == 1], fill=True, alpha=0.5, color='coral',     label='Origo', ax=ax)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Döntési határ (0.5)')
    ax.set_xlim(0, 1)
    ax.set_xlabel('P(Origo)', fontsize=12)
    ax.set_ylabel('Sűrűség', fontsize=12)
    ax.set_title(f'Becsült valószínűség eloszlása – Bidirectional LSTM ({year})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  ✓ probability_distribution.png')


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8 — LIME analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_lime_analysis(model, tokenizer, texts_test, year: int):
    print(f'\n  LIME magyarázatok számítása ({LIME_SAMPLES} mintán) …')

    def predict_fn(text_list):
        seqs = tokenizer.texts_to_sequences(text_list)
        padded = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LEN,
                               padding='post', truncating='post')
        proba = model.predict(padded, verbose=0).flatten()
        return np.column_stack([1 - proba, proba])

    explainer = LimeTextExplainer(class_names=['HVG', 'Origo'])
    sample_texts = texts_test[:LIME_SAMPLES]

    word_scores: dict[str, list[float]] = {}
    for i, text in enumerate(sample_texts):
        if (i + 1) % 50 == 0:
            print(f'    {i + 1}/{LIME_SAMPLES} …')
        exp = explainer.explain_instance(
            text, predict_fn, num_features=20, num_samples=500
        )
        for word, score in exp.as_list():
            word_scores.setdefault(word, []).append(score)

    # Mean score per word
    mean_scores = {w: float(np.mean(s)) for w, s in word_scores.items()}
    sorted_scores = sorted(mean_scores.items(), key=lambda x: x[1])

    # Save raw scores
    with open(OUTPUT_DIR / 'lime_scores.json', 'w', encoding='utf-8') as f:
        json.dump(mean_scores, f, ensure_ascii=False, indent=2)
    print('  ✓ lime_scores.json')

    # HVG bar chart (most negative scores)
    hvg_words  = sorted_scores[:20]
    hvg_labels = [w for w, _ in hvg_words]
    hvg_vals   = [s for _, s in hvg_words]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['steelblue' if v < 0 else 'coral' for v in hvg_vals]
    ax.barh(hvg_labels, hvg_vals, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Átlagos LIME-súly', fontsize=12)
    ax.set_title(f'Top 20 HVG-jelző szó – LIME ({year})', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lime_bar_hvg.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  ✓ lime_bar_hvg.png')

    # Origo bar chart (most positive scores)
    origo_words  = list(reversed(sorted_scores[-20:]))
    origo_labels = [w for w, _ in origo_words]
    origo_vals   = [s for _, s in origo_words]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['coral' if v > 0 else 'steelblue' for v in origo_vals]
    ax.barh(origo_labels, origo_vals, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Átlagos LIME-súly', fontsize=12)
    ax.set_title(f'Top 20 Origo-jelző szó – LIME ({year})', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lime_bar_origo.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  ✓ lime_bar_origo.png')


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — Save model & results
# ─────────────────────────────────────────────────────────────────────────────

def save_model_and_results(model, tokenizer, results, history,
                           best_params, grid_results, year: int, patience: int):
    model.save(OUTPUT_DIR / f'lstm_model_{year}.keras')
    print(f'  ✓ lstm_model_{year}.keras')

    with open(OUTPUT_DIR / f'tokenizer_{year}.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f'  ✓ tokenizer_{year}.pkl')

    epochs_run = len(history.history['loss'])
    summary = {
        'year':  year,
        'model': 'BidirectionalLSTM',
        'best_params': best_params,
        'architecture': {
            'max_sequence_len':  MAX_SEQUENCE_LEN,
            'max_vocab_size':    MAX_VOCAB_SIZE,
            'embedding_dim':     EMBEDDING_DIM,
            'dense_units':       DENSE_UNITS,
            'recurrent_dropout': RECURRENT_DROPOUT,
            **best_params,
        },
        'training': {
            'epochs_run':               epochs_run,
            'epochs_max':               EPOCHS,
            'batch_size':               BATCH_SIZE,
            'early_stopping_patience':  patience,
        },
        'baseline': {
            'test_accuracy': float(history.history['val_accuracy'][0]),
            'test_auc':      None,
        },
        'tuned': {
            'train_accuracy':   float(results['train_accuracy']),
            'test_accuracy':    float(results['test_accuracy']),
            'train_auc':        float(results['train_auc']),
            'test_auc':         float(results['test_auc']),
            'cv_mean':          None,
            'cv_std':           None,
            'confusion_matrix': results['confusion_matrix'].tolist(),
        },
        'grid_search': grid_results,
    }
    out = OUTPUT_DIR / f'results_summary_{year}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'  ✓ results_summary_{year}.json')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global OUTPUT_DIR

    parser = argparse.ArgumentParser(description='Bidirectional LSTM Portal Classifier')
    parser.add_argument('--year', type=int, default=YEAR,
                        help=f'Tanítási év (alapértelmezett: {YEAR})')
    parser.add_argument('--patience', type=int, default=EARLY_STOPPING_PATIENCE,
                        help=f'Early stopping türelem (alapértelmezett: {EARLY_STOPPING_PATIENCE})')
    args = parser.parse_args()
    year    = args.year
    patience = args.patience

    OUTPUT_DIR = OUTPUT_DIR_BASE / str(year)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    gpus = tf.config.list_physical_devices('GPU')
    print(f'  GPU: {gpus if gpus else "nincs (CPU módban fut)"}')

    print('\n' + '=' * 70)
    print(f'  BIDIRECTIONAL LSTM PORTÁLKLASSZIFIKÁTOR  (HVG vs Origo, {year})')
    print('=' * 70)

    # PHASE 1
    print('\nFÁZIS 1 — ADATOK BETÖLTÉSE ÉS SZŰRÉSE')
    df = load_and_filter_data(year)

    # PHASE 2
    print('\nFÁZIS 2 — TOKENIZÁLÁS ÉS PÁRNÁZÁS')
    X_train, X_test, y_train, y_test, tokenizer, vocab_size, texts_test = \
        tokenize_and_pad(df, TEST_SIZE, RANDOM_STATE)

    class_weights = compute_class_weight(
        'balanced', classes=np.array([0, 1]), y=y_train.values
    )
    class_weight_dict = {0: float(class_weights[0]), 1: float(class_weights[1])}
    print(f'  Osztálysúlyok: {class_weight_dict}')

    # PHASE 3
    print('\nFÁZIS 3 — BEÁGYAZÁSI MÁTRIX ÉPÍTÉSE')
    print(f'  FastText modell betöltése: {FASTTEXT_PATH}')
    print('  (Ez ~30–60 másodpercet vehet igénybe a 6.7 GB binárishoz…)')
    ft_model = fasttext.load_model(str(FASTTEXT_PATH))
    print('  ✓ FastText betöltve')
    embedding_matrix = build_embedding_matrix(tokenizer, ft_model, vocab_size)
    del ft_model
    print('  ✓ FastText modell felszabadítva')

    # PHASE 4 — Baseline (default params)
    print('\nFÁZIS 4a — ALAPMODELL (alapértelmezett paraméterekkel)')
    default_params = {'lstm_units': 128, 'dropout_rate': 0.3, 'learning_rate': 1e-3}
    baseline_model = build_model(vocab_size, embedding_matrix, **default_params)
    es_base = EarlyStopping(monitor='val_loss', patience=patience,
                            restore_best_weights=True, verbose=0)
    baseline_hist = baseline_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[es_base], class_weight=class_weight_dict, verbose=0,
    )
    baseline_val_acc  = max(baseline_hist.history['val_accuracy'])
    baseline_val_loss = min(baseline_hist.history['val_loss'])
    print(f'  Alap val_accuracy: {baseline_val_acc:.4f}  |  val_loss: {baseline_val_loss:.4f}')
    tf.keras.backend.clear_session()

    # PHASE 4b — Grid search
    print('\nFÁZIS 4b — RÁCSOS HIPERPARAMÉTER-KERESÉS')
    best_params, grid_results = tune_hyperparameters(
        X_train, y_train, X_test, y_test,
        vocab_size, embedding_matrix, class_weight_dict, patience,
    )

    # PHASE 4c — Final training
    print('\nFÁZIS 4c — VÉGSŐ MODELL BETANÍTÁSA')
    model, history = train_final_model(
        X_train, y_train, X_test, y_test,
        vocab_size, embedding_matrix,
        best_params, class_weight_dict, year, patience,
    )
    del embedding_matrix

    # PHASE 5 — Evaluate
    print('\nFÁZIS 5 — KIÉRTÉKELÉS')
    results = evaluate_model(model, X_train, X_test, y_train, y_test)

    # PHASE 6 — Visualize
    print('\nFÁZIS 6 — VIZUALIZÁCIÓK LÉTREHOZÁSA')
    create_visualizations(results, history, year)

    # PHASE 7 — LIME
    print('\nFÁZIS 7 — LIME ELEMZÉS')
    run_lime_analysis(model, tokenizer, texts_test, year)

    # PHASE 8 — Save
    print('\nFÁZIS 8 — MODELL ÉS EREDMÉNYEK MENTÉSE')
    # Patch baseline into results for save
    results['baseline_val_acc'] = baseline_val_acc
    save_model_and_results(model, tokenizer, results, history,
                           best_params, grid_results, year, patience)

    # Summary
    print('\n' + '=' * 70)
    print('  ELEMZÉS KÉSZ!')
    print('=' * 70)
    print(f'\n  Alapmodell  — val_accuracy: {baseline_val_acc:.4f}')
    print(f'  Legjobb     — Accuracy: {results["test_accuracy"]:.4f}'
          f'  |  AUC: {results["test_auc"]:.4f}')
    print(f'  Legjobb paraméterek: {best_params}')
    print(f'  Lefutott epochok: {len(history.history["loss"])} / {EPOCHS}')
    print(f'\n  Eredmények mentve: {OUTPUT_DIR}')
    print('=' * 70)


if __name__ == '__main__':
    main()
