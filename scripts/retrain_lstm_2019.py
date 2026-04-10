"""
Retrain the best 2019 LSTM model N_RUNS times with patience=3.
Best params from grid search: lstm_units=128, dropout=0.25, lr=0.001
Each run uses a different random seed. The best run (lowest val_loss)
is saved as the final model. The history plot shows mean ± std bands.
"""
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
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

sns.set_style('whitegrid')

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = Path(__file__).parent
PROJECT_ROOT    = SCRIPT_DIR.parent
DATA_PATH       = PROJECT_ROOT / 'processed_data' / 'final' / 'articles_lstm_lemmatized.json'
FASTTEXT_PATH   = PROJECT_ROOT / 'models' / 'cc.hu.300.bin'
OUTPUT_DIR      = PROJECT_ROOT / 'results' / 'lstm_results' / '2019'

# ── Configuration ─────────────────────────────────────────────────────────────
YEAR              = 2019
LSTM_UNITS        = 128
DROPOUT_RATE      = 0.3
LEARNING_RATE     = 1e-3
RECURRENT_DROPOUT = 0.2
DENSE_UNITS       = 64
MAX_SEQUENCE_LEN  = 330
MAX_VOCAB_SIZE    = 30_000
EMBEDDING_DIM     = 300
BATCH_SIZE        = 64
EPOCHS            = 50
PATIENCE          = 3
MIN_DELTA         = 0.001
TEST_SIZE         = 0.2
RANDOM_STATE      = 67
N_RUNS            = 5     # number of independent training runs


def build_model(vocab_size, embedding_matrix):
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix],
                  input_length=MAX_SEQUENCE_LEN, trainable=False),
        SpatialDropout1D(DROPOUT_RATE),
        Bidirectional(LSTM(LSTM_UNITS, dropout=DROPOUT_RATE,
                           recurrent_dropout=RECURRENT_DROPOUT)),
        Dense(DENSE_UNITS, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def pad_histories(histories: list[dict], key: str) -> np.ndarray:
    """Pad run histories to the same length with NaN for mean/std computation."""
    max_len = max(len(h[key]) for h in histories)
    arr = np.full((len(histories), max_len), np.nan)
    for i, h in enumerate(histories):
        vals = h[key]
        arr[i, :len(vals)] = vals
    return arr


def plot_mean_std(ax, arr: np.ndarray, color: str, label: str):
    mean = np.nanmean(arr, axis=0)
    std  = np.nanstd(arr,  axis=0)
    x    = np.arange(1, arr.shape[1] + 1)
    ax.plot(x, mean, color=color, linewidth=2, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('\n' + '=' * 65)
    print(f'  LSTM  {N_RUNS}× FUTTATÁS  —  {YEAR}  |  patience={PATIENCE}')
    print(f'  lstm_units={LSTM_UNITS}, dropout={DROPOUT_RATE}, lr={LEARNING_RATE}')
    print('=' * 65)

    # ── Load & prepare data (fixed split, same for all runs) ──────────────────
    print('\nAdatok betöltése …')
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    df = pd.DataFrame(articles)
    df = df[df['year'] == str(YEAR)].dropna(subset=['text']).reset_index(drop=True)
    print(f'  {len(df):,} cikk ({YEAR})')

    y     = (df['portal'] == 'origo').astype(int)
    texts = df['text'].tolist()
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print('\nTokenizálás …')
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts_train)
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE + 1)

    def encode(t):
        return pad_sequences(tokenizer.texts_to_sequences(t),
                             maxlen=MAX_SEQUENCE_LEN, padding='post', truncating='post')

    X_train = encode(texts_train)
    X_test  = encode(texts_test)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)
    print(f'  Szótár: {vocab_size:,}  |  Tanító: {X_train.shape}  |  Teszt: {X_test.shape}')

    print(f'\nFastText betöltése: {FASTTEXT_PATH}')
    ft = fasttext.load_model(str(FASTTEXT_PATH))
    matrix = np.zeros((vocab_size, EMBEDDING_DIM), dtype=np.float32)
    for word, idx in tokenizer.word_index.items():
        if idx < vocab_size:
            matrix[idx] = ft.get_word_vector(word)
    del ft
    print('  ✓ Beágyazási mátrix kész, FastText felszabadítva')

    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train.values)
    class_weight_dict = {0: float(cw[0]), 1: float(cw[1])}

    # ── N_RUNS independent training runs ─────────────────────────────────────
    all_histories  = []
    all_metrics    = []
    best_val_loss  = np.inf
    best_model     = None
    best_run_idx   = -1

    for run in range(N_RUNS):
        seed = RANDOM_STATE + run
        tf.random.set_seed(seed)
        np.random.seed(seed)
        print(f'\n{"─"*55}')
        print(f'  Futtatás {run + 1}/{N_RUNS}  (seed={seed})')
        print(f'{"─"*55}')

        model = build_model(vocab_size, matrix)
        ckpt_path = OUTPUT_DIR / f'_tmp_run{run}.keras'
        es   = EarlyStopping(monitor='val_loss', patience=PATIENCE,
                             min_delta=MIN_DELTA,
                             restore_best_weights=True, verbose=0)
        ckpt = ModelCheckpoint(filepath=str(ckpt_path),
                               monitor='val_loss', save_best_only=True, verbose=0)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[es, ckpt],
            class_weight=class_weight_dict,
            verbose=1,
        )

        run_best_loss  = min(history.history['val_loss'])
        run_best_epoch = int(np.argmin(history.history['val_loss'])) + 1
        epochs_run     = len(history.history['loss'])

        y_proba     = model.predict(X_test,  batch_size=BATCH_SIZE, verbose=0).flatten()
        y_pred      = (y_proba >= 0.5).astype(int)
        train_proba = model.predict(X_train, batch_size=BATCH_SIZE, verbose=0).flatten()

        metrics = {
            'run':        run + 1,
            'seed':       seed,
            'epochs_run': epochs_run,
            'best_epoch': run_best_epoch,
            'best_val_loss':  float(run_best_loss),
            'test_accuracy':  float(accuracy_score(y_test,  y_pred)),
            'test_auc':       float(roc_auc_score(y_test,   y_proba)),
            'train_accuracy': float(accuracy_score(y_train, (train_proba >= 0.5).astype(int))),
            'train_auc':      float(roc_auc_score(y_train,  train_proba)),
        }
        all_metrics.append(metrics)
        all_histories.append(history.history)

        print(f'  → Legjobb epoch: {run_best_epoch}  |  '
              f'val_loss: {run_best_loss:.4f}  |  '
              f'test_acc: {metrics["test_accuracy"]:.4f}  |  '
              f'AUC: {metrics["test_auc"]:.4f}')

        if run_best_loss < best_val_loss:
            best_val_loss = run_best_loss
            best_model    = model
            best_run_idx  = run

        tf.keras.backend.clear_session()

    del matrix

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print(f'  ÖSSZEFOGLALÓ  ({N_RUNS} futtatás)')
    print(f'{"="*65}')
    print(f'  {"Futtatás":>8}  {"Legjobb ep.":>11}  {"val_loss":>9}  {"test_acc":>9}  {"AUC":>7}')
    print(f'  {"─"*55}')
    for m in all_metrics:
        marker = ' ◀ legjobb' if m['run'] == best_run_idx + 1 else ''
        print(f'  {m["run"]:>8}  {m["best_epoch"]:>11}  '
              f'{m["best_val_loss"]:>9.4f}  {m["test_accuracy"]:>9.4f}  '
              f'{m["test_auc"]:>7.4f}{marker}')

    accs = [m['test_accuracy'] for m in all_metrics]
    aucs = [m['test_auc']      for m in all_metrics]
    print(f'  {"─"*55}')
    print(f'  Átlag ± szórás  —  '
          f'Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}  |  '
          f'AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
    print(f'{"="*65}')

    # ── Mean ± std history plot ───────────────────────────────────────────────
    train_loss_arr = pad_histories(all_histories, 'loss')
    val_loss_arr   = pad_histories(all_histories, 'val_loss')
    train_acc_arr  = pad_histories(all_histories, 'accuracy')
    val_acc_arr    = pad_histories(all_histories, 'val_accuracy')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_mean_std(axes[0], train_loss_arr, 'steelblue', f'Tanítási veszteség (átlag ± szórás)')
    plot_mean_std(axes[0], val_loss_arr,   'coral',     f'Validációs veszteség (átlag ± szórás)')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Veszteség', fontsize=11)
    axes[0].set_title(f'Tanítási és validációs veszteség\n({N_RUNS} futtatás átlaga ± szórása)',
                      fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    plot_mean_std(axes[1], train_acc_arr, 'steelblue', f'Tanítási pontosság (átlag ± szórás)')
    plot_mean_std(axes[1], val_acc_arr,   'coral',     f'Validációs pontosság (átlag ± szórás)')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Pontosság', fontsize=11)
    axes[1].set_title(f'Tanítási és validációs pontosság\n({N_RUNS} futtatás átlaga ± szórása)',
                      fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\n✓ training_history.png mentve')

    # ── Save best model & tokenizer ───────────────────────────────────────────
    best_model_path = OUTPUT_DIR / f'_tmp_run{best_run_idx}.keras'
    final_model = tf.keras.models.load_model(str(best_model_path))
    final_model.save(str(OUTPUT_DIR / f'lstm_model_{YEAR}.keras'))

    # Clean up temp checkpoints
    for run in range(N_RUNS):
        p = OUTPUT_DIR / f'_tmp_run{run}.keras'
        if p.exists():
            import shutil; shutil.rmtree(p) if p.is_dir() else p.unlink()

    with open(OUTPUT_DIR / f'tokenizer_{YEAR}.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    # Final evaluation on best model
    y_proba  = final_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()
    y_pred   = (y_proba >= 0.5).astype(int)
    best_m   = all_metrics[best_run_idx]

    print(classification_report(y_test, y_pred, target_names=['HVG', 'Origo'], digits=4))

    summary = {
        'year': YEAR, 'model': 'BidirectionalLSTM',
        'best_params': {'lstm_units': LSTM_UNITS, 'dropout_rate': DROPOUT_RATE,
                        'learning_rate': LEARNING_RATE},
        'training': {
            'n_runs': N_RUNS, 'epochs_max': EPOCHS, 'patience': PATIENCE,
            'batch_size': BATCH_SIZE,
            'best_run': best_run_idx + 1,
            'best_run_seed': best_m['seed'],
            'best_epoch': best_m['best_epoch'],
        },
        'tuned': {
            'train_accuracy': best_m['train_accuracy'],
            'test_accuracy':  best_m['test_accuracy'],
            'train_auc':      best_m['train_auc'],
            'test_auc':       best_m['test_auc'],
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        },
        'runs_summary': {
            'test_accuracy_mean': float(np.mean(accs)),
            'test_accuracy_std':  float(np.std(accs)),
            'test_auc_mean':      float(np.mean(aucs)),
            'test_auc_std':       float(np.std(aucs)),
        },
        'all_runs': all_metrics,
    }
    with open(OUTPUT_DIR / f'results_summary_{YEAR}.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 65)
    print(f'  KÉSZ  |  Legjobb futtatás: {best_run_idx + 1}  |  '
          f'Legjobb epoch: {best_m["best_epoch"]}')
    print(f'  Acc: {best_m["test_accuracy"]:.4f}  |  AUC: {best_m["test_auc"]:.4f}')
    print(f'  Átlag ± szórás  —  '
          f'Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}  |  '
          f'AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
    print(f'  Eredmények: {OUTPUT_DIR}')
    print('=' * 65)


if __name__ == '__main__':
    main()
