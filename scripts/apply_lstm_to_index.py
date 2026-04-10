import argparse
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer

sns.set_style('whitegrid')

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_BASE   = PROJECT_ROOT / 'results' / 'lstm_results'
INDEX_DATA   = PROJECT_ROOT / 'processed_data' / 'final' / 'index_lstm_lemmatized.json'
OUTPUT_BASE  = PROJECT_ROOT / 'results' / 'index_analysis'

MAX_SEQUENCE_LEN = 330   # must match training
BATCH_SIZE       = 64
LIME_SAMPLES     = 200   # articles explained per year


def load_model_and_tokenizer(model_year: int):
    model_path     = MODEL_BASE / str(model_year) / f'lstm_model_{model_year}.keras'
    tokenizer_path = MODEL_BASE / str(model_year) / f'tokenizer_{model_year}.pkl'

    print(f'Modell betöltése: {model_path}')
    model = tf.keras.models.load_model(str(model_path))

    print(f'Tokenizer betöltése: {tokenizer_path}')
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    print('✓ Modell és tokenizer betöltve')
    return model, tokenizer


def load_index_data() -> pd.DataFrame:
    print(f'\nIndex cikkek betöltése: {INDEX_DATA}')
    with open(INDEX_DATA, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f'✓ Betöltve: {len(df):,} cikk')
    print('\nÉv szerinti megoszlás:')
    print(df['year'].value_counts().sort_index().to_string())
    return df


def predict(model, tokenizer, texts: list) -> tuple[np.ndarray, np.ndarray]:
    print('\nTokenizálás és előrejelzés …')
    seqs   = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LEN,
                           padding='post', truncating='post')
    y_proba = model.predict(padded, batch_size=BATCH_SIZE, verbose=1).flatten()
    y_pred  = (y_proba >= 0.5).astype(int)
    print('✓ Előrejelzés kész')
    return y_pred, y_proba


def print_report(df: pd.DataFrame, y_pred: np.ndarray, y_proba: np.ndarray):
    n          = len(y_pred)
    origo_like = (y_pred == 1).sum()
    hvg_like   = (y_pred == 0).sum()

    print('\n' + '=' * 70)
    print('EREDMÉNYEK – INDEX CIKKEK OSZTÁLYOZÁSA (HVG vs Origo modell)')
    print('=' * 70)

    print(f'\nÖsszes cikk: {n:,}')
    print(f'\n  Origo-szerű (P(Origo) ≥ 0.5): {origo_like:>6,}  ({100*origo_like/n:.1f}%)')
    print(f'  HVG-szerű   (P(Origo) < 0.5): {hvg_like:>6,}  ({100*hvg_like/n:.1f}%)')
    print(f'\n  Átlagos P(Origo):  {y_proba.mean():.4f}')
    print(f'  Mediális P(Origo): {np.median(y_proba):.4f}')

    print('\n' + '-' * 70)
    print('ÉV SZERINTI BONTÁS')
    print('-' * 70)
    print(f"{'Év':<8} {'N':>6}  {'Origo-szerű':>12}  {'Origo %':>8}  {'Átlag P(Origo)':>15}")
    print('-' * 70)

    for yr in sorted(df['year'].unique()):
        mask    = df['year'].values == yr
        n_yr    = mask.sum()
        ol_yr   = (y_pred[mask] == 1).sum()
        mean_yr = y_proba[mask].mean()
        print(f'{yr:<8} {n_yr:>6}  {ol_yr:>12,}  {100*ol_yr/n_yr:>7.1f}%  {mean_yr:>15.4f}')

    print('=' * 70)


def save_results(df: pd.DataFrame, y_pred: np.ndarray, y_proba: np.ndarray,
                 model_year: int, output_dir: Path):
    n          = len(y_pred)
    origo_like = int((y_pred == 1).sum())

    by_year = {}
    for yr in sorted(df['year'].unique()):
        mask  = df['year'].values == yr
        n_yr  = int(mask.sum())
        ol_yr = int((y_pred[mask] == 1).sum())
        by_year[str(yr)] = {
            'n':                  n_yr,
            'origo_like_count':   ol_yr,
            'origo_like_pct':     round(100 * ol_yr / n_yr, 2),
            'hvg_like_count':     n_yr - ol_yr,
            'hvg_like_pct':       round(100 * (n_yr - ol_yr) / n_yr, 2),
            'mean_proba_origo':   round(float(y_proba[mask].mean()), 4),
            'median_proba_origo': round(float(np.median(y_proba[mask])), 4),
        }

    summary = {
        'model_year': model_year,
        'n_articles': n,
        'overall': {
            'origo_like_count':   origo_like,
            'origo_like_pct':     round(100 * origo_like / n, 2),
            'hvg_like_count':     n - origo_like,
            'hvg_like_pct':       round(100 * (n - origo_like) / n, 2),
            'mean_proba_origo':   round(float(y_proba.mean()), 4),
            'median_proba_origo': round(float(np.median(y_proba)), 4),
        },
        'by_year': by_year,
    }

    out = output_dir / f'results_index_{model_year}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n✓ Eredmények mentve: {out}')


def create_visualizations(df: pd.DataFrame, y_proba: np.ndarray,
                           model_year: int, output_dir: Path):
    print('\nVizualizációk készítése …')

    # 1. Overall KDE
    _, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(y_proba, fill=True, alpha=0.6, color='steelblue',
                label='Index cikkek', ax=ax)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1,
               alpha=0.7, label='Döntési határ (0.5)')
    ax.set_xlim(0, 1)
    ax.set_xlabel('P(Origo)', fontsize=12)
    ax.set_ylabel('Sűrűség', fontsize=12)
    ax.set_title(f'Index cikkek becsült valószínűség eloszlása\n'
                 f'(modell: {model_year}-es HVG–Origo LSTM osztályozó)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = output_dir / 'probability_distribution.png'
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Mentve: {p}')

    # 2. KDE by year
    years   = sorted(df['year'].unique())
    palette = sns.color_palette('tab10', len(years))

    _, ax = plt.subplots(figsize=(10, 6))
    for yr, color in zip(years, palette):
        mask = df['year'].values == yr
        sns.kdeplot(y_proba[mask], fill=True, alpha=0.3, color=color,
                    label=str(yr), ax=ax)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1,
               alpha=0.7, label='Döntési határ (0.5)')
    ax.set_xlim(0, 1)
    ax.set_xlabel('P(Origo)', fontsize=12)
    ax.set_ylabel('Sűrűség', fontsize=12)
    ax.set_title(f'Index cikkek becsült valószínűség eloszlása – év szerint\n'
                 f'(modell: {model_year}-es HVG–Origo LSTM osztályozó)',
                 fontsize=13, fontweight='bold')
    ax.legend(title='Év', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = output_dir / 'probability_distribution_by_year.png'
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Mentve: {p}')


def run_lime_by_year(df: pd.DataFrame, model, tokenizer,
                     model_year: int, output_dir: Path):
    """Run LIME separately for each publication year and save bar charts."""
    print('\nLIME elemzés évenkénti bontásban …')

    def predict_fn(texts):
        seqs   = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LEN,
                               padding='post', truncating='post')
        p = model.predict(padded, batch_size=BATCH_SIZE, verbose=0).flatten()
        return np.column_stack([1 - p, p])

    explainer = LimeTextExplainer(class_names=['HVG', 'Origo'])

    for yr in sorted(df['year'].unique()):
        print(f'\n  Év: {yr}')
        yr_texts = df[df['year'] == yr]['text'].tolist()
        n        = min(LIME_SAMPLES, len(yr_texts))
        rng      = np.random.default_rng(42)
        indices  = rng.choice(len(yr_texts), size=n, replace=False)
        sample   = [yr_texts[i] for i in indices]

        word_scores: dict[str, list[float]] = {}
        for i, text in enumerate(sample):
            if (i + 1) % 50 == 0:
                print(f'    {i + 1}/{n} …')
            exp = explainer.explain_instance(
                text, predict_fn, num_features=20, num_samples=500
            )
            for word, score in exp.as_list():
                word_scores.setdefault(word, []).append(score)

        mean_scores = {w: float(np.mean(s)) for w, s in word_scores.items()}

        # Save raw scores
        json_path = output_dir / f'lime_scores_{yr}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(mean_scores, f, ensure_ascii=False, indent=2)
        print(f'  ✓ lime_scores_{yr}.json mentve')

        sorted_scores = sorted(mean_scores.items(), key=lambda x: x[1])
        hvg_top   = sorted_scores[:20]          # most negative → HVG
        origo_top = sorted_scores[-20:][::-1]   # most positive → Origo

        for label, items, color, fname in [
            ('HVG',   hvg_top,   '#1f77b4', f'lime_bar_hvg_{yr}.png'),
            ('Origo', origo_top, '#d62728', f'lime_bar_origo_{yr}.png'),
        ]:
            words  = [w for w, _ in items]
            scores = [s for _, s in items]
            _, ax = plt.subplots(figsize=(8, 6))
            ax.barh(words, scores, color=color, alpha=0.8)
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_xlabel('Átlagos LIME súly', fontsize=11)
            ax.set_title(
                f'Top-20 {label}-jelző szó — Index {yr}\n'
                f'(modell: {model_year}-es LSTM)',
                fontsize=12, fontweight='bold'
            )
            ax.invert_yaxis()
            plt.tight_layout()
            out = output_dir / fname
            plt.savefig(out, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'  ✓ {fname} mentve')


def main():
    parser = argparse.ArgumentParser(
        description='Apply trained HVG–Origo LSTM model to Index articles')
    parser.add_argument('--model-year', type=int, default=2019,
                        help='Year of the trained model to use (2019 or 2021)')
    args = parser.parse_args()
    model_year = args.model_year

    output_dir = OUTPUT_BASE / f'lstm_{model_year}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '=' * 70)
    print(f'INDEX CIKKEK ELEMZÉSE – {model_year}-es LSTM modell')
    print('=' * 70)

    model, tokenizer = load_model_and_tokenizer(model_year)
    df               = load_index_data()
    y_pred, y_proba  = predict(model, tokenizer, df['text'].tolist())

    print_report(df, y_pred, y_proba)
    save_results(df, y_pred, y_proba, model_year, output_dir)
    create_visualizations(df, y_proba, model_year, output_dir)
    run_lime_by_year(df, model, tokenizer, model_year, output_dir)

    print(f'\nMinden eredmény mentve: {output_dir}')
    print('=' * 70)


if __name__ == '__main__':
    main()
