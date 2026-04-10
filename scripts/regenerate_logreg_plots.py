"""
Regenerate LogReg visualizations with Hungarian labels and consistent style.
Loads existing pipelines and reproduces the exact test split to get probabilities.
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH    = PROJECT_ROOT / 'processed_data' / 'final' / 'articles_tfidf_lemmatized.json'
RESULTS_BASE = PROJECT_ROOT / 'results' / 'logreg_results'

MODEL_YEARS  = [2019, 2021]
TEST_SIZE    = 0.2
RANDOM_STATE = 67


def load_year_data(year: int):
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df[df['year'] == str(year)].reset_index(drop=True)
    texts = df['text'].tolist()
    y     = (df['portal'] == 'origo').astype(int).values
    _, t_test, _, y_test = train_test_split(
        texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return t_test, y_test


def regenerate(year: int):
    out_dir  = RESULTS_BASE / str(year)
    pkl_path = out_dir / f'logreg_pipeline_{year}.pkl'
    json_path = out_dir / f'results_summary_{year}.json'

    print(f'\n{"="*60}')
    print(f'  {year}-es LogReg vizualizációk újragenerálása')
    print(f'{"="*60}')

    with open(pkl_path, 'rb') as f:
        pipeline = pickle.load(f)
    with open(json_path, 'r') as f:
        summary = json.load(f)

    t_test, y_test = load_year_data(year)
    y_proba = pipeline.predict_proba(t_test)[:, 1]

    test_auc = summary['tuned']['test_auc']
    cm       = np.array(summary['tuned']['confusion_matrix'])

    # Feature importances from pipeline
    feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
    coefs         = pipeline.named_steps['clf'].coef_[0]
    feat_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefs})
    top_hvg   = feat_df.nsmallest(20, 'coefficient')
    top_origo = feat_df.nlargest(20, 'coefficient')

    # ── 1. Confusion matrix ──────────────────────────────────────────────────
    _, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['HVG', 'Origo'],
                yticklabels=['HVG', 'Origo'], ax=ax)
    ax.set_ylabel('Tényleges', fontsize=12)
    ax.set_xlabel('Predikált', fontsize=12)
    ax.set_title(f'Konfúziós Mátrix – LogReg Portálklasszifikáció ({year})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  ✓ confusion_matrix.png')

    # ── 2. ROC curve ─────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    _, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Görbe (AUC = {test_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Véletlen osztályozó')
    ax.set_xlabel('Téves pozitív arány', fontsize=12)
    ax.set_ylabel('Valós pozitív arány', fontsize=12)
    ax.set_title(f'ROC Görbe – LogReg Portálklasszifikáció ({year})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  ✓ roc_curve.png')

    # ── 3. Feature importance ─────────────────────────────────────────────────
    _, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, df_feat, color, title in [
        (axes[0], top_hvg,   'steelblue', f'Top 20 HVG-jelző szó ({year})'),
        (axes[1], top_origo, 'coral',     f'Top 20 Origo-jelző szó ({year})'),
    ]:
        words  = df_feat['feature'].values
        values = np.abs(df_feat['coefficient'].values)
        y_pos  = np.arange(len(words))
        ax.barh(y_pos, values, color=color, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=9)
        ax.set_xlabel('Együttható nagysága', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  ✓ feature_importance.png')

    # ── 4. Probability distribution ───────────────────────────────────────────
    _, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(y_proba[y_test == 0], fill=True, alpha=0.5, color='steelblue',
                label='HVG', ax=ax)
    sns.kdeplot(y_proba[y_test == 1], fill=True, alpha=0.5, color='coral',
                label='Origo', ax=ax)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1,
               alpha=0.7, label='Döntési határ (0.5)')
    ax.set_xlim(0, 1)
    ax.set_xlabel('P(Origo)', fontsize=12)
    ax.set_ylabel('Sűrűség', fontsize=12)
    ax.set_title(f'Becsült valószínűség eloszlása – LogReg ({year})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  ✓ probability_distribution.png')


def main():
    print('\n' + '=' * 60)
    print('LOGREG VIZUALIZÁCIÓK ÚJRAGENERÁLÁSA')
    print('=' * 60)

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        _ = json.load(f)   # validate data path exists

    for year in MODEL_YEARS:
        regenerate(year)

    print('\n' + '=' * 60)
    print('  KÉSZ')
    print('=' * 60)


if __name__ == '__main__':
    main()
