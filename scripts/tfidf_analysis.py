import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
HVG_ORIGO    = PROJECT_ROOT / 'processed_data' / 'final' / 'articles_tfidf_lemmatized.json'
INDEX        = PROJECT_ROOT / 'processed_data' / 'final' / 'index_tfidf_lemmatized.json'
OUTPUT_DIR   = PROJECT_ROOT / 'results' / 'tfidf_results'

MAX_FEATURES = 5000
MIN_DF       = 5
MAX_DF       = 0.8
TOP_N        = 50
TOP_PORTAL   = 20
TOP_YEAR     = 20
YEARS        = ['2019', '2021']
PORTALS      = ['hvg', 'origo', 'index']
PORTAL_COLORS = {'hvg': 'steelblue', 'origo': 'coral', 'index': 'mediumseagreen'}


# Load data

def load_data() -> pd.DataFrame:

    frames = []
    for path in [HVG_ORIGO, INDEX]:
        with open(path, 'r', encoding='utf-8') as f:
            frames.append(pd.DataFrame(json.load(f)))

    df = pd.concat(frames, ignore_index=True)
    df = df[df['year'].isin(YEARS)].dropna(subset=['text']).reset_index(drop=True)

    return df


# TF-IDF vectorization

def perform_tfidf(df: pd.DataFrame):

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(df['text'])
    feature_names = vectorizer.get_feature_names_out()

    sparsity = 100 * (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
    return matrix, vectorizer, feature_names


# Analysis helpers

def top_terms_for(matrix, feature_names, n=20):
    avg = np.asarray(matrix.mean(axis=0)).flatten()
    idx = avg.argsort()[-n:][::-1]
    return [(feature_names[i], float(avg[i])) for i in idx]


def analyze_top_terms(matrix, feature_names, df):
    terms = top_terms_for(matrix, feature_names, TOP_N)
    for i, (t, s) in enumerate(terms, 1):

    out = OUTPUT_DIR / 'top_terms.txt'
    lines = ['TOP TERMS BY AVERAGE TF-IDF SCORE\n' + '=' * 70 + '\n']
    for i, (t, s) in enumerate(terms, 1):
        lines.append(f'{i:3d}. {t:<30} {s:.6f}')
    out.write_text('\n'.join(lines), encoding='utf-8')
    return terms


def analyze_by_portal(matrix, feature_names, df):
    results = {}
    for p in PORTALS:
        mask = (df['portal'] == p).values
        if mask.sum() == 0:
            continue
        terms = top_terms_for(matrix[mask], feature_names, TOP_PORTAL)
        results[p] = terms
        for i, (t, s) in enumerate(terms, 1):

    out = OUTPUT_DIR / 'top_terms_by_portal.txt'
    lines = ['TOP TERMS BY PORTAL\n' + '=' * 70]
    for p, terms in results.items():
        lines.append(f'\n{p.upper()}:\n' + '-' * 70)
        for i, (t, s) in enumerate(terms, 1):
            lines.append(f'{i:3d}. {t:<30} {s:.6f}')
    out.write_text('\n'.join(lines), encoding='utf-8')
    return results


def analyze_by_year(matrix, feature_names, df):
    results = {}
    for yr in YEARS:
        mask = (df['year'] == yr).values
        terms = top_terms_for(matrix[mask], feature_names, TOP_YEAR)
        results[yr] = terms
        for i, (t, s) in enumerate(terms[:10], 1):

    out = OUTPUT_DIR / 'top_terms_by_year.txt'
    lines = ['TOP TERMS BY YEAR\n' + '=' * 70]
    for yr, terms in results.items():
        lines.append(f'\nYear {yr}:\n' + '-' * 70)
        for i, (t, s) in enumerate(terms, 1):
            lines.append(f'{i:3d}. {t:<30} {s:.6f}')
    out.write_text('\n'.join(lines), encoding='utf-8')
    return results


def analyze_by_portal_and_year(matrix, feature_names, df):
    results = {}
    for p in PORTALS:
        results[p] = {}
        for yr in YEARS:
            mask = ((df['portal'] == p) & (df['year'] == yr)).values
            if mask.sum() < MIN_DF:
                continue
            terms = top_terms_for(matrix[mask], feature_names, 10)
            results[p][yr] = terms

    out = OUTPUT_DIR / 'top_terms_by_portal_year.txt'
    lines = ['TOP TERMS BY PORTAL × YEAR\n' + '=' * 70]
    for p in PORTALS:
        for yr in YEARS:
            terms = results.get(p, {}).get(yr)
            if not terms:
                continue
            n = ((df['portal'] == p) & (df['year'] == yr)).sum()
            lines.append(f'\n{p.upper()} — {yr}  (n={n:,}):\n' + '-' * 70)
            for i, (t, s) in enumerate(terms, 1):
                lines.append(f'{i:3d}. {t:<30} {s:.6f}')
    out.write_text('\n'.join(lines), encoding='utf-8')
    return results


# Text report (descriptive_statistics_report style)

def write_report(df, top_terms, portal_terms, year_terms, portal_year_terms, matrix, feature_names):
    sep  = '=' * 70
    dash = '-' * 70
    lines = []

    def h1(t): lines.append(f'\n{sep}\n{t}\n{sep}')
    def h2(t): lines.append(f'\n{dash}\n{t}\n{dash}')

    h1('TF-IDF ELEMZÉSI RIPORT — HVG, ORIGO, INDEX (2019 & 2021)')

    # Sample sizes
    h2('1. MINTAMÉRET')
    lines.append(f'\n  Összes cikk: {len(df):,}')
    lines.append(f'\n  {"Portál":<10} {"2019":>8} {"2021":>8} {"Összesen":>10}')
    lines.append(f'  {"-"*38}')
    for p in PORTALS:
        n19  = ((df['portal'] == p) & (df['year'] == '2019')).sum()
        n21  = ((df['portal'] == p) & (df['year'] == '2021')).sum()
        ntot = (df['portal'] == p).sum()
        lines.append(f'  {p:<10} {n19:>8,} {n21:>8,} {ntot:>10,}')

    # Matrix info
    h2('2. TF-IDF MÁTRIX')
    sparsity = 100 * (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
    lines.append(f'\n  Paraméterek: max_features={MAX_FEATURES}  min_df={MIN_DF}  max_df={MAX_DF}')
    lines.append(f'  Mátrix mérete: {matrix.shape[0]:,} dokumentum × {matrix.shape[1]:,} kifejezés')
    lines.append(f'  Nem-nulla elemek: {matrix.nnz:,}')
    lines.append(f'  Ritkasság: {sparsity:.2f}%')

    # Top overall terms
    h2(f'3. TOP {TOP_N} KIFEJEZÉS (átlagos TF-IDF)')
    lines.append(f'\n  {"Rang":<5} {"Kifejezés":<30} {"Átlag TF-IDF":>12}')
    lines.append(f'  {"-"*50}')
    for i, (t, s) in enumerate(top_terms, 1):
        lines.append(f'  {i:<5} {t:<30} {s:>12.6f}')

    # Top terms by portal
    h2(f'4. TOP {TOP_PORTAL} KIFEJEZÉS PORTÁLONKÉNT')
    for p in PORTALS:
        terms = portal_terms.get(p, [])
        n = (df['portal'] == p).sum()
        lines.append(f'\n  {p.upper()} (n={n:,}):')
        lines.append(f'  {"Rang":<5} {"Kifejezés":<30} {"Átlag TF-IDF":>12}')
        lines.append(f'  {"-"*50}')
        for i, (t, s) in enumerate(terms, 1):
            lines.append(f'  {i:<5} {t:<30} {s:>12.6f}')

    # Top terms by year
    h2(f'5. TOP {TOP_YEAR} KIFEJEZÉS ÉVENKÉNT')
    for yr in YEARS:
        terms = year_terms.get(yr, [])
        n = (df['year'] == yr).sum()
        lines.append(f'\n  {yr} (n={n:,}):')
        lines.append(f'  {"Rang":<5} {"Kifejezés":<30} {"Átlag TF-IDF":>12}')
        lines.append(f'  {"-"*50}')
        for i, (t, s) in enumerate(terms, 1):
            lines.append(f'  {i:<5} {t:<30} {s:>12.6f}')

    # Portal × year
    h2('6. TOP 10 KIFEJEZÉS PORTÁL × ÉV')
    for p in PORTALS:
        for yr in YEARS:
            terms = portal_year_terms.get(p, {}).get(yr, [])
            if not terms:
                continue
            n = ((df['portal'] == p) & (df['year'] == yr)).sum()
            lines.append(f'\n  {p.upper()} — {yr} (n={n:,}):')
            for i, (t, s) in enumerate(terms, 1):
                lines.append(f'    {i:2d}. {t:<30} {s:.6f}')

    out = OUTPUT_DIR / 'tfidf_analysis_report.txt'
    out.write_text('\n'.join(lines), encoding='utf-8')


# Visualizations

def create_visualizations(top_terms, portal_terms, year_terms):

    # 1. Top 30 overall
    fig, ax = plt.subplots(figsize=(12, 10))
    terms, scores = zip(*top_terms[:30])
    y_pos  = np.arange(len(terms))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(terms)))
    ax.barh(y_pos, scores, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms, fontsize=10)
    ax.set_xlabel('Átlagos TF-IDF pontszám', fontsize=12)
    ax.set_title('Top 30 kifejezés TF-IDF alapján', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top_30_terms.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Portal comparison — 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    for ax, p in zip(axes, PORTALS):
        terms_list = portal_terms.get(p, [])
        if not terms_list:
            continue
        terms, scores = zip(*terms_list[:20])
        y_pos = np.arange(len(terms))
        ax.barh(y_pos, scores, color=PORTAL_COLORS[p], alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms, fontsize=9)
        ax.set_xlabel('Átlagos TF-IDF pontszám', fontsize=10)
        ax.set_title(f'{p.upper()} – Top 20', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
    plt.suptitle('Top 20 kifejezés portálonként', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'portal_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Year comparison — 2 panels (2019 vs 2021)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    year_colors = {'2019': '#ff7f0e', '2021': '#9467bd'}
    for ax, yr in zip(axes, YEARS):
        terms_list = year_terms.get(yr, [])
        if not terms_list:
            continue
        terms, scores = zip(*terms_list[:20])
        y_pos = np.arange(len(terms))
        ax.barh(y_pos, scores, color=year_colors[yr], alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms, fontsize=9)
        ax.set_xlabel('Átlagos TF-IDF pontszám', fontsize=10)
        ax.set_title(f'{yr} – Top 20', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
    plt.suptitle('Top 20 kifejezés évenként', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'year_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Portal × year heatmap — top 20 shared terms, mean TF-IDF per group


def create_portal_year_heatmap(matrix, feature_names, df):
    avg_all  = np.asarray(matrix.mean(axis=0)).flatten()
    top_idx  = avg_all.argsort()[-30:][::-1]
    top_feat = [feature_names[i] for i in top_idx]

    groups   = [(p, yr) for p in PORTALS for yr in YEARS]
    heat     = np.zeros((len(top_feat), len(groups)))

    for j, (p, yr) in enumerate(groups):
        mask = ((df['portal'] == p) & (df['year'] == yr)).values
        if mask.sum() == 0:
            continue
        avg = np.asarray(matrix[mask].mean(axis=0)).flatten()
        heat[:, j] = avg[top_idx]

    col_labels = [f'{p.upper()}\n{yr}' for p, yr in groups]
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(heat, xticklabels=col_labels, yticklabels=top_feat,
                cmap='YlOrRd', linewidths=0.3, ax=ax,
                cbar_kws={'label': 'Átlagos TF-IDF'})
    ax.set_title('Top 30 kifejezés átlagos TF-IDF értéke – portál × év',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'portal_year_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


# Save matrix

def save_tfidf_matrix(matrix, feature_names, df):

    save_npz(OUTPUT_DIR / 'tfidf_matrix.npz', matrix)

    with open(OUTPUT_DIR / 'feature_names.json', 'w', encoding='utf-8') as f:
        json.dump(list(feature_names), f, ensure_ascii=False, indent=2)

    sparsity = 100 * (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
    metadata = {
        'n_documents':        matrix.shape[0],
        'n_features':         matrix.shape[1],
        'n_nonzero':          int(matrix.nnz),
        'sparsity':           float(sparsity),
        'documents_by_portal': df['portal'].value_counts().to_dict(),
        'documents_by_year':   df['year'].value_counts().to_dict(),
        'documents_by_portal_year': {
            f'{p}_{yr}': int(((df['portal'] == p) & (df['year'] == yr)).sum())
            for p in PORTALS for yr in YEARS
        },
    }
    with open(OUTPUT_DIR / 'tfidf_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# Main

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    matrix, vectorizer, feature_names = perform_tfidf(df)

    top_terms        = analyze_top_terms(matrix, feature_names, df)
    portal_terms     = analyze_by_portal(matrix, feature_names, df)
    year_terms       = analyze_by_year(matrix, feature_names, df)
    portal_year_terms = analyze_by_portal_and_year(matrix, feature_names, df)

    write_report(df, top_terms, portal_terms, year_terms, portal_year_terms, matrix, feature_names)
    create_visualizations(top_terms, portal_terms, year_terms)
    create_portal_year_heatmap(matrix, feature_names, df)
    save_tfidf_matrix(matrix, feature_names, df)

    for f in sorted(OUTPUT_DIR.iterdir()):


if __name__ == '__main__':
    main()
