import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PIPELINE_BASE = PROJECT_ROOT / 'results' / 'logreg_results'
INDEX_DATA    = PROJECT_ROOT / 'processed_data' / 'final' / 'index_tfidf_lemmatized.json'
OUTPUT_DIR    = PROJECT_ROOT / 'results' / 'index_analysis' / 'significance_tests'

MODEL_YEARS   = [2019, 2021]
ALPHA         = 0.05


# Helpers

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0


def effect_label(d: float) -> str:
    d = abs(d)
    if d < 0.2:   return 'elhanyagolható'
    if d < 0.5:   return 'kis'
    if d < 0.8:   return 'közepes'
    return 'nagy'


def run_tests(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> dict:
    mw  = stats.mannwhitneyu(a, b, alternative='two-sided')
    ks  = stats.ks_2samp(a, b)

    # Chi-square on proportions (classified as Origo)
    n_a, n_b   = len(a), len(b)
    k_a, k_b   = (a >= 0.5).sum(), (b >= 0.5).sum()
    contingency = np.array([[k_a, n_a - k_a], [k_b, n_b - k_b]])
    chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)

    d = cohens_d(a, b)

    return {
        'group_a': label_a,
        'group_b': label_b,
        'n_a': int(n_a),
        'n_b': int(n_b),
        'mean_a':   round(float(a.mean()),   4),
        'mean_b':   round(float(b.mean()),   4),
        'median_a': round(float(np.median(a)), 4),
        'median_b': round(float(np.median(b)), 4),
        'origo_pct_a': round(100 * k_a / n_a, 2),
        'origo_pct_b': round(100 * k_b / n_b, 2),
        'mann_whitney': {
            'statistic': float(mw.statistic),
            'p_value':   float(mw.pvalue),
            'significant': bool(mw.pvalue < ALPHA),
        },
        'kolmogorov_smirnov': {
            'statistic': float(ks.statistic),
            'p_value':   float(ks.pvalue),
            'significant': bool(ks.pvalue < ALPHA),
        },
        'chi_square': {
            'statistic': float(chi2),
            'p_value':   float(chi2_p),
            'significant': bool(chi2_p < ALPHA),
        },
        'cohens_d': {
            'value':  round(float(d), 4),
            'label':  effect_label(d),
        },
    }


# Report formatting

def format_test(name: str, stat: float, p: float, sig: bool) -> str:
    sig_str = '✓ SZIGNIFIKÁNS' if sig else '✗ nem szignifikáns'
    return f'  {name:<28} stat={stat:>12.4f}  p={p:.2e}  {sig_str}'


def print_and_collect(lines: list, text: str):
    lines.append(text)


def write_report(all_results: dict, path: Path):
    sep  = '=' * 70
    dash = '-' * 70
    lines = []

    def w(t=''):
        lines.append(t)

    w(sep)
    w('STATISZTIKAI SZIGNIFIKANCIA-VIZSGÁLAT')
    w('Index cikkek: 2019 vs 2021  |  HVG–Origo LogReg modellek')
    w(f'Szignifikancia-szint: α = {ALPHA}')
    w(sep)

    for model_year, res in all_results.items():
        w(f'\n{dash}')
        w(f'MODELL: {model_year}-es LogReg pipeline')
        w(dash)

        w(f'\n  Csoport A: Index {res["group_a"]}  (n={res["n_a"]:,})')
        w(f'  Csoport B: Index {res["group_b"]}  (n={res["n_b"]:,})')
        w()
        w(f'  Átlagos P(Origo):  {res["group_a"]} → {res["mean_a"]:.4f}  |  '
          f'{res["group_b"]} → {res["mean_b"]:.4f}  '
          f'(Δ = {res["mean_b"] - res["mean_a"]:+.4f})')
        w(f'  Mediális P(Origo): {res["group_a"]} → {res["median_a"]:.4f}  |  '
          f'{res["group_b"]} → {res["median_b"]:.4f}')
        w(f'  Origo-szerű arány: {res["group_a"]} → {res["origo_pct_a"]:.1f}%  |  '
          f'{res["group_b"]} → {res["origo_pct_b"]:.1f}%  '
          f'(Δ = {res["origo_pct_b"] - res["origo_pct_a"]:+.1f} pp)')
        w()
        w('  Tesztek:')
        w(format_test('Mann–Whitney U', res['mann_whitney']['statistic'],
                      res['mann_whitney']['p_value'], res['mann_whitney']['significant']))
        w(format_test('Kolmogorov–Smirnov', res['kolmogorov_smirnov']['statistic'],
                      res['kolmogorov_smirnov']['p_value'], res['kolmogorov_smirnov']['significant']))
        w(format_test('Khi-négyzet (arányok)', res['chi_square']['statistic'],
                      res['chi_square']['p_value'], res['chi_square']['significant']))
        w()
        d = res['cohens_d']
        w(f"  Hatásméret (Cohen's d): {d['value']:+.4f}  ({d['label']} hatás)")
        w()
        all_sig = all([
            res['mann_whitney']['significant'],
            res['kolmogorov_smirnov']['significant'],
            res['chi_square']['significant'],
        ])
        conclusion = ('Mindhárom teszt szignifikáns — a különbség statisztikailag igazolt.'
                      if all_sig else
                      'Nem minden teszt szignifikáns — az eredmény óvatosan értelmezendő.')
        w(f'  → {conclusion}')

    w()
    w(sep)
    path.write_text('\n'.join(lines), encoding='utf-8')


# Visualization

def create_plot(all_results: dict, probas: dict):
    fig, axes = plt.subplots(1, len(MODEL_YEARS), figsize=(14, 5), sharey=False)

    year_colors = {'2019': '#ff7f0e', '2021': '#9467bd'}

    for ax, model_year in zip(axes, MODEL_YEARS):
        p19, p21 = probas[model_year]['2019'], probas[model_year]['2021']
        res = all_results[model_year]

        for proba, yr in [(p19, '2019'), (p21, '2021')]:
            sns.kdeplot(proba, fill=True, alpha=0.45, color=year_colors[yr],
                        label=f'Index {yr}  (n={len(proba):,})', ax=ax)

        ax.axvline(0.5, color='black', linestyle='--', linewidth=1,
                   alpha=0.6, label='Döntési határ (0.5)')
        ax.set_xlim(0, 1)
        ax.set_xlabel('P(Origo)', fontsize=11)
        ax.set_ylabel('Sűrűség', fontsize=11)

        mw_p  = res['mann_whitney']['p_value']
        ks_p  = res['kolmogorov_smirnov']['p_value']
        chi_p = res['chi_square']['p_value']
        d_val = res['cohens_d']['value']
        info  = (f"MW p={mw_p:.2e}  KS p={ks_p:.2e}\n"
                 f"χ² p={chi_p:.2e}  d={d_val:+.3f}")
        ax.text(0.97, 0.97, info, transform=ax.transAxes,
                fontsize=8.5, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

        ax.set_title(f'{model_year}-es LogReg modell\n'
                     f'Δ átlag = {res["mean_b"] - res["mean_a"]:+.4f}  |  '
                     f'Δ arány = {res["origo_pct_b"] - res["origo_pct_a"]:+.1f} pp',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Index cikkek P(Origo) eloszlása — 2019 vs 2021',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = OUTPUT_DIR / 'significance_test_plot.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()


# Main

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    # Load Index data once
    with open(INDEX_DATA, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    mask_19 = df['year'] == '2019'
    mask_21 = df['year'] == '2021'

    all_results = {}
    all_probas  = {}

    for model_year in MODEL_YEARS:
        path = PIPELINE_BASE / str(model_year) / f'logreg_pipeline_{model_year}.pkl'
        with open(path, 'rb') as f:
            pipeline = pickle.load(f)

        proba = pipeline.predict_proba(df['text'].tolist())[:, 1]

        p19 = proba[mask_19.values]
        p21 = proba[mask_21.values]

        all_probas[model_year] = {'2019': p19, '2021': p21}
        all_results[model_year] = run_tests(p19, p21, '2019', '2021')

    # Report
    write_report(all_results, OUTPUT_DIR / 'significance_test_report.txt')

    # Plot
    create_plot(all_results, all_probas)

    # JSON
    json_out = {}
    for model_year, res in all_results.items():
        r = dict(res)
        json_out[str(model_year)] = r
    with open(OUTPUT_DIR / 'significance_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
