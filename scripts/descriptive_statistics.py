import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
HVG_ORIGO    = PROJECT_ROOT / 'processed_data' / 'final' / 'hvg_itthon_combined.json'
INDEX        = PROJECT_ROOT / 'processed_data' / 'final' / 'index_itthon_combined.json'
OUTPUT_DIR   = PROJECT_ROOT / 'results' / 'statistics'

PORTAL_COLORS  = {'hvg': '#1f77b4', 'origo': '#d62728', 'index': '#2ca02c'}
YEAR_COLORS    = {'2019': '#ff7f0e', '2021': '#9467bd'}
YEARS          = ['2019', '2021']
PORTALS        = ['hvg', 'origo', 'index']

_WORD_RE = re.compile(r'\b[a-záéíóöőúüű]{2,}\b', re.IGNORECASE)


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text)) if isinstance(text, str) else 0


def char_count(text: str) -> int:
    return len(text.strip()) if isinstance(text, str) else 0


def unique_words(text: str) -> int:
    return len(set(w.lower() for w in _WORD_RE.findall(text))) if isinstance(text, str) else 0


def describe(series: pd.Series) -> dict:
    return {
        'n':      int(len(series)),
        'mean':   float(series.mean()),
        'median': float(series.median()),
        'std':    float(series.std()),
        'min':    float(series.min()),
        'p25':    float(series.quantile(0.25)),
        'p75':    float(series.quantile(0.75)),
        'p95':    float(series.quantile(0.95)),
        'max':    float(series.max()),
    }


def fmt(d: dict) -> str:
    return (f"n={d['n']:,}  átlag={d['mean']:.1f}  med={d['median']:.1f}  "
            f"szórás={d['std']:.1f}  min={d['min']:.0f}  p95={d['p95']:.0f}  max={d['max']:.0f}")


# Load & prepare

def load_data() -> pd.DataFrame:
    frames = []
    for path in [HVG_ORIGO, INDEX]:
        with open(path, 'r', encoding='utf-8') as f:
            frames.append(pd.DataFrame(json.load(f)))
    df = pd.concat(frames, ignore_index=True)
    df = df[df['year'].isin(YEARS)].copy()
    df['full_text']    = df['title'].fillna('') + ' ' + df['content'].fillna('')
    df['word_count']   = df['full_text'].apply(word_count)
    df['char_count']   = df['full_text'].apply(char_count)
    df['title_words']  = df['title'].apply(word_count)
    df['unique_words'] = df['full_text'].apply(unique_words)
    df['ttr']          = df.apply(
        lambda r: r['unique_words'] / r['word_count'] if r['word_count'] > 0 else 0, axis=1
    )
    df['portal'] = df['portal'].str.lower()
    return df


# Text report

def write_report(df: pd.DataFrame, path: Path):
    lines = []
    sep  = '=' * 70
    dash = '-' * 70

    def h1(t): lines.append(f'\n{sep}\n{t}\n{sep}')
    def h2(t): lines.append(f'\n{dash}\n{t}\n{dash}')

    h1('LEÍRÓ STATISZTIKÁK — HVG, ORIGO, INDEX (2019 & 2021)')

    h2('1. MINTAMÉRET')
    total = len(df)
    lines.append(f'\nÖsszes cikk (2019+2021): {total:,}\n')

    lines.append(f'  {"Portál":<10} {"2019":>8} {"2021":>8} {"Összesen":>10}')
    lines.append(f'  {"-"*38}')
    for p in PORTALS:
        sub  = df[df['portal'] == p]
        n19  = (sub['year'] == '2019').sum()
        n21  = (sub['year'] == '2021').sum()
        ntot = len(sub)
        lines.append(f'  {p:<10} {n19:>8,} {n21:>8,} {ntot:>10,}')
    lines.append(f'  {"-"*38}')
    for yr in YEARS:
        n = (df['year'] == yr).sum()
        lines.append(f'  {"Összesen " + yr:<10} {n:>27,}')

    h2('2. SZÓSZÁM (cikk = cím + tartalom)')

    lines.append(f'\n  {"Portál":<10} {"n":>6}  {"átlag":>7}  {"medián":>7}  '
                 f'{"szórás":>7}  {"min":>5}  {"p95":>6}  {"max":>6}')
    lines.append(f'  {"-"*62}')
    for p in PORTALS:
        d = describe(df[df['portal'] == p]['word_count'])
        lines.append(f'  {p:<10} {d["n"]:>6,}  {d["mean"]:>7.1f}  {d["median"]:>7.1f}  '
                     f'{d["std"]:>7.1f}  {d["min"]:>5.0f}  {d["p95"]:>6.0f}  {d["max"]:>6.0f}')

    lines.append(f'\n  {"Év":<10} {"n":>6}  {"átlag":>7}  {"medián":>7}  '
                 f'{"szórás":>7}  {"min":>5}  {"p95":>6}  {"max":>6}')
    lines.append(f'  {"-"*62}')
    for yr in YEARS:
        d = describe(df[df['year'] == yr]['word_count'])
        lines.append(f'  {yr:<10} {d["n"]:>6,}  {d["mean"]:>7.1f}  {d["median"]:>7.1f}  '
                     f'{d["std"]:>7.1f}  {d["min"]:>5.0f}  {d["p95"]:>6.0f}  {d["max"]:>6.0f}')

    h2('3. SZÓSZÁM – PORTÁL × ÉV')
    lines.append(f'\n  {"Portál":<10} {"Év":<6} {"n":>6}  {"átlag":>7}  {"medián":>7}  {"szórás":>7}')
    lines.append(f'  {"-"*50}')
    for p in PORTALS:
        for yr in YEARS:
            sub = df[(df['portal'] == p) & (df['year'] == yr)]
            if len(sub) == 0:
                continue
            d = describe(sub['word_count'])
            lines.append(f'  {p:<10} {yr:<6} {d["n"]:>6,}  {d["mean"]:>7.1f}  '
                         f'{d["median"]:>7.1f}  {d["std"]:>7.1f}')

    h2('4. CІМSZÓSZÁM')
    lines.append(f'\n  {"Portál":<10} {"átlag":>7}  {"medián":>7}  {"szórás":>7}  {"p95":>6}')
    lines.append(f'  {"-"*42}')
    for p in PORTALS:
        d = describe(df[df['portal'] == p]['title_words'])
        lines.append(f'  {p:<10} {d["mean"]:>7.1f}  {d["median"]:>7.1f}  {d["std"]:>7.1f}  {d["p95"]:>6.0f}')

    h2('5. LEXIKAI DIVERZITÁS (Type-Token Ratio)')
    lines.append(f'\n  {"Portál":<10} {"átlag TTR":>10}  {"medián TTR":>11}  {"szórás":>8}')
    lines.append(f'  {"-"*44}')
    for p in PORTALS:
        d = describe(df[df['portal'] == p]['ttr'])
        lines.append(f'  {p:<10} {d["mean"]:>10.4f}  {d["median"]:>11.4f}  {d["std"]:>8.4f}')

    lines.append(f'\n  {"Év":<10} {"átlag TTR":>10}  {"medián TTR":>11}  {"szórás":>8}')
    lines.append(f'  {"-"*44}')
    for yr in YEARS:
        d = describe(df[df['year'] == yr]['ttr'])
        lines.append(f'  {yr:<10} {d["mean"]:>10.4f}  {d["median"]:>11.4f}  {d["std"]:>8.4f}')

    h2('6. LEGGYAKORIBB SZAVAK PORTÁLONKÉNT (top 20)')
    for p in PORTALS:
        lines.append(f'\n  {p.upper()}:')
        all_words = _WORD_RE.findall(
            ' '.join(df[df['portal'] == p]['full_text'].fillna(''))
        )
        top = Counter(w.lower() for w in all_words).most_common(20)
        for rank, (word, cnt) in enumerate(top, 1):
            lines.append(f'    {rank:>2}. {word:<20} {cnt:>7,}')

    h2('7. ÉVES VÁLTOZÁS PORTÁLONKÉNT (szószám mediánja)')
    lines.append(f'\n  {"Portál":<10} {"2019 med.":>10}  {"2021 med.":>10}  {"változás":>10}')
    lines.append(f'  {"-"*45}')
    for p in PORTALS:
        sub = df[df['portal'] == p]
        m19 = sub[sub['year'] == '2019']['word_count'].median()
        m21 = sub[sub['year'] == '2021']['word_count'].median()
        if not np.isnan(m19) and not np.isnan(m21):
            chg = m21 - m19
            lines.append(f'  {p:<10} {m19:>10.1f}  {m21:>10.1f}  {chg:>+10.1f}')

    path.write_text('\n'.join(lines), encoding='utf-8')


# Visualizations

def plot_article_counts(df: pd.DataFrame):
    counts = (df.groupby(['portal', 'year'])
                .size()
                .reset_index(name='n'))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: grouped bar by portal × year
    pivot = counts.pivot(index='portal', columns='year', values='n').fillna(0)
    pivot = pivot.reindex(PORTALS)
    x  = np.arange(len(PORTALS))
    w  = 0.35
    ax = axes[0]
    for i, yr in enumerate(YEARS):
        vals = pivot.get(yr, pd.Series([0]*len(PORTALS))).values
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=yr,
                      color=list(YEAR_COLORS.values())[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                    f'{int(v):,}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in PORTALS], fontsize=11)
    ax.set_ylabel('Cikkek száma', fontsize=11)
    ax.set_title('Cikkek száma portálonként és évenként', fontsize=12, fontweight='bold')
    ax.legend(title='Év', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: stacked bar total per portal
    ax2 = axes[1]
    bottom = np.zeros(len(PORTALS))
    for i, yr in enumerate(YEARS):
        vals = pivot.get(yr, pd.Series([0]*len(PORTALS))).values.astype(float)
        ax2.bar(x, vals, w * 2.2, bottom=bottom, label=yr,
                color=list(YEAR_COLORS.values())[i], alpha=0.85)
        for j, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0:
                ax2.text(x[j], b + v/2, f'{int(v):,}',
                         ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        bottom += vals
    ax2.set_xticks(x)
    ax2.set_xticklabels([p.upper() for p in PORTALS], fontsize=11)
    ax2.set_ylabel('Cikkek száma', fontsize=11)
    ax2.set_title('Cikkek összesített száma portálonként', fontsize=12, fontweight='bold')
    ax2.legend(title='Év', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    p = OUTPUT_DIR / 'article_counts.png'
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()


def plot_word_count_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: violin by portal
    ax = axes[0]
    data_portal = [df[df['portal'] == p]['word_count'].clip(upper=1500).values for p in PORTALS]
    parts = ax.violinplot(data_portal, positions=range(len(PORTALS)),
                          showmedians=True, showextrema=True)
    for i, (pc, p) in enumerate(zip(parts['bodies'], PORTALS)):
        pc.set_facecolor(PORTAL_COLORS[p])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)
    ax.set_xticks(range(len(PORTALS)))
    ax.set_xticklabels([p.upper() for p in PORTALS], fontsize=11)
    ax.set_ylabel('Szószám (max 1500)', fontsize=11)
    ax.set_title('Szószám eloszlása portálonként', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Right: violin by year
    ax2 = axes[1]
    data_year = [df[df['year'] == yr]['word_count'].clip(upper=1500).values for yr in YEARS]
    parts2 = ax2.violinplot(data_year, positions=range(len(YEARS)),
                            showmedians=True, showextrema=True)
    for pc, yr in zip(parts2['bodies'], YEARS):
        pc.set_facecolor(YEAR_COLORS[yr])
        pc.set_alpha(0.7)
    parts2['cmedians'].set_color('black')
    parts2['cmedians'].set_linewidth(2)
    ax2.set_xticks(range(len(YEARS)))
    ax2.set_xticklabels(YEARS, fontsize=11)
    ax2.set_ylabel('Szószám (max 1500)', fontsize=11)
    ax2.set_title('Szószám eloszlása évenként', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'word_count_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_word_count_by_portal_and_year(df: pd.DataFrame):
    fig, axes = plt.subplots(1, len(PORTALS), figsize=(15, 5), sharey=True)
    for ax, p in zip(axes, PORTALS):
        for yr in YEARS:
            sub = df[(df['portal'] == p) & (df['year'] == yr)]['word_count'].clip(upper=1500)
            sns.kdeplot(sub, ax=ax, fill=True, alpha=0.4,
                        color=YEAR_COLORS[yr], label=yr)
        ax.set_title(p.upper(), fontsize=12, fontweight='bold')
        ax.set_xlabel('Szószám', fontsize=10)
        ax.set_xlim(0, 1500)
        ax.legend(title='Év', fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Sűrűség', fontsize=11)
    fig.suptitle('Szószám eloszlása portál és év szerint', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'word_count_by_portal_year.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_length_kde_by_portal(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for p in PORTALS:
        sns.kdeplot(df[df['portal'] == p]['word_count'].clip(upper=1500),
                    ax=ax, fill=True, alpha=0.35, color=PORTAL_COLORS[p], label=p.upper())
    ax.set_xlabel('Szószám', fontsize=11)
    ax.set_ylabel('Sűrűség', fontsize=11)
    ax.set_title('Szószám eloszlása portálonként', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1500)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for p in PORTALS:
        sns.kdeplot(df[df['portal'] == p]['char_count'].clip(upper=10000),
                    ax=ax2, fill=True, alpha=0.35, color=PORTAL_COLORS[p], label=p.upper())
    ax2.set_xlabel('Karakterszám', fontsize=11)
    ax2.set_ylabel('Sűrűség', fontsize=11)
    ax2.set_title('Karakterszám eloszlása portálonként', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 10000)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'length_distribution_by_portal.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_ttr_comparison(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ttr_data = [df[df['portal'] == p]['ttr'].values for p in PORTALS]
    bp = ax.boxplot(ttr_data, patch_artist=True, medianprops=dict(color='black', linewidth=2))
    for patch, p in zip(bp['boxes'], PORTALS):
        patch.set_facecolor(PORTAL_COLORS[p])
        patch.set_alpha(0.7)
    ax.set_xticklabels([p.upper() for p in PORTALS], fontsize=11)
    ax.set_ylabel('Type-Token Ratio', fontsize=11)
    ax.set_title('Lexikai diverzitás portálonként', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax2 = axes[1]
    pivot_ttr = df.groupby(['portal', 'year'])['ttr'].mean().reset_index()
    x   = np.arange(len(PORTALS))
    w   = 0.35
    for i, yr in enumerate(YEARS):
        vals = [pivot_ttr[(pivot_ttr['portal'] == p) & (pivot_ttr['year'] == yr)]['ttr'].values
                for p in PORTALS]
        vals = [v[0] if len(v) > 0 else 0 for v in vals]
        ax2.bar(x + (i - 0.5) * w, vals, w, label=yr,
                color=list(YEAR_COLORS.values())[i], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([p.upper() for p in PORTALS], fontsize=11)
    ax2.set_ylabel('Átlagos Type-Token Ratio', fontsize=11)
    ax2.set_title('Lexikai diverzitás portál és év szerint', fontsize=12, fontweight='bold')
    ax2.legend(title='Év', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lexical_diversity.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_words_by_portal(df: pd.DataFrame):
    fig, axes = plt.subplots(1, len(PORTALS), figsize=(18, 6))
    for ax, p in zip(axes, PORTALS):
        all_words = _WORD_RE.findall(
            ' '.join(df[df['portal'] == p]['full_text'].fillna(''))
        )
        top = Counter(w.lower() for w in all_words).most_common(20)
        words, counts = zip(*top)
        ax.barh(list(reversed(words)), list(reversed(counts)),
                color=PORTAL_COLORS[p], alpha=0.8)
        ax.set_title(p.upper(), fontsize=12, fontweight='bold')
        ax.set_xlabel('Előfordulás', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
    fig.suptitle('Top 20 leggyakoribb szó portálonként', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top_words_by_portal.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_year_comparison_per_portal(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(PORTALS))
    w = 0.35
    for i, yr in enumerate(YEARS):
        medians = [df[(df['portal'] == p) & (df['year'] == yr)]['word_count'].median()
                   for p in PORTALS]
        bars = ax.bar(x + (i - 0.5) * w, medians, w, label=yr,
                      color=list(YEAR_COLORS.values())[i], alpha=0.85)
        for bar, v in zip(bars, medians):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{v:.0f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in PORTALS], fontsize=11)
    ax.set_ylabel('Mediális szószám', fontsize=11)
    ax.set_title('Mediális szószám változása 2019 → 2021', fontsize=12, fontweight='bold')
    ax.legend(title='Év', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'year_comparison_word_count.png', dpi=300, bbox_inches='tight')
    plt.close()


# Main

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    df = load_data()

    write_report(df, OUTPUT_DIR / 'descriptive_statistics_report.txt')

    plot_article_counts(df)
    plot_word_count_distribution(df)
    plot_word_count_by_portal_and_year(df)
    plot_length_kde_by_portal(df)
    plot_ttr_comparison(df)
    plot_top_words_by_portal(df)
    plot_year_comparison_per_portal(df)


if __name__ == '__main__':
    main()
