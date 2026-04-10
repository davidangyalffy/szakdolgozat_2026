import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_DATA   = PROJECT_ROOT / 'processed_data' / 'final' / 'articles_lstm_lemmatized.json'
OUTPUT_DIR   = PROJECT_ROOT / 'results' / 'lstm_analysis'


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {INPUT_DATA} …")
    with open(INPUT_DATA, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    print(f"✓ {len(articles):,} articles loaded")

    lengths = np.array([len(a['text'].split()) for a in articles])

    print("\n" + "=" * 55)
    print("SEQUENCE LENGTH STATISTICS")
    print("=" * 55)
    print(f"  Count   : {len(lengths):,}")
    print(f"  Min     : {lengths.min()}")
    print(f"  Max     : {lengths.max()}")
    print(f"  Mean    : {lengths.mean():.1f}")
    print(f"  Median  : {int(np.median(lengths))}")
    print(f"  Std     : {lengths.std():.1f}")
    print()
    print("  Percentiles:")
    for p in [50, 75, 90, 95, 99, 99.9]:
        val = int(np.percentile(lengths, p))
        covered = (lengths <= val).sum()
        print(f"    p{p:<5} : {val:>5}  tokens  "
              f"({100 * covered / len(lengths):.1f}% of articles covered)")
    print("=" * 55)

    # Recommended max_length based on 95th percentile
    max_len_95 = int(np.percentile(lengths, 95))
    print(f"\n→ Recommended max_length (95th pct): {max_len_95}")

    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(lengths, bins=100, color='steelblue', edgecolor='none', alpha=0.8)
    for p, color in [(50, 'green'), (95, 'orange'), (99, 'red')]:
        val = int(np.percentile(lengths, p))
        axes[0].axvline(val, color=color, linestyle='--', linewidth=1.5,
                        label=f'p{p} = {val}')
    axes[0].set_xlabel('Sequence length (tokens)', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Sequence length distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Cumulative coverage curve
    sorted_len = np.sort(lengths)
    coverage   = np.arange(1, len(sorted_len) + 1) / len(sorted_len) * 100
    axes[1].plot(sorted_len, coverage, color='steelblue', linewidth=1.5)
    for p, color in [(95, 'orange'), (99, 'red')]:
        val = int(np.percentile(lengths, p))
        axes[1].axvline(val, color=color, linestyle='--', linewidth=1.5,
                        label=f'p{p} = {val}')
    axes[1].axhline(95, color='orange', linestyle=':', linewidth=1, alpha=0.6)
    axes[1].axhline(99, color='red',    linestyle=':', linewidth=1, alpha=0.6)
    axes[1].set_xlabel('Max sequence length (tokens)', fontsize=11)
    axes[1].set_ylabel('Articles covered (%)', fontsize=11)
    axes[1].set_title('Cumulative coverage by max length', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / 'sequence_length_analysis.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved: {out}")


if __name__ == '__main__':
    main()
