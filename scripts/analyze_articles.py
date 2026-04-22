import json
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Hungarian stop words (common words to exclude from analysis)
HUNGARIAN_STOPWORDS = {
    'a', 'az', 'egy', 'és', 'hogy', 'is', 'volt', 'be', 'ki', 'le', 'fel', 'meg',
    'el', 'át', 'de', 'nem', 'van', 'amely', 'amit', 'aki', 'akik', 'azonban',
    'mint', 'már', 'csak', 'vagy', 'után', 'alatt', 'által', 'között', 'szerint',
    'miatt', 'során', 'ellen', 'minden', 'több', 'még', 'lehet', 'lesz', 'lett',
    'igen', 'így', 'ezt', 'azt', 'ezt', 'ezek', 'azok', 'ezen', 'azon', 'ebben',
    'abban', 'most', 'majd', 'amikor', 'ahol', 'ahhoz', 'ehhez', 'azért', 'ezért',
    'ha', 'pedig', 'is', 'sem', 'se', 'se', 'sem'
}

def load_data(filename: str) -> List[dict]:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    # Remove punctuation and convert to lowercase
    words = re.findall(r'\b[a-záéíóöőúüű]+\b', text.lower())
    # Remove stop words
    words = [w for w in words if w not in HUNGARIAN_STOPWORDS and len(w) > 2]
    return words

def calculate_basic_stats(data: List[dict]) -> dict:

    lengths = [len(article['content']) if article['content'] else 0 for article in data]
    word_counts = [len(tokenize(article['content'])) if article['content'] else 0 for article in data]

    # Find articles with min/max lengths
    min_idx = lengths.index(min(lengths))
    max_idx = lengths.index(max(lengths))

    stats = {
        'total_articles': len(data),
        'avg_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'std_length': np.std(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'avg_word_count': np.mean(word_counts),
        'median_word_count': np.median(word_counts),
        'shortest_article': data[min_idx],
        'longest_article': data[max_idx]
    }


    return stats, lengths, word_counts

def get_top_words(data: List[dict], top_n: int = 50) -> List[Tuple[str, int]]:

    all_words = []
    for article in data:
        if article['content']:
            all_words.extend(tokenize(article['content']))

    word_freq = Counter(all_words)
    top_words = word_freq.most_common(top_n)

    for i, (word, count) in enumerate(top_words[:20], 1):

    if top_n > 20:

    return top_words

def get_top_bigrams(data: List[dict], top_n: int = 20) -> List[Tuple[Tuple[str, str], int]]:

    all_bigrams = []
    for article in data:
        if article['content']:
            words = tokenize(article['content'])
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            all_bigrams.extend(bigrams)

    bigram_freq = Counter(all_bigrams)
    top_bigrams = bigram_freq.most_common(top_n)

    for i, (bigram, count) in enumerate(top_bigrams, 1):

    return top_bigrams

def analyze_by_year(data: List[dict]) -> dict:

    year_counts = Counter([article['year'] for article in data if article['year']])

    for year in sorted(year_counts.keys()):

    return year_counts

def create_visualizations(stats, lengths, word_counts, top_words, top_bigrams, year_counts):

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Content length distribution (histogram)
    ax1 = plt.subplot(2, 3, 1)
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(stats['avg_length'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["avg_length"]:.0f}')
    plt.axvline(stats['median_length'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats["median_length"]:.0f}')
    plt.xlabel('Content Length (characters)', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.title('Distribution of Article Content Length', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Word count distribution (histogram)
    ax2 = plt.subplot(2, 3, 2)
    plt.hist(word_counts, bins=50, color='lightcoral', edgecolor='black')
    plt.axvline(stats['avg_word_count'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["avg_word_count"]:.0f}')
    plt.axvline(stats['median_word_count'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats["median_word_count"]:.0f}')
    plt.xlabel('Word Count (excluding stop words)', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.title('Distribution of Article Word Count', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Top 20 words (bar chart)
    ax3 = plt.subplot(2, 3, 3)
    words, counts = zip(*top_words[:20])
    y_pos = np.arange(len(words))
    plt.barh(y_pos, counts, color='mediumseagreen')
    plt.yticks(y_pos, words, fontsize=10)
    plt.xlabel('Frequency', fontsize=12)
    plt.title('Top 20 Most Common Words', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')

    # 4. Top 20 bigrams (bar chart)
    ax4 = plt.subplot(2, 3, 4)
    bigrams, counts = zip(*top_bigrams[:20])
    bigram_labels = [f"{b[0]} {b[1]}" for b in bigrams]
    y_pos = np.arange(len(bigram_labels))
    plt.barh(y_pos, counts, color='mediumpurple')
    plt.yticks(y_pos, bigram_labels, fontsize=9)
    plt.xlabel('Frequency', fontsize=12)
    plt.title('Top 20 Most Common Bigrams', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')

    # 5. Articles by year (bar chart)
    ax5 = plt.subplot(2, 3, 5)
    years = sorted(year_counts.keys())
    counts = [year_counts[year] for year in years]
    plt.bar(years, counts, color='orange', edgecolor='black')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.title('Articles by Year', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

    # 6. Box plot for content length
    ax6 = plt.subplot(2, 3, 6)
    plt.boxplot([lengths], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))
    plt.ylabel('Content Length (characters)', fontsize=12)
    plt.title('Box Plot of Article Content Length', fontsize=14, fontweight='bold')
    plt.xticks([1], ['All Articles'])
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('article_statistics.png', dpi=300, bbox_inches='tight')

    # Additional visualization: Top 50 words (larger version)
    fig2, ax = plt.subplots(figsize=(12, 16))
    words, counts = zip(*top_words[:50])
    y_pos = np.arange(len(words))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(words)))
    plt.barh(y_pos, counts, color=colors)
    plt.yticks(y_pos, words, fontsize=11)
    plt.xlabel('Frequency', fontsize=14)
    plt.title('Top 50 Most Common Words', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('top_50_words.png', dpi=300, bbox_inches='tight')


def save_statistics_report(stats, top_words, top_bigrams, year_counts):

    with open('article_statistics_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("HVG ITTHON ARTICLES - DESCRIPTIVE STATISTICS REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("BASIC STATISTICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total articles: {stats['total_articles']}\n\n")
        f.write(f"Content Length (characters):\n")
        f.write(f"  Average: {stats['avg_length']:.2f}\n")
        f.write(f"  Median: {stats['median_length']:.2f}\n")
        f.write(f"  Std Dev: {stats['std_length']:.2f}\n")
        f.write(f"  Shortest: {stats['min_length']}\n")
        f.write(f"  Longest: {stats['max_length']}\n\n")

        f.write(f"Word Count (excluding stop words):\n")
        f.write(f"  Average: {stats['avg_word_count']:.2f}\n")
        f.write(f"  Median: {stats['median_word_count']:.2f}\n\n")

        f.write(f"Shortest article (ID: {stats['shortest_article']['id']}):\n")
        f.write(f"  Title: {stats['shortest_article']['title']}\n")
        f.write(f"  Length: {stats['min_length']} characters\n\n")

        f.write(f"Longest article (ID: {stats['longest_article']['id']}):\n")
        f.write(f"  Title: {stats['longest_article']['title']}\n")
        f.write(f"  Length: {stats['max_length']} characters\n\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("TOP 50 MOST COMMON WORDS\n")
        f.write("-" * 60 + "\n")
        for i, (word, count) in enumerate(top_words, 1):
            f.write(f"{i:2d}. {word:25s} - {count:5d} occurrences\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("TOP 20 MOST COMMON BIGRAMS\n")
        f.write("-" * 60 + "\n")
        for i, (bigram, count) in enumerate(top_bigrams, 1):
            f.write(f"{i:2d}. '{bigram[0]} {bigram[1]}' - {count:4d} occurrences\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("ARTICLES BY YEAR\n")
        f.write("-" * 60 + "\n")
        for year in sorted(year_counts.keys()):
            f.write(f"{year}: {year_counts[year]:5d} articles\n")


def main():
    # Load data
    data = load_data('hvg_itthon_combined.json')

    # Calculate basic statistics
    stats, lengths, word_counts = calculate_basic_stats(data)

    # Get top words and bigrams
    top_words = get_top_words(data, top_n=50)
    top_bigrams = get_top_bigrams(data, top_n=20)

    # Analyze by year
    year_counts = analyze_by_year(data)

    # Create visualizations
    create_visualizations(stats, lengths, word_counts, top_words, top_bigrams, year_counts)

    # Save report
    save_statistics_report(stats, top_words, top_bigrams, year_counts)


if __name__ == "__main__":
    main()
