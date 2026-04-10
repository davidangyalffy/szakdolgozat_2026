import argparse
import json
import re
import spacy
from pathlib import Path
from typing import List, Dict

# Project paths
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
_DEFAULT_INPUT  = PROJECT_ROOT / 'processed_data' / 'final' / 'hvg_itthon_combined.json'
_DEFAULT_OUTPUT = PROJECT_ROOT / 'processed_data' / 'final' / 'articles_tfidf_lemmatized.json'

# ── Stopwords ────────────────────────────────────────────────────────────────
from spacy.lang.hu.stop_words import STOP_WORDS as SPACY_STOPWORDS

PORTAL_STOPWORDS = {
    # Portal names
    'hvg', 'origo', 'index',
    # With domains
    'hvghu', 'origohu', 'indexhu',
    # Common variations — HVG
    'ahvg', 'hvgnak', 'hvgről', 'hvgtől', 'hvgnél',
    'hvgn', 'hvgs', 'hvgvel', 'hvgben',
    # Common variations — Origo
    'azorigo', 'origonak', 'origoről', 'origotól', 'origonál', 'origónak',
    'origon', 'origoval', 'origoban', 'origós',
    # Common variations — Index
    'azindex', 'indexnek', 'indexről', 'indextől', 'indexnél',
    'indexen', 'indexvel', 'indexben',
    # Common journalistic phrases
    'írja', 'közölte', 'portál', 'oldalon', 'cikk', 'cikkben',
    'számolt', 'beszámolt', 'tudósít', 'jelentette', 'hírportál',
    'news', 'hír', 'hírek', 'online', 'internetes',
    # Authorship
    'szerkesztőség', 'újságíró', 'riporter', 'tudósító',
    # Meta / media references
    'címlapkép', 'fotó', 'kép', 'illusztráció', 'videó',
    'frissítés', 'frissült', 'update', 'cikkünk', 'cikkünkben',
}

ALL_STOPWORDS = set(SPACY_STOPWORDS) | PORTAL_STOPWORDS
print(f"✓ Stopwords loaded: {len(ALL_STOPWORDS)} total")

# ── Load spaCy model ──────────────────────────────────────────────────────────
print("Loading hu_core_news_lg …")
# Keep: tok2vec, tagger, morphologizer, lookup_lemmatizer
# Disable: trainable_lemmatizer (extra neural layer, not needed), parser, ner
nlp = spacy.load(
    'hu_core_news_lg',
    disable=['trainable_lemmatizer', 'parser', 'ner', 'senter']
)
nlp.max_length = 2_000_000
print(f"✓ Model loaded  |  pipeline: {nlp.pipe_names}")


# ── Pre-cleaning (same as articles_tfidf.json) ────────────────────────────────
_URL_RE    = re.compile(r'http\S+|www\.\S+')
_EMAIL_RE  = re.compile(r'\S+@\S+')
_CHAR_RE   = re.compile(r'[^a-záéíóöőúüű\s]')
_SPACE_RE  = re.compile(r'\s+')

def pre_clean(text: str) -> str:
    """Remove noise before lemmatization."""
    if not text:
        return ''
    text = _URL_RE.sub('', text)
    text = _EMAIL_RE.sub('', text)
    text = text.lower()
    text = _CHAR_RE.sub(' ', text)
    text = _SPACE_RE.sub(' ', text).strip()
    return text


# ── Lemmatize + filter stopwords ──────────────────────────────────────────────
def lemmatize_and_filter(doc: spacy.tokens.Doc) -> str:
    """
    For every token in a parsed spaCy Doc:
      - take the lemma (base form)
      - lowercase it
      - drop stopwords, punctuation, spaces, and tokens ≤ 2 chars
    Returns a single joined string.
    """
    tokens = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        lemma = token.lemma_.lower().strip()
        if lemma in ALL_STOPWORDS:
            continue
        if len(lemma) <= 2:
            continue
        # Keep only Hungarian-alphabet lemmas
        if not re.match(r'^[a-záéíóöőúüű]+$', lemma):
            continue
        tokens.append(lemma)
    return ' '.join(tokens)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Create lemmatized TF-IDF dataset')
    parser.add_argument('--input',  type=Path, default=_DEFAULT_INPUT,
                        help='Path to source JSON (default: hvg_itthon_combined.json)')
    parser.add_argument('--output', type=Path, default=_DEFAULT_OUTPUT,
                        help='Path for output JSON (default: articles_tfidf_lemmatized.json)')
    args = parser.parse_args()
    INPUT_PATH  = args.input
    OUTPUT_PATH = args.output

    print('\n' + '=' * 65)
    print('CREATING LEMMATIZED TF-IDF DATASET')
    print('=' * 65)

    # Load source data
    print(f'\nLoading {INPUT_PATH} …')
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    print(f'Loaded {len(articles):,} articles')

    # Prepare (title + content) strings for batch processing
    raw_texts = [
        pre_clean(f"{a.get('title', '')} {a.get('content', '')}")
        for a in articles
    ]

    # Batch lemmatization via nlp.pipe — much faster than one-by-one
    batch_size = 500
    print(f'\nRunning lemmatization  batch_size={batch_size} …')
    lemmatized_texts = []
    total = len(raw_texts)

    for i, doc in enumerate(nlp.pipe(raw_texts, batch_size=batch_size)):
        lemmatized_texts.append(lemmatize_and_filter(doc))
        if (i + 1) % 1000 == 0:
            print(f'  {i + 1:>6,} / {total:,}  ({100*(i+1)/total:.1f}%)')

    print(f'  {total:>6,} / {total:,}  (100.0%)  ✓ done')

    # Build output records, drop articles too short after cleaning
    output = []
    skipped = 0
    new_id  = 1

    for article, text in zip(articles, lemmatized_texts):
        if len(text) < 50:
            skipped += 1
            continue
        output.append({
            'id':      new_id,
            'portal':  article['portal'],
            'year':    article['year'],
            'text':    text,
        })
        new_id += 1

    print(f'\nArticles kept:    {len(output):,}')
    print(f'Articles skipped: {skipped:,}  (text < 50 chars after lemmatization)')

    # Quick stats
    lengths = [len(r['text'].split()) for r in output]
    print(f'\nWord count after lemmatization:')
    print(f'  Average : {sum(lengths)/len(lengths):.1f}')
    print(f'  Median  : {sorted(lengths)[len(lengths)//2]}')
    print(f'  Min     : {min(lengths)}')
    print(f'  Max     : {max(lengths)}')

    # Save
    print(f'\nSaving to {OUTPUT_PATH} …')
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'✓ Saved  |  {len(output):,} articles  |  {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB')
    print('\n' + '=' * 65)
    print('DONE')
    print('=' * 65)


if __name__ == '__main__':
    main()
