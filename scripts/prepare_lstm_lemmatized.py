import argparse
import json
import re
import spacy
from pathlib import Path

# Project paths
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
_DEFAULT_INPUT  = PROJECT_ROOT / 'processed_data' / 'final' / 'hvg_itthon_combined.json'
_DEFAULT_OUTPUT = PROJECT_ROOT / 'processed_data' / 'final' / 'articles_lstm_lemmatized.json'

# General Hungarian stopwords are intentionally kept — LSTM needs sequential context.
# Only portal names and their inflected forms are removed to prevent data leakage.
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
}


nlp = spacy.load(
    'hu_core_news_lg',
    disable=['trainable_lemmatizer', 'parser', 'ner', 'senter']
)
nlp.max_length = 2_000_000


_URL_RE   = re.compile(r'http\S+|www\.\S+')
_EMAIL_RE = re.compile(r'\S+@\S+')
_CHAR_RE  = re.compile(r'[^a-záéíóöőúüű\s]')
_SPACE_RE = re.compile(r'\s+')

def pre_clean(text: str) -> str:
    if not text:
        return ''
    text = _URL_RE.sub('', text)
    text = _EMAIL_RE.sub('', text)
    text = text.lower()
    text = _CHAR_RE.sub(' ', text)
    text = _SPACE_RE.sub(' ', text).strip()
    return text


def lemmatize_and_filter(doc: spacy.tokens.Doc) -> str:
    tokens = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        lemma = token.lemma_.lower().strip()
        if lemma in PORTAL_STOPWORDS:
            continue
        if len(lemma) <= 1:
            continue
        if not re.match(r'^[a-záéíóöőúüű]+$', lemma):
            continue
        tokens.append(lemma)
    return ' '.join(tokens)


def main():
    parser = argparse.ArgumentParser(description='Create lemmatized dataset for LSTM')
    parser.add_argument('--input',  type=Path, default=_DEFAULT_INPUT,
                        help='Path to source JSON (default: hvg_itthon_combined.json)')
    parser.add_argument('--output', type=Path, default=_DEFAULT_OUTPUT,
                        help='Path for output JSON (default: articles_lstm_lemmatized.json)')
    args = parser.parse_args()
    INPUT_PATH  = args.input
    OUTPUT_PATH = args.output


    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    raw_texts = [
        pre_clean(f"{a.get('title', '')} {a.get('content', '')}")
        for a in articles
    ]

    batch_size = 500
    lemmatized_texts = []
    total = len(raw_texts)

    for i, doc in enumerate(nlp.pipe(raw_texts, batch_size=batch_size)):
        lemmatized_texts.append(lemmatize_and_filter(doc))
        if (i + 1) % 1000 == 0:


    output = []
    skipped = 0
    new_id  = 1

    for article, text in zip(articles, lemmatized_texts):
        if len(text) < 50:
            skipped += 1
            continue
        output.append({
            'id':     new_id,
            'portal': article['portal'],
            'year':   article['year'],
            'text':   text,
        })
        new_id += 1


    lengths = [len(r['text'].split()) for r in output]

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
