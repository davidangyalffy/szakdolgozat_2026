import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Get project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / 'processed_data' / 'final' / 'articles_tfidf_lemmatized.json'
OUTPUT_DIR_BASE = PROJECT_ROOT / 'results' / 'logreg_results'
OUTPUT_DIR = OUTPUT_DIR_BASE   # overwritten in main() once year is known

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_STATE = 67
TEST_SIZE    = 0.2

PARAM_GRID = {
    'tfidf__max_features': [3000, 4000, 5000, 6000, 7000],
    'clf__C':              [0.1, 1.0, 5.0, 10.0, 100.0],
}  # 25 combinations × 5-fold = 125 fits

def load_and_filter_data(year=2021):
    """Load data and filter for specific year"""
    print("=" * 70)
    print("LOADING AND FILTERING DATA")
    print("=" * 70)

    print(f"\nLoading from: {DATA_PATH}")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"Total articles loaded: {len(df)}")

    # Filter for specified year
    df = df[df['year'] == str(year)]
    print(f"Articles from {year}: {len(df)}")

    # Show distribution by portal
    print(f"\nDistribution by portal:")
    print(df['portal'].value_counts())

    # Remove any rows with missing text
    df = df.dropna(subset=['text'])
    print(f"\nArticles after removing missing text: {len(df)}")

    return df

def prepare_features(df):
    """Return raw texts and labels — vectorization handled inside Pipeline."""
    print("\n" + "=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)

    texts = df['text'].tolist()
    y = (df['portal'] == 'origo').astype(int)

    print(f"  Documents: {len(texts)}")
    print(f"\nLabel distribution:")
    print(f"  HVG   (0): {(y == 0).sum()}")
    print(f"  Origo (1): {(y == 1).sum()}")

    return texts, y


def split_data(texts, y):
    print("\n" + "=" * 70)
    print("SPLITTING DATA")
    print("=" * 70)
    t_train, t_test, y_train, y_test = train_test_split(
        texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Training set: {len(t_train)} samples")
    print(f"  Test set:     {len(t_test)} samples")
    return t_train, t_test, y_train, y_test


def _make_pipeline(max_features, C):
    """Build a TF-IDF → LogReg Pipeline with the given parameters."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            min_df=5, max_df=0.8,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ('clf', LogisticRegression(
            C=C,
            max_iter=1000,
            solver='lbfgs',
            random_state=RANDOM_STATE,
        )),
    ])


def train_baseline(texts_train, y_train):
    """Phase 1 — Pipeline with fixed defaults (max_features=5000, C=1.0)."""
    print("\n" + "=" * 70)
    print("PHASE 1 — BASELINE MODEL (default parameters)")
    print("=" * 70)
    print("\nTraining baseline pipeline  [max_features=5000, C=1.0] …")
    pipeline = _make_pipeline(max_features=5000, C=1.0)
    pipeline.fit(texts_train, y_train)
    print("✓ Baseline pipeline trained")
    return pipeline


def tune_hyperparameters(texts_train, y_train):
    """Phase 2 — GridSearchCV over TF-IDF max_features and LogReg C (5-fold, AUC-ROC)."""
    print("\n" + "=" * 70)
    print("PHASE 2 — HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 70)
    n_combos = len(PARAM_GRID['tfidf__max_features']) * len(PARAM_GRID['clf__C'])
    print(f"\nGrid: {PARAM_GRID}")
    print(f"Total fits: {n_combos} combinations × 5 folds = {n_combos * 5}\n")

    base = _make_pipeline(max_features=5000, C=1.0)
    gs = GridSearchCV(base, PARAM_GRID, cv=5, scoring='roc_auc', n_jobs=1, verbose=1)
    gs.fit(texts_train, y_train)

    print(f"\n✓ Grid search complete")
    print(f"  Best AUC (CV): {gs.best_score_:.4f}")
    print(f"  Best params:   {gs.best_params_}")
    return gs.best_params_, gs.cv_results_


def train_final_model(texts_train, y_train, best_params):
    """Phase 3 — Refit Pipeline with best params from grid search."""
    print("\n" + "=" * 70)
    print("PHASE 3 — FINAL MODEL (tuned parameters)")
    print("=" * 70)
    print(f"\nParameters: {best_params}")
    pipeline = _make_pipeline(
        max_features=best_params['tfidf__max_features'],
        C=best_params['clf__C'],
    )
    print("\nTraining final pipeline …")
    pipeline.fit(texts_train, y_train)
    print("✓ Final pipeline trained")
    return pipeline

def evaluate_model(pipeline, texts_train, texts_test, y_train, y_test):
    """Evaluate a fitted Pipeline on train and test texts."""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    y_train_pred  = pipeline.predict(texts_train)
    y_test_pred   = pipeline.predict(texts_test)
    y_train_proba = pipeline.predict_proba(texts_train)[:, 1]
    y_test_proba  = pipeline.predict_proba(texts_test)[:, 1]

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy  = accuracy_score(y_test,  y_test_pred)
    train_auc      = roc_auc_score(y_train, y_train_proba)
    test_auc       = roc_auc_score(y_test,  y_test_proba)

    print(f"\nAccuracy:")
    print(f"  Training: {train_accuracy:.4f}")
    print(f"  Test:     {test_accuracy:.4f}")
    print(f"\nAUC-ROC:")
    print(f"  Training: {train_auc:.4f}")
    print(f"  Test:     {test_auc:.4f}")

    cm = confusion_matrix(y_test, y_test_pred)
    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT (Test Set)")
    print("-" * 70)
    print(classification_report(y_test, y_test_pred,
                                target_names=['HVG', 'Origo'], digits=4))

    print("Confusion Matrix (Test Set):")
    print(f"              Predicted HVG  Predicted Origo")
    print(f"Actual HVG    {cm[0, 0]:>13d}  {cm[0, 1]:>15d}")
    print(f"Actual Origo  {cm[1, 0]:>13d}  {cm[1, 1]:>15d}")

    print("\n" + "-" * 70)
    print("5-FOLD CROSS-VALIDATION")
    print("-" * 70)
    cv_scores = cross_val_score(pipeline, texts_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy':  test_accuracy,
        'train_auc':      train_auc,
        'test_auc':       test_auc,
        'y_test':         y_test,
        'y_test_pred':    y_test_pred,
        'y_test_proba':   y_test_proba,
        'confusion_matrix': cm,
        'cv_scores':      cv_scores,
    }

def analyze_feature_importance(pipeline, top_n=30):
    """Analyze most important features for each class"""
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # Extract components from the fitted Pipeline
    vectorizer    = pipeline.named_steps['tfidf']
    model         = pipeline.named_steps['clf']

    feature_names = vectorizer.get_feature_names_out()
    coefficients  = model.coef_[0]

    # Create DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })

    # Top features for HVG (negative coefficients)
    top_hvg = feature_importance.nsmallest(top_n, 'coefficient')
    print(f"\nTop {top_n} features for HVG:")
    print("-" * 70)
    for i, row in enumerate(top_hvg.itertuples(), 1):
        print(f"{i:3d}. {row.feature:30s} - {row.coefficient:8.4f}")

    # Top features for Origo (positive coefficients)
    top_origo = feature_importance.nlargest(top_n, 'coefficient')
    print(f"\nTop {top_n} features for Origo:")
    print("-" * 70)
    for i, row in enumerate(top_origo.itertuples(), 1):
        print(f"{i:3d}. {row.feature:30s} - {row.coefficient:8.4f}")

    # Save to file
    results_file = OUTPUT_DIR / 'feature_importance.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("FEATURE IMPORTANCE ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Top {top_n} features for HVG (negative coefficients):\n")
        f.write("-" * 70 + "\n")
        for i, row in enumerate(top_hvg.itertuples(), 1):
            f.write(f"{i:3d}. {row.feature:30s} - {row.coefficient:8.4f}\n")

        f.write(f"\nTop {top_n} features for Origo (positive coefficients):\n")
        f.write("-" * 70 + "\n")
        for i, row in enumerate(top_origo.itertuples(), 1):
            f.write(f"{i:3d}. {row.feature:30s} - {row.coefficient:8.4f}\n")

    print(f"\n✓ Saved feature importance to: {results_file}")

    return top_hvg, top_origo

def create_visualizations(results, top_hvg, top_origo):
    """Create visualizations of model performance"""
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    # 1. Confusion Matrix Heatmap
    _, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        results['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['HVG', 'Origo'],
        yticklabels=['HVG', 'Origo'],
        ax=ax
    )
    ax.set_ylabel('Tényleges', fontsize=12)
    ax.set_xlabel('Predikált', fontsize=12)
    ax.set_title('Konfúziós Mátrix – LogReg Portálklasszifikáció', fontsize=14, fontweight='bold')
    plt.tight_layout()

    viz_file = OUTPUT_DIR / 'confusion_matrix.png'
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {viz_file}")
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(results['y_test'], results['y_test_proba'])

    _, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Görbe (AUC = {results["test_auc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Véletlen osztályozó')
    ax.set_xlabel('Téves pozitív arány', fontsize=12)
    ax.set_ylabel('Valós pozitív arány', fontsize=12)
    ax.set_title('ROC Görbe – LogReg Portálklasszifikáció', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    viz_file = OUTPUT_DIR / 'roc_curve.png'
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {viz_file}")
    plt.close()

    # 3. Feature Importance (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # HVG features
    ax = axes[0]
    features = top_hvg['feature'].values[:20]
    coefs = np.abs(top_hvg['coefficient'].values[:20])
    y_pos = np.arange(len(features))

    ax.barh(y_pos, coefs, color='steelblue', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel('Együttható nagysága', fontsize=11)
    ax.set_title('Top 20 HVG-jelző szó', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Origo features
    ax = axes[1]
    features = top_origo['feature'].values[:20]
    coefs = top_origo['coefficient'].values[:20]
    y_pos = np.arange(len(features))

    ax.barh(y_pos, coefs, color='coral', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel('Együttható nagysága', fontsize=11)
    ax.set_title('Top 20 Origo-jelző szó', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    viz_file = OUTPUT_DIR / 'feature_importance.png'
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {viz_file}")
    plt.close()

    # 4. Predicted probability distribution
    _, ax = plt.subplots(figsize=(10, 6))
    proba  = np.array(results['y_test_proba'])
    labels = np.array(results['y_test'])

    sns.kdeplot(proba[labels == 0], fill=True, alpha=0.5, color='steelblue',
                label='HVG', ax=ax)
    sns.kdeplot(proba[labels == 1], fill=True, alpha=0.5, color='coral',
                label='Origo', ax=ax)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1,
               alpha=0.7, label='Döntési határ (0.5)')
    ax.set_xlim(0, 1)
    ax.set_xlabel('P(Origo)', fontsize=12)
    ax.set_ylabel('Sűrűség', fontsize=12)
    ax.set_title('Becsült valószínűség eloszlása – LogReg', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    viz_file = OUTPUT_DIR / 'probability_distribution.png'
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {viz_file}")
    plt.close()

    print("\n✓ All visualizations created")

def save_model_and_results(pipeline, tuned_results, baseline_results, best_params, year):
    """Save the fitted Pipeline and results summary."""
    print("\n" + "=" * 70)
    print("SAVING MODEL AND RESULTS")
    print("=" * 70)

    # Save full pipeline (vectorizer + classifier in one object)
    pipeline_file = OUTPUT_DIR / f'logreg_pipeline_{year}.pkl'
    with open(pipeline_file, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"✓ Saved pipeline to: {pipeline_file}")

    summary = {
        'year':        year,
        'model':       'LogisticRegression',
        'best_params': best_params,
        'baseline': {
            'test_accuracy': float(baseline_results['test_accuracy']),
            'test_auc':      float(baseline_results['test_auc']),
            'train_accuracy': float(baseline_results['train_accuracy']),
            'train_auc':      float(baseline_results['train_auc']),

        },
        'tuned': {
            'train_accuracy': float(tuned_results['train_accuracy']),
            'test_accuracy':  float(tuned_results['test_accuracy']),
            'train_auc':      float(tuned_results['train_auc']),
            'test_auc':       float(tuned_results['test_auc']),
            'cv_mean':        float(tuned_results['cv_scores'].mean()),
            'cv_std':         float(tuned_results['cv_scores'].std()),
            'confusion_matrix': tuned_results['confusion_matrix'].tolist(),
        },
    }
    summary_file = OUTPUT_DIR / f'results_summary_{year}.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved results summary to: {summary_file}")

def main():
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION PORTAL CLASSIFIER")
    print("=" * 70)

    parser = argparse.ArgumentParser(description='Logistic Regression Portal Classifier')
    parser.add_argument('--year', type=int, default=2021,
                        help='Year of articles to train on (default: 2021)')
    args = parser.parse_args()
    YEAR = args.year

    global OUTPUT_DIR
    OUTPUT_DIR = OUTPUT_DIR_BASE / str(YEAR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training on {YEAR} articles to classify HVG vs Origo")

    df = load_and_filter_data(year=YEAR)
    texts, y = prepare_features(df)
    texts_train, texts_test, y_train, y_test = split_data(texts, y)

    # Phase 1 — baseline
    baseline_pipeline = train_baseline(texts_train, y_train)
    baseline_results  = evaluate_model(baseline_pipeline, texts_train, texts_test, y_train, y_test)

    # Phase 2 — grid search
    best_params, _ = tune_hyperparameters(texts_train, y_train)

    # Phase 3 — final tuned pipeline
    final_pipeline = train_final_model(texts_train, y_train, best_params)
    tuned_results  = evaluate_model(final_pipeline, texts_train, texts_test, y_train, y_test)

    top_hvg, top_origo = analyze_feature_importance(final_pipeline, top_n=30)
    create_visualizations(tuned_results, top_hvg, top_origo)
    save_model_and_results(final_pipeline, tuned_results, baseline_results, best_params, YEAR)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\n  Baseline — Accuracy: {baseline_results['test_accuracy']:.4f}"
          f"  AUC: {baseline_results['test_auc']:.4f}")
    print(f"  Tuned    — Accuracy: {tuned_results['test_accuracy']:.4f}"
          f"  AUC: {tuned_results['test_auc']:.4f}")
    print(f"  Best params: {best_params}")
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
