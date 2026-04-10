import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH        = PROJECT_ROOT / 'processed_data' / 'final' / 'articles_tfidf_lemmatized.json'
OUTPUT_DIR_BASE  = PROJECT_ROOT / 'results' / 'xgboost_results'

# ── Configuration ─────────────────────────────────────────────────────────────
YEAR         = 2019
MAX_FEATURES = 5000
TEST_SIZE    = 0.2
RANDOM_STATE = 67
SHAP_SAMPLES = 300  # number of test samples used for SHAP (memory / speed trade-off)

PARAM_GRID = {
    'max_depth':        [4, 6, 8],
    'learning_rate':    [0.05, 0.1],
    'n_estimators':     [200, 300, 500],
    'colsample_bytree': [0.3, 0.5],
    'subsample':        [0.7, 1.0],
}  # 72 combinations × 5-fold CV = 360 fits


# ─────────────────────────────────────────────────────────────────────────────

def load_and_filter_data(year=YEAR):
    print("=" * 70)
    print("LOADING AND FILTERING DATA")
    print("=" * 70)
    print(f"\nLoading from: {DATA_PATH}")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"Total articles loaded: {len(df)}")

    df = df[df['year'] == str(year)]
    print(f"Articles from {year}: {len(df)}")

    print(f"\nDistribution by portal:")
    print(df['portal'].value_counts())

    df = df.dropna(subset=['text'])
    print(f"\nArticles after removing missing text: {len(df)}")

    return df


def prepare_features(df, max_features=MAX_FEATURES):
    print("\n" + "=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    print(f"\nVectorizing text with max_features={max_features}...")
    X = vectorizer.fit_transform(df['text'])
    print(f"✓ Feature matrix shape: {X.shape}")

    y = (df['portal'] == 'origo').astype(int)
    print(f"\nLabel distribution:")
    print(f"  HVG   (0): {(y == 0).sum()}")
    print(f"  Origo (1): {(y == 1).sum()}")

    return X, y, vectorizer


def split_data(X, y):
    print("\n" + "=" * 70)
    print("SPLITTING DATA")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def train_baseline(X_train, y_train, X_test, y_test):
    """Train with XGBoost defaults — no tuning."""
    print("\n" + "=" * 70)
    print("PHASE 1 — BASELINE MODEL (default parameters)")
    print("=" * 70)

    model = XGBClassifier(
        n_estimators=100,       # XGBoost default
        tree_method='hist',
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    print("\nTraining baseline model...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("✓ Baseline model trained")
    return model


def tune_hyperparameters(X_train, y_train):
    """GridSearchCV over key XGBoost hyperparameters (5-fold, scoring=roc_auc)."""
    print("\n" + "=" * 70)
    print("PHASE 2 — HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 70)
    print(f"\nParameter grid: {PARAM_GRID}")
    n_fits = (
        len(PARAM_GRID['max_depth']) *
        len(PARAM_GRID['learning_rate']) *
        len(PARAM_GRID['n_estimators']) *
        len(PARAM_GRID['colsample_bytree']) *
        len(PARAM_GRID['subsample'])
    )
    print(f"Total fits: {n_fits} combinations × 5 folds = {n_fits * 5}\n")

    base = XGBClassifier(
        tree_method='hist',
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    gs = GridSearchCV(base, PARAM_GRID, cv=5, scoring='roc_auc',
                      n_jobs=1, verbose=1)
    gs.fit(X_train, y_train)

    print(f"\n✓ Grid search complete")
    print(f"  Best AUC (CV): {gs.best_score_:.4f}")
    print(f"  Best params:   {gs.best_params_}")
    return gs.best_params_, gs.cv_results_


def train_tuned_model(X_train, y_train, X_test, y_test, best_params):
    """Retrain on full training set with best params from grid search."""
    print("\n" + "=" * 70)
    print("PHASE 3 — FINAL MODEL (tuned parameters)")
    print("=" * 70)
    print(f"\nParameters: {best_params}")

    model = XGBClassifier(
        **best_params,
        tree_method='hist',
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    print("\nTraining tuned model...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("✓ Tuned model trained")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    y_train_pred  = model.predict(X_train)
    y_test_pred   = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba  = model.predict_proba(X_test)[:, 1]

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
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
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


def analyze_shap(model, vectorizer, X_test, n_samples=SHAP_SAMPLES):
    print("\n" + "=" * 70)
    print("SHAP ANALYSIS")
    print("=" * 70)

    feature_names = vectorizer.get_feature_names_out()

    # Use a fixed random sample so results are reproducible
    rng = np.random.default_rng(RANDOM_STATE)
    n = min(n_samples, X_test.shape[0])
    idx = rng.choice(X_test.shape[0], size=n, replace=False)
    X_sample = X_test[idx].toarray()   # TreeExplainer / summary_plot need dense

    print(f"\nComputing SHAP values on {n} test samples …")
    explainer   = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    shap_values = explainer.shap_values(X_sample)   # (n, n_features)
    print("✓ SHAP values computed")

    # ── Global importance ──────────────────────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
        'mean_shap':     shap_values.mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False)

    top_n = 30
    top_hvg   = importance_df[importance_df['mean_shap'] < 0].head(top_n)
    top_origo = importance_df[importance_df['mean_shap'] > 0].head(top_n)

    # Print & save text report
    report_path = OUTPUT_DIR / 'shap_importance.txt'
    lines = ["SHAP FEATURE IMPORTANCE", "=" * 70, ""]

    lines += [f"Top {top_n} features pushing toward HVG (negative SHAP):"]
    lines += ["-" * 70]
    for i, row in enumerate(top_hvg.itertuples(), 1):
        lines.append(f"{i:3d}. {row.feature:35s}  mean_shap={row.mean_shap:+.4f}")

    lines += ["", f"Top {top_n} features pushing toward Origo (positive SHAP):"]
    lines += ["-" * 70]
    for i, row in enumerate(top_origo.itertuples(), 1):
        lines.append(f"{i:3d}. {row.feature:35s}  mean_shap={row.mean_shap:+.4f}")

    text = "\n".join(lines)
    print(text)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\n✓ Saved: {report_path}")

    # Save raw SHAP array + feature names for future use
    np.save(OUTPUT_DIR / 'shap_values.npy', shap_values)
    with open(OUTPUT_DIR / 'shap_feature_names.json', 'w', encoding='utf-8') as f:
        json.dump(feature_names.tolist(), f, ensure_ascii=False)

    # ── Visualisations ─────────────────────────────────────────────────────────

    # 1. Beeswarm summary (top 20)
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        max_display=20,
        show=False,
        plot_type='dot',
    )
    plt.title('SHAP – Top 20 Jellemző\n(pozitív = Origo, negatív = HVG)',
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = OUTPUT_DIR / 'shap_summary_beeswarm.png'
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {p}")

    # 2. Bar chart – HVG top 20
    _plot_shap_bar(top_hvg.head(20), 'Top 20 HVG-t jelző jellemző',
                   'skyblue', OUTPUT_DIR / 'shap_bar_hvg.png')

    # 3. Bar chart – Origo top 20
    _plot_shap_bar(top_origo.head(20), 'Top 20 Origót jelző jellemző',
                   'lightcoral', OUTPUT_DIR / 'shap_bar_origo.png')

    return shap_values, importance_df


def _plot_shap_bar(df, title, color, path):
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, np.abs(df['mean_abs_shap'].values), color=color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['feature'].values, fontsize=9)
    ax.set_xlabel('Átlagos |SHAP érték|', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {path}")


def create_standard_visualizations(results):
    print("\n" + "=" * 70)
    print("CREATING STANDARD VISUALIZATIONS")
    print("=" * 70)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['HVG', 'Origo'], yticklabels=['HVG', 'Origo'], ax=ax)
    ax.set_ylabel('Tényleges', fontsize=12)
    ax.set_xlabel('Predikált', fontsize=12)
    ax.set_title('Konfúziós Mátrix – XGBoost Portálklasszifikáció',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = OUTPUT_DIR / 'confusion_matrix.png'
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {p}")

    # ROC curve
    fpr, tpr, _ = roc_curve(results['y_test'], results['y_test_proba'])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2,
            label=f'ROC Görbe (AUC = {results["test_auc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Véletlen osztályozó')
    ax.set_xlabel('Téves pozitív arány', fontsize=12)
    ax.set_ylabel('Valós pozitív arány', fontsize=12)
    ax.set_title('ROC Görbe – XGBoost Portálklasszifikáció',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = OUTPUT_DIR / 'roc_curve.png'
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {p}")

    # Predicted probability distribution
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
    ax.set_title('Becsült valószínűség eloszlása – XGBoost',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = OUTPUT_DIR / 'probability_distribution.png'
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {p}")


def save_model_and_results(model, vectorizer, tuned_results, baseline_results, best_params):
    print("\n" + "=" * 70)
    print("SAVING MODEL AND RESULTS")
    print("=" * 70)

    p = OUTPUT_DIR / f'xgboost_model_{YEAR}.pkl'
    with open(p, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved model:      {p}")

    p = OUTPUT_DIR / f'vectorizer_{YEAR}.pkl'
    with open(p, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✓ Saved vectorizer: {p}")

    summary = {
        'year':  YEAR,
        'model': 'XGBoost',
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
    p = OUTPUT_DIR / f'results_summary_{YEAR}.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary:    {p}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    global OUTPUT_DIR
    OUTPUT_DIR = OUTPUT_DIR_BASE / str(YEAR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"XGBOOST PORTAL CLASSIFIER  (HVG vs Origo, {YEAR})")
    print("=" * 70)

    df = load_and_filter_data(year=YEAR)
    X, y, vectorizer = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Phase 1 — baseline
    baseline_model = train_baseline(X_train, y_train, X_test, y_test)
    baseline_results = evaluate_model(baseline_model, X_train, X_test, y_train, y_test)

    # Phase 2 — grid search
    best_params, _ = tune_hyperparameters(X_train, y_train)

    # Phase 3 — final tuned model
    tuned_model = train_tuned_model(X_train, y_train, X_test, y_test, best_params)
    tuned_results = evaluate_model(tuned_model, X_train, X_test, y_train, y_test)

    # SHAP + visualisations on the tuned model
    analyze_shap(tuned_model, vectorizer, X_test, n_samples=SHAP_SAMPLES)
    create_standard_visualizations(tuned_results)
    save_model_and_results(tuned_model, vectorizer, tuned_results, baseline_results, best_params)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\n  Baseline  — Accuracy: {baseline_results['test_accuracy']:.4f}"
          f"  AUC: {baseline_results['test_auc']:.4f}")
    print(f"  Tuned     — Accuracy: {tuned_results['test_accuracy']:.4f}"
          f"  AUC: {tuned_results['test_auc']:.4f}")
    print(f"  Best params: {best_params}")
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
