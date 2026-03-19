import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data_prep import prepare_data

def perform_cv_classification(df, FEATURES, threshold, MODEL_PARAMS, n_splits=5):

    # Create binary target variable
    df['DoR_class'] = (df['DoR'] >= threshold).astype(int)

    # Extract features and target
    X = df[FEATURES]
    y = df['DoR_class']

    # Set up K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=MODEL_PARAMS['random_state'])

    # Initialize metrics lists
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    # SVC does not expose feature importances

    # Perform k-fold cross-validation
    for train_idx, test_idx in kf.split(X):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = SVC(**MODEL_PARAMS)
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

    # No feature importances for SVC
    feature_importances = None

    # Calculate class distribution
    class_dist = {
        'percent_high': (y == 1).mean() * 100,
        'percent_low': (y == 0).mean() * 100
    }

    # Build results dictionary
    cv_results = {
        'threshold': threshold,
        'accuracy': np.mean(accuracies),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1_scores),
        'feature_importances': feature_importances,
        'class_distribution': class_dist
    }

    return cv_results


def run_classification_analysis(thresholds, features=None, dataset_mode='mixed', dataset_label=None):

    # Nice display names for features
    feature_display_names = {
        'met': r'$\mathrm{[M/H]}$',
        'tau': r'$\mathrm{\tau_{\rm rel}}$',
        'met_err': r'$\Delta{\mathrm{[M/H]}}$',
        'lin_age_err': r'$\Delta{\mathrm{Age}}$',
        'logM': r'$\log(M/M_{\odot})$',
        'rad_kpc': r'$R \, \mathrm{(kpc)}$',
        'MgFe': r'$\mathrm{[Mg/Fe]}$',
        'vdisp': r'$\sigma \, \mathrm{(km/s)}$'
    }

    # Model parameters
    MODEL_PARAMS = {
        'C': 1.0,
        'gamma': 'scale',
        'kernel': 'rbf',
        'class_weight': 'balanced',
        'random_state': 42,
    }

    # Prepare data - only need train_df
    columns = ['vdisp', 'tau', 'MgFe', 'met_err', 'lin_age_err', 'met', 'rad_kpc', 'logM', 'DoR']
    mix_datasets = dataset_mode != 'domain_shift'
    label = dataset_label or ('Mixed' if mix_datasets else 'E-INSPIRE→INSPIRE')
    train_df, _ = prepare_data(columns, mix_datasets=mix_datasets)

    # Initialize results containers
    cv_results_all = []
    feature_importance_by_threshold = {}

    # Run analysis for each threshold
    for threshold in thresholds:
        print(f"Running classification analysis for threshold: {threshold} [{label}]")

        # Run cross-validation
        cv_result = perform_cv_classification(
            train_df,
            features,
            threshold,
            MODEL_PARAMS
        )
        cv_result['dataset'] = label
        cv_results_all.append(cv_result)

        # Store feature importances for this threshold
        feature_importance_by_threshold[threshold] = cv_result['feature_importances']

        # Print basic metrics
        print(f"Threshold: {threshold} [{label}]")
        print(f"Class Distribution: {cv_result['class_distribution']['percent_high']:.1f}% high, "
              f"{cv_result['class_distribution']['percent_low']:.1f}% low")
        print(f"CV Accuracy: {cv_result['accuracy']:.4f}")
        print(f"CV F1 Score: {cv_result['f1']:.4f}")
        print("-" * 50)

    # Convert results to DataFrame for easier analysis
    cv_results_df = pd.DataFrame(cv_results_all)

    # Create plots (saved with dataset-specific names)
    plot_classification_metrics(cv_results_df, label=label)
    plot_feature_importance_trends(feature_importance_by_threshold, feature_display_names, label=label)

    # Save results as CSV
    # cv_results_df.to_csv('classification_results/classification_metrics.csv', index=False)

    return cv_results_all


def plot_classification_metrics(cv_results_df, label=None, output_dir='outputs/paper_plots'):
    """
    Plot metrics for the classification analysis
    """
    # Set LaTeX style for plots
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "figure.dpi": 300,
        "font.size": 20,
    })

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Classification metrics
    thresholds = cv_results_df['threshold'].values
    ax1.plot(thresholds, cv_results_df['accuracy'], 'o-', label='Accuracy', linewidth=2, markersize=8)
    ax1.plot(thresholds, cv_results_df['precision'], 's-', label='Precision', linewidth=2, markersize=8)
    ax1.plot(thresholds, cv_results_df['recall'], '^-', label='Recall', linewidth=2, markersize=8)
    ax1.plot(thresholds, cv_results_df['f1'], 'D-', label='F1 Score', linewidth=2, markersize=8)

    ax1.set_xlabel(r'DoR Threshold', fontsize=20)
    ax1.set_ylabel('Metric Value', fontsize=20)
    ax1.set_title('Cross-Validation Classification Metrics', fontsize=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=20)
    ax1.set_xticks(thresholds)
    ax1.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax1.set_ylim(0, 1.05)

    # Plot 2: Class distribution
    high_percentages = [res['class_distribution']['percent_high'] for res in cv_results_df.to_dict('records')]
    low_percentages = [res['class_distribution']['percent_low'] for res in cv_results_df.to_dict('records')]

    x = np.arange(len(thresholds))
    width = 0.35

    ax2.bar(x - width / 2, high_percentages, width, label='High DoR class (\%)', color='firebrick', alpha=0.7)
    ax2.bar(x + width / 2, low_percentages, width, label='Low DoR class (\%)', color='steelblue', alpha=0.7)

    ax2.set_xlabel(r'DoR Threshold', fontsize=20)
    ax2.set_ylabel('Percentage (\%)', fontsize=20)
    ax2.set_title('Class Distribution', fontsize=20)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper center', fontsize=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax2.set_ylim(0, 100)

    plt.tight_layout()

    """# Save with dataset-aware name
    os.makedirs(output_dir, exist_ok=True)
    suffix = 'classification_metrics'
    if label:
        suffix += '_' + label.replace(' ', '_').replace('→', 'to')
    plt.savefig(os.path.join(output_dir, f'{suffix}.pdf'), bbox_inches='tight')
    plt.close()"""


def plot_feature_importance_trends(feature_importance_by_threshold, feature_display_names, label=None, output_dir='outputs/paper_plots'):
    """
    Plot how feature importance changes with different classification thresholds
    """
    # Skip plotting if we have no importances (e.g., SVC)
    if not feature_importance_by_threshold or all(v is None for v in feature_importance_by_threshold.values()):
        return
    # Set LaTeX style for plots
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "figure.dpi": 300,
        "font.size": 20,
    })

    # Convert the dictionary to a DataFrame for easier plotting
    data = []
    for threshold, importances in feature_importance_by_threshold.items():
        for feature, importance in importances.items():
            data.append({
                'Threshold': threshold,
                'Feature': feature,
                'Importance': importance
            })

    importance_df = pd.DataFrame(data)

    # Add display name column
    importance_df['Display_Name'] = importance_df['Feature'].map(
        lambda f: feature_display_names.get(f, f))

    # Create a figure for line plot
    plt.figure(figsize=(10, 8))

    # Plot feature importance trends
    for feature in importance_df['Feature'].unique():
        feature_data = importance_df[importance_df['Feature'] == feature]
        display_name = feature_data['Display_Name'].iloc[0]
        plt.plot(
            feature_data['Threshold'],
            feature_data['Importance'],
            'o-',
            label=display_name,
            linewidth=2,
            markersize=4
        )

    plt.xlabel(r'DoR Threshold', fontsize=20)
    plt.ylabel('Feature Importance', fontsize=20)
    #plt.title('Feature Importance vs. DoR Classification Threshold', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=20, bbox_to_anchor=(0.98, 0.98), borderaxespad=0.)
    plt.xticks(sorted(importance_df['Threshold'].unique()))
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    suffix = 'classification_features'
    if label:
        suffix += '_' + label.replace(' ', '_').replace('→', 'to')
    plt.savefig(os.path.join(output_dir, f'{suffix}.pdf'), bbox_inches='tight')
    plt.close()
