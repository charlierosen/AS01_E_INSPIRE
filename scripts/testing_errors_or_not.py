import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def compare_models_with_without_errors(df, base_features, error_features, target='DoR', n_splits=5, random_state=42):
    # Model parameters

    model_params = {
        'max_depth': 5,
        'max_features': 0.8,
        'max_samples': 0.7,
        'min_samples_leaf': 3,
        'min_samples_split': 8,
        'n_estimators': 30,
        'random_state': random_state
    }

    # Features with and without errors
    features_without_errors = base_features.copy()
    features_with_errors = base_features + error_features

    # Results dictionary
    results = {
        'without_errors': {
            'train_r2': [],
            'test_r2': [],
            'features': features_without_errors
        },
        'with_errors': {
            'train_r2': [],
            'test_r2': [],
            'features': features_with_errors
        }
    }

    # Set up K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Get target variable
    y = df[target]

    # Run cross-validation for both feature sets
    for model_type, features in [('without_errors', features_without_errors),
                                 ('with_errors', features_with_errors)]:
        print(f"\nRunning model {model_type} with {len(features)} features")

        # Get feature matrix
        X = df[features]

        # Store predictions for plotting
        all_true = np.zeros_like(y)
        all_pred = np.zeros_like(y)

        # Perform k-fold cross-validation
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            rf_model = RandomForestRegressor(**model_params)
            rf_model.fit(X_train_scaled, y_train)

            # Evaluate on training set
            y_train_pred = rf_model.predict(X_train_scaled)
            train_r2 = r2_score(y_train, y_train_pred)

            # Evaluate on test set
            y_test_pred = rf_model.predict(X_test_scaled)
            test_r2 = r2_score(y_test, y_test_pred)

            # Store R² scores
            results[model_type]['train_r2'].append(train_r2)
            results[model_type]['test_r2'].append(test_r2)

            # Store predictions for overall R²
            all_true[test_idx] = y_test
            all_pred[test_idx] = y_test_pred

            print(f"  Fold {fold}/{n_splits} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

        # Calculate overall test R²
        overall_r2 = r2_score(all_true, all_pred)

        # Calculate average R² scores and differences
        avg_train_r2 = np.mean(results[model_type]['train_r2'])
        avg_test_r2 = np.mean(results[model_type]['test_r2'])
        r2_diff = avg_train_r2 - avg_test_r2

        # Store aggregated metrics
        results[model_type]['avg_train_r2'] = avg_train_r2
        results[model_type]['avg_test_r2'] = avg_test_r2
        results[model_type]['r2_diff'] = r2_diff
        results[model_type]['overall_r2'] = overall_r2

        print(f"\n{model_type.replace('_', ' ').title()} Model Summary:")
        print(f"  Average Train R²: {avg_train_r2:.4f}")
        print(f"  Average Test R²: {avg_test_r2:.4f}")
        print(f"  Overfitting (Train-Test R² diff): {r2_diff:.4f}")
        print(f"  Overall Test R²: {overall_r2:.4f}")

    return results


def run_multiple_seeds(df, base_features, error_features, target='DoR', n_splits=5, n_seeds=5):

    # Seeds to use
    seeds = [42, 123, 456, 789, 101][:n_seeds]

    # Store results for each seed
    all_seed_results = []

    # Run for each seed
    for i, seed in enumerate(seeds):
        print(f"\n{'=' * 20} SEED {i + 1}/{n_seeds} (value: {seed}) {'=' * 20}")

        # Run comparison with this seed
        seed_results = compare_models_with_without_errors(
            df=df,
            base_features=base_features,
            error_features=error_features,
            target=target,
            n_splits=n_splits,
            random_state=seed
        )

        all_seed_results.append(seed_results)

    # Average results across seeds
    avg_results = {
        'without_errors': {
            'features': base_features,
            'avg_train_r2': np.mean([r['without_errors']['avg_train_r2'] for r in all_seed_results]),
            'avg_test_r2': np.mean([r['without_errors']['avg_test_r2'] for r in all_seed_results]),
            'r2_diff': np.mean([r['without_errors']['r2_diff'] for r in all_seed_results]),
            'overall_r2': np.mean([r['without_errors']['overall_r2'] for r in all_seed_results]),
            # Store individual seed values for standard deviation calculation
            'seed_train_r2': [r['without_errors']['avg_train_r2'] for r in all_seed_results],
            'seed_test_r2': [r['without_errors']['avg_test_r2'] for r in all_seed_results],
            'seed_r2_diff': [r['without_errors']['r2_diff'] for r in all_seed_results]
        },
        'with_errors': {
            'features': base_features + error_features,
            'avg_train_r2': np.mean([r['with_errors']['avg_train_r2'] for r in all_seed_results]),
            'avg_test_r2': np.mean([r['with_errors']['avg_test_r2'] for r in all_seed_results]),
            'r2_diff': np.mean([r['with_errors']['r2_diff'] for r in all_seed_results]),
            'overall_r2': np.mean([r['with_errors']['overall_r2'] for r in all_seed_results]),
            # Store individual seed values for standard deviation calculation
            'seed_train_r2': [r['with_errors']['avg_train_r2'] for r in all_seed_results],
            'seed_test_r2': [r['with_errors']['avg_test_r2'] for r in all_seed_results],
            'seed_r2_diff': [r['with_errors']['r2_diff'] for r in all_seed_results]
        }
    }

    # Calculate standard deviations
    for model_type in ['without_errors', 'with_errors']:
        avg_results[model_type]['std_train_r2'] = np.std(avg_results[model_type]['seed_train_r2'])
        avg_results[model_type]['std_test_r2'] = np.std(avg_results[model_type]['seed_test_r2'])
        avg_results[model_type]['std_r2_diff'] = np.std(avg_results[model_type]['seed_r2_diff'])

    # Print summary of averaged results
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS AVERAGED ACROSS ALL SEEDS")
    print("=" * 80)

    for model_type in ['without_errors', 'with_errors']:
        print(f"\n{model_type.replace('_', ' ').title()} Model:")
        print(
            f"  Average Train R²: {avg_results[model_type]['avg_train_r2']:.4f} ± {avg_results[model_type]['std_train_r2']:.4f}")
        print(
            f"  Average Test R²: {avg_results[model_type]['avg_test_r2']:.4f} ± {avg_results[model_type]['std_test_r2']:.4f}")
        print(
            f"  Overfitting (Train-Test R² diff): {avg_results[model_type]['r2_diff']:.4f} ± {avg_results[model_type]['std_r2_diff']:.4f}")
        print(f"  Overall Test R²: {avg_results[model_type]['overall_r2']:.4f}")

    return avg_results, all_seed_results


def visualize_model_comparison_with_error_bars(avg_results):

    plt.figure(figsize=(12, 8))

    # Set width of bars
    bar_width = 0.35

    # Set positions of bars on x-axis
    r1 = np.arange(len(avg_results.keys()))
    r2 = [x + bar_width for x in r1]

    # Extract data for plotting
    model_names = [m.replace('_', ' ').title() for m in avg_results.keys()]
    train_r2 = [avg_results[m]['avg_train_r2'] for m in avg_results.keys()]
    test_r2 = [avg_results[m]['avg_test_r2'] for m in avg_results.keys()]
    r2_diffs = [avg_results[m]['r2_diff'] for m in avg_results.keys()]

    # Extract standard deviations
    train_r2_std = [avg_results[m]['std_train_r2'] for m in avg_results.keys()]
    test_r2_std = [avg_results[m]['std_test_r2'] for m in avg_results.keys()]

    # Create bars with error bars
    plt.bar(r1, train_r2, width=bar_width, label='Train R²', color='skyblue', edgecolor='black', yerr=train_r2_std,
            capsize=5)
    plt.bar(r2, test_r2, width=bar_width, label='Test R²', color='lightgreen', edgecolor='black', yerr=test_r2_std,
            capsize=5)

    # Add overfitting labels with standard deviations
    for i, (diff, std) in enumerate(zip(r2_diffs, [avg_results[m]['std_r2_diff'] for m in avg_results.keys()])):
        plt.text(r1[i] + bar_width / 2, max(train_r2[i], test_r2[i]) + 0.04,
                 f'Δ = {diff:.4f} ± {std:.4f}', ha='center', va='bottom')

    # Add feature counts
    for i, model in enumerate(avg_results.keys()):
        plt.text(r1[i] + bar_width / 2, 0.05,
                 f'Features: {len(avg_results[model]["features"])}', ha='center', va='bottom', rotation=90)

    # Add labels and title
    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('Model Comparison: With vs. Without Error Features\n(Averaged across 5 seeds)', fontsize=14)

    # Set x-ticks
    plt.xticks([r + bar_width / 2 for r in range(len(model_names))], model_names, fontsize=12)

    # Add legend
    plt.legend(fontsize=12)

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Tight layout
    plt.tight_layout()

    return plt.gcf()


def analyze_decision_with_confidence(avg_results):

    without_diff = avg_results['without_errors']['r2_diff']
    with_diff = avg_results['with_errors']['r2_diff']
    without_test = avg_results['without_errors']['avg_test_r2']
    with_test = avg_results['with_errors']['avg_test_r2']

    # Get standard deviations
    without_diff_std = avg_results['without_errors']['std_r2_diff']
    with_diff_std = avg_results['with_errors']['std_r2_diff']
    without_test_std = avg_results['without_errors']['std_test_r2']
    with_test_std = avg_results['with_errors']['std_test_r2']

    diff_change = with_diff - without_diff
    test_change = with_test - without_test

    # Calculate if differences are statistically meaningful (using 1 std as threshold)
    diff_change_significant = abs(diff_change) > (without_diff_std + with_diff_std) / 2
    test_change_significant = test_change > (without_test_std + with_test_std) / 2

    recommendation = ""

    # Case 1: Higher overfitting with errors
    if with_diff > without_diff and diff_change_significant:
        recommendation = "RECOMMENDATION: Remove error features. The model with errors shows statistically higher overfitting."

    # Case 2: Similar overfitting levels (apply Occam's razor)
    elif not diff_change_significant:
        recommendation = "RECOMMENDATION: Remove error features based on Occam's razor. Both models show similar overfitting levels, so simpler is better."

    # Case 3: Lower overfitting with errors and significant quality improvement
    elif with_diff < without_diff and diff_change_significant and test_change_significant:
        recommendation = "RECOMMENDATION: Keep error features. The model with errors shows statistically lower overfitting and meaningful performance improvement."

    # Case 4: Lower overfitting with errors but no significant quality improvement
    elif with_diff < without_diff and diff_change_significant and not test_change_significant:
        recommendation = "RECOMMENDATION: Remove error features. Although the model with errors shows lower overfitting, it doesn't provide statistically significant quality improvement."

    # Default case
    else:
        recommendation = "RECOMMENDATION: Remove error features. No compelling reason to include them based on the given criteria."

    return recommendation


def main(file_path, base_features=None, error_features=None):

    # Default features if none provided
    if base_features is None:
        base_features = [
            'MgFe',
            '[M/H]_mean_mass',
            'velDisp_ppxf_res',
            'age_mean_mass',
        ]

    if error_features is None:
        error_features = [
            '[M/H]_err_mass',
            'velDisp_ppxf_err_res',
            'age_err_mass',
        ]

    # Read data
    print(f"Reading data from {file_path}")
    df = pd.read_csv(file_path)

    # Print dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"Base features: {base_features}")
    print(f"Error features: {error_features}")

    # Run comparison with multiple seeds
    avg_results, all_seed_results = run_multiple_seeds(
        df=df,
        base_features=base_features,
        error_features=error_features,
        target='DoR',
        n_splits=5,
        n_seeds=5
    )

    # Visualize results with error bars
    fig = visualize_model_comparison_with_error_bars(avg_results)

    # Get recommendation with confidence levels
    # recommendation = analyze_decision_with_confidence(avg_results)

    # Print recommendation
    print("\n" + "=" * 80)
    #print(recommendation)
    print("=" * 80)

    # Show plot
    plt.show()

    return avg_results, all_seed_results, fig


if __name__ == "__main__":
    # Example of how to use the code
    file_path = '../data/E-INSPIRE_I_master_catalogue.csv'

    # Define base features (without errors) and error features
    base_features = [
        'MgFe',
        '[M/H]_mean_mass',
        'velDisp_ppxf_res',
        'age_mean_mass',
    ]

    error_features = [
   #     '[M/H]_err_mass',
   #     'velDisp_ppxf_err_res',
        'age_err_mass',
    ]

    # Run test with multiple seeds
    avg_results, all_seed_results, fig = main(file_path, base_features, error_features)