import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb

from src.data_prep import prepare_data

def test_model_on_INSPIRE(name, model_class, model_params, FEATURES, test_df, train_df, random_state=42):

    # Extract features and target
    X_train = train_df[FEATURES]
    y_train = train_df['DoR']
    X_test = test_df[FEATURES]
    y_test = test_df['DoR']
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    if 'random_state' in model_params and random_state is not None:
        model_params['random_state'] = random_state
    
    model = model_class(**model_params)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on test data
    y_pred = model.predict(X_test_scaled)
    
    # Calculate test metrics
    test_r2 = r2_score(y_test, y_pred)
    
    # Now calculate cross-validation metrics on training data
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_r2_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        # Split and scale data
        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        scaler_cv = StandardScaler()
        X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
        X_val_cv_scaled = scaler_cv.transform(X_val_cv)
        
        # Create and train model
        cv_model = model_class(**model_params)
        cv_model.fit(X_train_cv_scaled, y_train_cv)
        
        # Predict and score
        y_val_pred = cv_model.predict(X_val_cv_scaled)
        cv_r2 = r2_score(y_val_cv, y_val_pred)
        cv_r2_scores.append(cv_r2)
    
    # Average CV R² score
    train_avg_r2 = np.mean(cv_r2_scores)
    
    return test_r2, train_avg_r2


def run_model_comparison(random_seeds=None, show_inline=False, dataset_mode='mixed', dataset_label=None):

    if random_seeds is None:
        random_seeds = [42, 123, 456, 789, 1010]
    
    print(f"\nRunning model comparison with {len(random_seeds)} random seeds: {random_seeds}...")
    
    # Prepare the data
    columns = ['vdisp','tau','MgFe', 'met_err', 'lin_age_err','met','rad_kpc','logM','DoR']
    mix_datasets = dataset_mode != 'domain_shift'

    label = dataset_label or ('Mixed' if mix_datasets else 'E-INSPIRE->INSPIRE')
    train_df, test_df = prepare_data(columns, mix_datasets=mix_datasets)
    
    feature_sets = {
    'Stel. pop.': ['met', 'tau'],
    'Stel. pop. with errors': ['met', 'tau', 'met_err', 'lin_age_err'],
    'Stel. pop. and kinematics': ['met', 'tau', 'met_err', 'lin_age_err', 'vdisp'],
    r'Stel. pop. and $\alpha$-abundance': ['met', 'tau', 'met_err', 'lin_age_err', 'MgFe'],
    r'Stel. pop., $\alpha$-abundance and kinematics': ['met', 'tau', 'met_err', 'lin_age_err', 'MgFe', 'vdisp'],
    'Stel. pop. and structural': ['met', 'tau', 'met_err', 'lin_age_err', 'logM', 'rad_kpc'],
    'Stel. pop., structural and kinematics': ['met', 'tau', 'met_err', 'lin_age_err', 'logM', 'rad_kpc', 'vdisp'],
    'Complete Set': ['met', 'tau', 'met_err', 'lin_age_err', 'logM', 'rad_kpc', 'MgFe', 'vdisp'],
    # 'New Chiara 1': ['logM', 'rad_kpc', 'vdisp'],
    # 'New Chiara 2': ['tau', 'logM', 'rad_kpc', 'vdisp', 'lin_age_err'],
    }
    
    # Define the models to compare
    models = {
        'RandomForest': {
            'class': RandomForestRegressor,
            'params': {
                'max_depth': 8,
                'max_features': 0.8,
                'max_samples': 0.7,
                'min_samples_leaf': 3,
                'min_samples_split': 5,
                'n_estimators': 50,
                'random_state': 42
            },
            'color': 'blue'
        },
        'XGBoost': {
            'class': xgb.XGBRegressor,
            'params': {
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 50,
                'min_child_weight': 3,
                'subsample': 0.7,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'color': 'red'
        },
        'Ridge': {
            'class': Ridge,
            'params': {
                'alpha': 1.0,
                'max_iter': 2000,
                'tol': 1e-4,
                'random_state': 42
            },
            'color': 'orange'
        },
        'SVR': {
            'class': SVR,
            'params': {
                'C': 5.0,
                'epsilon': 0.03,
                'kernel': 'poly',
                'gamma': 0.01,
                'degree': 3,
                'coef0': 0.5,
            },
            'color': 'brown'
        }
    }
    
    # Store results for each seed
    all_results = []
    
    # Run for each random seed
    for seed_idx, seed in enumerate(random_seeds):
        print(f"\nRandom seed {seed_idx+1}/{len(random_seeds)}: {seed}")
        
        # Results for this seed
        seed_results = {
            'model_id': [],
            'algorithm': [],
            'test_r2': [],
            'train_r2': [],
            'features': [],
            'seed': [],
            'dataset': []
        }
        
        # Run each model on each feature set
        for model_name, model_info in models.items():
            print(f"  Running {model_name}...")
            
            for model_id, features in feature_sets.items():
                start_time = time.time()
                
                # Run the model
                test_r2, train_r2 = test_model_on_INSPIRE(
                    model_id, 
                    model_info['class'], 
                    model_info['params'], 
                    features, 
                    test_df, 
                    train_df,
                    random_state=seed
                )
                
                # Store results
                seed_results['model_id'].append(model_id)
                seed_results['algorithm'].append(model_name)
                seed_results['test_r2'].append(test_r2)
                seed_results['train_r2'].append(train_r2)
                seed_results['features'].append(len(features))
                seed_results['seed'].append(seed)
                seed_results['dataset'].append(label)
                
                elapsed = time.time() - start_time
                print(f"    {model_id} ({len(features)} features): Test R²: {test_r2:.4f}, Train CV R²: {train_r2:.4f} ({elapsed:.2f}s)")
        
        # Convert results to DataFrame and add to all results
        all_results.append(pd.DataFrame(seed_results))
    
    # Combine all results
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Aggregate results across seeds
    agg_results = results_df.groupby(['dataset', 'model_id', 'algorithm', 'features']).agg({
        'test_r2': ['mean', 'std'],
        'train_r2': ['mean', 'std']
    }).reset_index()
    
    # Flatten the hierarchical column names
    agg_results.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in agg_results.columns.values]
    
    # Create the plots as separate figures
    create_performance_plots(agg_results, show_inline=show_inline, model_order=list(feature_sets.keys()))

    return results_df, agg_results


def create_performance_plots(results_df, show_inline=False, model_order=None):
    """
    Create separate plots for test and training performance
    """
    # Lazy import to avoid IPython dependency in pure script runs
    display = None
    if show_inline:
        try:
            from IPython.display import display as ipy_display
            display = ipy_display
        except Exception:
            display = None
    # Set plot style
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "figure.dpi": 300,
        "font.size": 30,
    })
    
    # Use provided order, or fall back to order of appearance in results_df
    if model_order is not None:
        ordered_model_ids = [m for m in model_order if m in results_df['model_id'].unique()]
    else:
        ordered_model_ids = list(results_df['model_id'].unique())
    
    output_dir = Path(__file__).resolve().parents[1] / 'outputs' / 'paper_plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    def slug(label: str) -> str:
        return label.replace(' ', '_').replace('→', 'to').replace('/', '-')

    for dataset_label in results_df['dataset'].unique():
        subset = results_df[results_df['dataset'] == dataset_label]
        suffix = slug(dataset_label)

        # Create and save test performance plot
        fig, ax = plt.subplots(figsize=(14, 8))
        create_single_performance_plot(
            subset,
            ax,
            ordered_model_ids,
            metric='test_r2_mean',
            err_metric='test_r2_std',
            title='',
            ylabel=r'Test $R^2$'
        )
        plt.tight_layout()
        plt.savefig(output_dir / f'model_comparison_test_{suffix}.pdf', bbox_inches='tight')
        if show_inline and display:
            display(fig)
        plt.close(fig)
        
        """
        # No need to save the train one really...
        # Create and save training performance plot
        fig, ax = plt.subplots(figsize=(14, 8))
        create_single_performance_plot(
            subset,
            ax,
            ordered_model_ids,
            metric='train_r2_mean',
            err_metric='train_r2_std',
            title='',
            ylabel=f'Train $R^2$ ({dataset_label})'
        )
        plt.tight_layout()
        plt.savefig(output_dir / f'model_comparison_train_{suffix}.pdf', bbox_inches='tight')
        if show_inline and display:
            display(fig)
        plt.close(fig)"""
    
    print("Created separate test and training performance plots")


def create_single_performance_plot(results_df, ax, ordered_model_ids, metric, err_metric=None, title=None, ylabel=None):
    """
    Helper function to create a performance plot on a given axis
    """
    # Get unique algorithms
    algorithms = results_df['algorithm'].unique()
    
    # Define colors and markers
    colors = {
        'RandomForest': 'blue',
        'XGBoost': 'red',
        'Ridge': 'orange',
        'SVR': 'green'
    }
    
    markers = ['o', 's', '^', 'D']
    
    # Map model_id to position on x-axis (using the fixed ordering)
    x_positions = {model_id: i for i, model_id in enumerate(ordered_model_ids)}
    
    # Plot each algorithm
    for i, algorithm in enumerate(algorithms):
        # Create a properly ordered dataset for this algorithm
        algo_data = []
        for model_id in ordered_model_ids:
            model_data = results_df[(results_df['algorithm'] == algorithm) & 
                                   (results_df['model_id'] == model_id)]
            if not model_data.empty:
                algo_data.append({
                    'model_id': model_id,
                    'x_pos': x_positions[model_id],
                    'y_val': model_data[metric].values[0],
                    'err_val': model_data[err_metric].values[0] if err_metric else None
                })
        
        # Convert to DataFrame for easier handling
        algo_df = pd.DataFrame(algo_data)
        
        # Extract x and y values (now properly ordered)
        x_values = algo_df['x_pos'].values
        y_values = algo_df['y_val'].values
        
        # Error bars if provided
        if err_metric:
            err_values = algo_df['err_val'].values
            ax.errorbar(
                x_values,
                y_values,
                yerr=err_values,
                fmt='none',
                ecolor=colors.get(algorithm, f'C{i}'),
                alpha=0.3,
                capsize=3,
                elinewidth=4,
                capthick=4,
            )
        
        # Plot the line and points
        ax.plot(
            x_values, y_values, 
            marker=markers[i % len(markers)], 
            linestyle='-', 
            linewidth=3,
            markersize=8,
            color=colors.get(algorithm, f'C{i}'),
            label=algorithm
        )
    
    # CHANGE 1: Use simple numbers 1-8 instead of model names
    feature_counts = []
    for i in range(len(ordered_model_ids)):
        feature_counts.append(f"{i+1}")
    
    # CHANGE 2: Set x-axis label to "Model #" instead of "Model ID"
    ax.set_xlabel('Model \#', fontsize=30)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=30)
    
    # Set x-axis ticks and labels
    ax.set_xticks(range(len(ordered_model_ids)))
    ax.set_xticklabels(feature_counts, rotation=0, ha='center', fontsize=26)  # Changed rotation and alignment too
    
    # Add grid and legend
    ax.legend(fontsize=22, loc='lower right', ncol=2)
    
    # Set y-axis limits to maximize visibility of differences
    min_r2 = max(0, results_df[metric].min() - results_df[err_metric].max() - 0.05) if err_metric else max(0, results_df[metric].min() - 0.05)
    max_r2 = min(1, results_df[metric].max() + results_df[err_metric].max() + 0.05) if err_metric else min(1, results_df[metric].max() + 0.05)
    # ax.set_ylim(min_r2, max_r2)

    ax.set_ylim(0.56, 0.86)
    #     ax.set_ylim(0.68, 0.87)
    yticks = [0.6,0.65, 0.7, 0.75, 0.8, 0.85]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.2f}" for y in yticks])
