import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Global stores mirror the original notebook state
results_summary = pd.DataFrame(columns=['name', 'features', 'test_r2', 'train_avg_r2', 'feature_importances'])

# Format: {name: {ID: {seed: prediction}}}
test_predictions = {}
# Format: {name: {index: {seed: prediction}}}
train_predictions = {}

# Optional references to the current datasets (set by the caller)
train_df = None
test_df = None

# ---- Functions copied from the original notebook ----
def test_regression_on_INSPIRE(name, FEATURES,test_df, train_df,MODEL_PARAMS, plotting=False, random_state=1):
    
    # Extract features and target
    X_train = train_df[FEATURES]
    y_train = train_df['DoR']
    X_test = test_df[FEATURES]
    y_test = test_df['DoR']
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    # print("Training the model...")
    rf_model = RandomForestRegressor(**MODEL_PARAMS)
    rf_model.fit(X_train_scaled, y_train)
    
        # Get feature importances and store as dictionary
    feature_importance_dict = dict(zip(FEATURES, rf_model.feature_importances_))
    
    
    # Make predictions on test data
    # print("Making predictions on test data...")
    y_pred = rf_model.predict(X_test_scaled)

    if name not in test_predictions:
        test_predictions[name] = {}
    for idx, id_val in enumerate(test_df['ID']):
        if id_val not in test_predictions[name]:
            test_predictions[name][id_val] = {}
        test_predictions[name][id_val][random_state] = y_pred[idx]
    
    # Calculate residuals
    residuals = y_test - y_pred
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # print(f"Test R²: {r2:.4f}")
    # print(f"Test RMSE: {rmse:.4f}")
    # print(f"Test MAE: {mae:.4f}")
    small_boundary = 0.35
    large_boundary = 0.6
    small_mask = y_pred <= small_boundary
    mid_mask = (y_pred > small_boundary) & (y_pred < large_boundary)
    large_mask = y_pred >= large_boundary
        
    small_rmse = np.sqrt(mean_squared_error(y_test[small_mask], y_pred[small_mask])) if np.any(small_mask) else np.nan
    mid_rmse = np.sqrt(mean_squared_error(y_test[mid_mask], y_pred[mid_mask])) if np.any(mid_mask) else np.nan
    large_rmse = np.sqrt(mean_squared_error(y_test[large_mask], y_pred[large_mask])) if np.any(large_mask) else np.nan
        
    
    if plotting:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.05})
        
        # Plot settings
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Computer Modern",
            "figure.dpi": 300,
            "font.size": 20,
        })
        
        SNR_values = test_df['SNR_MEAN']

        # ax1.scatter(y_pred, y_test, alpha=0.7, s=35, color='royalblue', edgecolors='k', linewidths=0.5)
        scatter = ax1.scatter(y_pred, y_test, alpha=0.7, s=35, c=SNR_values, 
                         cmap='viridis', edgecolors='k', linewidths=0.5)
        
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        buffer = (max_val - min_val) * 0.02
        plot_min = min_val - buffer
        plot_max = max_val + buffer
        
        ax1.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', 
                label='Perfect prediction', linewidth=1.5)
    
        
        ax1.set_ylabel('True DoR', fontsize=20)
        ax1.set_xlim(plot_min, plot_max)
        ax1.set_ylim(plot_min, plot_max)

        
        # Add textbox with performance metrics to top plot
        props = dict(boxstyle='square', facecolor='white', alpha=0.8, edgecolor='lightgray')
        textstr = '\n'.join((
            r'$\mathrm{R}^2 = %.4f$' % (r2,),
            r'$\mathrm{RMSE} = %.4f$' % (rmse,),
        ))
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=20,
                verticalalignment='top', bbox=props)
        
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='both', direction='in', labelsize=14)
        ax1.minorticks_on()
        for spine in ax1.spines.values():
            spine.set_color('lightgray')
        
        # ax2.scatter(y_pred, residuals, alpha=0.7, s=35, color='royalblue', edgecolors='k', linewidths=0.5)
        ax2.scatter(y_pred, residuals, alpha=0.7, s=35, c=SNR_values, 
                    cmap='viridis', edgecolors='k', linewidths=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
        
        
        ax2.set_xlabel('Predicted DoR', fontsize=20)
        ax2.set_ylabel('Residuals (True - Predicted)', fontsize=20)
        
                
        cbar = fig.colorbar(scatter, ax=ax2)
        cbar.set_label('SNR', rotation=270, labelpad=15, fontsize=20)
        
        #cbar = fig.colorbar(scatter, ax=ax2, orientation='horizontal', pad=0.15)
        #cbar.set_label('SNR', fontsize=20)
        
        # Add textbox with region-specific RMSE to bottom plot
        props = dict(boxstyle='square', facecolor='white', alpha=0.8, edgecolor='lightgray')
        textstr = '\n'.join((
            r'$\mathrm{Overall\ RMSE} = %.4f$' % (rmse,),
            r'$\mathrm{RMSE\ (DoR \leq %.2f)} = %.4f$' % (small_boundary, small_rmse),
            r'$\mathrm{RMSE\ (%.2f < DoR < %.2f)} = %.4f$' % (small_boundary, large_boundary, mid_rmse),
            r'$\mathrm{RMSE\ (DoR \geq %.2f)} = %.4f$' % (large_boundary, large_rmse)
        ))
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=20,
                verticalalignment='top', bbox=props)
        
        # Add grid and styling to bottom plot
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='both', direction='in', labelsize=14)
        ax2.minorticks_on()
        for spine in ax2.spines.values():
            spine.set_color('lightgray')
        
        plt.tight_layout()
        plt.savefig('tests/'+name+'_test.pdf', bbox_inches='tight')
        # plt.show()
    
    # print(f"\nRegion-specific RMSE:")
    # print(f"RMSE (DoR ≤ {small_boundary}): {small_rmse:.4f}")
    # print(f"RMSE ({small_boundary} < DoR < {large_boundary}): {mid_rmse:.4f}")
    # print(f"RMSE (DoR ≥ {large_boundary}): {large_rmse:.4f}")
    
    inspection_df = pd.DataFrame({
    'Index': test_df.index,
    'True_DoR': y_test,
    'Predicted_DoR': y_pred,
    'ID': test_df['ID']
    })

    inspection_df['absolute_residual'] = abs(inspection_df['True_DoR'] - inspection_df['Predicted_DoR'])
    inspection_df = inspection_df.sort_values('absolute_residual', ascending=False)
    #print("Largest absolute residuals:")
    #print(inspection_df.head(5)) # len(inspection_df)))

    global results_summary
    results_summary = pd.concat([results_summary, pd.DataFrame({
        'name': [name],
        'features': [', '.join(FEATURES)],
        'test_r2': [r2],
        'train_avg_r2': [None],  # Will be filled in by visualize_kfold_predictions_training
        'feature_importances': [feature_importance_dict],
        'random_seed': [random_state]  # Add random_state to track different runs
    })], ignore_index=True)
    
    visualize_kfold_predictions_training(train_df, FEATURES, name, model_params=MODEL_PARAMS, plotting=plotting)


def visualize_kfold_predictions_training(df, features, name, target='DoR', n_splits=5, model_params=None, 
                                 small_boundary=0.35, large_boundary=0.6, plotting=False):
    # print(model_params)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "figure.dpi": 300,
        "font.size": 20,
    })
    
    X = df[features]
    y = df[target]
    
    # Set up K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=model_params['random_state'])
    
    # Initialize arrays to store all predictions
    all_true = np.zeros_like(y)
    all_pred = np.zeros_like(y)
    all_residuals = np.zeros_like(y)
    fold_indices = np.zeros_like(y, dtype=int)  # Track which fold each point belongs to
    fold_results = {
        'r2_scores': [],
        'indices': [],
        'predictions': [],
        'true_values': [],
    }
    
    # Define colors for different folds
    fold_colors = plt.cm.tab10(np.linspace(0, 1, n_splits))
    
    # Perform k-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        # Split and scale data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model and make predictions
        rf_model = RandomForestRegressor(**model_params)
        rf_model.fit(X_train_scaled, y_train)
        y_pred = rf_model.predict(X_test_scaled)
        
        if name not in train_predictions:
            train_predictions[name] = {}
        actual_indices = df.index[test_idx]
        for idx, pred_val in zip(actual_indices, y_pred):
            if idx not in train_predictions[name]:
                train_predictions[name][idx] = {}
            train_predictions[name][idx][model_params['random_state']] = pred_val
        
        # Store results for this fold
        r2 = r2_score(y_test, y_pred)
        fold_results['r2_scores'].append(r2)
        fold_results['indices'].append(test_idx)
        fold_results['predictions'].append(y_pred)
        fold_results['true_values'].append(y_test.values)
        
        # Store predictions and residuals at the correct indices
        all_true[test_idx] = y_test
        all_pred[test_idx] = y_pred
        all_residuals[test_idx] = y_test - y_pred
        fold_indices[test_idx] = fold  # Track which fold this point belongs to
        
        # print(f"Fold {fold}/{n_splits} - R²: {r2:.4f}")
    
    # Calculate overall R² for all folds combined
    overall_r2 = r2_score(all_true, all_pred)
    mean_fold_r2 = np.mean(fold_results['r2_scores'])
    std_fold_r2 = np.std(fold_results['r2_scores'])
    
    mask = (results_summary['name'] == name) & (results_summary['random_seed'] == model_params['random_state'])
    results_summary.loc[mask, 'train_avg_r2'] = mean_fold_r2
    
    # Calculate overall RMSE and MAE
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    mae = mean_absolute_error(all_true, all_pred)
    
    # print(f"\nCross-validation results:")
    # print(f"Mean fold R²: {mean_fold_r2:.4f} (±{std_fold_r2:.4f})")
    # print(f"Overall R² (all predictions): {overall_r2:.4f}")
    # print(f"Overall RMSE: {rmse:.4f}")
    # print(f"Overall MAE: {mae:.4f}")
    
    if plotting:
        # Create figure with two subplots sharing x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.05})
        
        # Find plot limits
        min_val = min(min(all_true), min(all_pred))
        max_val = max(max(all_true), max(all_pred))
        buffer = (max_val - min_val) * 0.02
        plot_min = min_val - buffer
        plot_max = max_val + buffer
        
        # Top subplot: True vs Predicted
        # Plot each fold with different colors - x=predicted, y=true
        for fold in range(1, n_splits + 1):
            mask = fold_indices == fold
            ax1.scatter(all_pred[mask], all_true[mask], alpha=0.7, s=35,
                       color=fold_colors[fold - 1], edgecolors='k', linewidths=0.5,
                       label=f'Fold {fold} (R²: {fold_results["r2_scores"][fold - 1]:.4f})')
        
        # Add perfect prediction line
        ax1.plot([plot_min, plot_max], [plot_min, plot_max], 'r--',
                 label='Perfect prediction', linewidth=1.5)
        
        # Customize top plot
        ax1.set_ylabel('True DoR', fontsize=20)
        ax1.set_xlim(plot_min, plot_max)
        ax1.set_ylim(plot_min, plot_max)
        
        # Add textbox with performance metrics for top plot
        props = dict(boxstyle='square', facecolor='white', alpha=0.8, edgecolor='lightgray')
        textstr = '\n'.join((
            r'$\mathrm{Mean}\ R^2 = %.4f\ (\pm%.4f)$' % (mean_fold_r2, std_fold_r2),
        ))
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=20,
                 verticalalignment='top', bbox=props)
        
        ax1.legend(loc='lower right', fontsize=20)
        
        # Add grid and styling to top plot
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='both', direction='in', labelsize=14)
        ax1.minorticks_on()
        for spine in ax1.spines.values():
            spine.set_color('lightgray')
        
        # Bottom subplot: Residuals vs Predicted
        for fold in range(1, n_splits + 1):
            mask = fold_indices == fold
            ax2.scatter(all_pred[mask], all_residuals[mask], alpha=0.7, s=35,
                       color=fold_colors[fold - 1], edgecolors='k', linewidths=0.5,
                       label=f'Fold {fold}')
        
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
        
        # Calculate region-specific RMSE by predicted DoR
        small_mask = all_pred <= small_boundary
        mid_mask = (all_pred > small_boundary) & (all_pred < large_boundary)
        large_mask = all_pred >= large_boundary
        
        small_rmse = np.sqrt(mean_squared_error(all_true[small_mask], all_pred[small_mask])) if np.any(small_mask) else np.nan
        mid_rmse = np.sqrt(mean_squared_error(all_true[mid_mask], all_pred[mid_mask])) if np.any(mid_mask) else np.nan
        large_rmse = np.sqrt(mean_squared_error(all_true[large_mask], all_pred[large_mask])) if np.any(large_mask) else np.nan
        
        # Customize bottom plot
        ax2.set_xlabel('Predicted DoR', fontsize=20)
        ax2.set_ylabel('Residuals (True - Predicted)', fontsize=20)
        
        # Add region-specific metrics in a textbox for bottom plot
        props = dict(boxstyle='square', facecolor='white', alpha=0.8, edgecolor='lightgray')
        textstr = '\n'.join((
            r'$\mathrm{Overall\ RMSE} = %.4f$' % (rmse,),
            r'$\mathrm{RMSE\ (DoR \leq %.2f)} = %.4f$' % (small_boundary, small_rmse),
            r'$\mathrm{RMSE\ (%.2f < DoR < %.2f)} = %.4f$' % (small_boundary, large_boundary, mid_rmse),
            r'$\mathrm{RMSE\ (DoR \geq %.2f)} = %.4f$' % (large_boundary, large_rmse)
        ))
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=20,
                verticalalignment='top', bbox=props)
        
        # Add grid and styling to bottom plot
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='both', direction='in', labelsize=14)
        ax2.minorticks_on()
        for spine in ax2.spines.values():
            spine.set_color('lightgray')
        
        plt.tight_layout()
        plt.savefig('tests/'+name+'_train_cv.pdf', bbox_inches='tight')
        # plt.show()
    
        return fig


def visualize_ensemble_predictions_combined_residuals_DO_NOT_USE(model_name):
    
    top = 0.9
    bottom = 0
    # Check if predictions exist for this model
    if model_name not in test_predictions:
        print(f"No test predictions found for model: {model_name}")
        return
    
    if model_name not in train_predictions:
        print(f"No train predictions found for model: {model_name}")
        return
    
    # Set plotting style
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "figure.dpi": 300,
        "font.size": 26,
    })
    
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(12, 12))
    
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], 
                          hspace=0.0, wspace=0.0)
    
    # Create top-left subplot for train (swapped from original)
    ax_train = fig.add_subplot(gs[0, 0])
    
    # Top-right for test, shares y-axis with top-left
    ax_test = fig.add_subplot(gs[0, 1], sharey=ax_train)
    
    # Bottom-left for train residuals, shares x with top-left
    ax_train_res = fig.add_subplot(gs[1, 0], sharex=ax_train) # , sharey=ax_train)
    
    # Bottom-right for test residuals, shares x with top-right and y with bottom-left
    ax_test_res = fig.add_subplot(gs[1, 1], sharex=ax_test, sharey=ax_train_res)

    plt.setp(ax_test.get_yticklabels(), visible=False)        # top right y-ticks
    plt.setp(ax_test.get_xticklabels(), visible=False)        # top right x-ticks
    plt.setp(ax_train.get_xticklabels(), visible=False)       # top left x-ticks
    plt.setp(ax_test_res.get_yticklabels(), visible=False)    # bottom right y-ticks

    # Process and plot each dataset - train first, then test (swapped order)
    datasets = [
        {"name": "train", "ax": ax_train, "ax_res": ax_train_res, 
         "predictions": train_predictions, "df": train_df, "id_field": None},
        {"name": "test", "ax": ax_test, "ax_res": ax_test_res, 
         "predictions": test_predictions, "df": test_df, "id_field": "ID"}
    ]
    
    all_snr_values = []  # Collect all SNR values for shared colorbar
    scatter_artists = []  # Save scatter artists for colorbar
    
    for dataset in datasets:
        # Get data points
        predictions_dict = dataset["predictions"]
        df = dataset["df"]
        data_points = list(predictions_dict[model_name].keys())
        
        # Create mappings from data points to true values and SNR
        if dataset["id_field"] is not None:  # Test data
            point_to_true = {id_val: df.loc[df[dataset["id_field"]] == id_val, 'DoR'].values[0] 
                             for id_val in data_points}
            point_to_snr = {id_val: df.loc[df[dataset["id_field"]] == id_val, 'SNR'].values[0] 
                            for id_val in data_points}
        else:  # Train data
            point_to_true = {idx: df.loc[idx, 'DoR'] for idx in data_points}
            point_to_snr = {idx: df.loc[idx, 'SNR'] for idx in data_points}
        
        # Calculate average and std of predictions
        y_true = []
        y_pred_avg = []
        y_pred_std = []
        snr_values = []
        
        for id_val in data_points:
            predictions = list(predictions_dict[model_name][id_val].values())
            if predictions and id_val in point_to_true and id_val in point_to_snr:
                y_true.append(point_to_true[id_val])
                y_pred_avg.append(np.mean(predictions))
                y_pred_std.append(np.std(predictions))
                snr_values.append(point_to_snr[id_val])
        
        # Convert to numpy arrays for calculations
        y_true = np.array(y_true)
        y_pred_avg = np.array(y_pred_avg)
        y_pred_std = np.array(y_pred_std)
        snr_values = np.array(snr_values)
        
        # Add to overall SNR values for colorbar
        all_snr_values.extend(snr_values)
        
        # Remove NaN values
        valid_indices = ~np.isnan(y_pred_avg) & ~np.isnan(snr_values)
        if np.sum(valid_indices) < 10:  # Require at least 10 valid points
            print(f"Not enough valid predictions for {dataset['name']} data in model: {model_name}")
            continue
        
        y_true = y_true[valid_indices]
        y_pred_avg = y_pred_avg[valid_indices]
        y_pred_std = y_pred_std[valid_indices]
        snr_values = snr_values[valid_indices]
        
        """seed_r2_scores = {}
        for id_val in data_points:
            if id_val in point_to_true:
                true_val = point_to_true[id_val]
                # Group predictions by seed
                for seed, pred in predictions_dict[model_name][id_val].items():
                    if seed not in seed_r2_scores:
                        seed_r2_scores[seed] = {'true': [], 'pred': []}
                    seed_r2_scores[seed]['true'].append(true_val)
                    seed_r2_scores[seed]['pred'].append(pred)
        
        # Calculate R² for each seed, then average
        individual_r2_scores = []
        for seed, values in seed_r2_scores.items():
            if len(values['true']) > 1:  # Need at least 2 points to calculate R²
                seed_r2 = r2_score(values['true'], values['pred'])
                individual_r2_scores.append(seed_r2)
        
        r2 = np.mean(individual_r2_scores) if individual_r2_scores else 0
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_avg))"""
        
        r2 = r2_score(y_true, y_pred_avg)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_avg))
        
        # Plot True vs Predicted with SNR coloring
        ax = dataset["ax"]
        # scatter = ax.scatter(y_pred_avg, y_true, c=snr_values, cmap='viridis', s=50, alpha=0.7, edgecolors='none')
        
        low = 20
        high = 70
        snr_categories = np.zeros_like(snr_values, dtype=int)
        snr_categories[(snr_values >= low) & (snr_values < high)] = 1  # Medium SNR
        snr_categories[snr_values >= high] = 2  # High SNR
        
        category_colors = ['#d53e4f', '#fee08b','#3288bd']  # Blue, Yellow, Red for Low, Medium, High
        point_colors = [category_colors[cat] for cat in snr_categories]
        
        # Replace the scatter plot lines with this:
        scatter = ax.scatter(y_pred_avg, y_true, c=point_colors, s=50, alpha=0.7, edgecolors='none')
        
        scatter_artists.append(scatter)
        
        # Draw error bars using errorbar function with no markers
        ax.errorbar(y_pred_avg, y_true, xerr=y_pred_std, fmt='none', ecolor='#000000', 
                   elinewidth=0.2, capsize=0)
        
        ax.set_xlim(bottom, top)
        ax.set_ylim(bottom, top)
        
        # Create consistent tick marks with 0.1 increments
        ticks = np.arange(0.1, 1, 0.2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f'{t:.1f}' for t in ticks])
        ax.set_yticklabels([f'{t:.1f}' for t in ticks])
        
        # Add perfect prediction line
        ax.plot([0.0, 1], [0.0, 1], 'r--', 
               label='Perfect prediction', linewidth=1.5)
        
        # Add annotations
        props = dict(boxstyle='square', facecolor='white', alpha=0.8, edgecolor='lightgray')
        
        textstr = '\n'.join((
            r'$\mathrm{%s}$' % (dataset["name"].capitalize(),),
            r'$\mathrm{R}^2 = %.4f$' % (r2,),
            r'$\mathrm{RMSE} = %.4f$' % (rmse,),
        ))
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=20,
               verticalalignment='top', bbox=props)
        
        # Style plot
        ax.set_ylabel('True DoR', fontsize=20)
        if dataset["name"] == "test":
            ax.set_ylabel('')  # Remove label for test (right side)
        
        # ax.set_xlabel('Predicted DoR', fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
        ax.minorticks_on()
        
        # Plot residuals
        ax_res = dataset["ax_res"]
        residuals = y_true - y_pred_avg
        #scatter_res = ax_res.scatter(y_pred_avg, residuals, c=snr_values, cmap='viridis',s=50, alpha=0.7, edgecolors='none')
        scatter_res = ax_res.scatter(y_pred_avg, residuals, c=point_colors, s=50, alpha=0.7, edgecolors='none')

        # Draw error bars for residuals
        ax_res.errorbar(y_pred_avg, residuals, xerr=y_pred_std, fmt='none', ecolor='#000000', 
                       elinewidth=0.2, capsize=0)
        
        ax_res.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
        
        # Set limits for residuals
        residual_max = max(abs(np.min(residuals)), abs(np.max(residuals)))
        residual_limit = min(0.5, np.ceil(residual_max * 10) / 10)  # Round up to nearest 0.1
        ax_res.set_ylim(-residual_limit, residual_limit)
        
        # Create even tick marks for residuals
        residual_ticks = np.arange(-residual_limit, residual_limit + 0.1, 0.1).round(1)
        ax_res.set_yticks(residual_ticks)
        ax_res.set_yticklabels([f'{t:.1f}' for t in residual_ticks])
        
        # Keep consistent x-ticks
        ax_res.set_xlim(bottom, top)
        ax_res.set_xticks(ticks)
        ax_res.set_xticklabels([f'{t:.1f}' for t in ticks])
        
        # Style residual plot
        ax_res.set_xlabel('Predicted DoR', fontsize=20)
        ax_res.set_ylabel('Residuals', fontsize=20)
        if dataset["name"] == "test":
            ax_res.set_ylabel('')  # Remove label for test (right side)
        
        ax_res.grid(True, alpha=0.3)
        ax_res.tick_params(axis='both', which='both', direction='in', labelsize=12)
        ax_res.minorticks_on()
        
    yticks = ax_train_res.get_yticks()
    yticklabels = ax_train_res.get_yticklabels()
    if len(yticklabels) > 0:
        yticklabels[-1].set_visible(False)  # Hide the last (rightmost) tick label
        
    ax_test_res.set_xlabel('Predicted DoR', fontsize=20)
    ax_train_res.set_xlabel('Predicted DoR', fontsize=20)
    
    # Add a single colorbar to the right of the figure
    """cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    all_snr_values = np.array(all_snr_values)
    norm = plt.Normalize(vmin=min(all_snr_values), vmax=max(all_snr_values))
    sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('SNR', rotation=270, labelpad=15, fontsize=20)"""

    from matplotlib.patches import Patch
    legend_elements = [
    Patch(facecolor=category_colors[0], edgecolor='none', alpha=0.7, label=r'$\mathrm{SNR} < 20$'),
    Patch(facecolor=category_colors[1], edgecolor='none', alpha=0.7, label=r'$20 \leq \mathrm{SNR} < 70$'),
    Patch(facecolor=category_colors[2], edgecolor='none', alpha=0.7, label=r'$\mathrm{SNR} \geq 70$')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.08),
          ncol=3, frameon=True, fontsize=20, title='SNR Categories')
    
    # plt.show()
    plt.savefig(f'../outputs/tests/{model_name}_combined_ensemble.pdf', bbox_inches='tight')
    plt.close()
    
    return True


def visualize_ensemble_predictions_combined(model_name):
    
    top = 0.9
    bottom = 0
    
    # Use exact values from results_summary
    model_results = results_summary[results_summary['name'] == model_name]
    if len(model_results) > 0:
        # Round to 2 decimal places to match CSV format
        test_r2_value = round(model_results['test_r2'].mean(), 3)
        train_r2_value = round(model_results['train_avg_r2'].mean(), 3)
    else:
        print(f"Model {model_name} not found in results_summary")
        return False
    
    # Set plotting style
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "figure.dpi": 300,
        "font.size": 26,
    })
    
    # Create figure with 2x1 grid (train on top, test on bottom)
    fig = plt.figure(figsize=(10, 18))
    
    # Create gridspec with 2 rows, 1 column
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], 
                          hspace=0.1)
    
    # Create plots - train on top, test on bottom
    ax_train = fig.add_subplot(gs[0, 0])
    ax_test = fig.add_subplot(gs[1, 0], sharex=ax_train)

    # Process and plot each dataset - train first, then test
    datasets = [
        {"name": "train", "ax": ax_train, "predictions": train_predictions, 
         "df": train_df, "id_field": None},
        {"name": "test", "ax": ax_test, "predictions": test_predictions, 
         "df": test_df, "id_field": "ID"}
    ]
    
    all_snr_values = []  # Collect all SNR values for shared colorbar
    scatter_artists = []  # Save scatter artists for colorbar
    
    for dataset in datasets:
        # Get data points
        predictions_dict = dataset["predictions"]
        df = dataset["df"]
        data_points = list(predictions_dict[model_name].keys())
        
        # Create mappings from data points to true values and SNR
        if dataset["id_field"] is not None:  # Test data
            point_to_true = {id_val: df.loc[df[dataset["id_field"]] == id_val, 'DoR'].values[0] 
                             for id_val in data_points}
            point_to_snr = {id_val: df.loc[df[dataset["id_field"]] == id_val, 'SNR'].values[0] 
                            for id_val in data_points}
        else:  # Train data
            point_to_true = {idx: df.loc[idx, 'DoR'] for idx in data_points}
            point_to_snr = {idx: df.loc[idx, 'SNR'] for idx in data_points}
        
        # Calculate average and std of predictions
        y_true = []
        y_pred_avg = []
        y_pred_std = []
        snr_values = []
        
        for id_val in data_points:
            predictions = list(predictions_dict[model_name][id_val].values())
            if predictions and id_val in point_to_true and id_val in point_to_snr:
                y_true.append(point_to_true[id_val])
                y_pred_avg.append(np.mean(predictions))
                y_pred_std.append(np.std(predictions))
                snr_values.append(point_to_snr[id_val])
        
        # Convert to numpy arrays for calculations
        y_true = np.array(y_true)
        y_pred_avg = np.array(y_pred_avg)
        y_pred_std = np.array(y_pred_std)
        snr_values = np.array(snr_values)
        
        # Add to overall SNR values for colorbar
        all_snr_values.extend(snr_values)
        
        # Remove NaN values
        valid_indices = ~np.isnan(y_pred_avg) & ~np.isnan(snr_values)
        if np.sum(valid_indices) < 10:  # Require at least 10 valid points
            print(f"Not enough valid predictions for {dataset['name']} data in model: {model_name}")
            continue
        
        y_true = y_true[valid_indices]
        y_pred_avg = y_pred_avg[valid_indices]
        y_pred_std = y_pred_std[valid_indices]
        snr_values = snr_values[valid_indices]
        
        # Calculate RMSE only (R² comes from results_summary)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_avg))
        # Round RMSE to 2 decimal places
        rmse = round(rmse, 3)
        
        # Use the exact R² values from results_summary
        r2 = train_r2_value if dataset["name"] == "train" else test_r2_value
        
        # Plot True vs Predicted with SNR coloring
        ax = dataset["ax"]
        
        low = 20
        high = 70
        snr_categories = np.zeros_like(snr_values, dtype=int)
        snr_categories[(snr_values >= low) & (snr_values < high)] = 1  # Medium SNR
        snr_categories[snr_values >= high] = 2  # High SNR
        
        category_colors = ['#d53e4f', '#fee08b','#3288bd']  # Blue, Yellow, Red for Low, Medium, High
        point_colors = [category_colors[cat] for cat in snr_categories]
        
        # Create scatter plot
        scatter = ax.scatter(y_pred_avg, y_true, c=point_colors, s=50, alpha=0.7, edgecolors='none')
        scatter_artists.append(scatter)
        
        # Draw error bars using errorbar function with no markers
        ax.errorbar(y_pred_avg, y_true, xerr=y_pred_std, fmt='none', ecolor='#000000', 
                   elinewidth=0.2, capsize=0)
        
        ax.set_xlim(bottom, top)
        ax.set_ylim(bottom, top)
        
        # Create consistent tick marks with 0.1 increments
        ticks = np.arange(0.1, 1, 0.2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f'{t:.1f}' for t in ticks])
        ax.set_yticklabels([f'{t:.1f}' for t in ticks])
        
        # Add perfect prediction line
        ax.plot([0.0, 1], [0.0, 1], 'r--', 
               label='Perfect prediction', linewidth=1.5)
        
        # Add annotations
        props = dict(boxstyle='square', facecolor='white', alpha=0.8, edgecolor='lightgray')
            
        textstr = '\n'.join((
                r'$\mathrm{%s}$' % (dataset["name"].capitalize(),),
                r'$\mathrm{R}^2 = %.3f$' % (r2,),  
                r'$\mathrm{RMSE} = %.3f$' % (rmse,),
            ))

        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=20,
               verticalalignment='top', bbox=props)
        
        # Style plot
        ax.set_ylabel('True DoR', fontsize=20)
        
        # Only add x-label to the bottom plot
        if dataset["name"] == "test":
            ax.set_xlabel('Predicted DoR', fontsize=20)
        else:
            # Hide x-tick labels for top plot
            ax.set_xticklabels([])
            plt.setp(ax.get_xticklabels(), visible=False)
        
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
        ax.minorticks_on()
    
    # Add a legend for SNR categories at the bottom of the figure
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=category_colors[0], edgecolor='none', alpha=0.7, label=r'$\mathrm{SNR} < 20$'),
        Patch(facecolor=category_colors[1], edgecolor='none', alpha=0.7, label=r'$20 \leq \mathrm{SNR} < 70$'),
        Patch(facecolor=category_colors[2], edgecolor='none', alpha=0.7, label=r'$\mathrm{SNR} \geq 70$')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.08),
          ncol=3, frameon=True, fontsize=20, title='SNR Categories')
    
    plt.savefig(f'../outputs/tests/{model_name}_combined_ensemble.pdf', bbox_inches='tight')
    plt.close()
    
    return True


def calculate_ensemble_metrics():
    ensemble_results = pd.DataFrame(columns=[
        'name', 
        'features',
        'test_r2_avg', 'test_r2_std',
        'train_r2_avg', 'train_r2_std'
    ])
    
    # Group results by model name to handle different random seeds
    grouped_results = results_summary.groupby('name')
    
    for name, group in grouped_results:
        # Calculate test R² statistics
        test_r2_values = group['test_r2'].values
        test_r2_avg = np.mean(test_r2_values)
        test_r2_std = np.std(test_r2_values)
        
        # Calculate train R² statistics from cross-validation
        train_r2_values = group['train_avg_r2'].values
        train_r2_avg = np.mean(train_r2_values)
        train_r2_std = np.std(train_r2_values)
        
        # Get the features used for this model
        features = group['features'].iloc[0]
        
        # Create visualizations if desired
        """try:
            if name=='Stel. pop. and structural':
                visualize_ensemble_predictions_combined(name)
        except Exception as e:
            print(f"Error creating combined visualization for {name}: {e}")"""
        
        visualize_ensemble_predictions_combined(name)

        # Apply proper rounding for statistical significance
        # First round std to 1 significant figure
        test_std_rounded, test_decimal_places = round_to_sig_figs(test_r2_std)
        train_std_rounded, train_decimal_places = round_to_sig_figs(train_r2_std)
        
        # Then round averages to same decimal places as their std
        test_avg_rounded = round(test_r2_avg, test_decimal_places)
        train_avg_rounded = round(train_r2_avg, train_decimal_places)
        
        # Add to results with properly rounded values
        ensemble_results = pd.concat([ensemble_results, pd.DataFrame({
            'name': [name],
            'features': [features],
            'test_r2_avg': [test_avg_rounded],
            'test_r2_std': [test_std_rounded],
            'train_r2_avg': [train_avg_rounded],
            'train_r2_std': [train_std_rounded]
        })], ignore_index=True)
    
    # Print results
    print("\nR² Variation Metrics Across Random Seeds:")
    print("=" * 120)
    print(ensemble_results[['name', 'test_r2_avg', 'test_r2_std', 'train_r2_avg', 'train_r2_std']].to_string(
        index=False))
    print("=" * 120)
    
    # Save to CSV
    ensemble_results.to_csv('../outputs/tests/ensemble_results.csv', index=False)
    
    return ensemble_results


def round_to_sig_figs(num, sig_figs=1):
    if num == 0:
        return 0, 0
    
    # Find the position of the first significant digit
    pos = int(np.floor(np.log10(abs(num))))
    
    # Calculate the number of decimal places needed
    decimal_places = sig_figs - 1 - pos
    
    # Ensure decimal_places is non-negative for display purposes
    display_decimal_places = max(0, decimal_places)
    
    # Round the number
    factor = 10 ** decimal_places
    rounded_num = round(num * factor) / factor
    
    return rounded_num, display_decimal_places


def print_results_summary():
    # Group by model name and calculate statistics across random seeds
    aggregated_results = results_summary.groupby('name').agg({
        'test_r2': ['mean', 'std', 'min', 'max'],
        'train_avg_r2': ['mean', 'std', 'min', 'max']
    })
    
    # Flatten the multi-index columns
    aggregated_results.columns = [
        f"{col[0]}_{col[1]}" for col in aggregated_results.columns
    ]
    
    # Reset index to make 'name' a column instead of an index
    aggregated_results = aggregated_results.reset_index()
    
    # Format the output for display
    print(f"{'Model Name':<35} | {'Test R²: Mean ± Std':<25} | {'Test R²: Min-Max':<25} | {'Train R²: Mean ± Std':<25} | {'Train R²: Min-Max':<25}")
    print("-" * 120)
    
    for _, row in aggregated_results.iterrows():
        name = row['name']
        test_mean, test_std = row['test_r2_mean'], row['test_r2_std']
        test_min, test_max = row['test_r2_min'], row['test_r2_max']
        train_mean, train_std = row['train_avg_r2_mean'], row['train_avg_r2_std']
        train_min, train_max = row['train_avg_r2_min'], row['train_avg_r2_max']
        
        print(f"{name:<35} | {test_mean:.4f} ± {test_std:.4f} | {test_min:.4f} - {test_max:.4f} | {train_mean:.4f} ± {train_std:.4f} | {train_min:.4f} - {train_max:.4f}")


def plot_feature_importances():
    feature_names = {
        'met': r'$\mathrm{[M/H]}$',
        'tau': r'$\mathrm{\tau_{\rm rel}}$',
        'met_err': r'$\Delta{\mathrm{[M/H]}}$',
        'lin_age_err': r'$\Delta{\mathrm{Age}}$',
        'logM': r'$\log(M/M_{\odot})$',
        'rad_kpc': r'$R \, \mathrm{(kpc)}$',
        'MgFe': r'$\mathrm{[Mg/Fe]}$',
        'vdisp': r'$\sigma \, \mathrm{(km/s)}$',
        'DoR': r'$\mathrm{DoR}$'
    }
    # Create a DataFrame to store all feature importances
    all_importances = []

    # Group by model name to get unique models
    unique_models = results_summary['name'].unique()
    
    # For each unique model, calculate the average importance of each feature
    for model_name in unique_models:
        # Get all rows for this model
        model_rows = results_summary[results_summary['name'] == model_name]
        
        # Initialize a dictionary to store the sum of importances for each feature
        feature_sums = {}
        feature_counts = {}
        
        # Sum up the importances for each feature across all runs of this model
        for _, row in model_rows.iterrows():
            importances = row['feature_importances']
            
            if importances is not None:
                for feature, importance in importances.items():
                    if feature not in feature_sums:
                        feature_sums[feature] = 0
                        feature_counts[feature] = 0
                    
                    feature_sums[feature] += importance
                    feature_counts[feature] += 1
        
        # Calculate the average importance for each feature
        avg_importances = {feature: feature_sums[feature] / feature_counts[feature] 
                          for feature in feature_sums}
        
        # Add to the results
        for feature, avg_importance in avg_importances.items():
            all_importances.append({
                'Model': model_name,
                'Feature': feature,
                'Importance': avg_importance
            })

    # Convert to DataFrame
    importance_df = pd.DataFrame(all_importances)

    # Create feature category mapping
    feature_categories = {
        'met': 'Stellar Population',
        'tau': 'Stellar Population',
        'met_err': 'Measurement Error',
        'lin_age_err': 'Measurement Error',
        'vdisp': 'Kinematics',
        'MgFe': 'Chemical Composition',
        'logM': 'Structure',
        'rad_kpc': 'Structure',
        'DoR': 'Structure'
    }

    # Add category column
    importance_df['Category'] = importance_df['Feature'].map(feature_categories)

    # Set plot style
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "figure.dpi": 300,
        "font.size": 20,
    })

    # Create pivot table with actual NaN values for missing features (not 0s)
    pivot_df = importance_df.pivot_table(
        index='Model',
        columns='Feature',
        values='Importance',
        aggfunc='mean'
    )  # Remove fillna(0) to keep NaN values

    # Sort rows to match the order of unique_models
    pivot_df = pivot_df.reindex(unique_models)

    # Rename columns using the feature_names dictionary
    pivot_df = pivot_df.rename(columns=feature_names)

    # Create mask for NaN values AFTER renaming columns
    mask = pivot_df.isna()

    # Custom function to format annotations
    def format_val(val):
        return "{:.3f}".format(val) if not np.isnan(val) else ""
    
    # Create annotations array with formatted values
    annotations = np.vectorize(format_val)(pivot_df.values)

    # Create heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        pivot_df,
        annot=annotations,  # Use pre-formatted annotation array
        cmap='YlGnBu',
        fmt='',  # Empty format string since we're using pre-formatted annotations
        linewidths=0.5,
        cbar_kws={'label': 'Average Feature Importance'},
        annot_kws={"size": 10},
        mask=mask  # Apply mask for NaN values
    )
    
    #ax.set_ylabel("Model ID")
    ax.set_ylabel("Model ID")
    ax.set_xlabel("Feature")

    plt.tight_layout()
    plt.savefig('../outputs/tests/feature_imp_map.pdf', bbox_inches='tight')

    # Save the importance data to CSV with proper feature names
    # Create a copy of importance_df with renamed features
    export_df = importance_df.copy()
    export_df['Feature'] = export_df['Feature'].map(feature_names).fillna(export_df['Feature'])
    # export_df.to_csv('../outputs/tests/avg_feature_importances.csv', index=False)

    return importance_df


def create_corner_plots(df, features, feature_display_names=None, target='DoR', target_display_name=None):

    # Set LaTeX style to match the original code
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "figure.dpi": 300,
        "font.size": 20,
    })
    
    # Create display names dictionary if not provided
    if feature_display_names is None:
        feature_display_names = {f: f for f in features}
    
    # Set target display name if not provided
    if target_display_name is None:
        target_display_name = target
    
    # Create the first version with the target included as a feature
    # plot_features_with_target(df, features, feature_display_names, target, target_display_name)
    
    # Create the second version without the target but colored by the target
    plot_features_colored_by_target(df, features, feature_display_names, target, target_display_name)


def plot_features_with_target(df, features, feature_display_names, target='DoR', target_display_name=None):
    """Create a corner plot with all features including the target"""
    # Create a new figure
    fig = plt.figure(figsize=(16, 16))
    
    # If target display name not provided, use the target name
    if target_display_name is None:
        target_display_name = target
    
    # Include target in the feature list for this plot
    all_vars = features.copy()
    all_display_names = feature_display_names.copy()
    
    if target not in all_vars:
        all_vars.append(target)
        all_display_names[target] = target_display_name
    
    # Number of variables
    n_vars = len(all_vars)
    
    # Create subplots for each pair of variables
    for i in range(n_vars):
        for j in range(n_vars):
            # Create subplot
            ax = fig.add_subplot(n_vars, n_vars, i * n_vars + j + 1)
            
            if i == j:  # Diagonal: histogram
                # Draw the histogram
                sns.histplot(df[all_vars[i]], kde=True, ax=ax, color='steelblue', 
                            edgecolor='black', linewidth=0.8, alpha=0.7)
                
                # Remove title
                ax.set_title("")
                
                # Add "Count" as a title above the plot instead of as a y-label
                # ax.set_title("Count", fontsize=20, pad=10)
                
                # Remove x-label for diagonal histograms
                ax.set_xlabel("")
                
                # Only set x-label for the bottom row
                if i == n_vars - 1:
                    ax.set_xlabel(all_display_names[all_vars[i]], fontsize=20)
                
                # Remove y-label but keep y-tick labels
                ax.set_ylabel("")
                # Rotate y-ticks 45 degrees for better readability
                ax.tick_params(axis='y', rotation=45)
                
                # If not the bottom row, hide x-tick labels
                if i < n_vars - 1:
                    ax.set_xticklabels([])
                
                ax.grid(True, alpha=0.3)
            
            elif i > j:  # Lower triangle: scatter plots
                scatter = ax.scatter(df[all_vars[j]], df[all_vars[i]], 
                                    s=30, alpha=0.6, 
                                    edgecolors='k', linewidths=0.5)
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Set axis labels only for the bottom row and leftmost column
                # Use display names for labels
                if i == n_vars - 1:
                    ax.set_xlabel(all_display_names[all_vars[j]], fontsize=20)
                else:
                    ax.set_xlabel('')
                    
                if j == 0:
                    ax.set_ylabel(all_display_names[all_vars[i]], fontsize=20)
                else:
                    ax.set_ylabel('')
                
                # Only show tick labels on the edges
                if i < n_vars - 1:  # Not the bottom row
                    ax.set_xticklabels([])
                if j > 0:  # Not the leftmost column
                    ax.set_yticklabels([])
            
            else:  # Upper triangle: leave empty
                ax.axis('off')
            
            # Set ticks inward
            if i != j and i > j:
                ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
                ax.minorticks_on()
    
    # Use larger wspace to increase horizontal spacing between subplots
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3, hspace=0.1)
    plt.savefig(f'../outputs/tests/corner_plain.pdf')
    
    return fig


def plot_features_colored_by_target(df, features, feature_display_names, target='DoR', target_display_name=None):
    """Create a corner plot with features colored by the target variable"""
    # Create a new figure
    fig = plt.figure(figsize=(16, 16))
    
    # If target display name not provided, use the target name
    if target_display_name is None:
        target_display_name = target
    
    # Number of variables
    n_vars = len(features)
    
    # Create a custom colormap with red for high values (inverted heat colors)
    colors = ["navy", "blue", "dodgerblue", "deepskyblue", "cyan", 
              "yellowgreen", "yellow", "gold", "orange", "orangered", "red", "darkred"]
    cmap = LinearSegmentedColormap.from_list("DoR_cmap", colors) 
    
    # Normalize the target variable for coloring
    norm = plt.Normalize(df[target].min(), df[target].max())
    
    # Create subplots for each pair of variables
    for i in range(n_vars):
        for j in range(n_vars):
            # Create subplot
            ax = fig.add_subplot(n_vars, n_vars, i * n_vars + j + 1)
            
            if i == j:  # Diagonal: histogram

                if features[i] == 'MgFe':
                    mgfe_values = [0.0, 0.1, 0.2, 0.3, 0.4]
                    
                    mgfe_custom_bins = []
                    for value in mgfe_values:
                        mgfe_custom_bins.append(value - 0.005)  # Small buffer below the value
                        mgfe_custom_bins.append(value + 0.005)  # Small buffer above the value
                    
                    mgfe_custom_bins.append(0.405)
                    
                    #mgfe_custom_bins = [max(0, bin_edge) for bin_edge in mgfe_custom_bins]
                    #mgfe_custom_bins = sorted(list(set(mgfe_custom_bins)))
                    
                    counts, bins, patches = ax.hist(df[features[i]], bins=mgfe_custom_bins, alpha=0.7, linewidth=0.8)
                else:
                    counts, bins, patches = ax.hist(df[features[i]], bins=40, alpha=0.7, linewidth=0.8)
                
                
                
                """
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                for count, x, patch in zip(counts, bin_centers, patches):
                    # Find indices of points in this bin
                    bin_mask = (df[features[i]] >= x - (bins[1]-bins[0])/2) & \
                              (df[features[i]] < x + (bins[1]-bins[0])/2)
                    
                    if any(bin_mask):
                        # Color by average DoR in this bin
                        avg_DoR = df.loc[bin_mask, target].mean()
                        patch.set_facecolor(cmap(norm(avg_DoR)))
                    else:
                        patch.set_facecolor('gray')"""
                    
                for idx, (patch, left_edge, right_edge) in enumerate(zip(patches, bins[:-1], bins[1:])):
                    # For the last bin, include the right edge
                    if idx == len(patches) - 1:
                        bin_mask = (df[features[i]] >= left_edge) & (df[features[i]] <= right_edge)
                    else:
                        bin_mask = (df[features[i]] >= left_edge) & (df[features[i]] < right_edge)
                    
                    if any(bin_mask):
                        avg_DoR = df.loc[bin_mask, target].mean()
                        patch.set_facecolor(cmap(norm(avg_DoR)))
                    else:
                        patch.set_facecolor('gray')
                
                # Add "Count" as a title above the plot instead of as a y-label
                # ax.set_title("Count", fontsize=20, pad=10)
                
                # Remove x-label for diagonal histograms
                ax.set_xlabel("")
                
                # Only set x-label for the bottom row
                if i == n_vars - 1:
                    ax.set_xlabel(feature_display_names[features[i]], fontsize=20)
                
                # Remove y-label but keep y-tick labels
                ax.set_ylabel("")
                # Rotate y-ticks 45 degrees for better readability
                # ax.tick_params(axis='y', rotation=45)
                ax.set_yticklabels([])

                
                # If not the bottom row, hide x-tick labels
                if i < n_vars - 1:
                    ax.set_xticklabels([])
                
                ax.grid(True, alpha=0.3)
            
            elif i > j:  # Lower triangle: scatter plots colored by target
                scatter = ax.scatter(df[features[j]], df[features[i]], 
                                    c=df[target], cmap=cmap, 
                                    s=30, alpha=0.7, linewidths=0.5)
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Set axis labels only for the bottom row and leftmost column
                # Use display names for labels
                if i == n_vars - 1:
                    ax.set_xlabel(feature_display_names[features[j]], fontsize=20)
                else:
                    ax.set_xlabel('')
                    
                if j == 0:
                    ax.set_ylabel(feature_display_names[features[i]], fontsize=20)
                else:
                    ax.set_ylabel('')
                
                # Only show tick labels on the edges
                if i < n_vars - 1:  # Not the bottom row
                    ax.set_xticklabels([])
                if j > 0:  # Not the leftmost column
                    ax.set_yticklabels([])
            
            else:  # Upper triangle: leave empty
                ax.axis('off')
            
            # Set ticks inward
            if i != j and i > j:
                ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
                ax.minorticks_on()
    
    # Add colorbar to the right of the plot
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label(target_display_name, rotation=270, fontsize=20, labelpad=20)
    cbar_ax.tick_params(labelsize=12)
    
    # Use larger wspace to increase horizontal spacing between subplots
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)
    
    plt.savefig(f'../outputs/tests/corner_dor_coloured.pdf')
    
    return fig


def plot_features_by_dataset(train_df, test_df, features, feature_display_names=None):
    
    # Set LaTeX style to match the original code
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "figure.dpi": 300,
        "font.size": 20,
    })
    
    # Create display names dictionary if not provided
    if feature_display_names is None:
        feature_display_names = {f: f for f in features}
    
    # Number of variables
    n_vars = len(features)
    
    # Create a new figure with a custom GridSpec layout
    fig = plt.figure(figsize=(16, 16))
    
    # Create a GridSpec layout that's n_vars-1 x n_vars-1
    # This completely eliminates space for diagonal elements
    gs = plt.GridSpec(n_vars-1, n_vars-1, figure=fig)
    
    # Combine datasets with an identifier column
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()
    
    train_df_copy['dataset'] = 'Train'
    test_df_copy['dataset'] = 'Test'
    
    # Combine both datasets
    combined_df = pd.concat([train_df_copy, test_df_copy], ignore_index=True)
    
    # Define colors for each dataset
    dataset_colors = {'Train': 'blue', 'Test': 'red'}
    
    # Map the feature indices for our compact layout
    # We're using a n_vars-1 x n_vars-1 grid now
    plot_count = 0
    for i in range(1, n_vars):  # Start from 1 to skip first diagonal element
        for j in range(i):      # Only lower triangle
            # Calculate row and column for our compact grid
            row = i - 1  # Adjusted for 0-indexing 
            col = j
            
            # Create subplot at specific position
            ax = fig.add_subplot(gs[row, col])
            plot_count += 1
            
            # Plot scatter for both datasets
            for dataset, color in dataset_colors.items():
                subset = combined_df[combined_df['dataset'] == dataset]
                ax.scatter(subset[features[j]], subset[features[i]], 
                          c=color, label=dataset, 
                          s=30, alpha=0.6, 
                          edgecolors='k', linewidths=0.5)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Set axis labels only for the bottom row and leftmost column
            if i == n_vars - 1:  # Bottom row (last feature)
                ax.set_xlabel(feature_display_names[features[j]], fontsize=20)
            else:
                ax.set_xlabel('')
                
            if j == 0:  # Leftmost column
                ax.set_ylabel(feature_display_names[features[i]], fontsize=20)
            else:
                ax.set_ylabel('')
            
            # Only show tick labels on the edges
            if i < n_vars - 1:  # Not the bottom row
                ax.set_xticklabels([])
            if j > 0:  # Not the leftmost column
                ax.set_yticklabels([])
            
            # Add legend to first plot
            #if row == 0 and col == 0:  # First plot in grid
                #ax.legend(fontsize=20)
            
            # Set ticks inward
            ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
            ax.minorticks_on()
    
    # After creating the figure but before saving it:
    # Add a legend to the figure instead of to any specific subplot
    handles, labels = [], []
    for dataset, color in dataset_colors.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10, 
                                 markeredgecolor='k', markeredgewidth=0.5))
        labels.append(dataset)
    
    # Place legend in the top right corner of the figure
    fig.legend(handles, labels, loc='upper right', fontsize=20, 
               bbox_to_anchor=(0.35, 0.97), frameon=True)
    
    # Tighten layout
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    
    plt.savefig('../outputs/tests/combined_corner.pdf')
    #plt.savefig('../outputs/tests/combined_corner.png')

    return fig


def perform_residual_analysis(train_df, test_df, features, target='DoR', model_params=None):
    if model_params is None:
        model_params = {
            'max_depth': 13,
            'max_features': 0.8,
            'max_samples': 0.7,
            'min_samples_leaf': 3,
            'min_samples_split': 5,
            'n_estimators': 60,
            'random_state': 42
        }
    
    # Extract features and target
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    print("Training random forest model...")
    rf_model = RandomForestRegressor(**model_params)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = rf_model.predict(X_train_scaled)
    y_test_pred = rf_model.predict(X_test_scaled)
    
    # Calculate residuals
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"Train R2: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
    print(f"Test R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
    
    # Store results
    results = {
        'model': rf_model,
        'train_predictions': y_train_pred,
        'test_predictions': y_test_pred,
        'train_residuals': train_residuals,
        'test_residuals': test_residuals,
        'feature_importances': dict(zip(features, rf_model.feature_importances_)),
        'metrics': {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
    }
    
    return results


def plot_residuals_by_feature(test_df, features, results, output_dir='./residual_plots'):

    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a custom colormap
    colors = ["navy", "blue", "dodgerblue", "deepskyblue", "cyan", 
              "yellowgreen", "yellow", "gold", "orange", "orangered", "red", "darkred"]
    cmap = LinearSegmentedColormap.from_list("residual_cmap", colors)
    
    # Plot residuals against each feature
    for feature in features:
        plt.figure(figsize=(10, 6))
        
        # Scatter plot with coloring by absolute residual magnitude
        scatter = plt.scatter(
            test_df[feature],
            results['test_residuals'],
            c=np.abs(results['test_residuals']),
            cmap=cmap,
            alpha=0.7,
            s=40,
            edgecolors='k',
            linewidths=0.5
        )
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        
        # Add trend line to highlight patterns
        try:
            z = np.polyfit(test_df[feature], results['test_residuals'], 1)
            p = np.poly1d(z)
            plt.plot(
                np.sort(test_df[feature].values),
                p(np.sort(test_df[feature].values)),
                "r--", 
                linewidth=2,
                alpha=0.7
            )
            
            # Calculate correlation coefficient
            corr = np.corrcoef(test_df[feature], results['test_residuals'])[0, 1]
            plt.annotate(
                f"Correlation: {corr:.3f}",
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=20,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        except:
            print(f"Could not calculate trend line for {feature}")
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Absolute Residual', rotation=270, labelpad=20)
        
        # Set labels and title
        plt.xlabel(feature)
        plt.ylabel('Residual (True - Predicted)')
        plt.title(f'Residuals vs {feature}')
        plt.grid(alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        #plt.savefig(f"{output_dir}/residual_{feature}.pdf")
        plt.close()
    
    print(f"Individual residual plots saved to {output_dir}")


def create_feature_residual_grid(test_df, features, results, output_dir='./residual_plots'):    
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate number of rows and columns for the grid
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = int(np.ceil(n_features / n_cols))
    
    # Create the figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    # Plot for each feature
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # Scatter plot
        scatter = ax.scatter(
            test_df[feature],
            results['test_residuals'],
            c=np.abs(results['test_residuals']),
            cmap='viridis',
            alpha=0.7,
            s=30,
            edgecolors='none'
        )
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        
        # Add trend line
        try:
            z = np.polyfit(test_df[feature], results['test_residuals'], 1)
            p = np.poly1d(z)
            ax.plot(
                np.sort(test_df[feature].values),
                p(np.sort(test_df[feature].values)),
                "r--", 
                linewidth=1.5,
                alpha=0.7
            )
            
            # Calculate correlation coefficient
            corr = np.corrcoef(test_df[feature], results['test_residuals'])[0, 1]
            ax.annotate(
                f"Corr: {corr:.3f}",
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=20,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8)
            )
        except:
            pass
        
        # Set labels
        ax.set_xlabel(feature)
        ax.set_ylabel('Residual')
        ax.set_title(f'Residuals vs {feature}')
        ax.grid(alpha=0.3)
    
    # Hide any extra axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    # Add a colorbar for the entire figure
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Absolute Residual', rotation=270, labelpad=20)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(f"{output_dir}/residual_grid.pdf")
    plt.close()
    
    print(f"Residual grid plot saved to {output_dir}")


def create_residual_analysis_report(train_df, test_df, features, target='DoR', model_params=None):

    print("Starting residual analysis...")
    
    # Step 1: Train model and compute residuals
    results = perform_residual_analysis(train_df, test_df, features, target, model_params)
    
    # Step 3: Plot residuals against each feature
    print("\nCreating individual residual plots...")
    plot_residuals_by_feature(test_df, features, results)
    
    # Step 4: Create grid of residual plots
    print("\nCreating residual grid plot...")
    create_feature_residual_grid(test_df, features, results)
    
    
    # Step 6: Sort features by residual correlation
    residual_correlations = {}
    for feature in features:
        corr = np.corrcoef(test_df[feature], results['test_residuals'])[0, 1]
        residual_correlations[feature] = corr
    
    sorted_correlations = sorted(
        residual_correlations.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    print("\nFeatures ranked by absolute correlation with residuals:")
    for feature, corr in sorted_correlations:
        print(f"{feature}: {corr:.4f}")