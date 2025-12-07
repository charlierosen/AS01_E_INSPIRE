import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src import regression

FEATURE_SETS = [
    ('Stel. pop.', ['met', 'tau']),
    ('Stel. pop. with errors', ['met', 'tau', 'met_err', 'lin_age_err']),
    ('Stel. pop. and kinematics', ['met', 'tau', 'met_err', 'lin_age_err', 'vdisp']),
    (r'Stel. pop. and $\alpha$-abundance', ['met', 'tau', 'met_err', 'lin_age_err', 'MgFe']),
    ('Stel. pop. and structural', ['met', 'tau', 'met_err', 'lin_age_err', 'logM', 'rad_kpc']),
    (r'Stel. pop., $\alpha$-abundance and kinematics', ['met', 'tau', 'met_err', 'lin_age_err', 'MgFe', 'vdisp']),
    ('Stel. pop., structural and kinematics', ['met', 'tau', 'met_err', 'lin_age_err', 'logM', 'rad_kpc', 'vdisp']),
    ('Complete Set', ['met', 'tau', 'met_err', 'lin_age_err', 'logM', 'rad_kpc', 'MgFe', 'vdisp']),
]


def reset_output_dir(path='outputs/paper_plots'):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def run_baseline_regressions(train_df, test_df, random_states, plotting=False):
    for random_state in random_states:
        model_params = {
            'max_depth': 8,
            'max_features': 0.8,
            'max_samples': 0.7,
            'min_samples_leaf': 3,
            'min_samples_split': 5,
            'n_estimators': 50,
            'random_state': random_state,
        }
        for name, features in FEATURE_SETS:
            regression.test_regression_on_INSPIRE(
                name,
                features,
                test_df,
                train_df,
                model_params,
                plotting=plotting,
                random_state=random_state,
            )


def plot_feature_histograms(train_df, test_df):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "figure.dpi": 300,
        "font.size": 20,
    })

    # columns = ['Source', 'logM', 'vdisp', 'tau', 'lin_age_err', 'met', 'met_err', 'DoR']
    columns = ['rad_kpc', 'logM', 'vdisp', 'tau', 'lin_age_err', 'met', 'met_err', 'DoR']
    feature_names = {
        'met': r'$\mathrm{[M/H]}$',
        'tau': r'$\mathrm{\tau_{\rm rel}}$',
        'met_err': r'$\Delta{\mathrm{[M/H]}}$',
        'lin_age_err': r'$\Delta{\mathrm{Age}}$',
        'logM': r'$\log(M/M_{\odot})$',
        'rad_kpc': r'$R \, \mathrm{(kpc)}$',
        'MgFe': r'$\mathrm{[Mg/Fe]}$',
        'vdisp': r'$\sigma_{\star} \, \mathrm{(km/s)}$',
        'DoR': r'$\mathrm{DoR}$',
        'Source': 'Source',
    }

    def get_display_name(feature):
        return feature_names.get(feature, feature)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    sns.set_style("whitegrid")
    for i, feature in enumerate(columns):
        if i >= len(axes):
            break
        ax = axes[i]
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)

        min_val = min(train_df[feature].min(), test_df[feature].min())
        max_val = max(train_df[feature].max(), test_df[feature].max())
        num_bins = 20

        if feature == 'Source':
            bins = [-0.5, 0.5, 1.5]
        else:
            bins = np.linspace(min_val, max_val, num_bins)

        sns.histplot(
            train_df[feature],
            bins=bins,
            color='blue',
            alpha=0.6,
            label='E-INSPIRE',
            ax=ax,
            stat='density',
        )
        sns.histplot(
            test_df[feature],
            bins=bins,
            color='orange',
            alpha=0.6,
            label='INSPIRE',
            ax=ax,
            stat='density',
        )

        ax.set_xlabel(get_display_name(feature))
        ax.set_ylabel('Density')
        ax.set_xlim(min_val, max_val)
        ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/paper_plots/train_test_histo.pdf')
    plt.close()


__all__ = [
    'FEATURE_SETS',
    'reset_output_dir',
    'run_baseline_regressions',
    'plot_feature_histograms',
]
