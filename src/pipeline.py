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
    (r'Stel. pop., $\alpha$-abundance and kinematics', ['met', 'tau', 'met_err', 'lin_age_err', 'MgFe', 'vdisp']),
    ('Stel. pop. and structural', ['met', 'tau', 'met_err', 'lin_age_err', 'logM', 'rad_kpc']),
    ('Stel. pop., structural and kinematics', ['met', 'tau', 'met_err', 'lin_age_err', 'logM', 'rad_kpc', 'vdisp']),
    ('Complete Set', ['met', 'tau', 'met_err', 'lin_age_err', 'logM', 'rad_kpc', 'MgFe', 'vdisp']),
    # ('New Chiara 1', ['logM', 'rad_kpc', 'vdisp']),
    # ('New Chiara 2', ['tau', 'logM', 'rad_kpc', 'vdisp', 'lin_age_err'])
]


def reset_output_dir(path='outputs/paper_plots'):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def run_baseline_regressions(train_df, test_df, random_states, plotting=False, tag=None):
    for random_state in random_states:
        model_params = {
            'C': 5.0,
            'epsilon': 0.03,
            'kernel': 'poly',
            'gamma': 0.01,
            'degree': 3,
            'coef0': 0.5,
        }
        for name, features in FEATURE_SETS:
            labeled_name = f"{name} [{tag}]" if tag else name
            regression.test_regression_on_INSPIRE(
                labeled_name,
                features,
                test_df,
                train_df,
                model_params,
                plotting=plotting,
                random_state=random_state,
            )


def plot_feature_histograms(train_df, test_df, tag=None):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "figure.dpi": 300,
        "font.size": 20,
    })

    # columns = ['Source', 'logM', 'vdisp', 'tau', 'lin_age_err', 'met', 'met_err', 'DoR']
    columns = ['rad_kpc', 'logM', 'vdisp', 'tau', 'lin_age_err', 'met', 'met_err', 'DoR']
    feature_names = {
        'met': r'$\mathrm{[M/H] \, (dex)}$',
        'tau': r'$\tau_{\rm rel}$',
        'met_err': r'$\Delta\mathrm{[M/H] \, (dex)}$',
        'lin_age_err': r'$\Delta\mathrm{Age \, (Gyr)}$',
        'logM': r'$\log(M_{\star}/M_{\odot})$',
        'rad_kpc': r'$R_{\rm e} \, \mathrm{(kpc)}$',
        'MgFe': r'$\mathrm{[Mg/Fe] \, (dex)}$',
        'vdisp': r'$\sigma_{\star} \, \mathrm{(km/s)}$',
        'DoR': r'$\mathrm{DoR}$',
        'Source': 'Source',
    }

    def get_display_name(feature):
        return feature_names.get(feature, feature)

    if tag and tag.lower() == 'mixed':
        train_label = 'Train (mixed)'
        test_label = 'Test (mixed)'
    else:
        train_label = 'E-INSPIRE'
        test_label = 'INSPIRE'

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    sns.set_style("white")
    for i, feature in enumerate(columns):
        if i >= len(axes):
            break
        ax = axes[i]
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(axis='both', labelsize=14)

        min_val = min(train_df[feature].min(), test_df[feature].min())
        max_val = max(train_df[feature].max(), test_df[feature].max())
        num_bins = 32

        if feature == 'Source':
            bins = [-0.5, 0.5, 1.5]
            pad = 0
        else:
            rng = max_val - min_val
            bins = np.linspace(min_val, max_val, num_bins + 1)
            pad = rng * 0.05

        sns.histplot(
            train_df[feature],
            bins=bins,
            color='blue',
            alpha=0.6,
            label=train_label,
            ax=ax,
            stat='probability',
        )
        sns.histplot(
            test_df[feature],
            bins=bins,
            color='orange',
            alpha=0.6,
            label=test_label,
            ax=ax,
            stat='probability',
        )

        ax.set_xlabel(get_display_name(feature))
        if i in (0, 4):
            ax.set_ylabel(r'$\mathrm{Probability}$')
        else:
            ax.set_ylabel('')
        ax.set_xlim(bins[0] - pad, bins[-1] + pad)
        if i == 3:
            ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), framealpha=0.9, fontsize=14)
    plt.subplots_adjust(wspace=0.3, hspace=0.1)
    plt.tight_layout()
    suffix = ''
    if tag:
        safe_tag = tag.replace(' ', '_').replace('→', 'to')
        suffix = f'_{safe_tag}'
    plt.savefig(f'outputs/paper_plots/train_test_histo{suffix}.pdf')
    plt.close()


__all__ = [
    'FEATURE_SETS',
    'reset_output_dir',
    'run_baseline_regressions',
    'plot_feature_histograms',
]
