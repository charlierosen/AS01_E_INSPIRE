import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def create_corner_plots(df, features, feature_display_names=None, target='DoR', target_display_name=None, suffix=''):
    # Set LaTeX style
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "figure.dpi": 300,
        "font.size": 15,
    })

    if feature_display_names is None:
        feature_display_names = {f: f for f in features}
    if target_display_name is None:
        target_display_name = target

    # Create the first version with the target included as a feature
    plot_features_with_target(df, features, feature_display_names, target, target_display_name, suffix)

    # Create the second version, colored by the target
    plot_features_colored_by_target(df, features, feature_display_names, target, target_display_name, suffix)


def plot_features_with_target(df, features, feature_display_names, target='DoR', target_display_name=None, suffix=''):

    fig = plt.figure(figsize=(16, 16))

    if target_display_name is None:
        target_display_name = target

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
            ax = fig.add_subplot(n_vars, n_vars, i * n_vars + j + 1)

            if i == j:
                # Diagonal: histogram
                sns.histplot(df[all_vars[i]], kde=True, ax=ax, color='steelblue',
                             edgecolor='black', linewidth=0.8, alpha=0.7)
                ax.set_title("")

                if i == n_vars - 1:  # Bottom row
                    ax.set_xlabel(all_display_names[all_vars[i]], fontsize=16)
                    ax.set_ylabel("Count", fontsize=14)
                else:

                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_xticklabels([])

                ax.grid(True, alpha=0.3)

            elif i > j:  # Lower triangle: scatter plots
                scatter = ax.scatter(df[all_vars[j]], df[all_vars[i]],
                                     s=30, alpha=0.6,
                                     edgecolors='k', linewidths=0.5)

                ax.grid(True, alpha=0.3)

                # Set axis labels only for the bottom row and leftmost column
                if i == n_vars - 1:
                    ax.set_xlabel(all_display_names[all_vars[j]], fontsize=16)
                else:
                    ax.set_xlabel('')

                if j == 0:
                    ax.set_ylabel(all_display_names[all_vars[i]], fontsize=16)
                else:
                    ax.set_ylabel('')

                # Only show tick labels on the edges
                if i < n_vars - 1:  # Not the bottom row
                    ax.set_xticklabels([])
                if j > 0:  # Not the leftmost column
                    ax.set_yticklabels([])

            else:  # Upper triangle
                ax.axis('off')

            if i != j and i > j:
                ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
                ax.minorticks_on()

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.savefig(f'outputs/make_plots_output/corner_plain{suffix}.pdf')

    return fig


def plot_features_colored_by_target(df, features, feature_display_names, target='DoR', target_display_name=None,
                                    suffix=''):
    fig = plt.figure(figsize=(16, 16))

    if target_display_name is None:
        target_display_name = target

    # Number of variables
    n_vars = len(features)

    colors = ["navy", "blue", "dodgerblue", "deepskyblue", "cyan",
              "yellowgreen", "yellow", "gold", "orange", "orangered", "red", "darkred"]

    cmap = LinearSegmentedColormap.from_list("DoR_cmap", colors)

    # Normalize the target variable for coloring
    norm = plt.Normalize(df[target].min(), df[target].max())

    for i in range(n_vars):
        for j in range(n_vars):
            ax = fig.add_subplot(n_vars, n_vars, i * n_vars + j + 1)

            if i == j:  # Diagonal: histogram
                counts, bins, patches = ax.hist(df[features[i]], bins=20,
                                                alpha=0.7, edgecolor='black', linewidth=0.8)

                # Color the histogram bars by the average DoR for data in each bin
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                for count, x, patch in zip(counts, bin_centers, patches):
                    # Find indices of points in this bin
                    bin_mask = (df[features[i]] >= x - (bins[1] - bins[0]) / 2) & \
                               (df[features[i]] < x + (bins[1] - bins[0]) / 2)

                    if any(bin_mask):
                        # Color by average DoR in this bin
                        avg_DoR = df.loc[bin_mask, target].mean()
                        patch.set_facecolor(cmap(norm(avg_DoR)))
                    else:
                        patch.set_facecolor('gray')

                ax.set_title("")
                if i == n_vars - 1:  # Bottom row
                    ax.set_xlabel(feature_display_names[features[i]], fontsize=16)
                    ax.set_ylabel("Count", fontsize=14)
                else:
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_xticklabels([])

                ax.grid(True, alpha=0.3)

            elif i > j:  # Lower triangle: scatter plots
                scatter = ax.scatter(df[features[j]], df[features[i]],
                                     c=df[target], cmap=cmap,
                                     s=30, alpha=0.7,
                                     edgecolors='k', linewidths=0.5)
                ax.grid(True, alpha=0.3)

                if i == n_vars - 1:
                    ax.set_xlabel(feature_display_names[features[j]], fontsize=16)
                else:
                    ax.set_xlabel('')

                if j == 0:
                    ax.set_ylabel(feature_display_names[features[i]], fontsize=16)
                else:
                    ax.set_ylabel('')

                # Only show tick labels on the edges
                if i < n_vars - 1:  # Not the bottom row
                    ax.set_xticklabels([])
                if j > 0:  # Not the leftmost column
                    ax.set_yticklabels([])

            else:  # Upper triangle
                ax.axis('off')

            if i != j and i > j:
                ax.tick_params(axis='both', which='both', direction='in', labelsize=12)
                ax.minorticks_on()

    # Add colorbar to the right of the plot
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label(target_display_name, rotation=270, fontsize=20, labelpad=20)
    cbar_ax.tick_params(labelsize=12)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)

    # plt.savefig(f'outputs/make_plots_output/corner_dor_coloured{suffix}.pdf')

    return fig


if __name__ == "__main__":

    FEATURES = [
        'MgFe',
        '[M/H]_mean_mass',
        # '[M/H]_err_mass',
        'velDisp_ppxf_res',
        # 'velDisp_ppxf_err_res',
        'age_mean_mass',
        # 'age_err_mass',
    ]

    # Define a mapping between feature names and display names
    FEATURE_DISPLAY_NAMES = {
        'MgFe': r'$[\mathrm{Mg}/\mathrm{Fe}]$ (dex)',
        '[M/H]_mean_mass': r'$[\mathrm{M}/\mathrm{H}]$ (dex)',
        'velDisp_ppxf_res': r'$\sigma_{\star}$ (km/s)',
        'age_mean_mass': r'Age (Gyr)',
        # 'age_err_mass': r'Age Error (Gyr)',
        # 'velDisp_ppxf_err_res': r'vderr',
        # '[M/H]_err_mass':'met_err'
    }

    # Set a display name for the target variable
    TARGET_DISPLAY_NAME = r'$\mathrm{DoR}$'

    df = pd.read_csv('../data/E-INSPIRE_I_master_catalogue.csv')

    create_corner_plots(
        df,
        FEATURES,
        feature_display_names=FEATURE_DISPLAY_NAMES,
        target='DoR',
        target_display_name=TARGET_DISPLAY_NAME
    )