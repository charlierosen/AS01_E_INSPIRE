from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config.models import ClusteringConfig
from ..ml import (
    KFoldResults,
    plot_combined_regression_and_residuals,
    plot_feature_importance,
    plot_residuals,
    run_kfold_regression,
)


@dataclass
class ClusteringArtifacts:
    results: KFoldResults
    predictions: pd.DataFrame
    clusters_df: pd.DataFrame
    plot_paths: Dict[str, Path]


class DoRClusteringPipeline:
    """
    Execute the RandomForest regression on the INSPIRE catalogue and export
    DoR-based cluster assignments.
    """

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.paths = config.paths
        self.paths.ensure_directories()

    def _decorate_with_predictions(
        self, df: pd.DataFrame, predictions: pd.DataFrame
    ) -> pd.DataFrame:
        enriched = df.reset_index(drop=True).copy()
        enriched["Predicted_DoR"] = predictions["predicted_DoR"]
        enriched["Cluster"] = 1
        enriched.loc[
            enriched["Predicted_DoR"] <= self.config.model.small_boundary, "Cluster"
        ] = 0
        enriched.loc[
            enriched["Predicted_DoR"] >= self.config.model.large_boundary, "Cluster"
        ] = 2

        enriched["SDSS_ID"] = [
            f"spec-{int(plate):04d}-{int(mjd):05d}-{int(fiber):04d}.fits"
            for plate, mjd, fiber in zip(
                enriched["plate"], enriched["mjd"], enriched["fiberid"]
            )
        ]
        return enriched[["SDSS_ID", "Cluster", "DoR", "Predicted_DoR"]]

    def _write_outputs(
        self, results: KFoldResults, clusters: pd.DataFrame
    ) -> Dict[str, Path]:
        plot_paths: Dict[str, Path] = {}

        combined_plot = self.paths.plots_dir / self.config.model.combined_plot_filename
        residual_plot = self.paths.plots_dir / self.config.model.residuals_plot_filename
        importance_plot = self.paths.plots_dir / self.config.model.importance_plot_filename

        combined_plot.parent.mkdir(parents=True, exist_ok=True)
        residual_plot.parent.mkdir(parents=True, exist_ok=True)
        importance_plot.parent.mkdir(parents=True, exist_ok=True)

        plot_combined_regression_and_residuals(results, self.config.model, output_path=combined_plot)
        plot_residuals(results, self.config.model, output_path=residual_plot)
        plot_feature_importance(results, self.config.model, output_path=importance_plot)

        plot_paths["combined"] = combined_plot
        plot_paths["residuals"] = residual_plot
        plot_paths["feature_importance"] = importance_plot

        csv_path = (
            self.config.model.predictions_csv
            if self.config.model.predictions_csv.is_absolute()
            else Path(self.config.model.predictions_csv)
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        clusters.to_csv(csv_path, index=False)
        plot_paths["clusters_csv"] = csv_path

        return plot_paths

    def run(self, df: pd.DataFrame) -> ClusteringArtifacts:
        results = run_kfold_regression(df, self.config.model)
        predictions = results.predictions_dataframe()
        clusters_df = self._decorate_with_predictions(df, predictions)
        plot_paths = self._write_outputs(results, clusters_df)
        return ClusteringArtifacts(results, predictions, clusters_df, plot_paths)

    def read_cluster_groups(self, csv_path: Path) -> Dict[str, List[List[str]]]:
        """
        Read cluster assignment CSV files and return mapping of method to
        grouped SDSS spectra.
        """
        df = pd.read_csv(csv_path)
        groups = [
            df[df["Cluster"] == cluster]["SDSS_ID"].tolist()
            for cluster in sorted(df["Cluster"].unique())
        ]
        return {"REGRESSION": groups}
