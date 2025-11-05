from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from ..config.models import ClusteringConfig
from .clustering import DoRClusteringPipeline, ClusteringArtifacts
from .stacking import SpectralStacker, StackingSummary
from .ppxf import PPXFPipeline


@dataclass
class PipelineResult:
    clustering: ClusteringArtifacts
    stacking: Dict[str, List[StackingSummary]]


class FullPipeline:
    """
    Reproducible orchestration for clustering, stacking, and pPXF fitting.
    """

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.paths = config.paths
        self.clustering_pipeline = DoRClusteringPipeline(config)
        self.stacker = SpectralStacker(config)
        self.ppxf_pipeline = PPXFPipeline(config)

    def prepare_outputs(self, clean: bool = False) -> None:
        self.paths.ensure_directories()
        if clean:
            for directory in [
                self.paths.outputs_root,
                self.paths.cluster_results_dir,
                self.paths.stacked_fits_dir,
                self.paths.ppxf_fits_dir,
                self.paths.sfh_plots_dir,
                self.paths.stacked_catalogues_dir,
            ]:
                if directory.exists():
                    for item in directory.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)

    def run(self, inspire_df: pd.DataFrame, clean_outputs: bool = False) -> PipelineResult:
        self.prepare_outputs(clean=clean_outputs)
        clustering_artifacts = self.clustering_pipeline.run(inspire_df)

        csv_path = clustering_artifacts.plot_paths["clusters_csv"]
        cluster_groups = self.clustering_pipeline.read_cluster_groups(csv_path)

        summaries = self.stacker.stack_all(
            cluster_groups, scaling_factors=self.config.stacking.scaling_factors()
        )

        self.ppxf_pipeline.run()

        return PipelineResult(
            clustering=clustering_artifacts,
            stacking=summaries,
        )
