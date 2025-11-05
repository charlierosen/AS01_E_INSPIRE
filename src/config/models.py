from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator


class PipelinePaths(BaseModel):
    """Centralised definition of filesystem locations used by the pipelines."""

    data_root: Path = Field(default=Path("data"))
    outputs_root: Path = Field(default=Path("outputs"))
    cluster_results_dir: Path = Field(default=Path("outputs/cluster_results"))
    stacked_fits_dir: Path = Field(default=Path("outputs/stacked_fits"))
    ppxf_fits_dir: Path = Field(default=Path("outputs/ppxf_fits"))
    sfh_plots_dir: Path = Field(default=Path("outputs/sfh_plots"))
    stacked_catalogues_dir: Path = Field(default=Path("outputs/stacked_catalogues"))
    plots_dir: Path = Field(default=Path("outputs/make_plots_output"))

    def ensure_directories(self) -> None:
        for path in {
            self.outputs_root,
            self.cluster_results_dir,
            self.stacked_fits_dir,
            self.ppxf_fits_dir,
            self.sfh_plots_dir,
            self.stacked_catalogues_dir,
            self.plots_dir,
        }:
            path.mkdir(parents=True, exist_ok=True)

    def resolve_data(self, *parts: str) -> Path:
        return self.data_root.joinpath(*parts)

    def resolve_output(self, *parts: str) -> Path:
        return self.outputs_root.joinpath(*parts)


class RandomForestHyperparameters(BaseModel):
    n_estimators: int = 50
    max_depth: int = 8
    max_features: float = 0.8
    max_samples: float = 0.7
    min_samples_leaf: int = 3
    min_samples_split: int = 5
    random_state: int = 1


class ModelParameters(BaseModel):
    features: List[str]
    target: str = "DoR"
    n_splits: int = 5
    small_boundary: float = 0.35
    large_boundary: float = 0.6
    hyperparameters: RandomForestHyperparameters = Field(
        default_factory=RandomForestHyperparameters
    )
    feature_display_names: Dict[str, str] = Field(default_factory=dict)
    combined_plot_filename: str = "regression_performance.pdf"
    residuals_plot_filename: str = "residuals.pdf"
    importance_plot_filename: str = "feature_importance.pdf"
    predictions_csv: Path = Path("outputs/cluster_results/regression_clusters.csv")

    @validator("features")
    def ensure_features(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("At least one feature must be provided.")
        return value

    @validator("large_boundary")
    def validate_boundaries(
        cls, large_boundary: float, values: Dict[str, float]
    ) -> float:
        small_boundary = values.get("small_boundary", 0.0)
        if small_boundary >= large_boundary:
            raise ValueError("large_boundary must be greater than small_boundary.")
        return large_boundary


class StackConfig(BaseModel):
    """Configuration for spectral stacking."""

    smoothing_mode: str = "match_max_velocity"
    base_factor: float = 0.001635
    factor_scaling: Tuple[float, float, float] = (4.0, 2.7, 2.5)
    ensure_descending_dor: bool = True

    def scaling_factors(self) -> Tuple[float, float, float]:
        base = self.base_factor
        return tuple(base * scale for scale in self.factor_scaling)


class PPXFFitConfig(BaseModel):
    """Configuration for running the pPXF fitting on stacked spectra."""

    nrand: int = 9
    tie_balmer: bool = True
    limit_doublets: bool = True
    metal_range: Tuple[float, float] = (-2.0, 0.5)


class InspireDataFiles(BaseModel):
    master_catalogue: str = "E-INSPIRE_I_master_catalogue.csv"
    snr_table: str = "INSPIRE_SNR.csv"
    stellar_population: str = "stel_pop_fit_allZ.csv"
    refitting_results: str = "INSPIRE_stelpop.csv"
    restricted_output: str = "refitting_results/output.csv"
    dor_reference: str = "ppxf_stel_pop_test2_small.csv"


class TrainingDatasetConfig(BaseModel):
    features: List[str]
    target: str = "DoR"
    restricted: bool = False
    use_principal_components: bool = False
    random_states: List[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    data_files: InspireDataFiles = Field(default_factory=InspireDataFiles)

    @validator("features")
    def ensure_features(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("Training dataset requires at least one feature.")
        return value


class ModelTrainingConfig(BaseModel):
    paths: PipelinePaths = Field(default_factory=PipelinePaths)
    dataset: TrainingDatasetConfig = Field(default_factory=TrainingDatasetConfig)
    model: RandomForestHyperparameters = Field(default_factory=RandomForestHyperparameters)


class ClusteringConfig(BaseModel):
    """High level configuration for the full stacking + pPXF pipeline."""

    paths: PipelinePaths = Field(default_factory=PipelinePaths)
    model: ModelParameters = Field(default_factory=ModelParameters)
    stacking: StackConfig = Field(default_factory=StackConfig)
    ppxf: PPXFFitConfig = Field(default_factory=PPXFFitConfig)
