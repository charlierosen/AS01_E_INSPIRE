"""
Refactored pipeline package for the E-INSPIRE analysis.

This package provides a typed, modular interface that mirrors the legacy
workflows while exposing reusable building blocks for notebooks and scripts.
"""

from .config.models import (
    ClusteringConfig,
    ModelParameters,
    PipelinePaths,
    RandomForestHyperparameters,
    StackConfig,
    PPXFFitConfig,
    TrainingDatasetConfig,
    ModelTrainingConfig,
    InspireDataFiles,
)

__all__ = [
    "ClusteringConfig",
    "ModelParameters",
    "PipelinePaths",
    "RandomForestHyperparameters",
    "StackConfig",
    "PPXFFitConfig",
    "TrainingDatasetConfig",
    "ModelTrainingConfig",
    "InspireDataFiles",
]
