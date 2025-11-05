"""
Training workflow for machine learning models based on the INSPIRE catalogue.
"""

from .data import prepare_inspire_training_sets
from .model import (
    ModelSet,
    ModelTrainer,
    TrainingArtifacts,
    train_and_predict_desi,
)
from .pipeline import InspireDesiTrainingPipeline

__all__ = [
    "prepare_inspire_training_sets",
    "ModelSet",
    "ModelTrainer",
    "TrainingArtifacts",
    "train_and_predict_desi",
    "InspireDesiTrainingPipeline",
]
