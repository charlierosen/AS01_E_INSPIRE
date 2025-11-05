from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..config.models import ModelTrainingConfig
from ..data import load_desi_catalogue
from .data import prepare_inspire_training_sets
from .model import ModelSet, TrainingArtifacts, train_and_predict_desi


class InspireDesiTrainingPipeline:
    """
    End-to-end orchestrator for preparing INSPIRE/ DESI datasets and training
    the DoR regression model.
    """

    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def build_datasets(self) -> ModelSet:
        train_df, test_df = prepare_inspire_training_sets(self.config)
        keep_columns = list(dict.fromkeys(self.config.dataset.features + ["univ_age"]))
        desi_df = load_desi_catalogue(self.config.paths, keep_columns=keep_columns + ["tau"])
        return ModelSet(train=train_df, test=test_df, desi=desi_df)

    def run(self) -> TrainingArtifacts:
        model_set = self.build_datasets()
        return train_and_predict_desi(model_set, self.config)
