from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ..config.models import ModelTrainingConfig


@dataclass
class ModelSet:
    train: pd.DataFrame
    test: pd.DataFrame
    desi: pd.DataFrame


@dataclass
class TrainingArtifacts:
    model: RandomForestRegressor
    scaler: StandardScaler
    desi_predictions: pd.DataFrame
    feature_importance: Dict[str, float]


class ModelTrainer:
    """
    High-level interface mirroring the DESI notebook training logic.
    """

    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def _validate_features(self, df: pd.DataFrame) -> None:
        missing = [feature for feature in self.config.dataset.features if feature not in df.columns]
        if missing:
            raise ValueError(f"Missing features in dataset: {missing}")

    def train_final_model(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[RandomForestRegressor, StandardScaler]:
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        self._validate_features(combined_df)

        X = combined_df[self.config.dataset.features]
        y = combined_df[self.config.dataset.target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestRegressor(**self.config.model.dict())
        model.fit(X_scaled, y)
        return model, scaler

    def predict_desi(
        self,
        model: RandomForestRegressor,
        scaler: StandardScaler,
        desi_df: pd.DataFrame,
    ) -> pd.Series:
        self._validate_features(desi_df)
        X_desi = desi_df[self.config.dataset.features]
        X_scaled = scaler.transform(X_desi)
        return pd.Series(model.predict(X_scaled), index=desi_df.index, name="DoR")


def train_and_predict_desi(
    model_set: ModelSet,
    config: ModelTrainingConfig,
) -> TrainingArtifacts:
    trainer = ModelTrainer(config)
    model, scaler = trainer.train_final_model(model_set.train, model_set.test)
    predictions = trainer.predict_desi(model, scaler, model_set.desi)

    desi_with_predictions = model_set.desi.copy()
    desi_with_predictions["DoR"] = predictions

    feature_importance = dict(
        zip(config.dataset.features, model.feature_importances_)
    )

    return TrainingArtifacts(
        model=model,
        scaler=scaler,
        desi_predictions=desi_with_predictions,
        feature_importance=feature_importance,
    )
