from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from ..config.models import ModelParameters


@dataclass
class FoldResult:
    indices: np.ndarray
    predictions: np.ndarray
    true_values: np.ndarray
    r2: float
    feature_importances: Dict[str, float]


@dataclass
class KFoldResults:
    folds: List[FoldResult]
    all_true: np.ndarray
    all_pred: np.ndarray
    all_residuals: np.ndarray
    fold_indices: np.ndarray
    plot_limits: Dict[str, float]
    feature_importances: Dict[str, float] = field(default_factory=dict)

    def mean_r2(self) -> float:
        return float(np.mean([fold.r2 for fold in self.folds]))

    def std_r2(self) -> float:
        return float(np.std([fold.r2 for fold in self.folds]))

    def overall_r2(self) -> float:
        return float(r2_score(self.all_true, self.all_pred))

    def overall_rmse(self) -> float:
        return float(np.sqrt(mean_squared_error(self.all_true, self.all_pred)))

    def overall_mae(self) -> float:
        return float(mean_absolute_error(self.all_true, self.all_pred))

    def predictions_dataframe(self) -> pd.DataFrame:
        indices = np.concatenate([fold.indices for fold in self.folds])
        predictions = np.concatenate([fold.predictions for fold in self.folds])
        df = pd.DataFrame({"index": indices, "predicted_DoR": predictions})
        return df.sort_values("index").reset_index(drop=True)


def _aggregate_feature_importance(
    folds: List[FoldResult], features: List[str]
) -> Dict[str, float]:
    importances = {feature: [] for feature in features}
    for fold in folds:
        for feature, value in fold.feature_importances.items():
            importances[feature].append(value)
    return {feature: float(np.mean(values)) for feature, values in importances.items()}


def run_kfold_regression(df: pd.DataFrame, config: ModelParameters) -> KFoldResults:
    """
    Execute the RandomForest regression with stratified-like K-Fold splitting.
    """
    X = df[config.features]
    y = df[config.target]

    kf = KFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.hyperparameters.random_state,
    )

    all_true = np.zeros_like(y, dtype=float)
    all_pred = np.zeros_like(y, dtype=float)
    all_residuals = np.zeros_like(y, dtype=float)
    fold_indices = np.zeros_like(y, dtype=int)

    folds: List[FoldResult] = []

    scaler = StandardScaler()

    for fold_number, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rf = RandomForestRegressor(**config.hyperparameters.dict())
        rf.fit(X_train_scaled, y_train)
        y_pred = rf.predict(X_test_scaled)

        fold_result = FoldResult(
            indices=test_idx,
            predictions=y_pred,
            true_values=y_test.values,
            r2=float(r2_score(y_test, y_pred)),
            feature_importances=dict(zip(config.features, rf.feature_importances_)),
        )
        folds.append(fold_result)

        all_true[test_idx] = y_test
        all_pred[test_idx] = y_pred
        all_residuals[test_idx] = y_test - y_pred
        fold_indices[test_idx] = fold_number

    min_val = min(float(all_true.min()), float(all_pred.min()))
    max_val = max(float(all_true.max()), float(all_pred.max()))
    buffer = (max_val - min_val) * 0.02
    plot_limits = {"min": min_val - buffer, "max": max_val + buffer}

    aggregated_importance = _aggregate_feature_importance(folds, config.features)

    return KFoldResults(
        folds=folds,
        all_true=all_true,
        all_pred=all_pred,
        all_residuals=all_residuals,
        fold_indices=fold_indices,
        plot_limits=plot_limits,
        feature_importances=aggregated_importance,
    )
