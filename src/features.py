"""Leakage-safe preprocessing.

Everything that learns from data (imputation statistics, scaling, one-hot
categories) lives INSIDE an sklearn Pipeline, so it is fitted only when
``pipeline.fit(X_train, y_train)`` is called - never on validation/test data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ID_COLUMN, NUMERIC_FEATURES, REQUIRED_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSpec:
    """Ordered model input columns."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def all(self) -> list[str]:
        """Model input columns in the exact order the pipeline expects."""
        return [*self.numeric, *self.categorical]

    def to_dict(self) -> dict[str, list[str]]:
        return {"numeric": list(self.numeric), "categorical": list(self.categorical)}


def get_feature_spec(exclude: Iterable[str] = ()) -> FeatureSpec:
    """Define the model features.

    Numeric: tenure, MonthlyCharges, TotalCharges.
    Categorical: every other column except customerID (identifier) and Churn (target).
    ``exclude`` removes extra columns (e.g. for a fairness experiment).
    """
    excluded = set(exclude) | {ID_COLUMN, TARGET_COLUMN}
    numeric = tuple(c for c in NUMERIC_FEATURES if c not in excluded)
    categorical = tuple(
        c for c in REQUIRED_COLUMNS if c not in excluded and c not in NUMERIC_FEATURES
    )
    spec = FeatureSpec(numeric, categorical)
    assert ID_COLUMN not in spec.all and TARGET_COLUMN not in spec.all
    return spec


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    """ColumnTransformer: numeric -> impute(median)+scale; categorical -> impute(mode)+one-hot."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(spec.numeric)),
            ("cat", categorical_pipe, list(spec.categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator: BaseEstimator, spec: FeatureSpec) -> Pipeline:
    """Full pipeline: preprocessing + model. Step names: ``preprocess`` and ``model``."""
    return Pipeline([("preprocess", build_preprocessor(spec)), ("model", estimator)])


def split_xy(df: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y). X contains only model features in pipeline order; y is Churn (0/1)."""
    missing = [c for c in [*spec.all, TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    return df[spec.all].copy(), df[TARGET_COLUMN].astype(int)


def build_reference_values(X_train: pd.DataFrame, spec: FeatureSpec) -> dict[str, Any]:
    """Typical value per feature (median / mode) from TRAINING data, for explanations."""
    ref: dict[str, Any] = {}
    for col in spec.numeric:
        ref[col] = float(X_train[col].median())
    for col in spec.categorical:
        value = X_train[col].mode(dropna=True).iloc[0]
        ref[col] = value.item() if hasattr(value, "item") else value
    return ref


def feature_names_out(pipeline: Pipeline) -> Sequence[str]:
    """Expanded feature names after one-hot encoding (for fitted pipelines)."""
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())
