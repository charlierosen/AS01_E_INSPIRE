from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..config.models import PipelinePaths


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected file does not exist: {path}")
    return pd.read_csv(path, **kwargs)


def load_inspire_master_catalogue(
    paths: PipelinePaths,
    filename: str = "E-INSPIRE_I_master_catalogue.csv",
    *,
    usecols: Optional[list[str]] = None,
    dtype: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Load the master INSPIRE catalogue used throughout the analysis.

    Parameters
    ----------
    paths:
        Base paths configuration.
    filename:
        Name of the CSV file located under ``data/``.
    usecols:
        Optional subset of columns to load.
    dtype:
        Optional column type overrides.
    """

    csv_path = paths.resolve_data(filename)
    return _read_csv(csv_path, usecols=usecols, dtype=dtype)


def load_inspire_snr(
    paths: PipelinePaths,
    filename: str = "INSPIRE_SNR.csv",
    *,
    usecols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load the SNR ancillary table."""
    csv_path = paths.resolve_data(filename)
    return _read_csv(csv_path, usecols=usecols)


def load_inspire_stelpop(
    paths: PipelinePaths,
    filename: str = "stel_pop_fit_allZ.csv",
) -> pd.DataFrame:
    """Load the stellar population catalogue used for model training."""
    csv_path = paths.resolve_data(filename)
    return _read_csv(csv_path)


def load_refitting_results(
    paths: PipelinePaths,
    filename: str = "INSPIRE_stelpop.csv",
) -> pd.DataFrame:
    """
    Load the INSPIRE test set artefacts produced by the legacy pipeline.

    This corresponds to the CSV exported from the notebook refitting step.
    """
    csv_path = paths.resolve_data(filename)
    return _read_csv(csv_path)
