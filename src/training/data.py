from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..config.models import ModelTrainingConfig
from ..data import (
    load_inspire_master_catalogue,
    load_inspire_snr,
    load_inspire_stelpop,
    load_refitting_results,
)
from legacy.scripts.ned_calculator import NedCalculator


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected dataset not found: {path}")
    return pd.read_csv(path)


def _prepare_restricted(
    config: ModelTrainingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paths = config.paths
    files = config.dataset.data_files

    test_df = _load_csv(paths.resolve_output(files.restricted_output))
    train_df = load_inspire_master_catalogue(paths, files.master_catalogue)
    train_df["logAge"] = np.log10(train_df["age_mean_mass"]) + 9
    train_df["lin_age"] = train_df["age_mean_mass"]
    train_df["lin_age_err"] = train_df["age_err_mass"]

    train_mapping = {
        "logAge": "logAge",
        "age_err_mass": "age_err",
        "meanRadkpc_r": "rad_kpc",
        "logM*": "logM",
        "velDisp_ppxf_res": "vdisp",
        "[M/H]_mean_mass": "met",
        "[M/H]_err_mass": "met_err",
    }
    train_df = train_df.rename(columns=train_mapping)
    return train_df, test_df


def _prepare_unrestricted(
    config: ModelTrainingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paths = config.paths
    files = config.dataset.data_files

    test_df = load_refitting_results(paths, files.refitting_results)
    train_df = load_inspire_stelpop(paths, files.stellar_population)
    restricted_df = load_inspire_master_catalogue(paths, files.master_catalogue)

    restricted_cols = [
        "GALAXY ID",
        "univ_age",
        "MgFe",
        "velDisp_ppxf_res",
        "DoR",
        "logM*",
        "meanRadkpc_kids",
        "age_err_mass",
        "SNR",
        "[M/H]_mean_mass",
    ]
    restricted_df = restricted_df[restricted_cols]

    dor_df = _load_csv(paths.resolve_data(files.dor_reference))
    dor_df = dor_df.rename(columns={"dor_2": "dor_4"})
    restricted_df = restricted_df.rename(columns={"DoR": "dor_26"})

    restricted_df = restricted_df.sort_values("[M/H]_mean_mass", ascending=False)
    dor_df = dor_df.sort_values("[M/H]_mean_1", ascending=False)

    restricted_df["[M/H]_mean_mass_rounded"] = restricted_df["[M/H]_mean_mass"].round(5)
    dor_df["[M/H]_mean_1_rounded"] = dor_df["[M/H]_mean_1"].round(5)

    restricted_df = restricted_df.merge(
        dor_df,
        left_on="[M/H]_mean_mass_rounded",
        right_on="[M/H]_mean_1_rounded",
        how="left",
    ).drop(columns=["[M/H]_mean_mass_rounded", "[M/H]_mean_1_rounded"])

    train_df = train_df.merge(
        restricted_df, left_on="sexa_id", right_on="GALAXY ID", how="inner"
    )

    age_columns = ["age_noboot", "age_boot", "age_minus", "ages_plus"]
    train_df["logAge_err"] = np.log10(train_df["age_err_mass"]) + 9
    train_df["logAge"] = train_df[age_columns].mean(axis=1)
    train_df["lin_age"] = 10 ** (train_df["logAge"] - 9)
    train_df["lin_age_err"] = train_df["age_err_mass"]

    metal_columns = ["metals_noboot", "metals_boot", "metals_minus", "metal_plus"]
    train_df["met_err"] = train_df[metal_columns].std(axis=1)
    train_df["met"] = train_df[metal_columns].mean(axis=1)

    mapping = {
        "MgFe": "MgFe",
        "age_err": "age_err",
        "meanRadkpc_kids": "rad_kpc",
        "logM*": "logM",
        "velDisp_ppxf_res": "vdisp",
        "dor_4": "DoR",
        "met_err": "met_err",
    }
    train_df = train_df.rename(columns=mapping)

    return train_df, test_df


def prepare_inspire_training_sets(
    config: ModelTrainingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Re-implementation of the notebook ``prepare_data`` function with typing.
    """
    if config.dataset.restricted:
        train_df, test_df = _prepare_restricted(config)
    else:
        train_df, test_df = _prepare_unrestricted(config)

    test_df = test_df.copy()
    train_df = train_df.copy()

    # Test set feature engineering
    test_df["lin_age"] = (test_df["age_unr"] + test_df["age_rmax"]) / 2
    test_df["met"] = (test_df["metal_unr"] + test_df["metal_rmax"]) / 2

    snr_df = load_inspire_snr(config.paths)
    test_df = test_df.merge(
        snr_df[["ID_INSPIRE", "SNR_MEAN"]], on="ID_INSPIRE", how="left"
    )

    test_df["Reff_median_SDSS_arcsec"] = (
        (test_df["Reff_median_KIDS_arcsec"] + 0.35) / 0.86
    ) ** 2
    test_df["rad_kpc"] = test_df.apply(
        lambda row: row["Reff_median_SDSS_arcsec"]
        * NedCalculator(z=row["zspec_XSH"]).get_kpc_DA(),
        axis=1,
    )

    test_mapping = {
        "age_stdev": "lin_age_err",
        "rad_kpc": "rad_kpc",
        "logMstarzspecTOT_S20": "logM",
        "Vdisp_XSH": "vdisp",
        "AlphaFe": "MgFe",
        "met": "met",
        "metal_stdev": "met_err",
        "SNR_MEAN": "SNR",
    }
    test_df = test_df.rename(columns=test_mapping)

    # Tau calculations
    test_df["tau"] = (test_df["tuni"] - test_df["lin_age"]) / (test_df["tuni"])
    train_df["tau"] = (train_df["univ_age"] - train_df["lin_age"]) / (
        train_df["univ_age"]
    )

    if config.dataset.use_principal_components:
        test_df["lin_age_err"] = test_df["lin_age_err"] / test_df["lin_age"]
        train_df["lin_age_err"] = train_df["lin_age_err"] / train_df["lin_age"]
        test_df["met_err"] = test_df["met_err"] / test_df["met"]
        train_df["met_err"] = train_df["met_err"] / train_df["met"]

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
