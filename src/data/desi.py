from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.io import fits

from ..config.models import PipelinePaths


def load_desi_catalogue(
    paths: PipelinePaths,
    filename: str = "dr1_galaxy_lowZ_stellarmass_UCMGs_10.3_NOEMISSION_sigma150_with_tuni_andMH.fits",
    *,
    keep_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load and pre-process the DESI catalogue used for out-of-sample predictions.

    The transformation mirrors the original notebook implementation to ensure
    identical numerical results.
    """
    fits_path = paths.resolve_data(filename)
    if not fits_path.exists():
        raise FileNotFoundError(f"Expected DESI FITS file not found: {fits_path}")

    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None:
                data = hdu.data
                if data is not None:
                    break
        else:
            raise ValueError(f"No table HDU found in {fits_path}")

    df = pd.DataFrame(data)

    for column in df.columns:
        if df[column].dtype.kind in {"S", "U"}:
            df[column] = df[column].astype(str).str.strip()

    df = df.replace([-999, -99, -9999], pd.NA)

    desi_mapping = {
        "Metal_SL": "met",
        "MASS_CG": "logM",
        "SHAPE_R_kpc": "rad_kpc",
        "VD_SL": "vdisp",
        "AGE_SL": "lin_age",
        "t_uni": "univ_age",
    }

    df = df.rename(columns=desi_mapping)
    df["logM"] = np.log10(df["logM"])
    df["lin_age"] = df["lin_age"] / 1_000_000_000  # Convert years to Gyr
    df["tau"] = (df["univ_age"] - df["lin_age"]) / df["univ_age"]

    columns = keep_columns or ["met", "logM", "rad_kpc", "vdisp", "lin_age", "tau", "univ_age"]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected DESI columns: {missing}")

    return df[columns].copy()
