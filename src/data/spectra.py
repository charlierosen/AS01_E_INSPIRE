from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class FittedSpectrumMetadata:
    filename: str
    plate: int
    mjd: int
    fiber: int

    @classmethod
    def from_sdss_id(cls, sdss_id: str) -> "FittedSpectrumMetadata":
        match = re.match(r"spec-(\d{4})-(\d{5})-(\d{4})\.fits", sdss_id)
        if not match:
            raise ValueError(f"Invalid SDSS identifier: {sdss_id}")
        plate, mjd, fiber = map(int, match.groups())
        return cls(filename=sdss_id, plate=plate, mjd=mjd, fiber=fiber)


@dataclass
class Spectrum:
    wavelength: np.ndarray
    flux: np.ndarray
    inverse_variance: np.ndarray
    redshift: float
    metadata: FittedSpectrumMetadata


def _clean_and_normalise(
    wavelength: np.ndarray, flux: np.ndarray, inverse_variance: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror the legacy normalisation about 5300 Å."""
    good_idx = np.isfinite(flux) & np.isfinite(inverse_variance)
    wavelength, flux, inverse_variance = (
        wavelength[good_idx],
        flux[good_idx],
        inverse_variance[good_idx],
    )

    target_wavelength = 5300.0
    closest_idx = np.argmin(np.abs(wavelength - target_wavelength))
    normalising_wave = wavelength[closest_idx]

    if abs(normalising_wave - target_wavelength) > 2.0:
        raise ValueError(
            f"Closest wavelength {normalising_wave:.2f}Å deviates from target 5300Å"
        )

    normalising_flux = flux[closest_idx]
    if not np.isfinite(normalising_flux) or normalising_flux == 0:
        raise ValueError(f"Invalid normalising flux value: {normalising_flux}")

    flux = flux / normalising_flux
    inverse_variance = inverse_variance * (normalising_flux ** 2)
    return wavelength, flux, inverse_variance


def load_sdss_spectrum(base_dir: Path, sdss_id: str) -> Spectrum:
    """
    Load a single SDSS spectrum and normalise it following the legacy approach.
    """
    metadata = FittedSpectrumMetadata.from_sdss_id(sdss_id)
    fits_path = base_dir / sdss_id

    if not fits_path.exists():
        raise FileNotFoundError(f"SDSS spectrum not found: {fits_path}")

    with fits.open(fits_path) as hdul:
        coadd = hdul[1].data
        specobj = hdul[2].data

        flux = coadd["flux"]
        loglam = coadd["loglam"]
        inverse_variance = coadd["ivar"]

        redshift = float(specobj["Z"][0])

        wavelength = 10 ** loglam
        wavelength *= 1 / (1 + redshift)

        wavelength, flux, inverse_variance = _clean_and_normalise(
            wavelength, flux, inverse_variance
        )

    return Spectrum(
        wavelength=wavelength,
        flux=flux,
        inverse_variance=inverse_variance,
        redshift=redshift,
        metadata=metadata,
    )
