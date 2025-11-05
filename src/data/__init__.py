"""
Data loading utilities for the refactored pipelines.

This package provides typed helpers to read INSPIRE and DESI catalogues as well
as individual spectra used throughout the analysis.
"""

from .inspire import (
    load_inspire_master_catalogue,
    load_inspire_snr,
    load_inspire_stelpop,
    load_refitting_results,
)
from .desi import load_desi_catalogue
from .spectra import FittedSpectrumMetadata, load_sdss_spectrum

__all__ = [
    "load_inspire_master_catalogue",
    "load_inspire_snr",
    "load_inspire_stelpop",
    "load_refitting_results",
    "load_desi_catalogue",
    "load_sdss_spectrum",
    "FittedSpectrumMetadata",
]
