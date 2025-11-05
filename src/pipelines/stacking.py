from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from astropy.io import fits

from ..config.models import ClusteringConfig
from ..data import (
    load_inspire_master_catalogue,
    load_sdss_spectrum,
    FittedSpectrumMetadata,
)


@dataclass
class StackingSummary:
    cluster_label: str
    n_spectra: int
    mean_mgfe: float
    mean_age: float
    mean_metallicity: float
    mean_dor: float
    sigma_fin: float


class SpectralStacker:
    """
    Stack SDSS spectra belonging to a cluster and persist the combined FITS file.
    """

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.catalogue = load_inspire_master_catalogue(config.paths)
        self.spectra_root = config.paths.resolve_data("fits_shortlist")

    def _calculate_sigma_diff(self, wavelength: np.ndarray, sigma_gal: float, sigma_fin: float):
        c = 299_792.458
        ln_wave = np.log(wavelength)
        d_ln_wave = (ln_wave[-1] - ln_wave[0]) / (len(wavelength) - 1)
        velscale = c * d_ln_wave

        sigma_diff_kms = np.sqrt(max(sigma_fin ** 2 - sigma_gal ** 2, 0))
        sigma_gal_pxl = sigma_gal / velscale
        sigma_fin_pxl = sigma_fin / velscale
        sigma_diff = np.sqrt(max(sigma_fin_pxl ** 2 - sigma_gal_pxl ** 2, 0))
        return sigma_diff, sigma_diff_kms

    def _smooth_spectra(self, spectra, vdisps):
        sigma_fin = max(vdisps)
        smoothed = []

        for spectrum, sigma_gal in zip(spectra, vdisps):
            if sigma_fin <= sigma_gal:
                smoothed_flux = spectrum.flux
            else:
                sigma_diff, _ = self._calculate_sigma_diff(
                    spectrum.wavelength, sigma_gal, sigma_fin
                )
                smoothed_flux = gaussian_filter1d(spectrum.flux, sigma_diff)
            smoothed.append(
                (
                    spectrum.wavelength,
                    smoothed_flux,
                    spectrum.inverse_variance,
                )
            )

        return smoothed, sigma_fin

    def _resample_to_common_grid(self, smoothed_data):
        wave_min = max(s[0][0] for s in smoothed_data) + 0.1
        wave_max = min(s[0][-1] for s in smoothed_data) - 0.1
        wave_common = np.logspace(np.log10(wave_min), np.log10(wave_max), num=3828)

        resampled = []
        for wave, flux, ivar in smoothed_data:
            flux_interp = interpolate.interp1d(wave, flux, bounds_error=True)
            ivar_interp = interpolate.interp1d(wave, ivar, bounds_error=True)
            resampled.append(
                (
                    wave_common,
                    flux_interp(wave_common),
                    ivar_interp(wave_common),
                )
            )

        return resampled

    def _combine_spectra(self, aligned):
        wavelength = aligned[0][0]
        fluxes = np.array([spec[1] for spec in aligned])
        ivars = np.array([spec[2] for spec in aligned])

        combined_flux = np.mean(fluxes, axis=0)

        mask = np.all((ivars == 0) | np.isinf(ivars), axis=0)
        combined_ivar = np.zeros_like(ivars[0])

        valid = ~mask
        if np.any(valid):
            valid_mask = (ivars != 0) & np.isfinite(ivars)
            valid_count = np.sum(valid_mask, axis=0)
            sum_inv_ivar = np.sum(
                np.divide(1.0, np.where(valid_mask, ivars, np.inf)), axis=0
            )
            has_valid = valid_count > 0
            combined_ivar[has_valid] = (valid_count[has_valid] ** 2) / sum_inv_ivar[has_valid]

        return wavelength, combined_flux, combined_ivar

    def _save_fits(
        self,
        wavelength: np.ndarray,
        flux: np.ndarray,
        ivar: np.ndarray,
        summary: StackingSummary,
        vdisp_avg: float,
        vdisp_err: float,
        log_mass: float,
        log_mass_err: float,
        radius: float,
        radius_err: float,
    ) -> str:
        coadd_data = np.zeros(
            len(wavelength),
            dtype=[("flux", "f8"), ("wave", "f8"), ("ivar", "f8"), ("wdisp", "f8")],
        )
        coadd_data["flux"] = flux
        coadd_data["wave"] = wavelength
        coadd_data["ivar"] = ivar
        coadd_data["wdisp"] = np.full_like(wavelength, 2.76 / 2.355)

        primary = fits.PrimaryHDU()
        coadd_hdu = fits.BinTableHDU(data=coadd_data, name="COADD")

        primary.header["HIERARCH NAME"] = f"stacked_{summary.cluster_label}"
        primary.header["HIERARCH z"] = 0
        primary.header["HIERARCH ALPHA"] = summary.mean_mgfe
        primary.header["HIERARCH SIGMA_MAX"] = summary.sigma_fin
        primary.header["HIERARCH SIGMA"] = vdisp_avg
        primary.header["HIERARCH SIGMA_ERR"] = vdisp_err
        primary.header["HIERARCH logM"] = log_mass
        primary.header["HIERARCH errlogM"] = log_mass_err
        primary.header["HIERARCH meanRadkpc_r"] = radius
        primary.header["HIERARCH meanRadErrkpc_r"] = radius_err

        hdul = fits.HDUList([primary, coadd_hdu])
        output_file = self.config.paths.stacked_fits_dir / f"stacked_{summary.cluster_label}.fits"
        hdul.writeto(output_file, overwrite=True)
        return str(output_file)

    def _cluster_statistics(self, spectra_ids: Sequence[str]):
        mgfe = []
        vdisps = []
        vdisp_errs = []
        ages = []
        metallicity = []
        dors = []
        masses = []
        mass_errs = []
        radii = []
        radii_errs = []

        spectra = []

        for sdss_id in spectra_ids:
            spectrum = load_sdss_spectrum(self.spectra_root, sdss_id)
            spectra.append(spectrum)
            meta = spectrum.metadata
            match = (self.catalogue["plate"] == meta.plate) & (
                self.catalogue["mjd"] == meta.mjd
            ) & (self.catalogue["fiberid"] == meta.fiber)
            row = self.catalogue.loc[match].iloc[0]

            mgfe.append(float(row["MgFe"]))
            vdisps.append(float(row["velDisp_ppxf_res"]))
            vdisp_errs.append(float(row["velDisp_ppxf_err_res"]))
            ages.append(float(row["age_mean_mass"]))
            metallicity.append(float(row["[M/H]_mean_mass"]))
            dors.append(float(row["DoR"]))
            masses.append(float(row["logM*"]))
            mass_errs.append(float(row["errlogM*"]))
            radii.append(float(row["meanRadkpc_r"]))
            radii_errs.append(float(row["meanRadErrkpc_r"]))

        return {
            "spectra": spectra,
            "mgfe": np.array(mgfe),
            "vdisps": np.array(vdisps),
            "vdisp_errs": np.array(vdisp_errs),
            "ages": np.array(ages),
            "metallicity": np.array(metallicity),
            "dors": np.array(dors),
            "masses": np.array(masses),
            "mass_errs": np.array(mass_errs),
            "radii": np.array(radii),
            "radii_errs": np.array(radii_errs),
        }

    def stack_cluster(self, spectra_ids: Sequence[str], cluster_label: str) -> StackingSummary:
        stats = self._cluster_statistics(spectra_ids)
        smoothed, sigma_fin = self._smooth_spectra(stats["spectra"], stats["vdisps"])
        resampled = self._resample_to_common_grid(smoothed)
        wavelength, flux, ivar = self._combine_spectra(resampled)

        mgfe_avg = float(np.mean(stats["mgfe"]))
        vdisp_avg = float(np.mean(stats["vdisps"]))
        vdisp_err = float(np.sqrt(np.sum(stats["vdisp_errs"] ** 2)) / len(stats["vdisp_errs"]))
        log_mass = float(np.mean(stats["masses"]))
        log_mass_err = float(np.sqrt(np.sum(stats["mass_errs"] ** 2)) / len(stats["mass_errs"]))
        radius = float(np.mean(stats["radii"]))
        radius_err = float(np.sqrt(np.sum(stats["radii_errs"] ** 2)) / len(stats["radii_errs"]))

        summary = StackingSummary(
            cluster_label=cluster_label,
            n_spectra=len(spectra_ids),
            mean_mgfe=mgfe_avg,
            mean_age=float(np.mean(stats["ages"])),
            mean_metallicity=float(np.mean(stats["metallicity"])),
            mean_dor=float(np.mean(stats["dors"])),
            sigma_fin=float(sigma_fin),
        )

        self._save_fits(
            wavelength,
            flux,
            ivar,
            summary,
            vdisp_avg,
            vdisp_err,
            log_mass,
            log_mass_err,
            radius,
            radius_err,
        )
        return summary

    def verify_cluster_order(self, spectra_groups: List[List[str]]) -> List[List[str]]:
        if not self.config.stacking.ensure_descending_dor:
            return spectra_groups

        cluster_stats = []
        for cluster in spectra_groups:
            d_o_r_values = []
            for sdss_id in cluster:
                meta = FittedSpectrumMetadata.from_sdss_id(sdss_id)
                mask = (self.catalogue["plate"] == meta.plate) & (
                    self.catalogue["mjd"] == meta.mjd
                ) & (self.catalogue["fiberid"] == meta.fiber)
                row = self.catalogue.loc[mask]
                if not row.empty:
                    d_o_r_values.append(row["DoR"].iloc[0])
            cluster_stats.append(np.mean(d_o_r_values) if d_o_r_values else 0.0)

        order = np.argsort(cluster_stats)[::-1]
        return [spectra_groups[i] for i in order]

    def stack_all(
        self,
        clusters: Dict[str, List[List[str]]],
        *,
        scaling_factors: Sequence[float],
    ) -> Dict[str, List[StackingSummary]]:
        summaries: Dict[str, List[StackingSummary]] = {}
        for method, groups in clusters.items():
            method_summaries: List[StackingSummary] = []
            ordered_groups = self.verify_cluster_order(groups)
            labels = [f"{method}_{idx}" for idx in range(len(ordered_groups))]

            for spectra_ids, _factor, label in zip(
                ordered_groups, scaling_factors, labels
            ):
                summary = self.stack_cluster(spectra_ids, label)
                method_summaries.append(summary)
            summaries[method] = method_summaries
        return summaries
