import contextlib
import io
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # macOS Accelerate

import numpy as np
import pandas as pd
from copy import copy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

from astropy.io import fits
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib

from src.ned_calculator import NedCalculator
from src.der_snr import DER_SNR

tie_balmer = True
limit_doublets = True
c = 299792.458
regul_err = 0.1
vel = 0
moments = [4, 2, 2]
gas_reddening = 0


def _bootstrap_residuals(model, resid):
    eps = resid * (2 * np.random.randint(2, size=resid.size) - 1)
    return model + eps


def _sfh_from_weights(weights_star_flat, reg_dim, age_grid, univ_age):
    weights_2d = weights_star_flat.reshape(reg_dim)
    wei1 = weights_2d.sum(axis=1)
    wei1 /= wei1.sum()
    wei1_rev = copy(wei1[::-1])
    ages = age_grid[:, 0]
    ages1 = (ages[-1] - ages)[::-1] + (ages[1] - ages[0])
    agesplot = np.concatenate([[0.], ages1])
    weiplot = np.concatenate([[0.], np.cumsum(wei1_rev)])
    agesplot = np.concatenate([agesplot, [agesplot[-1] + (agesplot[-1] - agesplot[-2])]])
    weiplot = np.concatenate([weiplot, [weiplot[-1]]])
    agesplot = [a if a < univ_age else univ_age for a in agesplot]
    return agesplot, weiplot


def _line2p(p1, p2, x):
    x1, y1 = p1
    x2, y2 = p2
    return (y2 - y1) / (x2 - x1) * x + (-(y2 - y1) / (x2 - x1) * x1 + y1)


def _line2p_rev(p1, p2, y):
    x1, y1 = p1
    x2, y2 = p2
    return (y - y1) / (y2 - y1) * (x2 - x1) + x1 if y2 != y1 else x2


def _get_values_from_sfh(univ_age, sfh_table, ycol):
    tt, tt90, tt100 = [], [], []

    ii = 0
    while sfh_table.iloc[ii][ycol] < 0.75:
        tt.append((ii, sfh_table.iloc[ii][ycol]))
        ii += 1
    tt.append((ii, sfh_table.iloc[ii][ycol]))
    yy = sfh_table.iloc[np.array(tt[-2:])[:, 0]]
    xx = _line2p_rev(yy[['time', ycol]].iloc[0].values, yy[['time', ycol]].iloc[1].values, 0.75)

    ii = 0
    while sfh_table.iloc[ii][ycol] < 0.9:
        tt90.append((ii, sfh_table.iloc[ii][ycol]))
        ii += 1
    tt90.append((ii, sfh_table.iloc[ii][ycol]))
    yy90 = sfh_table.iloc[np.array(tt90[-2:])[:, 0]]
    xx90 = _line2p_rev(yy90[['time', ycol]].iloc[0].values, yy90[['time', ycol]].iloc[1].values, 0.9)

    ii = 0
    while sfh_table.iloc[ii][ycol] < 0.998:
        tt100.append((ii, sfh_table.iloc[ii][ycol]))
        ii += 1
    tt100.append((ii, sfh_table.iloc[ii][ycol]))
    yy100 = sfh_table.iloc[np.array(tt100[-2:])[:, 0]]
    xx100 = _line2p_rev(yy100[['time', ycol]].iloc[0].values, yy100[['time', ycol]].iloc[1].values, 0.998)

    tt_rev = [(0, 0)]
    for i in range(1, len(sfh_table['time'])):
        p1 = sfh_table['time'].iloc[i - 1], sfh_table[ycol].iloc[i - 1]
        p2 = sfh_table['time'].iloc[i], sfh_table[ycol].iloc[i]
        xs = np.arange(sfh_table['time'].iloc[i - 1], sfh_table['time'].iloc[i] + 0.1, 0.1)
        ys = _line2p(p1, p2, xs)
        for x, y in zip(xs, ys):
            tt_rev.append((round(x, 2), y))

    tt_rev = np.array(tt_rev)
    y_z2 = round(tt_rev[:, 1][np.where(tt_rev[:, 0] == 2.90)[0]][0], 5)
    x_075, x_090, x_100 = round(xx, 5), round(xx90, 5), round(xx100, 5)
    dor_90 = (y_z2 + 0.5 / x_075 + (0.7 + (univ_age - x_090) / univ_age)) / 3
    dor_100 = (y_z2 + 0.5 / x_075 + (0.7 + (univ_age - x_100) / univ_age)) / 3
    return y_z2, x_075, x_090, x_100, dor_90, dor_100


def _ppxf_regul_pair(templates, galaxy, noise, velscale, start, wave, sps,
                     reg_dim, component, gas_component, gas_names):
    """Two-pass ppxf: noise rescale then clean fit with regularization."""
    pp = ppxf(templates, galaxy, noise, velscale, start, moments=moments,
              degree=-1, mdegree=8, lam=wave, lam_temp=sps.lam_temp,
              regul=1 / regul_err, reg_dim=reg_dim, component=component,
              gas_component=gas_component, gas_names=gas_names,
              gas_reddening=gas_reddening, quiet=True)
    noise = noise * np.sqrt(pp.chi2)
    pp = ppxf(templates, galaxy, noise, velscale, start, moments=moments,
              degree=-1, mdegree=8, lam=wave, lam_temp=sps.lam_temp,
              regul=1 / regul_err, reg_dim=reg_dim, component=component,
              gas_component=gas_component, gas_names=gas_names,
              gas_reddening=gas_reddening, clean=True, quiet=True)
    return pp, noise


def _dor_for_alpha(alpha_str, galaxy, wave, velscale, FWHM_gal, univ_age,
                   start, data_dir):
    """Single ppxf fit (no bootstrap) for a given alpha model. Returns dor_100."""
    filename = str(data_dir / f'MILES_SSP/alpha{alpha_str}.npz')
    sps = lib.sps_lib(filename, velscale, FWHM_gal,
                      age_range=[0, univ_age], metal_range=[-2, 0.5])
    reg_dim = sps.templates.shape[1:]
    stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)
    lam_range_gal = np.array([wave.min(), wave.max()])
    with contextlib.redirect_stdout(io.StringIO()):
        gas_templates, gas_names, _ = util.emission_lines(
            sps.ln_lam_temp, lam_range_gal, FWHM_gal,
            tie_balmer=tie_balmer, limit_doublets=limit_doublets)
    templates = np.column_stack([stars_templates, gas_templates])
    n_temps = stars_templates.shape[1]
    n_forbidden = np.sum(["[" in a for a in gas_names])
    n_balmer = len(gas_names) - n_forbidden
    component = [0] * n_temps + [1] * n_balmer + [2] * n_forbidden
    gas_component = np.array(component) > 0

    noise = np.full_like(galaxy, 0.0163)
    pp, _ = _ppxf_regul_pair(templates, galaxy, noise, velscale, start, wave,
                              sps, reg_dim, component, gas_component, gas_names)

    weights_star = pp.weights[~gas_component].copy()
    agesplot, weiplot = _sfh_from_weights(weights_star, reg_dim, sps.age_grid, univ_age)
    df_temp = pd.DataFrame({'time': agesplot, 'sfh': weiplot})
    _, _, _, _, _, dor_100 = _get_values_from_sfh(univ_age, df_temp, 'sfh')
    return dor_100


def fit_galaxy(row):
    """Fit one galaxy: UNR + bootstrap (reg/unr DoR) + alpha±0.1 single fits (plus/min DoR).
    Returns a result dict. Designed for multiprocessing — no FITS/PDF output."""
    data_dir = Path('../data') if Path('../data').exists() else Path('data')
    nrand = row['_nrand']
    H0 = row.get('_H0', 75.0)
    name = row['ID_INSPIRE']

    try:
        plate, mjd, fiberid = int(row['plate']), int(row['mjd']), int(row['fiberid'])
        spec_path = str(data_dir / f'fits_shortlist/spec-{plate:04}-{mjd:05}-{fiberid:04}.fits')

        with fits.open(spec_path, ignore_missing_simple=True) as hdu:
            t = hdu['COADD'].data
            galaxy = t['flux'] / np.median(t['flux'])
            wave = np.exp(t['loglam'] * np.log(10))

        redshift = row['zspec_XSH']
        sigma = row['Vdisp_XSH']
        alpha_val = float(row['AlphaFe'])
        alpha_flag = int(row.get('alphaFe_flag', 0))

        if alpha_val < 0:
            alpha_str = '0'
        elif alpha_val > 0.4:
            alpha_str = '4'
        else:
            alpha_str = str(int(round(alpha_val * 10)))

        wave = wave / (1 + redshift)
        mask = (wave > 3600) & (wave < 6500)
        galaxy, wave = galaxy[mask], wave[mask]

        snr = DER_SNR(galaxy)
        wave *= np.median(util.vac_to_air(wave) / wave)

        velscale = c * np.log(wave[-1] / wave[0]) / (wave.size - 1)
        FWHM_gal = 2.76 / (1 + redshift)
        univ_age = NedCalculator(redshift, H0=H0).zage_Gyr
        start = [[vel, sigma], [vel, sigma], [vel, sigma]]

        # --- Build templates for main alpha ---
        filename = str(data_dir / f'MILES_SSP/alpha{alpha_str}.npz')
        sps = lib.sps_lib(filename, velscale, FWHM_gal,
                          age_range=[0, univ_age], metal_range=[-2, 0.5])
        reg_dim = sps.templates.shape[1:]
        stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)
        lam_range_gal = np.array([wave.min(), wave.max()])
        with contextlib.redirect_stdout(io.StringIO()):
            gas_templates, gas_names, _ = util.emission_lines(
                sps.ln_lam_temp, lam_range_gal, FWHM_gal,
                tie_balmer=tie_balmer, limit_doublets=limit_doublets)
        templates = np.column_stack([stars_templates, gas_templates])
        n_temps = stars_templates.shape[1]
        n_forbidden = np.sum(["[" in a for a in gas_names])
        n_balmer = len(gas_names) - n_forbidden
        component = [0] * n_temps + [1] * n_balmer + [2] * n_forbidden
        gas_component = np.array(component) > 0

        # --- UNR fit (regularized ppxf) ---
        noise = np.full_like(galaxy, 0.0163)
        pp, noise = _ppxf_regul_pair(templates, galaxy, noise, velscale, start,
                                     wave, sps, reg_dim, component, gas_component, gas_names)

        weights_unr = pp.weights[~gas_component].copy()
        mean_age = sps.mean_age_metal(weights_unr.reshape(reg_dim) / weights_unr.sum(), quiet=True)
        agesplot1, weiplot1 = _sfh_from_weights(weights_unr, reg_dim, sps.age_grid, univ_age)

        # --- Bootstrap (no regul) for REGUL fit ---
        bestfit0 = pp.bestfit.copy()
        resid = galaxy - bestfit0
        start_boot = pp.sol.copy()
        np.random.seed(123)
        weights_array = np.empty((nrand, pp.weights.size))
        for j in range(nrand):
            galaxy1 = _bootstrap_residuals(bestfit0, resid)
            pp_b = ppxf(templates, galaxy1, noise, velscale, start_boot, moments=moments,
                        degree=-1, mdegree=8, lam=wave, lam_temp=sps.lam_temp,
                        component=component, gas_component=gas_component,
                        gas_names=gas_names, gas_reddening=gas_reddening, quiet=True)
            noise = noise * np.sqrt(pp_b.chi2)
            pp_b = ppxf(templates, galaxy1, noise, velscale, start_boot, moments=moments,
                        degree=-1, mdegree=8, lam=wave, lam_temp=sps.lam_temp,
                        component=component, gas_component=gas_component,
                        gas_names=gas_names, gas_reddening=gas_reddening,
                        clean=True, quiet=True)
            weights_array[j] = pp_b.weights

        weights_regul = weights_array.sum(0)[~gas_component]
        agesplot2, weiplot2 = _sfh_from_weights(weights_regul, reg_dim, sps.age_grid, univ_age)

        df_temp = pd.DataFrame({'time': agesplot1, 'regul0': weiplot1, 'regul_max': weiplot2})
        y_z2u, x_075u, x_090u, x_100u, _, dor_100u = _get_values_from_sfh(univ_age, df_temp, 'regul0')
        y_z2r, x_075r, x_090r, x_100r, _, dor_100r = _get_values_from_sfh(univ_age, df_temp, 'regul_max')

        y_z2 = min(y_z2u, y_z2r)
        x_075 = max(x_075u, x_075r)
        x_090 = max(x_090u, x_090r)
        x_100 = max(x_100u, x_100r)
        dor_100 = (y_z2 + 0.5 / x_075 + (0.7 + (univ_age - x_100)) / univ_age) / 3

        # --- Alpha+0.1 single fit ---
        dor_100_plus = np.nan
        if alpha_val < 0.4:
            alpha_plus = alpha_val + 0.1
            if alpha_val == 0.0 and alpha_flag == -1:
                alpha_plus = 0.4
            alpha_plus_str = str(int(round(alpha_plus * 10)))
            dor_100_plus = _dor_for_alpha(alpha_plus_str, galaxy, wave, velscale,
                                          FWHM_gal, univ_age, start, data_dir)

        # --- Alpha-0.1 single fit ---
        dor_100_min = np.nan
        if alpha_val > 0.0:
            alpha_min = alpha_val - 0.1
            if alpha_val == 0.4 and alpha_flag == 1:
                alpha_min = 0.0
            alpha_min_str = str(int(round(alpha_min * 10)))
            dor_100_min = _dor_for_alpha(alpha_min_str, galaxy, wave, velscale,
                                         FWHM_gal, univ_age, start, data_dir)

        print(f'{name} done', flush=True)
        return {
            'ID_INSPIRE': name,
            'logAge': mean_age[0], '[M/H]': mean_age[1], 'SNR': snr,
            'mass_frac': y_z2, 'time_75': x_075, 'time_90': x_090,
            'time_100': x_100, 'dor_100': dor_100, 'univ_age': univ_age,
            'mass_frac_reg': y_z2r, 'time_75_reg': x_075r, 'time_90_reg': x_090r,
            'time_100_reg': x_100r, 'dor_100_reg': dor_100r,
            'mass_frac_unr': y_z2u, 'time_75_unr': x_075u, 'time_90_unr': x_090u,
            'time_100_unr': x_100u, 'dor_100_unr': dor_100u,
            'dor_100_plus': dor_100_plus,
            'dor_100_min': dor_100_min,
            'error': None,
        }

    except Exception as e:
        print(f'{name} FAILED: {e}', flush=True)
        return {'ID_INSPIRE': name, 'error': str(e)}
