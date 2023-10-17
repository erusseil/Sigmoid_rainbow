from typing import Dict, List, Union
from pathlib import Path
from argparse import ArgumentParser
import glob
from itertools import chain
from functools import lru_cache
from astropy.table import vstack, Table
import astropy.constants as c
import astropy.units as u
import os
import sys
import pickle
import numpy as np
import pandas as pd
import sncosmo
import subprocess
import shlex
from iminuit import Minuit
from iminuit.cost import LeastSquares
import kernel as kern
import time
import shutil
import random
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson


def Am_to_Hz(wl):
    """
    Convert Ångström to Hertz

    Parameters
    ----------
    wl : array
        Wave length array

    Returns
    -------
        Array converted to frequency
    """
    return c.c.value / (wl * 1e-10)


def normalize(lc):
    """
    Apply normalization transformation to
    an astropy table light curve
    
    Parameters
    ----------
    lc: astropy table
        Light curve

    Returns
    -------
    lc : astropy table
        Normalized light curve with :
        - Fluxes divided by max r band flux
        - Time 0 shifted by time of max r band flux
        - Added columns before normalization : max_flux, max_flux_time
    """

    normband = lc[(lc["BAND"] == "r ") | (lc["BAND"] == "PSr")]
    maxi = normband["FLUXCAL"].max()

    lc["max_flux"] = maxi
    t_maxi = normband["MJD"][normband["FLUXCAL"].argmax()]
    lc["max_flux_time"] = t_maxi
    lc["FLUXCAL"] = lc["FLUXCAL"] / maxi
    lc["FLUXCALERR"] = lc["FLUXCALERR"] / maxi
    lc["MJD"] = lc["MJD"] - t_maxi

    return lc


def Fsig(t, a, t0, trise):
    """
    Compute flux using Bazin with
    baseline fixed to 0.

    Parameters
    ----------
    t: array
        Time value of data points.
    a: float
        Bazin amplitude.
    t0: float
        Time value related to time of maximum flux.
    trise: float
        Value related to length of the rising part slope.

    Returns:
    --------
        Computed flux at each time t.
    """

    return a / (1 + np.exp((t + t0) / trise))


def Tsig(t, Tmin, dT, ksig, t0):
    """
    Compute temperature using sigmoid

    Parameters
    ----------
    t: array
        Time value of data points.
    Tmin: float
        Minimum temperature to reach at the end of the sigmoid
    dT: float
        Difference between beginning and end temperature of the sigmoid.
    ksig: float
        Slope parameter of the sigmoid.
    t0: float
        Bazin time value related to time of maximum flux.

    Returns:
    --------
        Computed temperature at each time t.
    """

    return Tmin + dT / (1 + np.exp((t - t0) / ksig))


def planck_nu(nu, T):
    """
    Compute blackbody intensity from temperature and frequency.

    Parameters
    ----------
    nu: array
        Frequency for which to compute intensity.
    T: array
        Temperature values at different times.

    Returns:
    --------
        Computed spectral radiance.
    """

    return (2 * c.h.value / c.c.value**2) * nu**3 / np.expm1(c.h.value * nu / (c.k_B.value * T))


@lru_cache(maxsize=64)
def planck_passband_spline(passband, T_min=1e2, T_max=1e6, T_steps=10_000):
    """
    Compute spline of passband intensity for the range of temperatures.

    Parameters
    ----------
    passband: string
        Name of the passband
    T_min: float
        Minimum temperature to consider
    T_max: float
        Maximum temperature to consider
    T_steps: int
        Step between temperatures to consider

    Returns
    -------
    scipy.iterpolate.UnivariateSpline
        Spline of the passband intensity versus temperature
    """
    passband = sncosmo.get_bandpass(passband)
    nu = (c.c / (passband.wave * u.AA)).to_value(u.Hz)
    trans = passband.trans  # photon counter transmission

    T = np.logspace(np.log10(T_min), np.log10(T_max), T_steps)
    black_body_integral = simpson(x=nu, y=planck_nu(nu, T[:, None]) * trans / nu, axis=-1)
    transmission_integral = simpson(x=nu, y=trans / nu)

    return UnivariateSpline(x=T, y=black_body_integral / transmission_integral, s=0, k=3, ext='raise')


def planck_passband(passband, T):
    """
    Compute spectral radiance from temperature and passband.

    Parameters
    ----------
    passband: array of strings
        Names of passband for which to compute intensity.
    T: array of floats
        Temperature values at different times.

    Returns:
    --------
    array of floats
        Computed intensity.
    """
    passband, T = np.broadcast_arrays(passband, T)

    # Speed up computation if only a single value is passed
    if passband.size == 1:
        return planck_passband_spline(passband.item())(T)

    unique_passbandes, indices = np.unique(passband, return_inverse=True)
    output = np.zeros_like(passband, dtype=float)
    for index, passband in enumerate(unique_passbandes):
        T_passband = T[indices == index]
        spline = planck_passband_spline(passband)
        output[indices == index] = spline(T_passband)

    return output


def planck(nu, T):
    """
    Compute intensity from temperature and frequency / passband.

    Parameters
    ----------
    nu: array of floats or strings
        Frequencies or passband names for which to compute intensity.
    T: array of floats
        Temperature values at different times.

    Returns:
    --------
    array of floats
        Computed intensity.
    """
    integrate = False
    if np.any(nu < 0):
        nu = np.vectorize(sncosmoband_convert.get)(nu)
        integrate = True
    nu = np.asarray(nu)
    
    if integrate:
        return planck_passband(nu, T)
   
    return planck_nu(nu, T)

def Fnu(x, a, t0, trise, Tmin, dT, ksig):
    """
    Complete fitting function. Used to compute flux at any
    frequency at any time (scaled by an arbitrary amplitude term).

    Parameters
    ----------
    x: ndarray
        Array of pair time and frequency from which to compute flux.
    a: float
        Bazin amplitude.
    t0: float
        Bazin time value related to time of maximum flux.
    trise: float
        Bazin value related to length of the rising part slope.
    Tmin: float
        Minimum temperature to reach at the end of the sigmoid
    dT: float
        Difference between beginning and end temperature of the sigmoid.
    ksig: float
        Slope parameter of the sigmoid.

    Returns:
    --------
        Computed flux at any frequency and at any time (scaled by
        an arbitrary amplitude term).
    """
    t, nu = x
    
    T = Tsig(t, Tmin, dT, ksig, t0)
    Fbol = Fsig(t, a, t0, trise)
    amplitude = 1e15

    return np.pi / c.sigma_sb.value * Fbol * planck(nu, T) / T**4 * amplitude



def elasticc_agg_to_plasticc(data):
    
    band_dic = {"u":0, "g":1, "r":2, "i":3, "z":4, "Y":5}
    all_obj = []
    for idx in range(len(data)):
        obj = data.iloc[idx]
        data_fmt = pd.DataFrame(data={'mjd':obj['cjd'], 'passband':obj['cfid'],\
                                      'flux':obj['cflux'], 'flux_err':obj['csigflux'],
                                     'detected_bool':(obj['cflux']/obj['csigflux']>4) + 0})
        data_fmt = data_fmt.replace({"passband":band_dic})
        data_fmt['object_id'] = obj['alertId']
        data_fmt = data_fmt.set_index('object_id')
        all_obj.append(data_fmt)
    return pd.concat(all_obj).sort_values(by='mjd')

def generate_plasticc_lcs(df): 
    for object_id, table in df.groupby('object_id'):
        assert np.all(np.diff(table['mjd']) >= 0), 'Light curve must be time-sorted'

        SNID = table.index[0]

        table = Table.from_pandas(table)
        table.rename_columns(['mjd', 'flux', 'flux_err'], ['MJD', 'FLUXCAL', 'FLUXCALERR'])
        table['BAND'] = kern.PASSBANDS[table['passband']]
        table.meta['SNID'] = SNID
        

        detections = table[table['detected_bool'] == 1]
        det_per_band = dict(zip(*np.unique(detections["BAND"], return_counts=True)))

        if any(det_per_band.get(band, 0) < min_det for band, min_det in kern.min_det_per_band.items()):
            continue
            
        yield table
        
def format_plasticc(df, max_n):

    lcs = []
    for idx, lc in zip(range(max_n), generate_plasticc_lcs(df)):
        lcs.append(lc)

    return lcs

def preprocess(lcs):
    new_lcs = [normalize(lc) for lc in lcs]
    return new_lcs


def extract_rainbow(lcs):
    all_param = []
    for obj in lcs:  
        obj["NU"] = np.vectorize(sncosmoband_convert_temporary.get)(obj["BAND"])
        extraction = perform_fit_rainbow(obj)
        all_param.append(list(extraction[0].to_dict().values()) + extraction[1])

    features = pd.DataFrame(
        columns=Fnu.__code__.co_varnames[1:7] + ("error", "max_flux", "max_time", "id"),
        data=all_param,
    )
    return features


def perform_fit_rainbow(obj):

    global_flux = obj["FLUXCAL"]
    global_fluxerr = obj["FLUXCALERR"]
    global_nu = obj["NU"]
    global_mjd = obj["MJD"]
    
    parameters_dict = {
        "a": global_flux.max(),
        "t0": global_mjd[np.argmax(global_flux)],
        "trise": -5,
        "Tmin": 4000,
        "dT": 7000,
        "ksig": 4,
    }
    
    least_squares = LeastSquares(
        np.array([global_mjd, global_nu]), global_flux, global_fluxerr, Fnu
    )

    fit = Minuit(
        least_squares,
        **parameters_dict,
    )

    fit.limits['Tmin'] = (100, 100000)
    fit.limits['dT'] = (0, 100000)
    fit.limits['t0'] = (-1000, 1000)
    fit.limits['a'] = (0.1, 1000)
    fit.limits['ksig'] = (1.5, 300)
    fit.limits['trise'] = (-100, -0.5)
    fit.migrad()

    max_flux = obj["max_flux"][0]
    max_time = obj["max_flux_time"][0]
    fit_error = fit.fval
    objid = obj.meta["SNID"]

    additionnal = [fit_error, max_flux, max_time, objid]

    return fit.values, additionnal


        
# __________________________USEFUL VALUES________________________

SATURATION_FLUX = 1e5

# Source of values used for the filters : http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=LSST&asttype=
nu_u = Am_to_Hz(3751)
nu_g = Am_to_Hz(4742)
nu_r = Am_to_Hz(6173)
nu_i = Am_to_Hz(7502)
nu_z = Am_to_Hz(8679)
nu_Y = Am_to_Hz(9711)
nu_PSg = Am_to_Hz(4811)
nu_PSr = Am_to_Hz(6156)
nu_PSi = Am_to_Hz(7504)
nu_ZTFg = Am_to_Hz(4723)
nu_ZTFr = Am_to_Hz(6340)
nu_ZTFi = Am_to_Hz(7886)
nu_B = Am_to_Hz(4450)


freq_dic = {"u ": nu_u, "g ": nu_g, "r ": nu_r, "i ": nu_i, "z ": nu_z, "Y ": nu_Y, "ZTFg": nu_ZTFg, "ZTFr": nu_ZTFr, "ZTFi": nu_ZTFi, "PSg":nu_PSg, "PSr":nu_PSr, "PSi":nu_PSi, "B":nu_B}

sncosmoband_convert_temporary = {"u ": -1, "g ": -2, "r ": -3, "i ": -4, "z ": -5, "Y ": -6,\
                                "ZTFg":-7, "ZTFr":-8, "ZTFi":-9, "PSg":-10, "PSr":-11, "PSi":-12, "B":-13}

sncosmoband_convert = {-1: "lsstu", -2: "lsstg", -3: "lsstr",\
                       -4: "lssti", -5: "lsstz", -6: "lssty",\
                       -7: "ztfg", -8: "ztfr", -9: "ztfi",\
                       -10: "ps1::g", -11: "ps1::r",\
                       -12: "ps1::i", -13: "standard::b"}
