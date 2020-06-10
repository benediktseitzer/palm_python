################
""" 
author: benedikt.seitzer
name: module_palm_pyplot
purpose: plot and process PALM-Data
"""
################

################
"""
IMPORTS
"""
################


import numpy as np
import pandas as pd

import palm_py as papy

################
"""
FUNCTIONS
"""
################

def calc_spectra(phi,t_eq,height,u_mean):
    """
    calculate spectra
    E_uu = discrete energy spectrum (discrete variance spectrum)
    S_uu = spectral energy density
    """
    # sample frequency 
    freq = np.fft.fftfreq(np.size(phi),t_eq[1]-t_eq[0])
    dt = np.diff(t_eq)
    # Fourier coefficients (normalized)
    fft_coeff = np.fft.fft(phi)/np.size(phi) 

    # Nyquist frequency
    freq_nyquist = np.abs(np.size(t_eq)//2)

    # spectral energy
    if np.size(phi)%2==0:
        E_uu = np.hstack((2. * np.abs(fft_coeff[0:freq_nyquist])**2., 
                            np.abs(fft_coeff[freq_nyquist])**2.))
        S_uu = E_uu * (t_eq[1]-t_eq[0]) * len(t_eq)
    else:
        E_uu = 2. * np.abs(fft_coeff[0:freq_nyquist+1])**2.
        S_uu = E_uu * (t_eq[1]-t_eq[0]) * len(t_eq)

    # dimensionless S_uu by sigma and frequency
    S_uu = np.abs(freq[0:freq_nyquist+1])*S_uu/(np.nanstd(phi)**2.)

    # reduced frequency
    freq_red = freq*height/u_mean 
    # freq_red = freq[0:freq_nyquist+1]

    smooth_spectra = True
    # spectral smoothing 
        # Maybe add other spectral smoothing-methods 
    if smooth_spectra:
        # use exponents for equi-distant bins in log plot
        freq_sm = np.hstack((np.array([0]),
                        np.log10(np.abs(freq_red[1:freq_nyquist]))))
        valcount, edges = np.histogram(freq_sm,bins=np.arange(
                                    freq_sm.min(),freq_sm.max()+10**(-5),0.05))
        freq_sm = np.zeros(valcount.shape)
        S_uu_sm = np.zeros(valcount.shape)

        vc = 0  # counting values
        for i,n in enumerate(valcount):
            if n>0:
                freq_sm[i] = np.mean(np.abs(freq_red)[vc:vc+n])
                S_uu_sm[i] = np.mean(S_uu[vc:vc+n])
                vc=vc+n   
    else:
        freq_sm = freq_red
        S_uu_sm = S_uu

    # sort frequencies for plotting    
    index_sort = np.argsort(freq_sm)
    freq_sm_sort = freq_sm[index_sort]
    S_uu_sort = S_uu_sm[index_sort]

    # aliasing
    phi_aliasing = freq_sm_sort.size-9+np.hstack((np.where
                                                (np.diff(S_uu_sort[-10:])>=0.)[0],[9]))[0]

    return freq_sm_sort, S_uu_sort, phi_aliasing


def calc_autocorr(timeseries, maxlags):
    """ Full autocorrelation of time series for lags up to maxlags.
    @parameter timeseries: np.array or list
    @parameter maxlags: int"""

    timeseries = timeseries[~np.isnan(timeseries)]
    acorr = np.asarray([1. if x == 0 else np.corrcoef(timeseries[x:], timeseries[:-x])[0][1] for x in range(maxlags)])
    return acorr


def calc_lux(dt, u_comp):
    """ Calculates the integral length scale according to R. Fischer (2011) 
    from an equidistant time series of the u component using time step dt.
    @parameter: t_eq, type = int or float
    @parameter: u_comp, type = np.array or list """

    if np.size(u_comp) < 5:
        raise Exception('Too few value to estimate Lux!')

    mask = np.where(~np.isnan(u_comp))

    u = u_comp[mask]

    initial_guess = 1000 / dt  # number of values for the length of the first autocorrelation
    if initial_guess >= len(u):
        initial_guess = int(len(u) / 2)
    lag_eq = np.arange(1, np.size(u) + 1) * dt  # array of time lags

    for lag in range(int(initial_guess), len(lag_eq), int(initial_guess)):
        u_eq_acorr = papy.calc_autocorr(u, lag)  # autocorrelation (one sided) of time series u_eq
        if np.any(np.diff(u_eq_acorr) > 0):  # if a single value of the derivation autocorrelation is smaller than 0
            # the iteration of the autocorrelation stops
            break

    lag_eq = lag_eq[:len(u_eq_acorr)]

    Lux = 0.
    # see dissertation R.Fischer (2011) for calculation method
    for i in range(np.size(u_eq_acorr) - 2):

        autc1 = u_eq_acorr[i]

        autc2 = u_eq_acorr[i + 1]

        Lux = Lux + (autc1 + autc2) * 0.5

        if autc2 > autc1:
            acorr_fit = np.polyfit(lag_eq[:i], np.log(abs(u_eq_acorr[:i])), deg=1)
            acorr_fit = np.exp(acorr_fit[0] * lag_eq + acorr_fit[1])

            if np.min(acorr_fit) < 0.001:
                ix = np.where(acorr_fit < 0.001)[0][0]

            else:
                ix = acorr_fit.size

            Lux = Lux + (np.sum(acorr_fit[i + 1:ix]) +
                         np.sum(acorr_fit[i + 2:ix + 1])) * 0.5
            break

        elif autc1 <= 0:
            break

    Lux = abs(Lux * np.mean(u_comp) * dt)
    return Lux


def calc_turbint(u_comp,v_comp):
    """
    calculate turbulence intensities for u and v-component
    """
    
    M = np.mean(np.sqrt(u_comp**2 +v_comp**2))
    u_std = np.std(u_comp)
    v_std = np.std(v_comp)
    ##  TURBULENCE INTENSITY
    I_u = u_std/np.mean(M)
    I_v = v_std/np.mean(M)

    # output array of function    
    data = np.array([I_u,I_v])
    
    return data


