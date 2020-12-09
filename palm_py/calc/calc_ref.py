# -*- coding: utf-8 -*-
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

__all__ = [
    'calc_theoretical_profile',
    'calc_ref_spectra',
    'get_reference_spectra',
    'get_lux_referencedata',
    'get_turbint_referencedata'
]

def calc_theoretical_profile(m_u, m_u_ref, m_z, z):
    """
    calculate theoretical u-profile using measurement data
    u_prandtl: Prandtl-Layer (neutral case boundary layer)
    u_powerlaw: Boundary layer derived after powerlaw
    u_fric: friction velocity
    """

    testi = papy.globals.z0

    # calculate friction velocity
    ident = 17
    u_fric = (papy.globals.ka * m_u[ident]) / (np.log(m_z[ident]/papy.globals.z0))

    # calculate theoretical profiles
    u_prandtl = u_fric/papy.globals.ka * np.log((z[1:]-papy.globals.d0)/papy.globals.z0)
    u_powerlaw = 1.* (z[1:]/200.)**papy.globals.alpha

    # no-slip bc:
    u_powerlaw = np.insert(u_powerlaw,0,0.)
    u_prandtl = np.insert(u_prandtl,0,0.)

    return u_prandtl, u_powerlaw, u_fric

def calc_ref_spectra(freq_red, ref_specs, var_name):
    """ 
    Calculate dimensionless reference spectra.
    E_kaimal => Kaimal et al. (1972)
    E_simiu => Simiu and Scanlan (1986)
    @parameter: 
    """
    modified = True

    # modified
    if var_name == 'u':
        a,b,c,d,e = ref_specs[0,:]
        f,g,h,i,j = ref_specs[1,:]
    elif var_name == 'v':
        a,b,c,d,e = ref_specs[2,:]
        f,g,h,i,j = ref_specs[3,:]
    elif var_name == 'w':
        a,b,c,d,e = ref_specs[4,:]
        f,g,h,i,j = ref_specs[5,:]                        
    else: 
        a,b,c,d,e = ref_specs[0,:]
        f,g,h,i,j = ref_specs[1,:]

    E_simiu = a* freq_red/(np.abs((e+0j) + b * freq_red**c)**d)
    E_kaimal = f* freq_red/(np.abs((j+0j) + g * freq_red**h)**i) 

    return E_simiu, E_kaimal

def get_reference_spectra(height, ref_path=None):
    """ Get referemce spectra from pre-defined location."""
    #  REFERENCE SPAECTRA RANGE FIT
    if ref_path == None:
        ref_path = 'reference_data/'

    ref_heights = np.array([7.00, 10.50, 14.00, 17.50, 22.75, 42.00, 70.00, 105.00])
    idx = (np.abs(ref_heights - height)).argmin()
    value = ref_heights[idx]
    value = '{:03.2f}'.format(value)
    ref_specs = np.genfromtxt(ref_path + 'ref_spectra_S_ii_z_10.50m.txt')

    return ref_specs

def get_lux_referencedata(ref_path=None):
    """
    Reads reference data for the integral length scale (Lux). 
    """
    if ref_path == None:
        ref_path = 'reference_data/'

    Lux_10 = np.genfromtxt(ref_path + 'Lux_data.dat', skip_header=7, skip_footer=421,
                           usecols=(0, 1), unpack=True)
    Lux_1 = np.genfromtxt(ref_path + 'Lux_data.dat', skip_header=32, skip_footer=402,
                          usecols=(0, 1), unpack=True)
    Lux_01 = np.genfromtxt(ref_path + 'Lux_data.dat', skip_header=51,
                           skip_footer=388, usecols=(0, 1), unpack=True)
    Lux_001 = np.genfromtxt(ref_path + 'Lux_data.dat', skip_header=65,
                            skip_footer=375, usecols=(0, 1), unpack=True)
    Lux_obs_smooth = np.genfromtxt(ref_path + 'Lux_data.dat', skip_header=78,
                                   skip_footer=317, usecols=(0, 1), unpack=True)
    Lux_obs_rough = np.genfromtxt(ref_path + 'Lux_data.dat', skip_header=136,
                                  skip_footer=276, usecols=(0, 1), unpack=True)

    return Lux_10, Lux_1, Lux_01, Lux_001, Lux_obs_smooth, Lux_obs_rough

def get_turbint_referencedata(var_name, ref_path=None):
    """
    Read reference data files for the turbulence intensity Iu, Iv
    """
    if ref_path == None:
        ref_path = 'reference_data/'

    if var_name == 'u':
        ref_dat = ref_path + 'Iu_data.dat'
        Iu_slight = np.genfromtxt(ref_dat, skip_header=11, skip_footer=367, 
                                    usecols=(0,1), unpack=True, encoding='latin1')  
        Iu_moderate = np.genfromtxt(ref_dat, skip_header=41, skip_footer=337, 
                                    usecols=(0,1), unpack=True, encoding='latin1')
        Iu_rough = np.genfromtxt(ref_dat, skip_header=69, skip_footer=310, 
                                    usecols=(0,1), unpack=True, encoding='latin1')
        Iu_very = np.genfromtxt(ref_dat, skip_header=103, skip_footer=269, 
                                    usecols=(0,1), unpack=True, encoding='latin1')
        return Iu_slight, Iu_moderate, Iu_rough, Iu_very
    if var_name == 'v': 
        ref_dat = ref_path + 'Iv_data.dat'
        Iv_slight = np.genfromtxt(ref_dat, skip_header=7, skip_footer=40, 
                                usecols=(0,1), unpack=True, encoding='latin1')
        Iv_moderate = np.genfromtxt(ref_dat, skip_header=20, skip_footer=29, 
                                usecols=(0,1), unpack=True, encoding='latin1')
        Iv_rough = np.genfromtxt(ref_dat, skip_header=31, skip_footer=15, 
                                usecols=(0,1), unpack=True, encoding='latin1')
        Iv_very = np.genfromtxt(ref_dat, skip_header=45, skip_footer=0, 
                                usecols=(0,1), unpack=True, encoding='latin1')
        return Iv_slight, Iv_moderate, Iv_rough, Iv_very
    if var_name == 'w': 
        ref_dat = ref_path + 'Iw_data.dat'
        Iw_slight = np.genfromtxt(ref_dat, skip_header=11, skip_footer=347, 
                                usecols=(0,1), unpack=True, encoding='latin1')
        Iw_moderate = np.genfromtxt(ref_dat, skip_header=37, skip_footer=321, 
                                usecols=(0,1), unpack=True, encoding='latin1')
        Iw_rough = np.genfromtxt(ref_dat, skip_header=63, skip_footer=295, 
                                usecols=(0,1), unpack=True, encoding='latin1')
        Iw_very = np.genfromtxt(ref_dat, skip_header=89, skip_footer=269, 
                                usecols=(0,1), unpack=True, encoding='latin1')
        return Iw_slight, Iw_moderate, Iw_rough, Iw_very