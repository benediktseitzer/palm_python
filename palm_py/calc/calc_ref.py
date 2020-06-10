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

################
"""
FUNCTIONS
"""
################

def calc_theoretical_profile(m_u, m_u_ref, m_z, z, par):
    """
    calculate theoretical u-profile using measurement data
    u_prandtl: Prandtl-Layer (neutral case boundary layer)
    u_powerlaw: Boundary layer derived after powerlaw
    u_fric: friction velocity
    """

    # get fitting parameters 
    z0 = par[0]
    alpha = par[1]
    ka = par[2]
    d0 = par[3]

    # calculate friction velocity
    ident = 17
    ident_low = 8
    u_fric = (ka*m_u[ident]) / (np.log(m_z[ident]/z0))

    # calculate theoretical profiles
    u_prandtl = u_fric/ka * np.log((z[1:]-d0)/z0)
    u_powerlaw = 1.* (z[1:]/200.)**alpha

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
        ref_path = '../../Documents/phd/palm/input_data/reference_spectra/'
    ref_heights = np.array([7.00, 10.50, 14.00, 17.50, 22.75, 42.00, 70.00, 105.00])
    idx = (np.abs(ref_heights - height)).argmin()
    value = ref_heights[idx]
    value = '{:03.2f}'.format(value)
    ref_specs = np.genfromtxt(ref_path + 'ref_spectra_S_ii_z_10.50m.txt')

    return ref_specs


def get_lux_referencedata(ref_path=None):
    """
    Reads and returns reference data for the integral length scale (Lux). 
    """
    if ref_path == None:
        ref_path = '../../Documents/phd/palm/input_data/reference_lux/'

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



