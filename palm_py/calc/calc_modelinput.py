################
""" 
author: benedikt.seitzer
name: read_nc
purpose: read netCDF-files
"""
################

################
"""
IMPORTS
"""
################

import numpy as np

import palm_py as papy

################
"""
FUNCTIONS
"""
################

__all__ = [
    'calc_input_profile'
]

def calc_input_profile(m_u, m_z, z, ref_height):
    """
    calculate u- and v-profile using measurement data

    -----------
    Parameters
    m_u: array-like
    m_z: array-like
    z: array-like
    ref_height: float

    ---------
    Returns
    u_prandtl: float
    u_fric: float
    """
    # calculate friction velocity
    i_upper = np.where(m_z == ref_height[1])
    u_fric = (papy.globals.ka * m_u[i_upper]) / (np.log(m_z[i_upper]/papy.globals.z0))
    # calculate theoretical profiles
    u_prandtl = u_fric/papy.globals.ka * np.log((z[1:]-papy.globals.d0)/papy.globals.z0)
    # no-slip bc:
    u_prandtl = np.insert(u_prandtl,0,0.)
    print('     z0 = {} m'.format(papy.globals.z0))
    print('     u* = {} with reference height at {} m \n'.format(u_fric,m_z[i_upper]))

    return u_prandtl, u_fric
