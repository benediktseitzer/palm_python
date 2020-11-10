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
import os

import numpy as np
import pandas as pd

import palm_py as papy

################
"""
FUNCTIONS
"""
################

def calc_input_profile(m_u, m_z, z):
    """
    calculate u- and v-profile using measurement data

    -----------
    Parameters
    m_u: array-like
    m_z: array-like
    z: array-like

    ---------
    Returns
    u_prandtl: float
    u_fric: float
    """

    print(papy.globals.z0)

    # calculate friction velocity
    ident = 17
    u_fric = (papy.globals.ka * m_u[ident]) / (np.log(m_z[ident]/papy.globals.z0))

    # calculate theoretical profiles
    u_prandtl = u_fric/papy.globals.ka * np.log((z[1:]-papy.globals.d0)/papy.globals.z0)

    # no-slip bc:
    u_prandtl = np.insert(u_prandtl,0,0.)

    return u_prandtl, u_fric
