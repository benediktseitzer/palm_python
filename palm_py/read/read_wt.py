################
""" 
author: benedikt.seitzer
name: read_wt
purpose: read windtunnel data
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

__all__ = [
    'read_wt_ts',
    'read_wt_ver_pr'
]

def read_wt_ts(wt_file):
    """
    read wind tunnel profile
    """

    df = pd.read_table(wt_file,delimiter=' ', usecols=[0,1,2])

    t = df.iloc[:,0].to_numpy()    
    u = df.iloc[:,1].to_numpy()
    v = df.iloc[:,2].to_numpy()

    # there are NaNs in timeseries
    mask = np.where(~np.isnan(u))
    t = t[mask]
    u = u[mask]
    v = v[mask]

    return u, v, t

def read_wt_ver_pr(wt_file, wt_ref_file, scale):
    """ 
    Create Timeseries object from file.
    
    ----------
    Parameters:
    
    wt_file: str
    wt_ref_file: str
    scale: float

    ----------
    Returns:

    u: array-like
    u_ref: array-like
    z: array-like

    """

    z, u = np.genfromtxt(wt_file, usecols=(3, 8),
                        skip_header=6, unpack=True)

    u_ref = np.genfromtxt(wt_ref_file, usecols=3, 
                        skip_header=1, unpack=True)

    z = z*scale/1000

    return u, u_ref, z