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
import os

import numpy as np
import pandas as pd

################
"""
FUNCTIONS
"""
################

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


def read_wt_ver_pr(wt_file):
    """
    read wind tunnel profile-data
    """

    df = pd.read_table(wt_file,delimiter=',',header=1, usecols=[3,4,5])
    
    u_ref_dat = df.iloc[:,1].to_numpy()
    
    u_ref = np.mean(u_ref_dat)

    u = df.iloc[:,2].to_numpy()
    z = df.iloc[:,0].to_numpy()
    
    return u, u_ref, z