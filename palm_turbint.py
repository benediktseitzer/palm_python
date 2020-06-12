# -*- coding: utf-8 -*-
################
""" 
author: benedikt.seitzer
purpose: - process timeseries and calculate Turbulence intensities
"""
################


################
"""
IMPORTS
"""
################

import os
import numpy as np
import math as m
import netCDF4
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import palm_py as papy

import warnings
warnings.simplefilter("ignore")

################
"""
FUNCTIONS
"""
################


def plot_turbint_profile(turbint, height, var_name, run_name, run_number):
    """
    Plot turbulence intensities Iu or Iv.
    @parameter ax: axis passed to function
    """

    ref_path = None
    I_slight, I_moderate, I_rough, I_very = papy.get_turbint_referencedata(var_name, ref_path)

    plt.style.use('classic')
    fig, ax = plt.subplots()

    err = 0.1 * turbint
    if run_number == '':
        run_number = '.000'

    # plot data
    h1 = ax.errorbar(turbint, height_list, xerr=err, fmt='o', markersize=3,
                label=r'PALM - $I _{}$'.format(var_name))
    # plot ref-data
    r1 = ax.plot(I_slight[1,:],I_slight[0,:],'k-', linewidth=0.5,
                label= 'VDI slightly rough (lower bound)')
    r2 = ax.plot(I_moderate[1,:],I_moderate[0,:],'k-.', linewidth=0.5,
                label= 'VDI moderately rough (lower bound)')
    r3 = ax.plot(I_rough[1,:],I_rough[0,:],'k--', linewidth=0.5,
                label= 'VDI rough (lower bound)')
    r4 = ax.plot(I_very[1,:],I_very[0,:],'k:', linewidth=0.5,
                label= 'VDI very rough (lower bound)')                
    set_limits = True
    if set_limits:
        ax.set_xlim(0,0.3)
        ax.set_ylim(0,300)

    ax.set_xlabel(r"$I _{}$ [-]".format(var_name))
    ax.set_ylabel(r"$z$ [m]" )
    ax.legend(loc='upper left', fontsize=11)
    ax.grid()

    if testing:
        fig.savefig('../palm_results/testing/turbint/testing_{}_turbint.png'.format(var_name), 
                    bbox_inches='tight')
    else:
        plt.savefig('../palm_results/{}/run_{}/turbint/{}_{}_turbint.png'.format(run_name,
                    run_number[-3:], run_name, var_name), bbox_inches='tight')


################
"""
GLOBAL VARIABLES
"""
################

run_name = 'thunder_balcony_resstudy_precursor'
run_number = '.013'
nc_file = '{}_masked_M02{}.nc'.format(run_name,run_number)
nc_file_grid = '{}_pr{}.nc'.format(run_name,run_number)
nc_file_path = '../current_version/JOBS/{}/OUTPUT/'.format(run_name)

mask_name_list = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 
                    'M10','M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20']
height_list = [5., 10., 12.5, 15., 17.5, 20., 25., 30., 35., 40., 45., 50., 60.,
                     70., 80., 90., 100., 110., 120., 130.]

wt_file = '../../Documents/phd/palm/input_data/windtunnel_data/HG_BL_MR_DOK_UV_014_000001_timeseries_test.txt'

# testing parameters
testing = False

################
"""
MAIN
"""
################

# prepare the outputfolders
papy.prepare_plotfolder(run_name,run_number)

Iu = np.zeros(len(height_list))
Iv = np.zeros(len(height_list))
Iw = np.zeros(len(height_list))

grid_name = 'zu'
z, z_unit = papy.read_nc_grid(nc_file_path,nc_file_grid,grid_name)
i = 0 

for mask_name in mask_name_list: 
    nc_file = '{}_masked_{}{}.nc'.format(run_name,mask_name,run_number)
    height = height_list[i]
    
    var_u, var_unit_u = papy.read_nc_var_ms(nc_file_path, nc_file, 'u')
    var_v, var_unit_v = papy.read_nc_var_ms(nc_file_path, nc_file, 'v')
    var_w, var_unit_w = papy.read_nc_var_ms(nc_file_path, nc_file, 'w')

    turbint_dat = papy.calc_turbint(var_u, var_v, var_w)

    Iu[i] = turbint_dat[0]
    Iv[i] = turbint_dat[1]
    Iw[i] = turbint_dat[2]
    i = i + 1
    print('\n calculated turbulence intensities scale for {}'.format(str(height)))

plot_turbint_profile(Iu, height_list, 'u', run_name, run_number)
print('\n plotted turbulence intensity profiles for u-component')

plot_turbint_profile(Iv, height_list, 'v', run_name, run_number)
print('\n plotted turbulence intensity profiles for v-component')

plot_turbint_profile(Iw, height_list, 'w', run_name, run_number)
print('\n plotted turbulence intensity profiles for w-component')