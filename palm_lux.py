################
""" 
author: benedikt.seitzer
purpose: - process timeseries and calculate spectra
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


def plot_lux_profile(lux, height, var_name, run_name, run_number):
    """
    Plot Lux-profiles.
    @parameter ax: axis passed to function
    """

    ref_path = None
    Lux_10,Lux_1,Lux_01,Lux_001,Lux_obs_smooth,Lux_obs_rough = \
    papy.get_lux_referencedata(ref_path)

    plt.style.use('classic')
    fig, ax = plt.subplots()

    err = 0.1 * lux
    if run_number == '':
        run_number = '.000'

    h1 = ax.errorbar(lux, height_list, xerr=err, fmt='o', markersize=3,
                label=r'PALM - $u$')
    ref1 = ax.plot(Lux_10[1,:],Lux_10[0,:],'k-',linewidth=1,label=r'$z_0=10\ m$ (theory)')
    ref2 = ax.plot(Lux_1[1,:],Lux_1[0,:],'k--',linewidth=1,label=r'$z_0=1\ m$ (theory)')
    ref3 = ax.plot(Lux_01[1,:],Lux_01[0,:],'k-.',linewidth=1,label=r'$z_0=0.1\ m$ (theory)')
    ref4 = ax.plot(Lux_001[1,:],Lux_001[0,:],'k:',linewidth=1,label=r'$z_0=0.01\ m$ (theory)')
    ref5 = ax.plot(Lux_obs_smooth[1,:],Lux_obs_smooth[0,:],'k+',
                    linewidth=1,label='observations smooth surface')
    ref6 = ax.plot(Lux_obs_rough[1,:],Lux_obs_rough[0,:],'kx',linewidth=1,label='observations rough surface')
    
    set_limits = True
    if set_limits:
        ax.set_xlim(10,1000)
        ax.set_ylim(10,1000)


    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r"$L _{ux}$ [m]")
    ax.set_ylabel(r"$z$ [m]" )
    ax.legend(loc='upper left', fontsize=11)
    ax.grid()

    if testing:
        fig.savefig('../palm_results/testing/lux/testing_{}_lux.png'.format(var_name), bbox_inches='tight')
    else:
        plt.savefig('../palm_results/{}/run_{}/lux/{}_{}_lux.png'.format(run_name,run_number[-3:],
                    run_name,var_name), bbox_inches='tight')


################
"""
GLOBAL VARIABLES
"""
################

run_name = 'thunder_balcony_resstudy_precursor'
run_number = '.014'
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

lux = np.zeros(len(height_list))

grid_name = 'zu'
z, z_unit = papy.read_nc_grid(nc_file_path,nc_file_grid,grid_name)
var_name = 'u'
i = 0 

for mask_name in mask_name_list: 
    nc_file = '{}_masked_{}{}.nc'.format(run_name,mask_name,run_number)
    height = height_list[i]
        

    time, time_unit = papy.read_nc_var_ms(nc_file_path,nc_file,'time')        
    var, var_unit = papy.read_nc_var_ms(nc_file_path,nc_file,var_name)
    
    lux[i] = papy.calc_lux(np.abs(time[1]-time[0]),var)
    
    i = i + 1
    print('\n calculated integral length scale for {}'.format(str(height)))

plot_lux_profile(lux, height_list, var_name, run_name, run_number)
print('\n plotted integral length scale profiles')
