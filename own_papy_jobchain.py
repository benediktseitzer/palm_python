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

import numpy as np
import math as m
import pandas as pd
import sys
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import palm_py as papy

sys.path.append('/home/bene/Documents/phd/windtunnel_py/windtunnel/')    
import windtunnel as wt

import warnings
warnings.simplefilter("ignore")

################
"""
GLOBAL VARIABLES
"""
################
# PALM input files
papy.globals.run_name = 'SB_SI_back'
papy.globals.run_number = '.030'
# papy.globals.run_numbers = ['.029', '.028']
papy.globals.jobchain_numbers = ['.001', '.002', '.003', '.004', '.005',
                                '.006', '.007', '.008', '.009', '.010', '.011', 
                                '.012', '.013', '.014', '.015', '.016', '.017', 
                                '.018', '.019', '.020', '.021', '.022', '.023', 
                                '.024', '.025', '.026', '.027', '.028', '.029',
                                '.030']
nc_file_grid = '{}_pr{}.nc'.format(papy.globals.run_name,papy.globals.run_number)
nc_file_path = '../palm/current_version/JOBS/{}/OUTPUT/'.format(papy.globals.run_name)

# wind tunnel input files
experiment = 'single_building'
wt_filename = 'SB_BL_UV_001'
wt_path = '../../Documents/phd/experiments/{}/{}'.format(experiment, wt_filename[3:5])
wt_file = '{}/coincidence/timeseries/{}.txt'.format(wt_path, wt_filename)
wt_file_pr = '{}/coincidence/mean/{}.000001.txt'.format(wt_path, wt_filename)
wt_file_ref = '{}/wtref/{}_wtref.txt'.format(wt_path, wt_filename)
wt_scale = 150.

# PHYSICS
papy.globals.z0 = 0.03
papy.globals.z0_wt = 0.071
papy.globals.alpha = 0.18
papy.globals.ka = 0.41
papy.globals.d0 = 0.
if papy.globals.run_name == 'single_building_ABL_2m':
    papy.globals.nx = 512
    papy.globals.ny = 512
    papy.globals.dx = 2.
elif papy.globals.run_name == 'single_building_ABL_4m':    
    papy.globals.nx = 256
    papy.globals.ny = 256
    papy.globals.dx = 4.
else:    
    papy.globals.nx = 1024
    papy.globals.ny = 1024
    papy.globals.dx = 1.

# test-cases for spectral analysis testing
test_case_list = ['frequency_peak']
# spectra mode to run scrupt in
mode_list = ['testing', 'heights', 'compare', 'filtercheck'] 
mode = mode_list[1]

# Steeringflags
compute_timeseries = True

################
"""
MAIN
"""
################

# prepare the outputfolders
papy.prepare_plotfolder(papy.globals.run_name,papy.globals.run_number)
plt.style.use('classic')

################
# Timeseries of several measures
if compute_timeseries:
    nc_file = '{}_ts{}.nc'.format(papy.globals.run_name, papy.globals.run_number)
    var_name_list = ['umax', 'w"u"0', 'E', 'E*', 'div_old', 'div_new', 'dt', 'us*']
    # var_name_list = ['umax']

    for var_name in var_name_list:
        time_total = np.zeros(1)
        var_total = np.zeros(1)
        plt.style.use('classic')
        fig, ax = plt.subplots()
        for run in papy.globals.jobchain_numbers:
            nc_file = '{}_ts{}.nc'.format(papy.globals.run_name, run)
            time, time_unit = papy.read_nc_time(nc_file_path,nc_file)
            var, var_unit = papy.read_nc_var_ts(nc_file_path,nc_file,var_name)
            time_total = np.append(time_total, time)
            var_total = np.append(var_total, var)
            if var_name == 'umax':
                ax.plot(time, var)
                ax.set_xlabel(r'$t$ ({})'.format('s'), fontsize = 18)
                ax.set_ylabel(r'{} ({})'.format(var_name,var_unit), fontsize = 18)
        if var_name == 'umax':
            ax.grid()
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
            plt.xlim(250., max(time_total))
            fig.savefig('../palm_results/{}/run_{}/timeseries/{}_{}_colour_ts.png'.format(papy.globals.run_name,papy.globals.run_number[-3:],
                        papy.globals.run_name,var_name), bbox_inches='tight',dpi=500)
        print('\n READ {} from {}{} \n'.format(var_name, nc_file_path, nc_file))
        papy.plot_timeseries(var_total, var_unit, var_name, time_total, time_unit)
        print('\n plotted {} \n'.format(var_name))
    print(' Finished Timeseries')

print('')
print('Finished processing of: {}{}'.format(papy.globals.run_name, papy.globals.run_number))