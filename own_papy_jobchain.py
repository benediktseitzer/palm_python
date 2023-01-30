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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import palm_py as papy

sys.path.append('/home/bene/Documents/phd/windtunnel_py/windtunnel/')    
import windtunnel as wt

import warnings
warnings.simplefilter("ignore")

plotformat = 'pgf'
# plotformat = 'png'
# plotformat = 'pdf'
if plotformat == 'pgf':
    plt.style.use('default')
    matplotlib.use('pgf')
    matplotlib.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        'font.family': 'sans-serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'xtick.labelsize' : 11,
        'ytick.labelsize' : 11,
        'legend.fontsize' : 11,
        'lines.linewidth' : 0.75,
        'lines.markersize' : 2.5,
        'figure.dpi' : 300,
    })
    print('Textwidth in inch = ' + str(426/72.27))
    # 5.89 inch
    textwidth = 5
    textwidth_half = 0.5*textwidth
else:
    plt.style.use('default')
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'text.usetex': False,
        'mathtext.fontset': 'cm',
        'xtick.labelsize' : 11,
        'ytick.labelsize' : 11,
        'legend.fontsize' : 11,
        'lines.linewidth' : 0.75,
        'lines.markersize' : 2.5,
        'figure.dpi' : 300,
    })

################
"""
GLOBAL VARIABLES
"""
################
# PALM input files
papy.globals.run_name = 'SB_SI_BL'
# papy.globals.run_name = 'yshift_SB_BL_corr'
papy.globals.run_numbers = ['.000', '.001', '.002', '.003', '.004', '.005', '.006', '.007',
                            '.008', '.009', '.010', '.011', '.012', 
                            '.013', '.014', '.015', '.016', '.017', '.018',
                            '.019', '.020', '.021', '.022', '.023', '.024',
                            '.025', '.026', '.027', '.028', '.029', '.030',
                            '.031', '.032', '.033', '.034', '.035', '.036',
                            '.037', '.038', '.039', '.040', '.041', '.042',
                            '.043', '.044', '.045', '.046', '.047']
# papy.globals.run_name = 'SB_SI_front'
# papy.globals.run_name = 'SB_SI_back'
# papy.globals.run_numbers = ['.007', '.008', '.009', '.010', '.011', '.012', 
#                         '.013', '.014', '.015', '.016', '.017', '.018',
#                         '.019', '.020', '.021', '.022', '.023', '.024',
#                         '.025', '.026', '.027', '.028', '.029', '.030', 
#                         '.031', '.032', '.033', '.034', '.035', '.036',
#                         '.037', '.038', '.039', '.040', '.041', '.042',
#                         '.043', '.044', '.045', '.046']
# papy.globals.run_name = 'SB_LE'
# papy.globals.run_numbers = ['.008', '.009', '.010', '.011', '.012', 
#                         '.013', '.014', '.015', '.016', '.017', '.018',
#                         '.019', '.020', '.021', '.022', '.023', '.024',
#                         '.025', '.026', '.027', '.028', '.029', '.030',
#                         '.031', '.032', '.033', '.034', '.035', '.036',
#                         '.037', '.038', '.039', '.040', '.041', '.042',
#                         '.043', '.044', '.045', '.046', '.047', '.048',
#                         '.049', '.050',]

# papy.globals.run_name = 'SB_LU'
# papy.globals.run_numbers = ['.008', '.009', '.010', '.011', '.012', 
#                             '.013', '.014', '.015', '.016', '.017', '.018',
#                             '.019', '.020', '.021', '.022', '.023', '.024',
#                             '.025', '.026', '.027', '.028', '.029', '.030', 
#                             '.031', '.032', '.033', '.034', '.035', '.036',
#                             '.037', '.038', '.039', '.040', '.041', '.042',
#                             '.043', '.044', '.045', '.046', '.047']

papy.globals.run_number = papy.globals.run_numbers[-1]
nc_file_grid = '{}_pr{}.nc'.format(papy.globals.run_name,papy.globals.run_number)
nc_file_path = '../palm/current_version/JOBS/{}/OUTPUT/'.format(papy.globals.run_name)
file_type = 'png'
file_type = plotformat

# wind tunnel input files
experiment = 'single_building'
wt_filename = 'SB_BL_UV_001'
wt_path = '../../Documents/phd/experiments/{}/{}'.format(experiment, wt_filename[3:5])
wt_file = '{}/coincidence/timeseries/{}.txt'.format(wt_path, wt_filename)
wt_file_pr = '{}/coincidence/mean/{}.000001.txt'.format(wt_path, wt_filename)
wt_file_ref = '{}/wtref/{}_wtref.txt'.format(wt_path, wt_filename)
wt_scale = 150.

# PHYSICS
papy.globals.z0 = 0.021
papy.globals.z0_wt = 0.071
papy.globals.alpha = 0.18
papy.globals.ka = 0.41
papy.globals.d0 = 0.
if papy.globals.run_name in ['single_building_ABL_2m', 'BA_BL_z0_z02', 'BA_BL_z0_z021', 'BA_BL_z0_z06']:
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

################
# Timeseries of several measures
if compute_timeseries:
    nc_file = '{}_ts{}.nc'.format(papy.globals.run_name, papy.globals.run_number)
    var_name_list = ['umax', 'w"u"0', 'E', 'E*', 'div_old', 'div_new', 'dt', 'us*']
    # var_name_list = ['umax']

    for var_name in var_name_list:
        time_total = np.zeros(1)
        var_total = np.zeros(1)
        fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
        for i,run in enumerate(papy.globals.run_numbers):
            nc_file = '{}_ts{}.nc'.format(papy.globals.run_name, run)
            time, time_unit = papy.read_nc_time(nc_file_path,nc_file)
            var, var_unit = papy.read_nc_var_ts(nc_file_path,nc_file,var_name)
            time_total = np.append(time_total, time)
            var_total = np.append(var_total, var)
            if var_name == 'umax':
                ax.plot(time, var)
                ax.set_xlabel(r'$t$ ({})'.format('s'))
                ax.set_ylabel(r'$u_{max}$ ' + r'({})'.format(var_unit))
        if var_name == 'dt':
            print('     mean time step = ' + str(np.mean(var_total)))
        if var_name == 'umax':
            # ax.grid()
            ax.vlines(5400., 4, 9, 
                        color='tab:red', linestyles='--',
                        label='onset of data output')
            ax.legend(loc='lower center', numpoints=1, 
                bbox_to_anchor=(0.5, 1.0), fontsize=11)
            # ax.set_ylim(5., 6.25)
            ax.set_xlim(0., max(time_total))
            fig.savefig('../palm_results/{}/run_{}/timeseries/{}_{}_{}_ts.{}'.format(papy.globals.run_name,papy.globals.run_number[-3:],
                        papy.globals.run_name, var_name, papy.globals.run_name, file_type), bbox_inches='tight',dpi=300)
        print('\n READ {} from {}{} \n'.format(var_name, nc_file_path, nc_file))
        papy.plot_timeseries(var_total, var_unit, var_name, time_total, time_unit, file_type)
        print('\n plotted {} \n'.format(var_name))


    # Divergence comparison
    time_total = np.zeros(1)
    var_old_total = np.zeros(1)
    var_new_total = np.zeros(1)
    for i,run in enumerate(papy.globals.run_numbers):
        nc_file = '{}_ts{}.nc'.format(papy.globals.run_name, run)
        time, time_unit = papy.read_nc_time(nc_file_path,nc_file)
        var_old, var_unit = papy.read_nc_var_ts(nc_file_path,nc_file,'div_old')
        var_new, var_unit = papy.read_nc_var_ts(nc_file_path,nc_file,'div_new')        
        time_total = np.append(time_total, time)
        var_old_total = np.append(var_old_total, var_old)
        var_new_total = np.append(var_new_total, var_new)

    print('     div_old = ' + str(np.mean(var_old_total)))
    print('     div_new = ' + str(np.mean(var_new_total)))
    print('     div_red = ' + str(np.mean(var_new_total)-np.mean(var_old_total)))        
    fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))   
    ax.plot(time_total, var_old_total, color='tab:blue', label=r'$(\nabla \cdot \vec u)_{old}$  (s$^{-1}$)')
    ax.plot(time_total, var_new_total, color='tab:green', label=r'$(\nabla \cdot \vec u)_{new}$  (s$^{-1}$)')
    ax.vlines(5400., 10**(-7), 10**(-2), 
                color='tab:red', linestyles='--',
                label='onset of data output')
    ax.set_xlabel(r'$t$ (s)', fontsize=11)
    ax.set_ylabel(r'$(\nabla \cdot \vec u)$  (s$^{-1}$)', fontsize=11)    
    ax.legend(loc='lower center', numpoints=1, 
                bbox_to_anchor=(0.5, 1.0), fontsize=11)
    # ax.set_ylim(4.75, 6.25)
    ax.set_xlim(0., max(time_total))
    ax.set_yscale('log')
    fig.savefig('../palm_results/{}/run_{}/timeseries/divergence_comp_{}_ts.{}'.format(
                papy.globals.run_name,papy.globals.run_number[-3:], papy.globals.run_name, file_type), 
                bbox_inches='tight',dpi=300)    

    # Energy Comparison
    time_total = np.zeros(1)
    var_old_total = np.zeros(1)
    var_new_total = np.zeros(1)
    for i,run in enumerate(papy.globals.run_numbers):
        nc_file = '{}_ts{}.nc'.format(papy.globals.run_name, run)
        time, time_unit = papy.read_nc_time(nc_file_path,nc_file)
        var_old, var_unit = papy.read_nc_var_ts(nc_file_path,nc_file,'E*')
        var_new, var_unit = papy.read_nc_var_ts(nc_file_path,nc_file,'E')        
        time_total = np.append(time_total, time)
        var_old_total = np.append(var_old_total, var_old)
        var_new_total = np.append(var_new_total, var_new)

    var_new_total_mean = np.mean(var_new_total)
    var_old_total_mean = np.mean(var_old_total)

    fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))   
    ax.plot(time_total, var_old_total, color='tab:olive', label=r'$\overline{e}_{SGS}=$' + r'${}$'.format(str(var_old_total_mean)[0:4]) + r' m$^2$s$^{-2}$')
    ax.plot(time_total, var_new_total, color='tab:purple', label=r'$\overline{e}_{total}=$' + r'${}$'.format(str(var_new_total_mean)[0:4]) + r' m$^2$s$^{-2}$')
    ax.vlines(5400., 10**(-3), 100, 
                color='tab:red', linestyles='--',
                label='onset of data output')
    ax.set_xlabel(r'$t$ (s)', fontsize=11)
    ax.set_ylabel(r'$E$ (m$^2$ s$^{-2}$)', fontsize=11)    
    ax.legend(loc='lower center', numpoints=1, 
                bbox_to_anchor=(0.5, 1.0), fontsize=11)
    # ax.set_ylim(4.75, 6.25)
    ax.set_xlim(0., max(time_total))
    ax.set_yscale('log')
    fig.savefig('../palm_results/{}/run_{}/timeseries/energy_comp_{}_ts.{}'.format(
                papy.globals.run_name,papy.globals.run_number[-3:], papy.globals.run_name, file_type), 
                bbox_inches='tight',dpi=300)    


    print(' Finished Timeseries')

print('')
print('Finished processing of: {}{}'.format(papy.globals.run_name, papy.globals.run_number))