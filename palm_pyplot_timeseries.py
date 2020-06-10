################
""" 
author: benedikt.seitzer
purpose: - read palm output *_ts.nc file and plot specified (run_name, var_name) timeseries.
         - volume-averaged quantities.
"""
################


################
"""
IMPORTS
"""
################

import os
import numpy as np
import netCDF4

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import palm_py as papy

################
"""
FUNCTIONS
"""
################

def plot_timeseries(var, var_unit, time, time_unit,run_number):
    """
    plot height profile for all available times
    """    

    if run_number == '':
        run_number = '.000'

    plt.style.use('classic')
    fig, ax = plt.subplots()
    ax.plot(time, var, color='green')
    

    ax.set(xlabel=r'$t$ $[{}]$'.format(time_unit), ylabel=r'{} $[{}]$'.format(var_name,var_unit), 
            title= 'Timeseries {}'.format(var_name))

    ax.grid()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

    plt.xlim(min(time),max(time))
    fig.savefig('../palm_results/{}/run_{}/timeseries/{}_{}_ts.png'.format(run_name,run_number[-3:],
                run_name,var_name), bbox_inches='tight')
    # plt.show()


################
"""
GLOBAL VARIABLES
"""
################

run_name = 'thunder_balcony_resstudy_precursor'
run_number = '.012'
nc_file = '{}_ts{}.nc'.format(run_name,run_number)
nc_file_path = '../current_version/JOBS/{}/OUTPUT/'.format(run_name)

# var_name_list = ['umax', 'w"u"0', 'E', 'E*', 'div_old', 'div_new', 'dt', 'us*', 'u_p1']
var_name_list = ['umax', 'w"u"0', 'E', 'E*', 'div_old', 'div_new', 'dt', 'us*']

################
"""
MAIN
"""
################

# prepare the outputfolders
papy.prepare_plotfolder(run_name,run_number)

# read variables for plot and call plot-function
time, time_unit = papy.read_nc_time(nc_file_path,nc_file)

for var_name in var_name_list:
    var, var_unit = papy.read_nc_var_ts(nc_file_path,nc_file,var_name)
    print('\n READ {} from {}{} \n'.format(var_name, nc_file_path, nc_file))
    plot_timeseries(var, var_unit, time, time_unit, run_number)
    print('\n plotted {} \n'.format(var_name))