################
""" 
author: benedikt.seitzer
purpose: read palm output _3d.nc file and plot horizontal profile.
"""
################


################
"""
IMPORTS
"""
################

import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt

import module_palm_py as papy

################
"""
FUNCTIONS
"""
################

def read_wt_hor_pr(wt_file):
    """
    read wind tunnel profile
    """
    df = pd.read_table(wt_file,delimiter=',',header=1, usecols=[2,4,5])
    u_ref_dat = df.iloc[:,1].to_numpy()
    u_ref = np.mean(u_ref_dat)
    u = df.iloc[:,2].to_numpy()
    y = (df.iloc[:,0].to_numpy()/100.+2.)
    return u, u_ref, y

def plot_hor_profile(var, var_unit, y, y_unit, time, time_unit, run_number):
    """
    plot height profile for all available times
    """    

    xerror = 0.03*wt_pr
    if run_number == '':
        run_number = '.000'

    plt.style.use('classic')
    fig, ax = plt.subplots()
    jet = plt.get_cmap('viridis')
    colors = iter(jet(np.linspace(0,1,10)))
    ax.grid()
    # ax.xaxis.set_major_locator(plt.MaxNLocator(7))

    for i in range(len(time)-1,len(time)):
        try:
            ax.plot(y[:-1], var[:-1], label='{} [{}]'.format(str(round(time[i],2)),time_unit), 
                    color=next(colors))
        except:
            print('Exception has occurred: StopIteration')
    if var_name == 'u':
        try:
            ax.errorbar(wt_y[:], wt_pr[:], yerr=xerror[:], label='wind tunnel', fmt='x')
        except:
            print('Exception has occurred: Stop wt-plotting')

    ax.set(xlabel=r'$y$ $[{}]$'.format(y_unit), ylabel=r'$u/u_{ref}$ $[-]$', 
    title= r'Horizontal profile of $u/u_{ref}$')
    
    ax.legend(loc='best')
    plt.xlim(min(y),max(y[:-1]))
    fig.savefig('../../palm_results/{}/run_{}/profiles/{}_{}_horpr.png'.format(run_name,run_number[-3:],
                run_name,var_name), bbox_inches='tight')
    # plt.show()

################
"""
GLOBAL VARIABLES
"""
################

run_name = 'thunder_balcony_resstudy_precursor'
run_number = '.007'

nc_file = '{}_av_3d{}.nc'.format(run_name, run_number)
nc_file_path = '../../current_version/JOBS/{}/OUTPUT/'.format(run_name)

wt_file = '../../../Documents/phd/palm/input_data/HG_BL_MR_DOK_UV_004_means.txt'

x_grid_name = 'xu'
y_grid_name = 'y'
z_grid_name = 'zu_3d'


################
"""
MAIN
"""
################

# prepare the outputfolders
papy.prepare_plotfolder(run_name,run_number)

# read variables for plot
x_grid, x_unit = papy.read_nc_grid(nc_file_path,nc_file,x_grid_name)
y_grid, y_unit = papy.read_nc_grid(nc_file_path,nc_file,y_grid_name)
z_grid, z_unit = papy.read_nc_grid(nc_file_path,nc_file,z_grid_name)


time, time_unit = papy.read_nc_time(nc_file_path,nc_file)
print('\n READ {}{}\n'.format(nc_file_path,nc_file))


# read wind tunnel profile
wt_pr, wt_u_ref, wt_y = read_wt_hor_pr(wt_file)
print('\n wind tunnel profile loaded \n') 


# x-component of velocity
print('\n x-component of velocity \n')
var_name = 'u'
z_level = 3
x_level = 32
time_show = len(time)-1
vert_gridname = y_grid_name
cut_gridname = 'z'

var, var_max, var_unit = papy.read_nc_var_hor_pr(nc_file_path,nc_file,var_name, z_level, x_level, time_show)
print('u_max = {}'.format(var_max))
plot_hor_profile(var/var_max, var_unit, y_grid, y_unit, time, time_unit, run_number)
print('\n -> plotted \n')