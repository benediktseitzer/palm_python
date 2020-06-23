################
""" 
author: benedikt.seitzer

purpose: - read palm output _pr.nc file and plot specified (run_name, var_name) height profiles.
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

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import palm_py as papy

################
"""
FUNCTIONS
"""
################

def plot_semilog_u(var, var_unit, z, z_unit, time, time_unit,run_number):
    """
    semilog-plot u-profile for all available times
    """    

    xerror = 0.03*wt_pr
    if run_number == '':
        run_number = '.000'

    fig2, ax2 = plt.subplots()    
    ax2.set_yscale("log", nonposy='clip')    
    jet= plt.get_cmap('viridis')
    colors = iter(jet(np.linspace(0,1,10)))
    for i in range(len(time)-1,len(time)):
        try:
            ax2.plot(var[i,1:-1], z[1:-1], label='PALM', color=next(colors))
            ax2.plot(u_pw[1:-1], z[1:-1], label='power law', color='red',linestyle='--')
            ax2.plot(u_pr[1:-1], z[1:-1], label='prandtls law', color='blue',linestyle='--')
            ax2.errorbar(wt_pr[:],wt_z[:],xerr=xerror[:],label='wind tunnel',fmt='x',color='grey')
        except:
            print('Exception has occurred: StopIteration - plot_semilog-plot_u')

    ax2.xaxis.set_major_locator(plt.MaxNLocator(7))

    plt.style.use('classic')

    ax2.legend(loc='best')
    ax2.set(xlabel=r'$u/u_{ref}$ $[-]$', ylabel=r'$z$ $[{}]$'.format(z_unit), 
            title= r'Logarithmic profile of $u/u_{ref}$')
    ax2.grid()
    ax2.grid(which='minor', alpha=0.2)
    ax2.grid(which='major', alpha=0.2)
    fig2.savefig('../palm_results/{}/run_{}/profiles/{}_{}_pr_verpr_log.png'.format(run_name,run_number[-3:],
                    run_name,var_name), bbox_inches='tight')
    # plt.show()

def plot_ver_profile(var_plt, var_unit, z, z_unit, time, time_unit,run_number):
    """
    plot height profile for all available times
    """    

    xerror = 0.03*wt_pr
    if run_number == '':
        run_number = '.000'

    plt.style.use('classic')
    fig, ax = plt.subplots()
    jet= plt.get_cmap('viridis')
    colors = iter(jet(np.linspace(0,1,10)))
    ax.grid()
    ax.xaxis.set_major_locator(plt.MaxNLocator(7))

    for i in range(len(time)-1,len(time)):
        try:
            ax.plot(var_plt[i,:-1], z[:-1], label='PALM', 
                    color=next(colors))
        except:
            print('Exception has occurred: StopIteration - plot_ver_profile')
    if var_name == 'u':
        try:
            ax.plot(u_pw[:-1], z[:-1], label='power law', color='red',linestyle='--')
            ax.plot(u_pr[:-1], z[:-1], label='prandtls law', color='blue',linestyle='--')
            ax.errorbar(wt_pr[:-1],wt_z[:-1],xerr=xerror[:-1],label='wind tunnel',fmt='x',c='gray')
        except:
            print('Exception has occurred: Stop wt-plotting - plot_ver_profile')

    if var_name == 'w"u"':
        ax.set(xlabel=r'$\tau _{31}$'+'$[m^2/s^2]$', 
                ylabel=r'$z$ $[m]$', title= r'Height profile of $\tau _{31}$')
    elif var_name == 'w*u*':
        ax.set(xlabel=r'$\tau^* _{31}$'+'$[m^2/s^2]$', 
                ylabel=r'$z$ $[m]$', title= r'Height profile of $\tau^* _{31}$')
    elif var_name == 'u':
        ax.set(xlabel=r'$u/u_{ref}$'+'$[-]$', 
                ylabel=r'$z$ $[m]$', title= r'Height profile of $u/u_{ref}$')
    elif var_name == 'e*':
        ax.set(xlabel=r'$e^*$'+'$[m^2/s^2]$', 
                ylabel=r'$z$ $[m]$', title= r'Height profile of $e^*$')
    elif var_name == 'u*2':
        ax.set(xlabel=r'$\tau _{11}$'+'$[m^2/s^2]$', 
                ylabel=r'$z$ $[m]$', title= r'Height profile of $\tau _{11}$')                
    elif var_name == 'e':
        ax.set(xlabel=r'$e$'+'$[m^2/s^2]$', 
                ylabel=r'$z$ $[m]$', title= r'Height profile of $e$')                
    else:     
        ax.set(xlabel=r'${}$ $[{}]$'.format(var_name,var_unit), 
                ylabel=r'$z$ $[{}]$'.format(z_unit), title= r'Height profile of ${}$'.format(var_name))
     
    ax.legend(loc='best')
    plt.ylim(min(z),max(z[:-1]))
    fig.savefig('../palm_results/{}/run_{}/profiles/{}_{}_verpr.png'.format(run_name,run_number[-3:],
                run_name,var_name), bbox_inches='tight')
    # plt.show()


################
"""
GLOBAL VARIABLES
"""
################

# PATHS
run_name = 'thunder_balcony_resstudy_precursor'
run_number = '.014'
nc_file = '{}_pr{}.nc'.format(run_name,run_number)
nc_file_path = '../current_version/JOBS/{}/OUTPUT/'.format(run_name)
#wt_file = '../../Documents/phd/palm/input_data/u_profile.txt'
wt_file = '../../Documents/phd/palm/input_data/windtunnel_data/HG_BL_MR_DOK_UV_015_means.txt'

# PHYSICS
# roughness length
z0 = 0.066
# exponent for powerlaw-fit
alpha = 0.17
# von Karman constant
ka = 0.41
# displacement height
d0 = 0.
# save physical parameters to list
phys_params = [z0,alpha,ka,d0]

################
"""
MAIN
"""
################

print('\n READ {}{}\n'.format(nc_file_path,nc_file))

papy.prepare_plotfolder(run_name,run_number)

# read variables for plot
time, time_unit = papy.read_nc_time(nc_file_path,nc_file)

# read wind tunnel profile
wt_pr, wt_u_ref, wt_z = papy.read_wt_ver_pr(wt_file)
print('\n wind tunnel profile loaded \n') 


# call plot-functions
var_name_list = ['w*u*', 'w"u"', 'e', 'e*', 'u*2', 'u']
for var_name in var_name_list:
    if var_name == 'u':
        grid_name = 'z{}'.format(var_name)
        var, var_max, var_unit = papy.read_nc_var_ver_pr(nc_file_path,nc_file,var_name)
        print('\n       u_max = {} \n'.format(var_max))

        z, z_unit = papy.read_nc_grid(nc_file_path,nc_file,grid_name)

        # calculate theoretical wind profile
        u_pr, u_pw, u_fric = papy.calc_theoretical_profile(wt_pr, wt_u_ref, wt_z,z,phys_params)

        print('\n       wt_u_ref = {} \n'.format(wt_u_ref))
        print('\n       th_u_fric = {} \n'.format(u_fric))

        plot_ver_profile(var/wt_u_ref, var_unit, z, z_unit, time, time_unit,run_number)
        plot_semilog_u(var/wt_u_ref, var_unit, z, z_unit, time, time_unit,run_number)
     
        print('\n --> plottet {} \n'.format(var_name))
    else:
        grid_name = 'z{}'.format(var_name)
        var, var_max, var_unit = papy.read_nc_var_ver_pr(nc_file_path,nc_file,var_name)
        z, z_unit = papy.read_nc_grid(nc_file_path,nc_file,grid_name)
        plot_ver_profile(var, var_unit, z, z_unit, time, time_unit,run_number)
        print('\n --> plottet {} \n'.format(var_name))
