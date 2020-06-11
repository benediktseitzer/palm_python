################
""" 
author: benedikt.seitzer
purpose: read palm output _pr.nc file and plot crosssection vor every second gridpoint.
"""
################


################
"""
IMPORTS
"""
################

import numpy as np
import os

import matplotlib.pyplot as plt

import palm_py as papy

################
"""
FUNCTIONS
"""
################

def plot_contour_crosssection(x,y,var,z_level,vert_gridname, cut_gridname, run_number):
    """
    plot cross sections
    """    

    if run_number == '':
        run_number = '.000'

    plt.style.use('classic')
    fig, ax = plt.subplots()


    # estimate bounds of colorbar
    if abs(np.min(var)) > abs(np.max(var)):
        v_bound = np.min(var)
    elif abs(np.min(var)) < abs(np.max(var)):
        v_bound = np.max(var)

    # set colorbar and mark masked buildings in grey
    current_cmap = plt.cm.seismic
    current_cmap.set_bad(color='gray')
    # plot the 2D-array var
    im = ax.imshow(var, interpolation='bilinear', cmap=current_cmap, 
                    extent=(np.min(x), np.max(x), np.min(y), np.max(y)), 
                    vmin=-v_bound, vmax=v_bound, origin='lower')

    # labeling 
    fig.colorbar(im, label=r'${}$ $[{}]$'.format(var_name,var_unit),orientation="horizontal")
    ax.set_title(r'Crosssection of ${}$ at ${}={}$ ${}$'.format(var_name,cut_gridname, 
                    round(z_grid[z_level],2), z_unit))
    ax.set(xlabel=r'${}$ $[{}]$'.format(x_grid_name,x_unit), 
            ylabel=r'${}$ $[{}]$'.format(vert_gridname,y_unit))

    # file output
    fig.savefig('../palm_results/{}/run_{}/crosssections/{}_{}_cs_{}_{}.png'.format(run_name,run_number[-3:],
                run_name,var_name,str(round(z_grid[z_level],2)),crosssection), bbox_inches='tight')
    # plt.show()
    plt.close()


################
"""
GLOBAL VARIABLES
"""
################

run_name = 'thunder_balcony_resstudy_precursor'
run_number = '.013'
crosssection = 'xy'

nc_file = '{}_3d{}.nc'.format(run_name, run_number)
nc_file_path = '../current_version/JOBS/{}/OUTPUT/'.format(run_name)

x_grid_name = 'x'
y_grid_name = 'y'
z_grid_name = 'zw_3d'


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

time_show = len(time)-1

print('\n READ {}{}\n'.format(nc_file_path,nc_file))

var_name_list = ['u', 'v', 'w']

for var_name in var_name_list:
    print('\n --> plot {}: \n'.format(var_name))
    
    # if crosssection == 'xz':
    crosssection = 'xz'
    y_level = int(len(y_grid)/2)
    print('     y={}    level={}'.format(round(y_grid[y_level],2), y_level))
    vert_gridname = 'z'
    cut_gridname = y_grid_name
    var, var_unit = papy.read_nc_var_ver_3d(nc_file_path,nc_file,var_name, y_level, time_show)
    plot_contour_crosssection(x_grid, z_grid, var, y_level, vert_gridname, cut_gridname, run_number)
    
    # elif crosssection == 'xy':
    crosssection = 'xy'
    z_level = int(len(z_grid)/6)
    vert_gridname = y_grid_name
    cut_gridname = 'z'
    print('     z={}    level={}'.format(round(z_grid[z_level],2), z_level))
    var, var_unit = papy.read_nc_var_hor_3d(nc_file_path,nc_file,var_name, z_level, time_show)
    plot_contour_crosssection(x_grid, y_grid, var, z_level, vert_gridname, cut_gridname, run_number)
