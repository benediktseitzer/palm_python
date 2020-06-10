################
""" 
author: benedikt.seitzer
name: read_nc
purpose: read netCDF-files
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
import netCDF4

################
"""
FUNCTIONS
"""
################

# netCDF-readers
def read_nc_var_hor_3d(nc_file_path,nc_file,var_name,z_level,time_val):
    """
    read xy-plane of predefined variable 
    var_name = variable identifier
    z_level = height of crosssection
    time_val = timestamp of read data

    var = crosssection of var_name
    var_unit = units of var
    """
    fh = netCDF4.Dataset('{}{}'.format(nc_file_path,nc_file), mode='r')

    var = fh.variables[var_name][time_val,z_level,:,:]
    var_unit = fh.variables[var_name].units
    fh.close()

    return var, var_unit


def read_nc_var_ver_3d(nc_file_path,nc_file,var_name,y_level,time_val):
    """
    read xz-plane of predefined variable 
    var_name = variable identifier
    z_level = height of crosssection
    time_val = timestamp of read data

    var = crosssection of var_name
    var_unit = units of var    
    """
    fh = netCDF4.Dataset('{}{}'.format(nc_file_path,nc_file), mode='r')

    var = fh.variables[var_name][time_val,:,y_level,:]
    var_unit = fh.variables[var_name].units
    fh.close()

    return var, var_unit


def read_nc_var_hor_pr(nc_file_path,nc_file,var_name,z_level,x_level,time_val):
    """
    read xy-plane of predefined variable 
    var_name = variable identifier
    z_level = height of profile
    x_level = x-position of profile
    time_val = timestamp of read data    
    
    var = horizontally averaged values
    var_max = maximum value
    var_unit = units of var
    """
    fh = netCDF4.Dataset('{}{}'.format(nc_file_path,nc_file), mode='r')

    var = fh.variables[var_name][time_val,z_level,:,x_level]
    var_max = np.amax(fh.variables[var_name][time_val,:,:,:])
    var_unit = fh.variables[var_name].units
    fh.close()

    return var, var_max, var_unit


def read_nc_var_ver_pr(nc_file_path,nc_file,var_name):
    """
    read palm horizontally averaged vertical profiles
    var_name = variable identifier
    var = horizontally averaged values
    var_max = maximum value
    var_unit = units of var
    """

    fh = netCDF4.Dataset('{}{}'.format(nc_file_path,nc_file), mode='r')

    var = fh.variables[var_name][:,:]
    var_max = np.amax(var)
    var_unit = fh.variables[var_name].units
    fh.close()

    return var, var_max, var_unit


def read_nc_var_ts(nc_file_path,nc_file,var_name):
    """
    read palm timeseries 
    var_name = variable identifier
    var = timeseries of var_name
    var_unit = units of var
    """

    try:
        fh = netCDF4.Dataset('{}{}'.format(nc_file_path, nc_file), mode='r')
    except:
        print('\n Exception occured: {} not found \n'.format(nc_file))

    try:    
        var = fh.variables[var_name][:]
        var_unit = fh.variables[var_name].units
        fh.close()
        return var, var_unit
    except:
        print('\n Exception occured: no variable called {} in {} \n'.format(var_name,nc_file))


def read_nc_var_av_3d(var_name,z_level):
    """
    read horizontal slice of variable at certain height

    var_name = predefined variable 
    z_level = index of height-gridpoint.
    """
    try:
        fh = netCDF4.Dataset('{}{}'.format(nc_file_path,nc_file), mode='r')
    except:
        print('\n Exception occured: {} not found \n'.format(nc_file))

    try: 
        var = fh.variables[var_name][1,z_level,:,:]
        var_unit = fh.variables[var_name].units
        fh.close()
        return var, var_unit
    except: 
        print('\n Exception occured: no variable called {} in {} \n'.format(var_name,nc_file))        


def read_nc_grid(nc_file_path,nc_file,grid_name):
    """
    read grid belonging to variable
    grid_name = coordinate-name

    grid = grid-values
    grid_unit = unit of the grid
    """    
    try:
        fh = netCDF4.Dataset('{}{}'.format(nc_file_path,nc_file), mode='r')
    except: 
        print('\n Exception occured: {} not found \n'.format(nc_file))

    try: 
        grid = fh.variables[grid_name][:]
        grid_unit = fh.variables[grid_name].units
        fh.close()
        return grid, grid_unit
    except:
        print('\n Exception occured: no variable called {} in {} \n'.format(grid_name,nc_file))


def read_nc_time(nc_file_path,nc_file):
    """
    read times vector

    time = time-value
    time_unit = unit of time 
    """
    try:
        fh = netCDF4.Dataset('{}{}'.format(nc_file_path,nc_file), mode='r')
    except:
        print('\n Exception occured: {} not found \n'.format(nc_file))

    time = fh.variables['time'][:]
    time_unit = fh.variables['time'].units
    fh.close()

    return time, time_unit


def read_nc_var_ms(nc_file_path,nc_file,var_name):
    """
    read palm timeseries  from masked data
    var_name = variable identifier
    var = timeseries of var_name
    var_unit = units of var
    """

    try:
        fh = netCDF4.Dataset('{}{}'.format(nc_file_path, nc_file), mode='r')
    except:
        print('\n Exception occured: {} not found \n'.format(nc_file))

    try:    
        if var_name == 'time':
            var = fh.variables[var_name][:]
        else:    
            var = fh.variables[var_name][:,0,0,0]
            
        var_unit = fh.variables[var_name].units
        fh.close()
        return var, var_unit
    except:
        print('\n Exception occured: no variable called {} in {} \n'.format(var_name,nc_file))
