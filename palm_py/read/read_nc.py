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

import numpy as np
import netCDF4

################
"""
FUNCTIONS
"""
################

__all__ = [
    'read_nc_var_hor_3d',
    'read_nc_var_ver_3d',
    'read_nc_var_hor_pr',
    'read_nc_var_ver_pr',
    'read_nc_var_ts',
    'read_nc_var_av_3d',
    'read_nc_grid',
    'read_nc_time',
    'read_nc_var_ms'
]

# netCDF-readers
def read_nc_var_hor_3d(nc_file_path, nc_file, var_name, z_level, time_val):
    """
    read xy-plane of predefined variable 

    ----------
    Parameters:
    nc_file_path: str
    nc_file: str
    var_name: str
    z_level : int
    time_val: float

    ----------
    Returns:
    var: array-like
    var_unit: str
    """
    fh = netCDF4.Dataset('{}{}'.format(nc_file_path,nc_file), mode='r')

    var = fh.variables[var_name][time_val,z_level,:,:]
    var_unit = fh.variables[var_name].units
    fh.close()

    return var, var_unit

def read_nc_var_ver_3d(nc_file_path, nc_file, var_name, y_level, time_val):
    """
    read xz-plane of predefined variable 

    ----------
    Parameters:
    nc_file_path: str
    nc_file: str
    var_name: str
    y_level : int
    time_val: float

    ----------
    Returns:
    var: array-like
    var_unit: str
    """
    fh = netCDF4.Dataset('{}{}'.format(nc_file_path,nc_file), mode='r')

    var = fh.variables[var_name][time_val,:,y_level,:]
    var_unit = fh.variables[var_name].units
    fh.close()

    return var, var_unit

def read_nc_var_hor_pr(nc_file_path, nc_file, var_name, z_level, x_level, time_val):
    """
    read xy-plane of predefined variable 

    ----------
    Parameters:
    nc_file_path: str
    nc_file: str
    var_name: str
    z_level : int
    x_level : int
    time_val: float

    ----------
    Returns:
    var: array-like
    var_max: float
    var_unit: str
    """
    fh = netCDF4.Dataset('{}{}'.format(nc_file_path,nc_file), mode='r')

    var = fh.variables[var_name][time_val,z_level,:,x_level]
    var_max = np.amax(fh.variables[var_name][time_val,:,:,:])
    var_unit = fh.variables[var_name].units
    fh.close()

    return var, var_max, var_unit

def read_nc_var_ver_pr(nc_file_path, nc_file, var_name):
    """
    Read palm horizontally averaged vertical profiles.

    ----------
    Parameters:
    nc_file_path: str
    nc_file: str
    var_name: str

    ----------
    Returns:
    var: array-like
    var_max: float
    var_unit: str
    """

    fh = netCDF4.Dataset('{}{}'.format(nc_file_path,nc_file), mode='r')

    var = fh.variables[var_name][:,:]
    var_max = np.amax(var)
    var_unit = fh.variables[var_name].units
    fh.close()

    return var, var_max, var_unit

def read_nc_var_ts(nc_file_path, nc_file, var_name):
    """
    Read palm timeseries.

    ----------
    Parameters:
    nc_file_path: str
    nc_file: str
    var_name: str

    ----------
    Returns:
    var: array-like
    var_unit: str
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
        
def read_nc_var_av_3d(var_name, z_level):
    """
    read horizontal slice of variable at certain height

    ----------
    Parameters:
    var_name: str
    z_level : int

    ----------
    Returns:
    var: array-like
    var_unit: str
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
        
def read_nc_grid(nc_file_path, nc_file, grid_name):
    """
    read grid belonging to variable.
    
    ----------
    Parameters:
    nc_file_path: str
    nc_file: str
    grid_name: str
    
    ----------
    Returns:
    grid: array-like
    grid_unit: str
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

def read_nc_time(nc_file_path, nc_file):
    """
    Read times vector.

    ----------
    Parameters:
    nc_file_path: str
    nc_file: str
    
    ----------
    Returns:
    time: array-like
    time_unit: str
    """
    try:
        fh = netCDF4.Dataset('{}{}'.format(nc_file_path,nc_file), mode='r')
    except:
        print('\n Exception occured: /{}{} not found \n'.format(nc_file_path, nc_file))
        

    time = fh.variables['time'][:]
    time_unit = fh.variables['time'].units
    fh.close()

    return time, time_unit

def read_nc_var_ms(nc_file_path,nc_file,var_name):
    """
    Read palm timeseries from masked data.

    ----------
    Parameters:
    nc_file_path: str
    nc_file: str
    var_name: str
    
    ----------
    Returns:
    var: array-like
    var_unit: str
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
        
