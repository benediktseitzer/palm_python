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
import matplotlib.pyplot as plt

import palm_py as papy

################
"""
FUNCTIONS
"""
################

__all__ = [
    'calc_input_profile',
    'calc_topofile_building',
    'calc_topofile_roughness',
    'calc_topofile_roughness_building']

def calc_input_profile(m_u, m_z, z, ref_height):
    """
    calculate u- and v-profile using measurement data

    -----------
    Parameters
    m_u: array-like
    m_z: array-like
    z: array-like
    ref_height: float

    ---------
    Returns
    u_prandtl: float
    u_fric: float
    """
    # calculate friction velocity
    i_upper = np.where(m_z == ref_height[1])
    u_fric = (papy.globals.ka * m_u[i_upper]) / (np.log(m_z[i_upper]/papy.globals.z0))
    # calculate theoretical profiles
    u_prandtl = u_fric/papy.globals.ka * np.log((z[1:]-papy.globals.d0)/papy.globals.z0)
    # no-slip bc:
    u_prandtl = np.insert(u_prandtl,0,0.)
    print('     z0 = {} m'.format(papy.globals.z0))
    print('     u* = {} with reference height at {} m \n'.format(u_fric,m_z[i_upper]))

    return u_prandtl, u_fric


def calc_topofile_building(building_height, building_x_length, building_y_length):
    """
    Construct simple topography ascii-file for single building 

    -----------
    Parameters
    nx: nx
    ny: ny
    building_height: float
    building_x_edge: integer
    building_y_edge: integer

    ---------
    Returns

    """
    nx = papy.globals.nx + 1
    ny = papy.globals.ny + 1
    dx = papy.globals.dx
    topo_matrix = np.zeros((nx,ny))
    for i in range(nx):
        if dx*i>((dx*nx/2.)-building_x_length/2.) and i<((dx*nx/2.)+building_x_length/2.):
            for j in range(ny):
                if j>((dx*ny/2.)-building_y_length/2.) and j<((dx*ny/2.)+building_y_length/2.):
                    topo_matrix[i,j] = building_height
                    # print(i, j, topo_matrix[i,j])
    plt.figure(11)
    plt.imshow(topo_matrix)
    plt.xlabel('x in m')
    plt.xlabel('y in m')    
    plt.show()
    plt.close(11)
    np.savetxt('{}_topo'.format(papy.globals.run_name), topo_matrix, fmt='%1.0f', delimiter=' ')
    print('     Saved topo-file to {}_topo'.format(papy.globals.run_name))


def calc_topofile_roughness(rough_dist_x, rough_dist_y, rough_height):
    """
    Construct simple topography ascii-file for wind tunnel roughness

    -----------
    Parameters
    nx: nx
    ny: ny
    rough_dist_x: float
    rough_dist_y: float
    rough_height: float

    ---------
    Returns

    """
    nx = papy.globals.nx + 1
    ny = papy.globals.ny + 1
    dx = papy.globals.dx
    topo_matrix = np.zeros((nx,ny))

    for i in range(nx):
        if dx*i % rough_dist_x == 0:
            for j in range(ny):
                if dx*j % rough_dist_y == 0:
                    topo_matrix[j,i] = rough_height
                    topo_matrix[j+1,i] = rough_height
                    topo_matrix[j-1,i] = rough_height
        elif (dx*i+15) % rough_dist_x == 0:
            for j in range(ny):
                if (dx*j-7) % rough_dist_y == 0:
                    topo_matrix[j,i] = rough_height
                    topo_matrix[j+1,i] = rough_height
                    topo_matrix[j-1,i] = rough_height
        elif (dx*i+30) % rough_dist_x == 0:
            for j in range(ny):
                if (dx*j-14) % rough_dist_y == 0:
                    topo_matrix[j,i] = rough_height
                    topo_matrix[j+1,i] = rough_height
                    topo_matrix[j-1,i] = rough_height
    plt.figure(11)
    plt.imshow(topo_matrix, cmap='binary', interpolation='none')
    plt.xlabel('x in m')
    plt.xlabel('y in m')    
    plt.show()
    plt.close(11)
    np.savetxt('{}_topo'.format(papy.globals.run_name), topo_matrix, fmt='%1.0f', delimiter=' ')
    print('     Saved topo-file to {}_topo'.format(papy.globals.run_name))


def calc_topofile_roughness_building(rough_dist_x, rough_dist_y, rough_height, building_height, building_x_length, building_y_length):
    """
    Construct simple topography ascii-file for wind tunnel roughness

    -----------
    Parameters
    nx: nx
    ny: ny
    rough_dist_x: float
    rough_dist_y: float
    rough_height: float

    ---------
    Returns

    """
    nx = papy.globals.nx + 1
    ny = papy.globals.ny + 1
    dx = papy.globals.dx
    topo_matrix = np.zeros((nx,ny))
    # wind tunnel roughness elements
    for i in range(nx):
        if dx*i % rough_dist_x == 0:
            for j in range(ny):
                if dx*j % rough_dist_y == 0:
                    topo_matrix[j,i] = rough_height
                    topo_matrix[j+1,i] = rough_height
                    topo_matrix[j-1,i] = rough_height
        elif (dx*i+15) % rough_dist_x == 0:
            for j in range(ny):
                if (dx*j-7) % rough_dist_y == 0:
                    topo_matrix[j,i] = rough_height
                    topo_matrix[j+1,i] = rough_height
                    topo_matrix[j-1,i] = rough_height
        elif (dx*i+30) % rough_dist_x == 0:
            for j in range(ny):
                if (dx*j-14) % rough_dist_y == 0:
                    topo_matrix[j,i] = rough_height
                    topo_matrix[j+1,i] = rough_height
                    topo_matrix[j-1,i] = rough_height
    # building geometry
    for i in range(nx):
        if dx*i>((dx*nx/2.)-building_x_length/2.) and i<((dx*nx/2.)+building_x_length/2.):
            for j in range(ny):
                if j>((dx*ny/2.)-building_y_length/2.) and j<((dx*ny/2.)+building_y_length/2.):
                    topo_matrix[i,j] = building_height

    plt.figure(11)
    plt.imshow(topo_matrix, cmap='binary', interpolation='none')
    plt.xlabel('x in m')
    plt.xlabel('y in m')    
    plt.show()
    plt.close(11)
    np.savetxt('{}_topo'.format(papy.globals.run_name), topo_matrix, fmt='%1.0f', delimiter=' ')
    print('     Saved topo-file to {}_topo'.format(papy.globals.run_name))