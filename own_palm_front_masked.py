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
plt.style.use('classic')

import palm_py as papy

sys.path.append('/home/bene/Documents/phd/windtunnel_py/windtunnel/')    
import windtunnel as wt

import warnings
warnings.simplefilter("ignore")

################
"""
FUNCTIONS
"""
################

################
"""
GLOBAL VARIABLES
"""
################
# PALM input files
papy.globals.run_name = 'SB_SI_front'
papy.globals.run_number = '.031'
papy.globals.run_numbers = ['.007', '.008', '.009', '.010', '.011', '.012', 
                            '.013', '.014', '.015', '.016', '.017', '.018',
                            '.019', '.020', '.021', '.022', '.023', '.024',
                            '.025', '.026', '.027', '.028', '.029', '.030', 
                            '.031']
nc_file_grid = '{}_pr{}.nc'.format(papy.globals.run_name,papy.globals.run_number)
nc_file_path = '../palm/current_version/JOBS/{}/OUTPUT/'.format(papy.globals.run_name)
mask_name_list = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08',
                    'M09', 'M10', 'M11', 'M12']
distance_list = []
height_list = []

# WIND TUNNEL INPIUT FILES
experiment = 'single_building'
# experiment = 'balcony'
# wt_filename = 'BA_BL_UW_001'
wt_path = '../../Documents/phd/experiments/{}/{}'.format(experiment, 'CO_REF')
wt_scale = 150.

# PHYSICS
papy.globals.z0 = 0.03
papy.globals.z0_wt = 0.071
papy.globals.alpha = 0.18
papy.globals.ka = 0.41
papy.globals.d0 = 0.
papy.globals.nx = 1024
papy.globals.ny = 1024
papy.globals.dx = 1.

# Steeringflags
compute_front_mean = True
compute_front_var = True
compute_front_covar = True
compute_spectra = False

compute_lux = False
compute_turbint_masked = False

################
"""
MAIN
"""
################

# prepare the outputfolders
papy.prepare_plotfolder(papy.globals.run_name,papy.globals.run_number)
plt.style.use('classic')

# get wtref from boundary layer PALM-RUN
palm_ref_run_numbers = ['.007', '.008', '.009', '.010', '.011', '.012', 
                        '.013', '.014', '.015', '.016', '.017', '.018',
                        '.019', '.020', '.021', '.022', '.023', '.024',
                        '.025', '.026', '.027', '.028', '.029', '.030', 
                        '.031']
palm_ref_file_path = '../palm/current_version/JOBS/{}/OUTPUT/'.format('SB_SI_BL')
for run_no in palm_ref_run_numbers:
    palm_ref_file = '{}_masked_{}{}.nc'.format('SB_SI_BL', 'M10', run_no)
    palm_u, var_unit = papy.read_nc_var_ms(palm_ref_file_path, palm_ref_file, 'u')
data_nd = 1
if data_nd == 1:
    palm_ref = np.mean(palm_u)
else:
    palm_ref = 1.
print('     PALM REFERENCE VELOCITY: {} m/s'.format(palm_ref))
# wind tunnel data
namelist = ['SB_FL_SI_UV_023',
            'SB_BR_SI_UV_012',
            'SB_WB_SI_UV_013']
config = 'CO_REF'
path = '{}/coincidence/timeseries/'.format(wt_path) # path to timeseries folder
wtref_path = '{}/wtref/'.format(wt_path)
wtref_factor = 0.738
scale = wt_scale

time_series = {}
time_series.fromkeys(namelist)
# Gather all files into Timeseries objects
for name in namelist:
    files = wt.get_files(path,name)
    time_series[name] = {}
    time_series[name].fromkeys(files)
    for i,file in enumerate(files):
        ts = wt.Timeseries.from_file(path+file)            
        ts.get_wind_comps(path+file)
        ts.get_wtref(wtref_path,name,index=i)
        ts.wtref = ts.wtref*wtref_factor
        # edit 6/20/19: Assume that input data is dimensional, not non-dimensional
        if data_nd == 0:
            print('Warning: Assuming that data is dimensional. If using non-dimensional input data, set variable data_nd to 1')
            ts.nondimensionalise()
        else:
            if data_nd == 1:
                []
            else:
                print('Warning: data_nd can only be 1 (for non-dimensional input data) or 0 (for dimensional input data)')        
        #edit 06/20/19: added seperate functionto  calculate equidistant timesteps             
        ts.adapt_scale(scale)         
        ts.mask_outliers()
        ts.index = ts.t_arr         
        ts.weighted_component_mean
        ts.weighted_component_variance
        time_series[name][file] = ts

# plotting colors and markers
c_list = ['forestgreen', 'darkorange', 'navy', 'tab:red', 'tab:olive']
marker_list = ['^', 'o', 'd', 'x', '8']

######################################################
# compute u-mean alongside building
######################################################
if compute_front_mean:
    print('\n     compute means')
    var_name_list = ['u', 'v']
    for var_name in var_name_list:
        mean_vars = np.array([])
        wall_dists = np.array([])
        for mask in mask_name_list:
            total_var = np.array([])
            total_time = np.array([])
            for run_no in papy.globals.run_numbers:
                nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask, run_no)
                time, time_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'time')
                var, var_unit = papy.read_nc_var_ms(nc_file_path, nc_file, var_name)
                y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'y')
                total_time = np.concatenate([total_time, time])
                total_var = np.concatenate([total_var, var])
            # gather values
            var_mean = np.asarray([np.mean(total_var)])
            wall_dist = np.asarray([abs(y[0]-530.)])
            mean_vars = np.concatenate([mean_vars, var_mean])
            wall_dists = np.concatenate([wall_dists, wall_dist])

        #plot profiles
        err = np.mean(mean_vars/palm_ref)*0.05
        fig, ax = plt.subplots()
        # plot PALM masked output
        ax.errorbar(wall_dists, mean_vars/palm_ref, yerr=err, 
                    label= r'PALM', 
                    fmt='o', c='darkmagenta')
        #plot wt_data
        for i,name in enumerate(namelist):
            wt_var1 = []
            wt_var2 = []        
            wt_z = []
            files = wt.get_files(path,name)            
            for file in files:
                wt_var1.append(time_series[name][file].weighted_component_mean[0]/(time_series[name][file].wtref))
                wt_var2.append(time_series[name][file].weighted_component_mean[1]/(time_series[name][file].wtref))
                wt_z.append(time_series[name][file].y)
            wt_z_plot = np.asarray(wt_z)-0.115*scale
            if var_name == 'u':
                wt_var_plot = wt_var1
                ax.errorbar(wt_z_plot, wt_var_plot, yerr = 0.025,
                            label=name, 
                            fmt=marker_list[i], color=c_list[i])
                if i==1:
                    ax.vlines(0.0066*150.*5., -0.5, 1.5, colors='tab:red', 
                            linestyles='dashed', 
                            label=r'$5 \cdot h_{r}$')
                ax.set_ylabel(r'$\overline{u}$ $u_{ref}^{-1}$ (-)', fontsize = 18)
            elif var_name == 'v':
                wt_var_plot = wt_var2                
                ax.errorbar(wt_z_plot, wt_var_plot, yerr = 0.025,
                            label=name, 
                            fmt=marker_list[i], color=c_list[i])
                if i==1:
                    ax.vlines(0.0066*150.*5., -0.05, 0.25, colors='tab:red', 
                            linestyles='dashed', 
                            label=r'$5 \cdot h_{r}$')
                ax.set_ylabel(r'$\overline{v}$ $u_{ref}^{-1}$ (-)', fontsize = 18)
                
        ax.grid(True, 'both', 'both')
        ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                    borderaxespad = 0., ncol = 2, 
                    numpoints = 1, fontsize = 18)
        ax.set_xlabel(r'$\Delta y$ (m)', fontsize = 18)
        # save plots
        ax.set_xscale('log')
        fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_mean_{}_mask_log.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:],
                    papy.globals.run_name, var_name), bbox_inches='tight', dpi=500)
        print('     SAVED TO: ' 
                + '../palm_results/{}/run_{}/maskprofiles/{}_mean_{}_mask_log.png'.format(papy.globals.run_name,
                papy.globals.run_number[-3:],
                papy.globals.run_name, var_name))                    
        plt.close(12)

######################################################
# compute variances alongside building
######################################################
if compute_front_var:
    var_name_list = ['u', 'v']
    print('\n     compute variances')

    for var_name in var_name_list:
        var_vars = np.array([])
        wall_dists = np.array([])
        for mask in mask_name_list:
            total_var = np.array([])
            total_time = np.array([])
            for run_no in papy.globals.run_numbers:
                nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask, run_no)
                # var_name = 'u'
                time, time_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'time')
                var, var_unit = papy.read_nc_var_ms(nc_file_path, nc_file, var_name)
                y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'y')
                total_time = np.concatenate([total_time, time])
                total_var = np.concatenate([total_var, var])
            # gather values
            var_variance = np.asarray([np.std(total_var)**2.])
            wall_dist = np.asarray([abs(y[0]-530.)])
            var_vars = np.concatenate([var_vars, var_variance])
            wall_dists = np.concatenate([wall_dists, wall_dist])

        #plot profiles
        err = np.mean(var_vars/palm_ref**2)*0.05
        fig, ax = plt.subplots()
        # plot PALM masked output
        ax.errorbar(wall_dists, var_vars/palm_ref**2., yerr=err, 
                    label= r'PALM', 
                    fmt='o', c='darkmagenta')
        #plot wt_data
        for i,name in enumerate(namelist):
            wt_var1 = []
            wt_var2 = []        
            wt_z = []
            files = wt.get_files(path,name)            
            for file in files:
                wt_var1.append(time_series[name][file].weighted_component_variance[0]/(time_series[name][file].wtref)**2.)
                wt_var2.append(time_series[name][file].weighted_component_variance[1]/(time_series[name][file].wtref)**2.)
                wt_z.append(time_series[name][file].y)
            wt_z_plot = np.asarray(wt_z)-0.115*scale
            if var_name == 'u':
                wt_var_plot = wt_var1
                ax.errorbar(wt_z_plot, wt_var_plot, yerr = 0.025/palm_ref**2.,
                            label=name, 
                            fmt=marker_list[i], color=c_list[i])
                if i==1:                            
                    # vertical line
                    ax.vlines(0.0066*150.*5., 0., 0.4, colors='tab:red', 
                            linestyles='dashed', 
                            label=r'$5 \cdot h_{r}$')
                ax.set_ylabel(r'$\overline{u^\prime u^\prime}$ $u_{ref}^{-2}$ (-)', fontsize = 18)
            elif var_name == 'v':
                wt_var_plot = wt_var2                
                ax.errorbar(wt_z_plot, wt_var_plot, yerr = 0.025/palm_ref**2.,
                            label=name, 
                            fmt=marker_list[i], color=c_list[i])
                if i==1:
                    # vertical line
                    ax.vlines(0.0066*150.*5., 0., 0.08, colors='tab:red', 
                            linestyles='dashed', 
                            label=r'$5 \cdot h_{r}$')
                ax.set_ylabel(r'$\overline{v^\prime v^\prime}$ $u_{ref}^{-2}$ (-)', fontsize = 18)
        ax.grid(True, 'both', 'both')
        ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                    borderaxespad = 0., ncol = 2, 
                    numpoints = 1, fontsize = 18)
        ax.set_xlabel(r'$\Delta y$ (m)', fontsize = 18)
        ax.set_xscale('log')
        fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_variance_{}_mask_log.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:],
                    papy.globals.run_name,var_name), bbox_inches='tight', dpi=500)
        print('     SAVED TO: ' 
                    + '../palm_results/{}/run_{}/maskprofiles/{}_variance_{}_mask_log.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:],
                    papy.globals.run_name,var_name))
        plt.close(12)


######################################################
# compute covariance in back of building
######################################################
if compute_front_covar:
    print('\n     compute co-variance')
    var_vars = np.array([])
    wall_dists = np.array([])
    for mask in mask_name_list:
        total_var1 = np.array([])
        total_var2 = np.array([])
        total_time = np.array([])
        for run_no in papy.globals.run_numbers:
            nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask, run_no)
            # var_name = 'u'
            time, time_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'time')
            var1, var1_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'u')
            var2, var2_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'v')
            y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'y')
            total_time = np.concatenate([total_time, time])
            total_var1 = np.concatenate([total_var1, var1])
            total_var2 = np.concatenate([total_var2, var2])            
        # gather values
        var1_fluc = np.asarray([np.mean(total_var1)]-total_var1)
        var2_fluc = np.asarray([np.mean(total_var2)]-total_var2)
        var_flux = np.asarray([np.mean(var1_fluc*var2_fluc)/palm_ref**2.])
        wall_dist = np.asarray([abs(y[0]-530.)])
        # wall_dist = np.asarray([abs(y[0])])
        var_vars = np.concatenate([var_vars, var_flux])
        wall_dists = np.concatenate([wall_dists, wall_dist])

    #plot profiles
    err = np.mean(var_vars)*0.1
    fig, ax = plt.subplots()
    # plot PALM masked output
    ax.errorbar(wall_dists, var_vars, yerr=err, 
                label= r'PALM', fmt='o', c='darkmagenta')
    # vertical line
    ax.vlines(0.0066*150.*5., -0.06, 0.01, colors='tab:red', 
                linestyles='dashed', 
                label=r'$5 \cdot h_{r}$')                
    # plot wind tunnel data
    for i,name in enumerate(namelist):
        files = wt.get_files(path,name)
        wt_flux = []   
        wt_z = []
        files = wt.get_files(path,name)            
        for file in files:
            wt_flux.append(wt.transit_time_weighted_flux(
                                    time_series[name][file].t_transit,
                                    time_series[name][file].u.dropna(),
                                    time_series[name][file].v.dropna())/time_series[name][file].wtref**2.)
            wt_z.append(time_series[name][file].y)
        wt_z_plot = np.asarray(wt_z)-0.115*scale
        ax.errorbar(wt_z_plot, wt_flux, yerr = 0.025/palm_ref**2.,
                    label=name, 
                    fmt=marker_list[i], color=c_list[i])
    ax.grid(True, 'both')
    ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                borderaxespad = 0., ncol = 2, 
                numpoints = 1, fontsize = 18)
    ax.set_xlabel(r'$\Delta y$ (m)', fontsize = 18)
    ax.set_ylabel(r'$\overline{u^\prime v^\prime}$ ' + r'(m$^2$ s$^{-2}$)', fontsize = 18)
    ax.set_xscale('log')
    fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_covariance_{}_mask_log.png'.format(papy.globals.run_name,
                papy.globals.run_number[-3:],
                papy.globals.run_name,'uw'), bbox_inches='tight', dpi=500)
    print('     SAVED TO: ' 
                + '../palm_results/{}/run_{}/maskprofiles/{}_covariance_{}_mask_log.png'.format(papy.globals.run_name,
                papy.globals.run_number[-3:],
                papy.globals.run_name,'uw'))
    plt.close(12)

##################
# Copmute spectra
##################
if compute_spectra:
    # heights mode
    print('\n Compute at different heights: \n')

    var_name_list = ['u', 'v']
    for var_name in var_name_list:
        mean_vars = np.array([])
        wall_dists = np.array([])
        for mask in mask_name_list:
            total_var = np.array([])
            total_time = np.array([])
            for run_no in papy.globals.run_numbers:
                nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask, run_no)
                time, time_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'time')
                var, var_unit = papy.read_nc_var_ms(nc_file_path, nc_file, var_name)
                y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'y')
                total_time = np.concatenate([total_time, time])
                total_var = np.concatenate([total_var, var])
            # gather values
            var_mean = np.asarray([np.mean(total_var)])
            wall_dist = np.asarray([abs(y[0]-530.)])
            mean_vars = np.concatenate([mean_vars, var_mean])
            wall_dists = np.concatenate([wall_dists, wall_dist])
            print('\n HEIGHT = {} m'.format(wall_dist))
            u_mean  = np.mean(total_var)      
            f_sm, S_uu_sm, u_aliasing = papy.calc_spectra(total_var, total_time, wall_dist, u_mean)
            print('    calculated spectra for {}'.format(var_name))
            papy.plot_spectra(f_sm, S_uu_sm, u_aliasing, u_mean, wall_dist, var_name, mask)
            print('    plotted spectra for {} \n'.format(var_name))


################
# Intergral length scale Lux
if compute_lux:
    nc_file = '{}_masked_M02{}.nc'.format(papy.globals.run_name,papy.globals.run_number)
    lux = np.zeros(len(height_list))
    var_name = 'u'

    for i,mask_name in enumerate(mask_name_list): 
        nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name,mask_name,papy.globals.run_number)
        height = height_list[i]
        time, time_unit = papy.read_nc_var_ms(nc_file_path,nc_file,'time')        
        var, var_unit = papy.read_nc_var_ms(nc_file_path,nc_file,var_name)        
        lux[i] = papy.calc_lux(np.abs(time[1]-time[0]),var)
        print('\n calculated integral length scale for {}'.format(str(height)))

    papy.plot_lux_profile(lux, height_list)
    print('\n plotted integral length scale profiles')

################
# Compute turbulence intensities
if compute_turbint_masked:
    nc_file = '{}_masked_M02{}.nc'.format(papy.globals.run_name, papy.globals.run_number)

    Iu = np.zeros(len(height_list))
    Iv = np.zeros(len(height_list))
    Iw = np.zeros(len(height_list))

    grid_name = 'zu'
    z, z_unit = papy.read_nc_grid(nc_file_path, nc_file_grid, grid_name)

    for i,mask_name in enumerate(mask_name_list): 
        nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask_name, papy.globals.run_number)
        height = height_list[i]
        
        var_u, var_unit_u = papy.read_nc_var_ms(nc_file_path, nc_file, 'u')
        var_v, var_unit_v = papy.read_nc_var_ms(nc_file_path, nc_file, 'v')
        var_w, var_unit_w = papy.read_nc_var_ms(nc_file_path, nc_file, 'w')

        turbint_dat = papy.calc_turbint(var_u, var_v, var_w)

        Iu[i] = turbint_dat[0]
        Iv[i] = turbint_dat[1]
        Iw[i] = turbint_dat[2]
        print('\n calculated turbulence intensities scale for {}'.format(str(height)))

    papy.plot_turbint_profile(Iu, height_list, 'u')
    print('\n plotted turbulence intensity profiles for u-component')

    papy.plot_turbint_profile(Iv, height_list, 'v')
    print('\n plotted turbulence intensity profiles for v-component')

    papy.plot_turbint_profile(Iw, height_list, 'w')
    print('\n plotted turbulence intensity profiles for w-component')

print('')
print('Finished processing of: {}{}'.format(papy.globals.run_name, papy.globals.run_number))