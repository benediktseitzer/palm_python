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
import scipy.stats as stats

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
papy.globals.run_number = '.046'
papy.globals.run_numbers = ['.007', '.008', '.009', '.010', '.011', '.012', 
                        '.013', '.014', '.015', '.016', '.017', '.018',
                        '.019', '.020', '.021', '.022', '.023', '.024',
                        '.025', '.026', '.027', '.028', '.029', '.030', 
                        '.031', '.032', '.033', '.034', '.035', '.036',
                        '.037', '.038', '.039', '.040', '.041', '.042',
                        '.043', '.044', '.045', '.046']
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
compute_front_pdfs = False
compute_front_highermoments = True
compute_front_var = True
compute_front_covar = True
compute_spectra = False
compute_front_lux = True

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
                        '.031', '.032', '.033', '.034', '.035', '.036',
                        '.037', '.038', '.039', '.040', '.041', '.042',
                        '.043', '.044', '.045', '.046', '.047']
palm_ref_file_path = '../palm/current_version/JOBS/{}/OUTPUT/'.format('SB_SI_BL')
total_palm_u = np.array([])
for run_no in palm_ref_run_numbers:
    palm_ref_file = '{}_masked_{}{}.nc'.format('SB_SI_BL', 'M10', run_no)
    palm_u, var_unit = papy.read_nc_var_ms(palm_ref_file_path, palm_ref_file, 'u')
    total_palm_u = np.concatenate([total_palm_u, palm_u])
data_nd = 1
if data_nd == 1:
    palm_ref = np.mean(total_palm_u)
else:
    palm_ref = 1.
print('     PALM REFERENCE VELOCITY: {} m/s \n'.format(palm_ref))
# wind tunnel data
namelist = ['SB_FL_SI_UV_023',
            'SB_BR_SI_UV_012',
            'SB_WB_SI_UV_013']
config = 'CO_REF'
path = '{}/coincidence/timeseries/'.format(wt_path) # path to timeseries folder
wtref_path = '{}/wtref/'.format(wt_path)
wtref_factor = 0.738
scale = wt_scale

wt_err = {}
wt_err.fromkeys(namelist)
for name in namelist:
    files = wt.get_files(path,name)
    var_names = ['umean', 'vmean', 'u_var', 'v_var', 'covar', 'lux']    
    wt_err[name] = {}
    wt_err[name].fromkeys(var_names)
    if name[3:5] == 'FL':
        wt_err[name]['umean'] = [0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 
                                0.0395, 0.0395, 0.0395, 0.0217, 0.0217, 0.0217, 0.0167, 0.0167, 0.0229, 0.0229, 0.0229, 0.0173]
        wt_err[name]['vmean'] = [0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 
                                0.0107, 0.0107, 0.0107, 0.0101, 0.0101, 0.0101, 0.0152, 0.0152, 0.0081, 0.0081, 0.0081, 0.008]
        wt_err[name]['u_var'] = [0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 
                                0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0039, 0.0039, 0.0047, 0.0047, 0.0047, 0.0006]
        wt_err[name]['v_var'] = [0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 
                                0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0039, 0.0039, 0.0047, 0.0047, 0.0047, 0.0006]
        wt_err[name]['covar'] = [0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 
                                0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0039, 0.0039, 0.0047, 0.0047, 0.0047, 0.0006]
        wt_err[name]['lux'] =   [3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 
                                3.1814, 3.1814, 3.1814, 1.5144, 1.5144, 1.5144, 2.9411, 2.9411, 2.2647, 2.2647, 2.2647, 26.5786]
    if name[3:5] == 'BR':
        wt_err[name]['umean'] = [0.0255, 0.0255, 0.0255, 0.0255, 0.0255, 0.0255, 0.0255, 0.0255, 0.0465, 0.0465, 
                                0.0465, 0.0292, 0.0292, 0.0179, 0.0179, 0.0179, 0.0202]
        wt_err[name]['vmean'] = [0.0156, 0.0156, 0.0156, 0.0156, 0.0156, 0.0156, 0.0156, 0.0156, 0.0116, 0.0116, 
                                0.0116, 0.0101, 0.0101, 0.0114, 0.0114, 0.0114, 0.0073]
        wt_err[name]['u_var'] = [0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 
                                0.0029, 0.0048, 0.0048, 0.0037, 0.0037, 0.0037, 0.0007]
        wt_err[name]['v_var'] = [0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 
                                0.0029, 0.0048, 0.0048, 0.0037, 0.0037, 0.0037, 0.0007]
        wt_err[name]['covar'] = [0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 
                                0.0029, 0.0048, 0.0048, 0.0037, 0.0037, 0.0037, 0.0007]
        wt_err[name]['lux'] =   [2.6852, 2.6852, 2.6852, 2.6852, 2.6852, 2.6852, 2.6852, 2.6852, 3.3587, 3.3587, 
                                3.3587, 1.9594, 1.9594, 4.7631, 4.7631, 4.7631, 22.5726]
    if name[3:5] == 'WB':
        wt_err[name]['umean'] = [0.0171, 0.0171, 0.0171, 0.0171, 0.0171, 0.0171, 0.0171, 0.0171, 0.0245, 0.0245, 
                                0.0245, 0.0335, 0.0335, 0.0175, 0.0175, 0.0175, 0.0202]
        wt_err[name]['vmean'] = [0.0133, 0.0133, 0.0133, 0.0133, 0.0133, 0.0133, 0.0133, 0.0133, 0.016, 0.016, 
                                0.016, 0.0106, 0.0106, 0.007, 0.007, 0.007, 0.0006]
        wt_err[name]['u_var'] = [0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0028, 0.0028, 
                                0.0028, 0.004, 0.004, 0.0029, 0.0029, 0.0029, 0.0008]
        wt_err[name]['v_var'] = [0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0028, 0.0028, 
                                0.0028, 0.004, 0.004, 0.0029, 0.0029, 0.0029, 0.0008]
        wt_err[name]['covar'] = [0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0028, 0.0028, 
                                0.0028, 0.004, 0.004, 0.0029, 0.0029, 0.0029, 0.0008]
        wt_err[name]['lux'] =   [1.9007, 1.9007, 1.9007, 1.9007, 1.9007, 1.9007, 1.9007, 1.9007, 2.2369, 2.2369, 
                                2.2369, 4.4863, 4.4863, 2.6004, 2.6004, 2.6004, 33.5205]





if compute_front_covar or compute_front_var or compute_front_mean or compute_front_lux or compute_front_highermoments:
    time_series = {}
    time_series.fromkeys(namelist)
    # Gather all files into Timeseries objects
    for name in namelist:
        print('READ        : {}'.format(name))
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
            ts_eq = ts
            ts_eq.calc_equidistant_timesteps()            
            ts.index = ts.t_arr         
            ts.weighted_component_mean
            ts.weighted_component_variance
            time_series[name][file] = ts
# plotting colors and markers
c_list = ['forestgreen', 'darkorange', 'navy', 'tab:red', 'tab:olive']
marker_list = ['^', 'o', 'd', 'x', '8']
label_list = ['flat facade', 'rough facade', 'medium rough facade', '{}']


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
                ax.errorbar(wt_z_plot, wt_var_plot, yerr = wt_err[name]['umean'],
                            label=label_list[i], 
                            fmt=marker_list[i], color=c_list[i])
                if i==1:
                    ax.vlines(0.0066*150.*5., -0.5, 1.5, colors='tab:red', 
                            linestyles='dashed', 
                            label=r'$5 \cdot h_{r}$')
                ax.set_ylabel(r'$\overline{u}$ $u_{ref}^{-1}$ (-)', fontsize = 18)
            elif var_name == 'v':
                wt_var_plot = wt_var2                
                ax.errorbar(wt_z_plot, wt_var_plot, yerr = wt_err[name]['vmean'],
                            label=label_list[i], 
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
                    'front', var_name), bbox_inches='tight', dpi=500)
        print('     SAVED TO: ' 
                + '../palm_results/{}/run_{}/maskprofiles/{}_mean_{}_mask_log.png'.format(papy.globals.run_name,
                papy.globals.run_number[-3:],
                'front', var_name))                    
        plt.close(12)


######################################################
# compute PDFs alongside the building
######################################################
if compute_front_pdfs:
    print('\n     compute PDF')    
    # velocity and variance PDFs
    var_name_list = ['u', 'v', 'w']
    for var_name in var_name_list:
        mean_vars = np.array([])
        wall_dists = np.array([])
        for mask in mask_name_list:
            total_var = np.array([])
            total_variance = np.array([])
            total_time = np.array([])
            for run_no in papy.globals.run_numbers:
                nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask, run_no)
                time, time_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'time')
                var, var_unit = papy.read_nc_var_ms(nc_file_path, nc_file, var_name)
                y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'y')
                total_time = np.concatenate([total_time, time])
                total_var = np.concatenate([total_var, var])
            # gather values
            var_mean = np.asarray([np.mean(total_var/palm_ref)])
            total_variance = (total_var-np.mean(total_var))**2.
            wall_dist = np.asarray([abs(y[0]-530.)])

            #plot PDF
            fig, ax = plt.subplots()
            # plot PALM masked output
            ax.hist(total_var/palm_ref, bins=100, density=True,
                    label=r'${}$ at $\Delta y={}$ m'.format(var_name, wall_dist[0]))
            if var_name == 'u':
                ax.vlines(var_mean, 0., 2., colors='tab:red', 
                            linestyles='dashed', 
                            label=r'$\overline{u}$' + r'$u_{ref}^{-1}$')
            elif var_name == 'v':
                ax.vlines(var_mean, 0., 2., colors='tab:red', 
                            linestyles='dashed', 
                            label=r'$\overline{v}$' + r'$u_{ref}^{-1}$')
            elif var_name == 'w':
                ax.vlines(var_mean, 0., 2., colors='tab:red', 
                            linestyles='dashed', 
                            label=r'$\overline{w}$' + r'$u_{ref}^{-1}$')
            ax.grid(True, 'both', 'both')
            ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                        borderaxespad = 0., ncol = 2, 
                        numpoints = 1, fontsize = 18)
            ax.set_xlabel(r'${}$'.format(var_name) + r'$u_{ref}^{-1}$ (-)', fontsize = 18)
            ax.set_ylabel(r'relative frequency', fontsize = 18)            
            if abs(min(total_var/palm_ref))<abs(max(total_var/palm_ref)):
                ax.set_xlim(-abs(max(total_var/palm_ref)), abs(max(total_var/palm_ref)))
            else:
                ax.set_xlim(-abs(min(total_var/palm_ref)), abs(min(total_var/palm_ref)))
            # save plots
            fig.savefig('../palm_results/{}/run_{}/histogram/{}_hist_{}_{}.png'.format(papy.globals.run_name,
                        papy.globals.run_number[-3:],
                        'front', var_name, mask), bbox_inches='tight', dpi=500)
            print('     SAVED TO: ' 
                    + '../palm_results/{}/run_{}/histogram/{}_hist_{}_{}.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:],
                    'front', var_name, mask))
            plt.close(12)
            fig, ax = plt.subplots()
            # plot PALM masked output
            ax.hist(total_variance/palm_ref**2., bins=100, density=True,
                    label=r'${}$ at $\Delta y={}$ m'.format(var_name, wall_dist[0]))
            if var_name == 'u':
                ax.vlines(np.mean(total_variance/palm_ref**2.), 0., 2., colors='tab:red', 
                            linestyles='dashed', 
                            label=r'$\overline{u}$ ' + r'$u_{ref}^{-2}$')
                ax.set_xlabel(r'$u^\prime u^\prime$ ' + r'$u_{ref}^{-2}$ (-)', fontsize = 18)
            elif var_name == 'v':
                ax.vlines(np.mean(total_variance/palm_ref**2.), 0., 2., colors='tab:red', 
                            linestyles='dashed', 
                            label=r'$\overline{v}$ ' + r'$u_{ref}^{-2}$')
                ax.set_xlabel(r'$v^\prime v^\prime$ ' + r'$u_{ref}^{-2}$ (-)', fontsize = 18)
            elif var_name == 'w':
                ax.vlines(np.mean(total_variance/palm_ref**2.), 0., 2., colors='tab:red', 
                            linestyles='dashed', 
                            label=r'$\overline{w}$ ' + r'$u_{ref}^{-2}$')
                ax.set_xlabel(r'$w^\prime w^\prime$ ' + r'$u_{ref}^{-2}$ (-)', fontsize = 18)
            ax.grid(True, 'both', 'both')
            ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                        borderaxespad = 0., ncol = 2, 
                        numpoints = 1, fontsize = 18)
            ax.set_ylabel(r'relative frequency', fontsize = 18)
            # save plots
            fig.savefig('../palm_results/{}/run_{}/histogram/{}_hist_{}{}_{}.png'.format(papy.globals.run_name,
                        papy.globals.run_number[-3:],
                        'front', var_name, var_name, mask), bbox_inches='tight', dpi=500)
            print('     SAVED TO: ' 
                    + '../palm_results/{}/run_{}/histogram/{}_hist_{}{}_{}.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:],
                    'front', var_name, var_name, mask))
            plt.close(13)
    # flux PDFs
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
        var_flux = np.asarray(var1_fluc*var2_fluc/palm_ref**2.)
        wall_dist = np.asarray([abs(y[0]-530.)])
        #plot PDF
        fig, ax = plt.subplots()
        # plot PALM masked output
        ax.hist(var_flux, bins=100, density=True,
                label=r'$u^\prime v^\prime$ at $\Delta y={}$ m'.format(wall_dist[0]))
        ax.vlines(np.mean(var_flux), 0., 2., colors='tab:red', 
                        linestyles='dashed', 
                        label=r'$\overline{u^\prime v^\prime}$ ' + r'$u_{ref}^{-2}$')
        ax.grid(True, 'both', 'both')
        ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                    borderaxespad = 0., ncol = 2, 
                    numpoints = 1, fontsize = 18)
        ax.set_xlabel(r'$u^\prime v^\prime$ ' + r'$u_{ref}^{-2}$ (-)', fontsize = 18)
        ax.set_ylabel(r'relative frequency', fontsize = 18)
        if abs(min(var_flux))<abs(max(var_flux)):        
            ax.set_xlim(-abs(max(var_flux)), abs(max(var_flux)))
        else:
            ax.set_xlim(-abs(min(var_flux)), abs(min(var_flux)))        
        # save plots
        fig.savefig('../palm_results/{}/run_{}/histogram/{}_hist_flux_{}.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:],
                    papy.globals.run_name, mask), bbox_inches='tight', dpi=500)
        print('     SAVED TO: ' 
                + '../palm_results/{}/run_{}/histogram/{}_hist_flux_{}.png'.format(papy.globals.run_name,
                papy.globals.run_number[-3:],
                papy.globals.run_name, mask))
        plt.close(13)        


######################################################
# compute skewness and kurtosis profiles
######################################################
if compute_front_highermoments:
    print('\n     compute higher statistical moments')    
    # velocity and variance PDFs
    var_name_list = ['u', 'v']
    for var_name in var_name_list:
        skew_vars = np.array([])
        kurt_vars = np.array([])
        wall_dists = np.array([])
        for mask in mask_name_list:
            total_var = np.array([])
            total_variance = np.array([])
            total_time = np.array([])
            for run_no in papy.globals.run_numbers:
                nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask, run_no)
                time, time_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'time')
                var, var_unit = papy.read_nc_var_ms(nc_file_path, nc_file, var_name)
                y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'y')
                total_time = np.concatenate([total_time, time])
                total_var = np.concatenate([total_var, var])
            # gather values
            var_mean = np.asarray([np.mean(total_var/palm_ref)])
            total_skew = np.asarray([stats.skew(total_var)])
            total_kurt = np.asarray([stats.kurtosis(total_var, fisher=False)])
            wall_dist = np.asarray([abs(y[0]-530.)])
            wall_dist = np.asarray([abs(y[0]-530.)])
            skew_vars = np.concatenate([skew_vars, total_skew])
            kurt_vars = np.concatenate([kurt_vars, total_kurt])            
            wall_dists = np.concatenate([wall_dists, wall_dist])

        #plot profiles
        err = 0.05
        fig, ax = plt.subplots()
        # plot PALM masked output
        ax.errorbar(wall_dists, skew_vars, yerr=err, 
                    label= r'PALM', 
                    fmt='o', c='darkmagenta')                        
        #plot wt_data
        for i,name in enumerate(namelist):
            wt_skew = []        
            wt_z = []
            files = wt.get_files(path,name)            
            for file in files:
                if var_name == 'u':
                    wt_skew.append(stats.skew(time_series[name][file].u.dropna()))
                elif var_name == 'v':
                    wt_skew.append(stats.skew(time_series[name][file].v.dropna()))
                wt_z.append(time_series[name][file].y)
            wt_z_plot = np.asarray(wt_z)-0.115*scale
            if var_name == 'u':
                ax.errorbar(wt_z_plot, wt_skew, yerr = 0.05,
                            label=label_list[i], 
                            fmt=marker_list[i], color=c_list[i])
                if i==1:
                    ax.vlines(0.0066*150.*5., -1.5, 1.5, colors='tab:red', 
                            linestyles='dashed', 
                            label=r'$5 \cdot h_{r}$')
                ax.set_ylabel(r'$\gamma_u$ (-)', fontsize = 18)
            elif var_name == 'v':             
                ax.errorbar(wt_z_plot, wt_skew, yerr = 0.05,
                            label=label_list[i], 
                            fmt=marker_list[i], color=c_list[i])
                if i==1:
                    ax.vlines(0.0066*150.*5., -0.8, 1., colors='tab:red', 
                            linestyles='dashed', 
                            label=r'$5 \cdot h_{r}$')
                ax.set_ylabel(r'$\gamma_v$ (-)', fontsize = 18)
        ax.grid(True, 'both', 'both')
        ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                    borderaxespad = 0., ncol = 2, 
                    numpoints = 1, fontsize = 18)
        ax.set_xlabel(r'$\Delta y$ (m)', fontsize = 18)
        # save plots
        ax.set_xscale('log')
        fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_skewness_{}_mask_log.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:],
                    'front', var_name), bbox_inches='tight', dpi=500)
        print('     SAVED TO: ' 
                + '../palm_results/{}/run_{}/maskprofiles/{}_skewness_{}_mask_log.png'.format(papy.globals.run_name,
                papy.globals.run_number[-3:],
                'front', var_name))                    
        plt.close(12)

        #plot profiles
        err = 0.1
        fig, ax = plt.subplots()
        # plot PALM masked output
        ax.errorbar(wall_dists, kurt_vars, yerr=err, 
                    label= r'PALM', 
                    fmt='o', c='darkmagenta')                        
        #plot wt_data
        for i,name in enumerate(namelist):
            wt_kurt = []
            wt_z = []
            files = wt.get_files(path,name)            
            for file in files:
                if var_name == 'u':
                    wt_kurt.append(stats.kurtosis(time_series[name][file].u.dropna(), fisher=False))
                elif var_name == 'v':
                    wt_kurt.append(stats.kurtosis(time_series[name][file].v.dropna(), fisher=False))
                wt_z.append(time_series[name][file].y)
            wt_z_plot = np.asarray(wt_z)-0.115*scale
            if var_name == 'u':
                ax.errorbar(wt_z_plot, wt_kurt, yerr = 0.1,
                            label=label_list[i], 
                            fmt=marker_list[i], color=c_list[i])
                if i==1:
                    ax.vlines(0.0066*150.*5., 1, 7, colors='tab:red', 
                            linestyles='dashed', 
                            label=r'$5 \cdot h_{r}$')
                ax.set_ylabel(r'$\beta_u$ (-)', fontsize = 18)
            elif var_name == 'v':             
                ax.errorbar(wt_z_plot, wt_kurt, yerr = 0.1,
                            label=label_list[i], 
                            fmt=marker_list[i], color=c_list[i])
                if i==1:
                    ax.vlines(0.0066*150.*5., 1, 7, colors='tab:red', 
                            linestyles='dashed', 
                            label=r'$5 \cdot h_{r}$')
                ax.set_ylabel(r'$\beta_v$ (-)', fontsize = 18)
        ax.grid(True, 'both', 'both')
        ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                    borderaxespad = 0., ncol = 2, 
                    numpoints = 1, fontsize = 18)
        ax.set_xlabel(r'$\Delta y$ (m)', fontsize = 18)
        # save plots
        ax.set_xscale('log')
        fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_kurtosis_{}_mask_log.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:],
                    'front', var_name), bbox_inches='tight', dpi=600)
        print('     SAVED TO: ' 
                + '../palm_results/{}/run_{}/maskprofiles/{}_kurtosis_{}_mask_log.png'.format(papy.globals.run_name,
                papy.globals.run_number[-3:],
                'front', var_name))                    
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
                ax.errorbar(wt_z_plot, wt_var_plot, yerr = wt_err[name]['u_var'],
                            label=label_list[i], 
                            fmt=marker_list[i], color=c_list[i])
                if i==1:                            
                    # vertical line
                    ax.vlines(0.0066*150.*5., 0., 0.35, colors='tab:red', 
                            linestyles='dashed', 
                            label=r'$5 \cdot h_{r}$')
                ax.set_ylabel(r'$\overline{u^\prime u^\prime}$ $u_{ref}^{-2}$ (-)', fontsize = 18)
            elif var_name == 'v':
                wt_var_plot = wt_var2                
                ax.errorbar(wt_z_plot, wt_var_plot, yerr = wt_err[name]['v_var'],
                            label=label_list[i], 
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
                    'front', var_name), bbox_inches='tight', dpi=500)
        print('     SAVED TO: ' 
                    + '../palm_results/{}/run_{}/maskprofiles/{}_variance_{}_mask_log.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:],
                    'front', var_name))
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
        ax.errorbar(wt_z_plot, wt_flux, yerr = wt_err[name]['covar'],
                    label=label_list[i], 
                    fmt=marker_list[i], color=c_list[i])
    ax.grid(True, 'both')
    ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                borderaxespad = 0., ncol = 2, 
                numpoints = 1, fontsize = 18)
    ax.set_xlabel(r'$\Delta y$ (m)', fontsize = 18)
    ax.set_ylabel(r'$\overline{u^\prime v^\prime} u_{ref}^{-2}$ ' + r'(-)', fontsize = 18)
    ax.set_xscale('log')
    fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_covariance_{}_mask_log.png'.format(papy.globals.run_name,
                papy.globals.run_number[-3:],
                'front', 'uv'), bbox_inches='tight', dpi=500)
    print('     SAVED TO: ' 
                + '../palm_results/{}/run_{}/maskprofiles/{}_covariance_{}_mask_log.png'.format(papy.globals.run_name,
                papy.globals.run_number[-3:],
                'front', 'uv'))
    plt.close(12)


######################################################
# Copmute spectra
######################################################
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


######################################################
# Intergral length scale Lux
######################################################  
if compute_front_lux:
    print('     compute Lux-profiles')
    lux = np.zeros(len(mask_name_list))
    var_name = 'u'
    wall_dists = np.array([])
    for i,mask in enumerate(mask_name_list):
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
        wall_dist = np.asarray([abs(y[0]-530.)])
        wall_dists = np.concatenate([wall_dists, wall_dist])
        lux[i] = papy.calc_lux(np.abs(total_time[1]-total_time[0]),total_var)
    print('    calculated palm-LUX for {}'.format(nc_file))
        
    # calculate wt-LUX
    wt_lux = {}
    wt_lux.fromkeys(namelist)
    wt_z = {}
    wt_z.fromkeys(namelist)
    for i,name in enumerate(namelist):
        files = wt.get_files(path,name)
        wt_lux[name] = []
        wt_z[name] = []
        for file in files:
            # equidistant timestepping
            dt = time_series[name][file].t_eq[1] - time_series[name][file].t_eq[0]
            wt_lux[name].append(papy.calc_lux(dt, time_series[name][file].u_eq.dropna().values))
            wt_z[name].append(time_series[name][file].y-0.115*scale)        
        # wt_z_plot = np.asarray(wt_z)-0.115*scale
        print('    calculated wt-LUX for {}'.format(name))

    #plot profiles
    err = lux*0.1
    fig, ax = plt.subplots()
    # plot PALM-LUX
    ax.errorbar(wall_dists, lux, yerr=err, 
                label= r'PALM', 
                fmt='o', c='darkmagenta')
    # plot wt-LUX
    for j,name in enumerate(namelist):
        err = np.asarray(wt_lux[name]) * 0.1
        ax.errorbar(np.asarray(wt_z[name]), np.asarray(wt_lux[name]), 
                yerr=wt_err[name]['lux'], label=label_list[j], 
                fmt=marker_list[j], color=c_list[j])
    ax.vlines(0.0066*150.*5., 0, 140, colors='tab:red', 
            linestyles='dashed', 
            label=r'$5 \cdot h_{r}$')

    ax.grid(True, 'both', 'both')
    ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                borderaxespad = 0., ncol = 2, 
                numpoints = 1, fontsize = 18)
    ax.set_ylabel(r'$L_{u}^x$ (m)', fontsize = 18)
    ax.set_xlabel(r'$\Delta y$ (m)', fontsize = 18)
    # save plots
    ax.set_xscale('log')
    fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_lux_{}_mask_log.png'.format(papy.globals.run_name,
                papy.globals.run_number[-3:],
                'front', var_name), bbox_inches='tight', dpi=500)
    print('     SAVED TO: ' 
            + '../palm_results/{}/run_{}/maskprofiles/{}_lux_{}_mask_log.png'.format(papy.globals.run_name,
            papy.globals.run_number[-3:],
            'front', var_name))
    ax.set_yscale('log')
    fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_lux_{}_mask_loglog.png'.format(papy.globals.run_name,
                papy.globals.run_number[-3:],
                'front', var_name), bbox_inches='tight', dpi=500)
    print('     SAVED TO: ' 
            + '../palm_results/{}/run_{}/maskprofiles/{}_lux_{}_mask_loglog.png'.format(papy.globals.run_name,
            papy.globals.run_number[-3:],
            'front', var_name))    

print('')
print('Finished processing of: {}{}'.format(papy.globals.run_name, papy.globals.run_number))