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
# papy.globals.run_name = 'SB_SI'
papy.globals.run_numbers = ['.007', '.008', '.009', '.010', '.011', '.012', 
                        '.013', '.014', '.015', '.016', '.017', '.018',
                        '.019', '.020', '.021', '.022', '.023', '.024',
                        '.025', '.026', '.027', '.028', '.029', '.030', 
                        '.031', '.032', '.033', '.034', '.035', '.036',
                        '.037', '.038', '.039', '.040', '.041', '.042',
                        '.043', '.044', '.045', '.046']
papy.globals.run_name = 'yshift_SB_SI'
papy.globals.run_numbers = ['.008', '.009', '.010', '.011', '.012', 
                            '.013', '.014', '.015', '.016', '.017', '.018',
                            '.019', '.020', '.021', '.022', '.023', '.024',
                            '.025', '.026', '.027', '.028', '.029', '.030',
                            '.031', '.032', '.033', '.034', '.035', '.036',
                            '.037']
papy.globals.run_number = papy.globals.run_numbers[-1]
print('Analyze PALM-run up to: ' + papy.globals.run_number)
nc_file_grid = '{}_pr{}.nc'.format(papy.globals.run_name,papy.globals.run_number)
nc_file_path = '../palm/current_version/JOBS/{}/OUTPUT/'.format(papy.globals.run_name)
if papy.globals.run_name == 'SB_SI_front':
    mask_name_list = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08',
                    'M09', 'M10', 'M11', 'M12']
elif papy.globals.run_name == 'SB_SI':
    mask_name_list = ['M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20',
                    'M21', 'M22', 'M23', 'M24']
elif papy.globals.run_name == 'yshift_SB_SI':
    mask_name_list = ['M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20',
                    'M21', 'M22', 'M23', 'M24']

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
compute_front_lux = True
compute_quadrant_analysis = True
################
"""
MAIN
"""
################

# prepare the outputfolders
papy.prepare_plotfolder(papy.globals.run_name,papy.globals.run_number)

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
        wt_err[name]['umean'] = 0.5 * np.asarray([0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 
                                0.0395, 0.0395, 0.0395, 0.0217, 0.0217, 0.0217, 0.0167, 0.0167, 0.0229, 0.0229, 0.0229, 0.0173])
        wt_err[name]['vmean'] = 0.5 * np.asarray([0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 
                                0.0107, 0.0107, 0.0107, 0.0101, 0.0101, 0.0101, 0.0152, 0.0152, 0.0081, 0.0081, 0.0081, 0.008])
        wt_err[name]['u_var'] = 0.5 * np.asarray([0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 
                                0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0039, 0.0039, 0.0047, 0.0047, 0.0047, 0.0006])
        wt_err[name]['v_var'] = 0.5 * np.asarray([0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 
                                0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0039, 0.0039, 0.0047, 0.0047, 0.0047, 0.0006])
        wt_err[name]['covar'] = 0.5 * np.asarray([0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 
                                0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0039, 0.0039, 0.0047, 0.0047, 0.0047, 0.0006])
        wt_err[name]['lux'] =   0.5 * np.asarray([3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 
                                3.1814, 3.1814, 3.1814, 1.5144, 1.5144, 1.5144, 2.9411, 2.9411, 2.2647, 2.2647, 2.2647, 26.5786])
    if name[3:5] == 'BR':
        wt_err[name]['umean'] = 0.5 * np.asarray([0.0255, 0.0255, 0.0255, 0.0255, 0.0255, 0.0255, 0.0255, 0.0255, 0.0465, 0.0465, 
                                0.0465, 0.0292, 0.0292, 0.0179, 0.0179, 0.0179, 0.0202])
        wt_err[name]['vmean'] = 0.5 * np.asarray([0.0156, 0.0156, 0.0156, 0.0156, 0.0156, 0.0156, 0.0156, 0.0156, 0.0116, 0.0116, 
                                0.0116, 0.0101, 0.0101, 0.0114, 0.0114, 0.0114, 0.0073])
        wt_err[name]['u_var'] = 0.5 * np.asarray([0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 
                                0.0029, 0.0048, 0.0048, 0.0037, 0.0037, 0.0037, 0.0007])
        wt_err[name]['v_var'] = 0.5 * np.asarray([0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 
                                0.0029, 0.0048, 0.0048, 0.0037, 0.0037, 0.0037, 0.0007])
        wt_err[name]['covar'] = 0.5 * np.asarray([0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 
                                0.0029, 0.0048, 0.0048, 0.0037, 0.0037, 0.0037, 0.0007])
        wt_err[name]['lux'] =   0.5 * np.asarray([2.6852, 2.6852, 2.6852, 2.6852, 2.6852, 2.6852, 2.6852, 2.6852, 3.3587, 3.3587, 
                                3.3587, 1.9594, 1.9594, 4.7631, 4.7631, 4.7631, 22.5726])
    if name[3:5] == 'WB':
        wt_err[name]['umean'] = 0.5 * np.asarray([0.0171, 0.0171, 0.0171, 0.0171, 0.0171, 0.0171, 0.0171, 0.0171, 0.0245, 0.0245, 
                                0.0245, 0.0335, 0.0335, 0.0175, 0.0175, 0.0175, 0.0202])
        wt_err[name]['vmean'] = 0.5 * np.asarray([0.0133, 0.0133, 0.0133, 0.0133, 0.0133, 0.0133, 0.0133, 0.0133, 0.016, 0.016, 
                                0.016, 0.0106, 0.0106, 0.007, 0.007, 0.007, 0.0006])
        wt_err[name]['u_var'] = 0.5 * np.asarray([0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0028, 0.0028, 
                                0.0028, 0.004, 0.004, 0.0029, 0.0029, 0.0029, 0.0008])
        wt_err[name]['v_var'] = 0.5 * np.asarray([0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0028, 0.0028, 
                                0.0028, 0.004, 0.004, 0.0029, 0.0029, 0.0029, 0.0008])
        wt_err[name]['covar'] = 0.5 * np.asarray([0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0028, 0.0028, 
                                0.0028, 0.004, 0.004, 0.0029, 0.0029, 0.0029, 0.0008])
        wt_err[name]['lux'] =   0.5 * np.asarray([1.9007, 1.9007, 1.9007, 1.9007, 1.9007, 1.9007, 1.9007, 1.9007, 2.2369, 2.2369, 
                                2.2369, 4.4863, 4.4863, 2.6004, 2.6004, 2.6004, 33.5205])

# get wind tunnel data
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
    err = 0.001
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
    # ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
    #             borderaxespad = 0., ncol = 2, 
    #             numpoints = 1, fontsize = 18)
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

    ax.grid(True)
    ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                borderaxespad = 0., ncol = 2, 
                numpoints = 1, fontsize = 18)
    ax.set_ylabel(r'$L_{u}^x$ (m)', fontsize = 18)
    ax.set_xlabel(r'$\Delta y$ (m)', fontsize = 18)
    # save plots
    ax.set_ylim(10.**-2., 150.)
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


######################################################
# Quadrant Analysis
######################################################  
if compute_quadrant_analysis:
    print('     compute Quadrant Analysis')
    wall_dists = np.array([])
    q1_fluxes = np.array([])
    q2_fluxes = np.array([])
    q3_fluxes = np.array([])
    q4_fluxes = np.array([])
    s1_all = np.array([])
    s2_all = np.array([])
    s3_all = np.array([])
    s4_all = np.array([])
    for i,mask in enumerate(mask_name_list):
        total_varu = np.array([])
        total_varv = np.array([])
        for run_no in papy.globals.run_numbers:
            nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask, run_no)
            varu, varu_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'u')
            varv, varv_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'v')
            y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'y')
            total_varu = np.concatenate([total_varu, varu/palm_ref])
            total_varv = np.concatenate([total_varv, varv/palm_ref])

        varu_fluc = np.asarray(np.mean(total_varu)-total_varu)
        varv_fluc = np.asarray(np.mean(total_varv)-total_varv)
        total_flux = np.asarray(varu_fluc * varv_fluc)

        q1_ind = np.where(np.logical_and(varu_fluc>0, varv_fluc>0))
        q2_ind = np.where(np.logical_and(varu_fluc<0, varv_fluc>0))
        q3_ind = np.where(np.logical_and(varu_fluc<0, varv_fluc<0))
        q4_ind = np.where(np.logical_and(varu_fluc>0, varv_fluc<0))

        q1_flux = np.asarray([np.mean(total_flux[q1_ind])])
        q2_flux = np.asarray([np.mean(total_flux[q2_ind])])
        q3_flux = np.asarray([np.mean(total_flux[q3_ind])])
        q4_flux = np.asarray([np.mean(total_flux[q4_ind])])

        q1_fluxes = np.concatenate([q1_fluxes, q1_flux])
        q2_fluxes = np.concatenate([q2_fluxes, q2_flux])
        q3_fluxes = np.concatenate([q3_fluxes, q3_flux])
        q4_fluxes = np.concatenate([q4_fluxes, q4_flux])

        wall_dist = np.asarray([abs(y[0])-530.])
        wall_dists = np.concatenate([wall_dists, wall_dist])

        s1 = np.asarray([q1_flux[0]/np.mean(total_flux) * len(q1_ind[0])/len(total_flux)])
        s2 = np.asarray([q2_flux[0]/np.mean(total_flux) * len(q2_ind[0])/len(total_flux)])
        s3 = np.asarray([q3_flux[0]/np.mean(total_flux) * len(q3_ind[0])/len(total_flux)])
        s4 = np.asarray([q4_flux[0]/np.mean(total_flux) * len(q4_ind[0])/len(total_flux)])

        s1_all = np.concatenate([s1_all, s1])
        s2_all = np.concatenate([s2_all, s2])
        s3_all = np.concatenate([s3_all, s3])
        s4_all = np.concatenate([s4_all, s4])

        print('\n S1 = {}'.format(str(s1[0])[:6]) + '   N1 = {}'.format(len(q1_ind[0])))
        print(' S2 = {}'.format(str(s2[0])[:6]) + '   N2 = {}'.format(len(q2_ind[0])))
        print(' S3 = {}'.format(str(s3[0])[:6]) + '   N3 = {}'.format(len(q3_ind[0])))
        print(' S4 = {}'.format(str(s4[0])[:6]) + '   N4 = {}'.format(len(q4_ind[0])))
        print(' Flux = {}'.format(str(np.mean(total_flux))[:6]) + '   N = {}'.format(len(total_flux)))        
        print(' SUM = {}'.format(str(s1[0] + 
                                    s2[0]  + 
                                    s3[0]  + 
                                    s4[0])))
        extent_val = 1.2
        plot_QA_PALM = False
        if plot_QA_PALM:
            # PLOT SINGLE Quadrant-scatterplots
            fig, ax = plt.subplots()
            fig.gca().set_aspect('equal', adjustable='box')
            ax.plot(varu_fluc[q1_ind], varv_fluc[q1_ind] ,'o', color='blue',
                    markersize=2,label='Q1')
            ax.plot(varu_fluc[q2_ind], varv_fluc[q2_ind] ,'o', color='darkorange',
                    markersize=2, label='Q2')
            ax.plot(varu_fluc[q3_ind], varv_fluc[q3_ind] ,'o', color='cyan',
                    markersize=2, label='Q3')
            ax.plot(varu_fluc[q4_ind], varv_fluc[q4_ind] ,'o', color='red',
                    markersize=2, label='Q4')
            ax.grid(True, 'both', 'both')
            ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                        borderaxespad = 0., ncol = 2, 
                        numpoints = 1, fontsize = 18)
            ax.set_xlabel(r'$u^\prime$ $u_{ref}^{-1}$ (-)', fontsize = 18)
            ax.set_ylabel(r'$v^\prime$ $u_{ref}^{-1}$ (-)', fontsize = 18)
            # save plots
            fig.savefig('../palm_results/{}/run_{}/quadrant_analysis/scatter/{}_QA_scatter_mask_{}.png'.format(papy.globals.run_name,
                        papy.globals.run_number[-3:],
                        'front', mask), bbox_inches='tight', dpi=500)
            print('     SAVED TO: ' 
                        + '../palm_results/{}/run_{}/quadrant_analysis/scatter/{}_QA_scatter_mask_{}.png'.format(papy.globals.run_name,
                        papy.globals.run_number[-3:],
                        'front', mask))
            plt.close()

            # PLOT JOINT PROBABILITY DENSITY FUNCTIONS
            umin = varu_fluc.min()
            umax = varu_fluc.max()
            vmin = varv_fluc.min()
            vmax = varv_fluc.max()
            u_jpdf, v_jpdf = np.mgrid[umin:umax:100j, vmin:vmax:100j]
            positions = np.vstack([u_jpdf.ravel(), v_jpdf.ravel()])
            values = np.vstack([varu_fluc, varv_fluc])
            kernel = stats.gaussian_kde(values)
            jpdf = np.reshape(kernel.evaluate(positions).T, u_jpdf.shape)        
            # plot
            fig, ax = plt.subplots()
            fig.gca().set_aspect('equal', adjustable='box')
            ax.set_xlim(-extent_val, extent_val)
            ax.set_ylim(-extent_val, extent_val)
            array = np.full((100,100), np.min(jpdf))
            try:
                im0 = ax.contourf(array, colors='lemonchiffon',
                                extent=[-extent_val, extent_val, -extent_val, extent_val], levels = 1)            
            except:
                print('too many levels')
            im1 = ax.contourf(jpdf.T, cmap='YlGnBu',
                    extent=[umin, umax, vmin, vmax], levels = 15)
            im2 = ax.contour(jpdf.T, extent=[umin, umax, vmin, vmax], levels = 15,
                    colors='gray')

            ax.vlines(0., -extent_val, extent_val, colors='darkgray', 
                    linestyles='dashed')
            ax.hlines(0., -extent_val, extent_val, colors='darkgray', 
                    linestyles='dashed')
            ax.grid(True, 'both', 'both')
            plt.colorbar(im1, label=r'$\rho (u^\prime_{q_i},  v^\prime_{q_i})$ (-)')
            ax.set_xlabel(r'$u^\prime$ $u_{ref}^{-1}$ (-)', fontsize = 18)
            ax.set_ylabel(r'$v^\prime$ $u_{ref}^{-1}$ (-)', fontsize = 18)
            ax.set_title(r'PALM - $\Delta y = {} m$'.format(wall_dist[0]))
            # save plots
            fig.savefig('../palm_results/{}/run_{}/quadrant_analysis/jpdf/{}_QA_jpdf_mask_{}.png'.format(papy.globals.run_name,
                        papy.globals.run_number[-3:],
                        'front', mask), bbox_inches='tight', dpi=500)
            print('     SAVED TO: ' 
                        + '../palm_results/{}/run_{}/quadrant_analysis/jpdf/{}_QA_jpdf_mask_{}.png'.format(papy.globals.run_name,
                        papy.globals.run_number[-3:],
                        'front', mask))
            plt.close()

    # plot wind tunnel data
    for i,name in enumerate(namelist):
        files = wt.get_files(path,name)
        wt_wall_dists = np.array([])
        wt_q1_fluxes = np.array([])
        wt_q2_fluxes = np.array([])
        wt_q3_fluxes = np.array([])
        wt_q4_fluxes = np.array([])
        wt_s1_all = np.array([])
        wt_s2_all = np.array([])
        wt_s3_all = np.array([])
        wt_s4_all = np.array([])
        files = wt.get_files(path,name)            
        for j,file in enumerate(files):
            wt_varu_fluc = (time_series[name][file].weighted_component_mean[0] - time_series[name][file].u.dropna().values)/time_series[name][file].wtref
            wt_varv_fluc = (time_series[name][file].weighted_component_mean[1] - time_series[name][file].v.dropna().values)/time_series[name][file].wtref
            wt_flux = np.asarray(wt_varu_fluc * wt_varv_fluc)

            wt_q1_ind = np.where(np.logical_and(wt_varu_fluc>0, wt_varv_fluc>0))
            wt_q2_ind = np.where(np.logical_and(wt_varu_fluc<0, wt_varv_fluc>0))
            wt_q3_ind = np.where(np.logical_and(wt_varu_fluc<0, wt_varv_fluc<0))
            wt_q4_ind = np.where(np.logical_and(wt_varu_fluc>0, wt_varv_fluc<0))

            wt_q1_flux = np.asarray([np.mean(wt_flux[wt_q1_ind])])
            wt_q2_flux = np.asarray([np.mean(wt_flux[wt_q2_ind])])
            wt_q3_flux = np.asarray([np.mean(wt_flux[wt_q3_ind])])
            wt_q4_flux = np.asarray([np.mean(wt_flux[wt_q4_ind])])

            wt_q1_fluxes = np.concatenate([wt_q1_fluxes, wt_q1_flux])
            wt_q2_fluxes = np.concatenate([wt_q2_fluxes, wt_q2_flux])
            wt_q3_fluxes = np.concatenate([wt_q3_fluxes, wt_q3_flux])
            wt_q4_fluxes = np.concatenate([wt_q4_fluxes, wt_q4_flux])

            wt_wall_dist = np.asarray([abs(time_series[name][file].y)-0.115*scale])
            wt_wall_dists = np.concatenate([wt_wall_dists, wt_wall_dist])

            wt_s1 = np.asarray([wt_q1_flux[0]/np.mean(wt_flux) * len(wt_q1_ind[0])/len(wt_flux)])
            wt_s2 = np.asarray([wt_q2_flux[0]/np.mean(wt_flux) * len(wt_q2_ind[0])/len(wt_flux)])
            wt_s3 = np.asarray([wt_q3_flux[0]/np.mean(wt_flux) * len(wt_q3_ind[0])/len(wt_flux)])
            wt_s4 = np.asarray([wt_q4_flux[0]/np.mean(wt_flux) * len(wt_q4_ind[0])/len(wt_flux)])

            wt_s1_all = np.concatenate([wt_s1_all, wt_s1])
            wt_s2_all = np.concatenate([wt_s2_all, wt_s2])
            wt_s3_all = np.concatenate([wt_s3_all, wt_s3])
            wt_s4_all = np.concatenate([wt_s4_all, wt_s4])

            print('\n S1 = {}'.format(str(wt_s1[0])[:6]) + '   N1 = {}'.format(len(wt_q1_ind[0])))
            print(' S2 = {}'.format(str(wt_s2[0])[:6]) + '   N2 = {}'.format(len(wt_q2_ind[0])))
            print(' S3 = {}'.format(str(wt_s3[0])[:6]) + '   N3 = {}'.format(len(wt_q3_ind[0])))
            print(' S4 = {}'.format(str(wt_s4[0])[:6]) + '   N4 = {}'.format(len(wt_q4_ind[0])))
            print(' Flux = {}'.format(str(np.mean(wt_flux))[:6]) + '   N = {}'.format(len(wt_flux)))        
            print(' SUM = {}'.format(str(wt_s1[0] + 
                                        wt_s2[0] + 
                                        wt_s3[0] + 
                                        wt_s4[0])))

            # PLOT SINGLE Quadrant-scatterplots
            plot_WT_QA = False
            if plot_WT_QA:
                fig, ax = plt.subplots()
                fig.gca().set_aspect('equal', adjustable='box')
                ax.plot(wt_varu_fluc[wt_q1_ind], wt_varv_fluc[wt_q1_ind] ,'o', color='blue',
                        markersize=2,label='Q1')
                ax.plot(wt_varu_fluc[wt_q2_ind], wt_varv_fluc[wt_q2_ind] ,'o', color='darkorange',
                        markersize=2, label='Q2')
                ax.plot(wt_varu_fluc[wt_q3_ind], wt_varv_fluc[wt_q3_ind] ,'o', color='cyan',
                        markersize=2, label='Q3')
                ax.plot(wt_varu_fluc[wt_q4_ind], wt_varv_fluc[wt_q4_ind] ,'o', color='red',
                        markersize=2, label='Q4')
                ax.grid(True, 'both', 'both')
                ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                            borderaxespad = 0., ncol = 2, 
                            numpoints = 1, fontsize = 18)
                ax.set_xlabel(r'$u^\prime$ $u_{ref}^{-1}$ (-)', fontsize = 18)
                ax.set_ylabel(r'$v^\prime$ $u_{ref}^{-1}$ (-)', fontsize = 18)
                # save plots
                fig.savefig('../palm_results/{}/run_{}/quadrant_analysis/scatter/{}_QA_scatter_WT_{}.png'.format(papy.globals.run_name,
                            papy.globals.run_number[-3:],
                            'front', file), bbox_inches='tight', dpi=500)
                print('     SAVED TO: ' 
                            + '../palm_results/{}/run_{}/quadrant_analysis/scatter/{}_QA_scatter_WT_{}.png'.format(papy.globals.run_name,
                            papy.globals.run_number[-3:],
                            'front', file))
                plt.close()

                # PLOT JOINT PROBABILITY DENSITY FUNCTIONS
                umin = wt_varu_fluc.min()
                umax = wt_varu_fluc.max()
                vmin = wt_varv_fluc.min()
                vmax = wt_varv_fluc.max()
                u_jpdf, v_jpdf = np.mgrid[umin:umax:100j, vmin:vmax:100j]
                positions = np.vstack([u_jpdf.ravel(), v_jpdf.ravel()])
                values = np.vstack([wt_varu_fluc, wt_varv_fluc])
                kernel = stats.gaussian_kde(values)
                jpdf = np.reshape(kernel.evaluate(positions).T, u_jpdf.shape)
                # plot
                fig, ax = plt.subplots()
                fig.gca().set_aspect('equal', adjustable='box')
                ax.set_xlim(-extent_val, extent_val)
                ax.set_ylim(-extent_val, extent_val)
                array = np.full((100,100), np.min(jpdf))
                try:
                    im0 = ax.contourf(array, colors='lemonchiffon',
                                    extent=[-extent_val, extent_val, -extent_val, extent_val], levels = 1)            
                except:
                    print('too many levels')                
                    im0 = ax.contourf(array, colors='lemonchiffon',
                                        extent=[-extent_val, extent_val, -extent_val, extent_val], levels = 2)                
                im1 = ax.contourf(jpdf.T, cmap='YlGnBu',
                        extent=[umin, umax, vmin, vmax], levels = 15)
                im2 = ax.contour(jpdf.T, extent=[umin, umax, vmin, vmax], levels = 15,
                        colors='gray')

                ax.vlines(0., -extent_val, extent_val, colors='darkgray', 
                        linestyles='dashed')
                ax.hlines(0., -extent_val, extent_val, colors='darkgray', 
                        linestyles='dashed')
                ax.grid(True, 'both', 'both')
                if name[3:5] == 'FL':
                    ax.set_title(r'Flat - $\Delta y = {} m$'.format(str(wt_wall_dist[0])[:5]))
                elif name[3:5] == 'WB':
                    ax.set_title(r'Medium Rough - $\Delta y = {} m$'.format(str(wt_wall_dist[0])[:5]))
                elif name[3:5] == 'BR':
                    ax.set_title(r'Rough - $\Delta y = {} m$'.format(str(wt_wall_dist[0])[:5]))
                else:
                    ax.set_title(r'Wind tunnel - $\Delta y = {} m$'.format(str(wt_wall_dist[0])[:5]))
                plt.colorbar(im1, 
                            label=r'$\rho (u^\prime_{q_i},  v^\prime_{q_i})$ (-)')
                ax.set_xlabel(r'$u^\prime$ $u_{ref}^{-1}$ (-)', fontsize = 18)
                ax.set_ylabel(r'$v^\prime$ $u_{ref}^{-1}$ (-)', fontsize = 18)
                # save plots
                fig.savefig('../palm_results/{}/run_{}/quadrant_analysis/jpdf/{}_QA_jpdf_WT_{}.png'.format(papy.globals.run_name,
                            papy.globals.run_number[-3:],
                            'front', file), bbox_inches='tight', dpi=500)
                print('     SAVED TO: ' 
                            + '../palm_results/{}/run_{}/quadrant_analysis/jpdf/{}_QA_jpdf_WT_{}.png'.format(papy.globals.run_name,
                            papy.globals.run_number[-3:],
                            'front', file))
                plt.close()

        # quadrant contributions
        fig, ax = plt.subplots()
        ax.errorbar(wt_wall_dists, wt_s1_all, yerr=0.1,
                label = 'Q1', fmt='o', c='blue')
        ax.errorbar(wt_wall_dists, wt_s2_all, yerr=0.1,
                label = 'Q2', fmt='o', c='darkorange')
        ax.errorbar(wt_wall_dists, wt_s3_all, yerr=0.1,
                label = 'Q3', fmt='o', c='cyan')
        ax.errorbar(wt_wall_dists, wt_s4_all, yerr=0.1,
                label = 'Q4', fmt='o', c='red')
        ax.vlines(0.0066*150.*5., -5., 5., colors='tab:red', 
                    linestyles='dashed', 
                    label=r'$5 \cdot h_{r}$')
        ax.hlines(0., 0.1, 100, colors='darkgray', 
                    linestyles='dashed')                
        ax.grid(True, 'both', 'both')
        ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                    borderaxespad = 0., ncol = 2, 
                    numpoints = 1, fontsize = 18)
        ax.set_ylim(-5., 5.)
        ax.set_ylabel(r'$\overline{u^\prime v^\prime_{q_i}}$ $\overline{u^\prime v^\prime}^{-1}$ (-)', fontsize = 18)
        ax.set_xlabel(r'$\Delta y$ (m)', fontsize = 18)
        ax.set_xscale('log')
        # save plots
        fig.savefig('../palm_results/{}/run_{}/quadrant_analysis/{}_quadrantcontribution_profile_{}.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:],
                    'front', name), bbox_inches='tight', dpi=500)
        print('     SAVED TO: ' 
                    + '../palm_results/{}/run_{}/quadrant_analysis/{}_quadrantcontribution_profile_{}.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:],
                    'front', name))

    # quadrant contributions
    fig, ax = plt.subplots()
    ax.errorbar(wall_dists, s1_all, yerr=0.1,
            label = 'Q1', fmt='o', c='blue')
    ax.errorbar(wall_dists, s2_all, yerr=0.1,
            label = 'Q2', fmt='o', c='darkorange')
    ax.errorbar(wall_dists, s3_all, yerr=0.1,
            label = 'Q3', fmt='o', c='cyan')
    ax.errorbar(wall_dists, s4_all, yerr=0.1,
            label = 'Q4', fmt='o', c='red')
    ax.vlines(0.0066*150.*5., -5., 5., colors='tab:red', 
                linestyles='dashed', 
                label=r'$5 \cdot h_{r}$')
    ax.hlines(0., 0.1, 100, colors='darkgray', 
                linestyles='dashed')                
    ax.grid(True, 'both', 'both')
    ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                borderaxespad = 0., ncol = 2, 
                numpoints = 1, fontsize = 18)
    ax.set_ylim(-5., 5.)
    ax.set_ylabel(r'$\overline{u^\prime v^\prime_{q_i}}$ $\overline{u^\prime v^\prime}^{-1}$ (-)', fontsize = 18)
    ax.set_xlabel(r'$\Delta y$ (m)', fontsize = 18)
    ax.set_xscale('log')
    # save plots
    fig.savefig('../palm_results/{}/run_{}/quadrant_analysis/{}_quadrantcontribution_profile_PALM.png'.format(papy.globals.run_name,
                papy.globals.run_number[-3:],
                'front'), bbox_inches='tight', dpi=500)
    print('     SAVED TO: ' 
                + '../palm_results/{}/run_{}/quadrant_analysis/{}_quadrantcontribution_profile_PALM.png'.format(papy.globals.run_name,
                papy.globals.run_number[-3:],
                'front'))


print('\n Finished processing of: {}{}'.format(papy.globals.run_name, papy.globals.run_number))