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
papy.globals.run_name = 'SB_SI_BL'
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
wt_filename = 'SB_BL_UV_001'
# experiment = 'balcony'
# wt_filename = 'BA_BL_UW_001'
wt_path = '../../Documents/phd/experiments/{}/{}'.format(experiment, wt_filename[3:5])
wt_file = '{}/coincidence/timeseries/{}.txt'.format(wt_path, wt_filename)
wt_file_pr = '{}/coincidence/mean/{}.000001.txt'.format(wt_path, wt_filename)
wt_file_ref = '{}/wtref/{}_wtref.txt'.format(wt_path, wt_filename)
wt_scale = 150.

# PHYSICS
papy.globals.z0 = 0.03
papy.globals.z0_wt = 0.071
papy.globals.alpha = 0.18
papy.globals.ka = 0.41
papy.globals.d0 = 0.
if papy.globals.run_name == 'single_building_ABL_2m':
    papy.globals.nx = 512
    papy.globals.ny = 512
    papy.globals.dx = 2.
elif papy.globals.run_name == 'single_building_ABL_4m':    
    papy.globals.nx = 256
    papy.globals.ny = 256
    papy.globals.dx = 4.
else:
    papy.globals.nx = 1024
    papy.globals.ny = 1024
    papy.globals.dx = 1.


# test-cases for spectral analysis testing
test_case_list = ['frequency_peak']
# spectra mode to run scrupt in
mode_list = ['testing', 'heights', 'compare', 'filtercheck'] 
mode = mode_list[1]

# Steeringflags
compute_BL_mean = False
compute_BL_var = False
compute_BL_covar = False
compute_spectra = True

compute_lux = False

################
"""
MAIN
"""
################

# prepare the outputfolders
papy.prepare_plotfolder(papy.globals.run_name,papy.globals.run_number)
plt.style.use('classic')

################
# compute BL mean in front of building
if compute_BL_mean:
    namelist = [wt_filename]
    path = '{}/coincidence/timeseries/'.format(wt_path) # path to timeseries folder
    wtref_path = '{}/wtref/'.format(wt_path)
    if wt_filename == 'SB_BL_UV_001':
        wtref_factor = 0.738
    elif wt_filename == 'BA_BL_UW_001':
        wtref_factor = 1.    
    scale = wt_scale
    data_nd = 1
    time_series = {}
    time_series.fromkeys(namelist)
    # Gather all files into Timeseries objects
    for name in namelist:
        files = wt.get_files(path,name)
        time_series[name] = {}
        time_series[name].fromkeys(files)
        wt_var1 = []
        wt_var2 = []        
        wt_z_SB = []
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
            wt_var1.append(time_series[name][file].weighted_component_mean[0])
            wt_var2.append(time_series[name][file].weighted_component_mean[1])
            wt_z_SB.append(time_series[name][file].z)

    experiment = 'balcony'
    wt_filename = 'BA_BL_UW_001'
    namelist = [wt_filename]
    wt_path = '../../Documents/phd/experiments/{}/{}'.format(experiment, wt_filename[3:5])
    wt_file = '{}/coincidence/timeseries/{}.txt'.format(wt_path, wt_filename)
    path = '{}/coincidence/timeseries/'.format(wt_path) # path to timeseries folder
    wtref_path = '{}/wtref/'.format(wt_path)
    if wt_filename == 'SB_BL_UV_001':
        wtref_factor = 0.738
    elif wt_filename == 'BA_BL_UW_001':
        wtref_factor = 1.    
    scale = wt_scale
    data_nd = 1
    time_series = {}
    time_series.fromkeys(namelist)
    # Gather all files into Timeseries objects
    for name in namelist:
        files = wt.get_files(path,name)
        time_series[name] = {}
        time_series[name].fromkeys(files)
        wt_var3 = []
        wt_z_BL = []
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
            wt_var3.append(time_series[name][file].weighted_component_mean[1])
            wt_z_BL.append(time_series[name][file].z)

    var_name_list = ['u', 'v', 'w']
    # GATHER masked PALM data
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
                if var_name == 'w':
                    y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'zw_3d')
                else: 
                    y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'zu_3d')
                total_time = np.concatenate([total_time, time])
                total_var = np.concatenate([total_var, var])
            # gather values
            var_mean = np.asarray([np.mean(total_var)])
            # wall_dist = np.asarray([abs(y[0]-530.)])
            wall_dist = np.asarray([abs(y[0])])
            mean_vars = np.concatenate([mean_vars, var_mean])
            wall_dists = np.concatenate([wall_dists, wall_dist])

        #plot profiles
        plot_wn_profiles = True
        if plot_wn_profiles:
            # read wind tunnel profile
            wt_pr, wt_u_ref, wt_z = papy.read_wt_ver_pr(wt_file_pr, wt_file_ref ,wt_scale)
            print('\n wind tunnel profile loaded \n') 
            # calculate theoretical wind profile
            u_pr, u_pw, u_fric = papy.calc_theoretical_profile(wt_pr, wt_u_ref, wt_z, wall_dists)

            err = 0.05*mean_vars
            fig, ax = plt.subplots()
            # plot PALM masked output
            ax.errorbar(mean_vars, wall_dists, xerr=err, 
                        label= r'PALM', fmt='o', c='darkmagenta', markersize=3)
            if var_name == 'u':
                # plot wind tunnel data
                ax.errorbar(wt_var1, wt_z_SB, xerr=0.035, 
                        label=r'wind tunnel', fmt='^', c='orangered')                        
                # plot theoretical profile                
                ax.plot(u_pr[1:], wall_dists[1:], label=r'fit: $z_0=({} \pm 0.003)$m'.format(papy.globals.z0_wt), 
                    color='darkorange', linestyle='--', linewidth = 2)
            elif var_name == 'v':
                # plot wind tunnel data                      
                ax.errorbar(wt_var2, wt_z_SB, xerr=0.04, 
                           label=r'wind tunnel', fmt='^', c='orangered')
            elif var_name == 'w':
                # plot wind tunnel data                      
                ax.errorbar(wt_var3, wt_z_BL, xerr=0.04,
                            label=r'wind tunnel', fmt='^', c='orangered')

            ax.set_ylim(0.,140.)
            ax.grid()
            ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                        borderaxespad = 0., ncol = 3, 
                        numpoints = 1, fontsize = 18)
            ax.set_xlabel(r'$\Delta y$ (m)', fontsize = 18)
            ax.set_ylabel(r'${}$ '.format(var_name) + r'(m s$^{-1}$)', fontsize = 18)
            
            # save plots
            fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_mean_{}_mask.png'.format(papy.globals.run_name,
                        papy.globals.run_number[-3:],
                        papy.globals.run_name,var_name), bbox_inches='tight', dpi=500)
            if var_name == 'u':
                ax.set_yscale('log')
                ax.set_ylim(0.1,140.)
                fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_mean_{}_mask_log.png'.format(papy.globals.run_name,
                            papy.globals.run_number[-3:],
                            papy.globals.run_name,var_name), bbox_inches='tight', dpi=500)            
            plt.close(12)

################
# compute BL var in front of building
if compute_BL_var:
    namelist = [wt_filename]
    path = '{}/coincidence/timeseries/'.format(wt_path) # path to timeseries folder
    wtref_path = '{}/wtref/'.format(wt_path)
    if wt_filename == 'SB_BL_UV_001':
        wtref_factor = 0.738
    elif wt_filename == 'BA_BL_UW_001':
        wtref_factor = 1.    
    scale = wt_scale
    data_nd = 1
    time_series = {}
    time_series.fromkeys(namelist)
    # Gather all files into Timeseries objects
    for name in namelist:
        files = wt.get_files(path,name)
        time_series[name] = {}
        time_series[name].fromkeys(files)
        wt_var1 = []
        wt_var2 = []        
        wt_z_SB = []
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
            wt_var1.append(time_series[name][file].weighted_component_variance[0])
            wt_var2.append(time_series[name][file].weighted_component_variance[1])
            wt_z_SB.append(time_series[name][file].z)

    experiment = 'balcony'
    wt_filename = 'BA_BL_UW_001'
    namelist = [wt_filename]
    wt_path = '../../Documents/phd/experiments/{}/{}'.format(experiment, wt_filename[3:5])
    wt_file = '{}/coincidence/timeseries/{}.txt'.format(wt_path, wt_filename)
    path = '{}/coincidence/timeseries/'.format(wt_path) # path to timeseries folder
    wtref_path = '{}/wtref/'.format(wt_path)
    if wt_filename == 'SB_BL_UV_001':
        wtref_factor = 0.738
    elif wt_filename == 'BA_BL_UW_001':
        wtref_factor = 1.    
    scale = wt_scale
    data_nd = 1
    time_series = {}
    time_series.fromkeys(namelist)
    # Gather all files into Timeseries objects
    for name in namelist:
        files = wt.get_files(path,name)
        time_series[name] = {}
        time_series[name].fromkeys(files)
        wt_var3 = []
        wt_z_BL = []
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
            wt_var3.append(time_series[name][file].weighted_component_variance[1])
            wt_z_BL.append(time_series[name][file].z)

    var_name_list = ['u', 'v', 'w']
    print('     compute variances')
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
                if var_name == 'w':
                    y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'zw_3d')
                else: 
                    y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'zu_3d')
                total_time = np.concatenate([total_time, time])
                total_var = np.concatenate([total_var, var])
            # gather values
            var_variance = np.asarray([np.std(total_var)**2.])
            # wall_dist = np.asarray([abs(y[0]-530.)])
            wall_dist = np.asarray([abs(y[0])])
            var_vars = np.concatenate([var_vars, var_variance])
            wall_dists = np.concatenate([wall_dists, wall_dist])
        ABL_file = 'single_building_ABL_1m_RE_z03_pr.015.nc'.format(papy.globals.run_name,papy.globals.run_number)
        ABL_file_path = '../palm/current_version/JOBS/single_building_ABL_1m_RE_z03/OUTPUT/'
        var_e, var_e_max, var_e_unit = papy.read_nc_var_ver_pr(ABL_file_path, ABL_file, 'e')
        z_e, z_unit_e = papy.read_nc_grid(ABL_file_path, ABL_file, 'ze')
        ABL_time, ABL_time_unit = papy.read_nc_time(ABL_file_path,ABL_file)
        time_show = time.nonzero()[0][0]

        #plot profiles
        plot_wn_profiles = True
        if plot_wn_profiles:
            err = np.mean(var_vars)*0.05
            fig, ax = plt.subplots()
            # plot PALM masked output
            ax.errorbar(var_vars, wall_dists, xerr=err, 
                        label= r'PALM', fmt='o', c='darkmagenta', markersize=3)
            ax.plot(1./3.*var_e[time_show,:-1], z_e[:-1],
                    label = r'PALM: $1/3$ $e_{SGS}$', 
                    color = 'plum',
                    linewidth = 2)
            
            #plot wt_data
            wt_z_plot = wt_z_SB[:5] +  wt_z_SB[7:]
            if var_name == 'u':
                wt_var_plot = wt_var1[:5] +  wt_var1[7:]
                ax.errorbar(wt_var_plot, wt_z_plot, xerr = 0.025,
                            label='wind tunnel', 
                            fmt='^', 
                            c='orangered')
            elif var_name == 'v':
                wt_var_plot = wt_var2[:5] +  wt_var2[7:]
                ax.errorbar(wt_var_plot, wt_z_plot, xerr = 0.025,
                            label='wind tunnel', 
                            fmt='^', 
                            c='orangered')
            elif var_name == 'w':
                wt_z_plot = wt_z_BL[:5] +  wt_z_BL[7:]
                wt_var_plot = wt_var3[:5] +  wt_var3[7:]
                ax.errorbar(wt_var_plot, wt_z_plot, xerr = 0.025,
                            label='wind tunnel', 
                            fmt='^', 
                            c='orangered')
            ax.set_ylim(0.,140.)
            ax.grid()
            ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                        borderaxespad = 0., ncol = 3, 
                        numpoints = 1, fontsize = 18)
            ax.set_xlabel(r'$\Delta y$ (m)', fontsize = 18)
            ax.set_ylabel(r'${}^\prime {}^\prime$ '.format(var_name, var_name) + r'(m$^2$ s$^{-2}$)', fontsize = 18)
            # fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_variance_{}_mask.png'.format(papy.globals.run_name,
            #             papy.globals.run_number[-3:],
            #             papy.globals.run_name,var_name), bbox_inches='tight', dpi=500)
            ax.set_yscale('log')
            ax.set_ylim(0.1,140.)
            fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_variance_{}_mask_log.png'.format(papy.globals.run_name,
                        papy.globals.run_number[-3:],
                        papy.globals.run_name,var_name), bbox_inches='tight', dpi=500)
            plt.close(12)
            print('         plotted variance of {}'.format(var_name))

################
# compute BL var in front of building
if compute_BL_covar:
    experiment = 'balcony'
    wt_filename = 'BA_BL_UW_001'
    namelist = [wt_filename]
    wt_path = '../../Documents/phd/experiments/{}/{}'.format(experiment, wt_filename[3:5])
    wt_file = '{}/coincidence/timeseries/{}.txt'.format(wt_path, wt_filename)
    path = '{}/coincidence/timeseries/'.format(wt_path) # path to timeseries folder
    wtref_path = '{}/wtref/'.format(wt_path)
    if wt_filename == 'SB_BL_UV_001':
        wtref_factor = 0.738
    elif wt_filename == 'BA_BL_UW_001':
        wtref_factor = 1.    
    scale = wt_scale
    data_nd = 1
    time_series = {}
    time_series.fromkeys(namelist)
    # Gather all files into Timeseries objects
    for name in namelist:
        files = wt.get_files(path,name)
        time_series[name] = {}
        time_series[name].fromkeys(files)
        wt_flux = []
        wt_z = []
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
            wt_flux.append(wt.transit_time_weighted_flux(
                                    time_series[name][file].t_transit,
                                    time_series[name][file].u.dropna(),
                                    time_series[name][file].v.dropna()))
            wt_z.append(time_series[name][file].z)

    var_name_list = ['u', 'v', 'w']
    print('     compute co-variance')
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
            var2, var2_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'w')            
            y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'zw_3d')
            y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'zu_3d')
            total_time = np.concatenate([total_time, time])
            total_var1 = np.concatenate([total_var1, var1])
            total_var2 = np.concatenate([total_var2, var2])            
        # gather values
        var1_fluc = np.asarray([np.mean(total_var1)]-total_var1)
        var2_fluc = np.asarray([np.mean(total_var2)]-total_var2)
        var_flux = np.asarray([np.mean(var1_fluc*var2_fluc)])
        # wall_dist = np.asarray([abs(y[0]-530.)])
        wall_dist = np.asarray([abs(y[0])])
        var_vars = np.concatenate([var_vars, var_flux])
        wall_dists = np.concatenate([wall_dists, wall_dist])

    ABL_file = 'single_building_ABL_1m_RE_z03_pr.015.nc'.format(papy.globals.run_name,papy.globals.run_number)
    ABL_file_path = '../palm/current_version/JOBS/single_building_ABL_1m_RE_z03/OUTPUT/'
    var_e, var_e_max, var_e_unit = papy.read_nc_var_ver_pr(ABL_file_path, ABL_file, 'w"u"')
    z_e, z_unit_e = papy.read_nc_grid(ABL_file_path, ABL_file, 'zw*u*')
    ABL_time, ABL_time_unit = papy.read_nc_time(ABL_file_path,ABL_file)
    time_show = time.nonzero()[0][0]


    #plot profiles
    plot_wn_profiles = True
    if plot_wn_profiles:
        err = np.mean(var_vars)*0.05
        fig, ax = plt.subplots()
        # plot PALM masked output
        ax.errorbar(var_vars, wall_dists, xerr=err, 
                    label= r'PALM', fmt='o', c='darkmagenta', markersize=3)
        # plot SGS-Fluxes
        ax.plot(var_e[time_show,:-1], z_e[:-1],
                label = r'$\overline{u^\prime w^\prime}_{SGS}$', 
                color = 'plum',
                linewidth = 2)
        # plot wt_data
        ax.errorbar(wt_flux, wt_z, xerr = 0.005,
                    label='wind tunnel',
                    fmt='^', 
                    c='orangered')
        ax.set_ylim(0.,140.)
        ax.grid()
        ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                    borderaxespad = 0., ncol = 3, 
                    numpoints = 1, fontsize = 18)
        ax.set_xlabel(r'$\Delta y$ (m)', fontsize = 18)
        ax.set_ylabel(r'$\overline{u^\prime w^\prime}$ ' + r'(m$^2$ s$^{-2}$)', fontsize = 18)
        ax.set_yscale('log')
        ax.set_ylim(0.1,140.)
        fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_covariance_{}_mask_log.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:],
                    papy.globals.run_name,'uw'), bbox_inches='tight', dpi=500)
        plt.close(12)

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
# Copmute spectra
if compute_spectra:
    print('\n Compute at different heights: \n')
    var_name_list = ['u', 'v', 'w']
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
                if var_name == 'w':
                    y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'zw_3d')
                else:
                    y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'zu_3d')
                total_time = np.concatenate([total_time, time])
                total_var = np.concatenate([total_var, var])
            # gather values
            var_mean = np.asarray([np.mean(total_var)])
            wall_dist = np.asarray([abs(y[0])])
            print('\n HEIGHT = {} m'.format(wall_dist))
            # # equidistant timestepping
            time_eq = np.linspace(total_time[0], total_time[-1], len(total_time))
            var_eq = wt.equ_dist_ts(total_time, time_eq, total_var)
            if var_name == 'u':
                u_mean = var_mean[0]
            # f_sm, S_uu_sm, u_aliasing = papy.calc_spectra(total_var, total_time, wall_dist, u_mean)
            f_sm, S_uu_sm, u_aliasing = papy.calc_spectra(var_eq, time_eq, wall_dist, u_mean)
            print('    calculated spectra for {}'.format(var_name))
            papy.plot_spectra(f_sm, S_uu_sm, u_aliasing, u_mean, wall_dist[0], var_name, mask)
            print('    plotted spectra for {} \n'.format(var_name))

print('')
print('Finished processing of: {}{}'.format(papy.globals.run_name, papy.globals.run_number))