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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import palm_py as papy

sys.path.append('/home/bene/Documents/phd/windtunnel_py/windtunnel/')    
import windtunnel as wt

import warnings
warnings.simplefilter("ignore")

# plotformat = 'pgf'
plotformat = 'png'
# plotformat = 'pdf'
if plotformat == 'pgf':
    plt.style.use('default')
    matplotlib.use('pgf')
    matplotlib.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        'font.family': 'sans-serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'mathtext.fontset': 'cm',        
        'xtick.labelsize' : 11,
        'ytick.labelsize' : 11,
        'legend.fontsize' : 11,
        'lines.linewidth' : 0.75,
        'lines.markersize' : 2.,
        'figure.dpi' : 300,
    })
    print('Textwidth in inch = ' + str(426/72.27))
    # 5.89 inch
    textwidth = 5
    textwidth_half = 0.5*textwidth
else:
    plt.style.use('default')
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'text.usetex': True,
        'mathtext.fontset': 'cm',
        'xtick.labelsize' : 11,
        'ytick.labelsize' : 11,
        'legend.fontsize' : 11,
        'lines.linewidth' : 0.75,
        'lines.markersize' : 2.,
        'figure.dpi' : 300,
    })
    print('Textwidth in inch = ' + str(426/72.27))
    # 5.89 inch
    textwidth = 5
    textwidth_half = 0.5*textwidth
    

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
# papy.globals.run_name = 'yshift_SB_BL_corr'
papy.globals.run_numbers = ['.008', '.009', '.010', '.011', '.012', 
                            '.013', '.014', '.015', '.016', '.017', '.018',
                            '.019', '.020', '.021', '.022', '.023', '.024',
                            '.025', '.026', '.027', '.028', '.029', '.030',
                            '.031', '.032', '.033', '.034', '.035', '.036',
                            '.037', '.038', '.039', '.040', '.041', '.042',
                            '.043', '.044', '.045', '.046', '.047']
papy.globals.run_number = papy.globals.run_numbers[-1]
print('Analyze PALM-run up to: ' + papy.globals.run_number)
nc_file_grid = '{}_pr{}.nc'.format(papy.globals.run_name,papy.globals.run_number)
nc_file_path = '../palm/current_version/JOBS/{}/OUTPUT/'.format(papy.globals.run_name)
mask_name_list = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08',
                    'M09', 'M10', 'M11', 'M12']
file_type = 'png'
file_type = plotformat

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
papy.globals.z0_wt = 0.084
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
data_nd = 0
if data_nd == 0:
    palm_ref = np.mean(total_palm_u)
else:
    palm_ref = 1.
print('     PALM REFERENCE VELOCITY: {} m/s \n'.format(palm_ref))


# test-cases for spectral analysis testing
test_case_list = ['frequency_peak']
# spectra mode to run scrupt in
mode_list = ['testing', 'heights', 'compare', 'filtercheck'] 
mode = mode_list[1]

#set data_nd to 1 if using non-dimensional data
if data_nd == 1:
    u_err = 0.009552720502061733
    v_err = 0.0011673986655155527
    flux_err = 0.0005127975028476365
    I_u_err = 0.004383914423607249
    I_v_err = 0.002449995506950048
    lux_err = 8.298036937278763

    PALM_u_err = 0.015858/2.
    PALM_v_err = 0.004881333/2.
    PALM_w_err = 0.006863333/2.
    PALM_flux_err = 0.000780667/2.
    PALM_I_u_err = 0.003302333*palm_ref/2.
    PALM_I_v_err = 0.001813333*palm_ref/2.
    PALM_I_w_err = 0.0034*palm_ref/2.    
    PALM_lux_err = 8.298036937278763/2.

elif data_nd == 0:
    u_err = 0.009552720502061733
    v_err = 0.0011673986655155527
    flux_err = 0.0005127975028476365
    I_u_err = 0.004383914423607249
    I_v_err = 0.002449995506950048
    lux_err = 8.298036937278763

    PALM_u_err = 0.015858/2.
    PALM_v_err = 0.004881333/2.
    PALM_w_err = 0.006863333/2.
    PALM_flux_err = 0.000780667/2.
    PALM_I_u_err = 0.003302333*palm_ref/2.
    PALM_I_v_err = 0.001813333*palm_ref/2.
    PALM_I_w_err = 0.0034*palm_ref/2.    
    PALM_lux_err = 8.298036937278763/2.


# Steeringflags
compute_BL_mean = True
compute_BL_var = True
compute_BL_covar = True
compute_BL_turbint = True
compute_BL_lux = False
compute_quadrant_analysis = False

compute_BL_correlation = False
################
"""
MAIN
"""
################

# prepare the outputfolders
papy.prepare_plotfolder(papy.globals.run_name,papy.globals.run_number)

namelist = [wt_filename]
path = '{}/coincidence/timeseries/'.format(wt_path) # path to timeseries folder
wtref_path = '{}/wtref/'.format(wt_path)
if wt_filename == 'SB_BL_UV_001':
    wtref_factor = 0.738
elif wt_filename == 'BA_BL_UW_001':
    wtref_factor = 0.731 
    # wtref_factor = 1.
scale = wt_scale
# data_nd = 1
# if not compute_BL_correlation:
#     time_series = {}
#     time_series.fromkeys(namelist)
#     # Gather all files into Timeseries objects
#     for name in namelist:
#         files = wt.get_files(path,name)
#         time_series[name] = {}
#         time_series[name].fromkeys(files)
#         wt_var1 = []
#         wt_var2 = []        
#         wt_z_SB = []
#         for i,file in enumerate(files):
#             ts = wt.Timeseries.from_file(path+file)            
#             ts.get_wind_comps(path+file)
#             ts.get_wtref(wtref_path,name,index=i)
#             ts.wtref = ts.wtref*wtref_factor
#             # edit 6/20/19: Assume that input data is dimensional, not non-dimensional
#             if data_nd == 0:
#                 print('Warning: Assuming that data is dimensional. If using non-dimensional input data, set variable data_nd to 1')
#                 ts.nondimensionalise()
#             else:
#                 if data_nd == 1:
#                     []
#                 else:
#                     print('Warning: data_nd can only be 1 (for non-dimensional input data) or 0 (for dimensional input data)')        
#             #edit 06/20/19: added seperate functionto  calculate equidistant timesteps             
#             ts.adapt_scale(scale)         
#             ts.mask_outliers()
#             ts.index = ts.t_arr         
#             ts.weighted_component_mean
#             ts.weighted_component_variance
#             time_series[name][file] = ts

######################################################
# compute BL mean in front of building
######################################################
if compute_BL_mean:
    namelist = [wt_filename]
    path = '{}/coincidence/timeseries/'.format(wt_path) # path to timeseries folder
    wtref_path = '{}/wtref/'.format(wt_path)
    if wt_filename == 'SB_BL_UV_001':
        wtref_factor = 0.738
    elif wt_filename == 'BA_BL_UW_001':
        wtref_factor = 0.731 
    scale = wt_scale
    # data_nd = 1
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
                # ts.nondimensionalise()
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
    wt_ref_dcf = time_series['SB_BL_UV_001']['SB_BL_UV_001.000016.txt'].wtref
    print('\n     windtunnelref = ' + str(wt_ref_dcf))
    wt_ref_dcf = time_series['SB_BL_UV_001']['SB_BL_UV_001.000016.txt'].weighted_component_mean[0]
    print('\n     windtunnelref = ' + str(wt_ref_dcf))    
    for name in namelist:
        files = wt.get_files(path,name)
        for file in files:    
            time_series[name][file].u = time_series[name][file].u/time_series[name][file].wtref
            time_series[name][file].v = time_series[name][file].v/time_series[name][file].wtref
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
        wtref_factor = 0.731  
    scale = wt_scale
    # data_nd = 1
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
                # ts.nondimensionalise()
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
    for name in namelist:
        files = wt.get_files(path,name)
        for file in files:    
            time_series[name][file].u = time_series[name][file].u/time_series[name][file].wtref
            time_series[name][file].v = time_series[name][file].v/time_series[name][file].wtref
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
                total_var = np.concatenate([total_var, var/palm_ref])
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
            print('\n wind tunnel profile loaded: {}'.format(var_name)) 
            # calculate theoretical wind profile
            sfc_height = 45
            sfc_layer_l = np.where(np.asarray(wt_z_SB) >= 7.*1.5)
            sfc_layer_u = np.where(np.asarray(wt_z_SB) <= sfc_height)
            sfc_layer = np.intersect1d(sfc_layer_l,sfc_layer_u)
            u_pr, u_pw, u_fric = papy.calc_theoretical_profile(wt_pr, wt_u_ref, wt_z, wall_dists)
            z0, z0_err, z1 = wt.calc_z0(wt_var1, wt_z_SB, 
                                sfc_height=sfc_height, sfc_layer=sfc_layer)
            print('     z0 = ', z0.round(3), '+-', z0_err.round(3))
            z0_x, z0_y = zip(*sorted(zip(wt_var1, z1)))

            err = 0.01
            fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))

            # plot PALM masked output
            if var_name == 'u':
                ax.errorbar(mean_vars, wall_dists, xerr=PALM_u_err, 
                        label= r'PALM', fmt='o', c='darkmagenta')
                # plot wind tunnel data
                ax.errorbar(wt_var1, wt_z_SB, xerr=u_err, 
                        label=r'reference profile', fmt='^', c='orangered')
                # plot theoretical profile
                ax.plot(np.asarray(z0_x), np.asarray(z0_y), 
                            color='darkorange', 
                            linestyle='--',
                            label=r'fit: $z_0=({} \pm 0.002)$m'.format(papy.globals.z0_wt))
                ax.hlines(45., 0.5 ,1.2, 
                            linestyle='--', 
                            colors = 'cornflowerblue', 
                            label = r'$\delta_{cf}$')
                ax.set_xlim(0.5,1.2)
            elif var_name == 'v':
                ax.errorbar(mean_vars, wall_dists, xerr=PALM_v_err, 
                        label= r'PALM', fmt='o', c='darkmagenta')
                # plot wind tunnel data
                ax.errorbar(wt_var2, wt_z_SB, xerr=v_err, 
                           label=r'reference profile', fmt='^', c='orangered')
                ax.hlines(45., -0.015, 0.025, 
                            linestyle='--', 
                            colors = 'cornflowerblue', 
                            label = r'$\delta_{cf}$')
                ax.set_xlim(-0.015, 0.025)
            elif var_name == 'w':
                ax.errorbar(mean_vars, wall_dists, xerr=PALM_v_err, 
                        label= r'PALM', fmt='o', c='darkmagenta')
                # plot wind tunnel data
                ax.errorbar(wt_var3, wt_z_BL, xerr=v_err,
                            label=r'reference profile', fmt='^', c='orangered')
                ax.hlines(45., -0.1 ,0., 
                                linestyle='--', 
                                colors = 'cornflowerblue', 
                                label = r'$\delta_{cf}$')
                ax.set_xlim(-0.1 ,0.)
            ax.set_ylim(4.,140.)
            # ax.grid()
            ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                        borderaxespad = 0., 
                        numpoints = 1)
            ax.set_ylabel(r'$z$ (m)')
            ax.set_xlabel(r'$\overline {}$ '.format(var_name) + r' $u_{ref}^{-1}$ (-)')
            
            # save plots
            fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_mean_{}_mask.{}'.format(papy.globals.run_name,
                        papy.globals.run_number[-3:], 'BL',var_name, file_type), bbox_inches='tight', dpi=300)
            ax.set_yscale('log')
            ax.set_ylim(4.,140.)
            fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_mean_{}_mask_log.{}'.format(papy.globals.run_name,
                        papy.globals.run_number[-3:], 'BL',var_name, file_type), bbox_inches='tight', dpi=300)

            plt.close(12)


######################################################
# compute BL var in front of building
######################################################
if compute_BL_var:
    experiment = 'single_building'
    wt_filename = 'SB_BL_UV_001'    
    namelist = [wt_filename]
    wt_path = '../../Documents/phd/experiments/{}/{}'.format(experiment, wt_filename[3:5])
    wt_file = '{}/coincidence/timeseries/{}.txt'.format(wt_path, wt_filename)
    path = '{}/coincidence/timeseries/'.format(wt_path) # path to timeseries folder
    wtref_path = '{}/wtref/'.format(wt_path)
    if wt_filename == 'SB_BL_UV_001':
        wtref_factor = 0.738
    elif wt_filename == 'BA_BL_UW_001':
        wtref_factor = 0.731    
    scale = wt_scale
    # data_nd = 1
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
                # ts.nondimensionalise()
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
    for name in namelist:
        files = wt.get_files(path,name)
        for file in files:    
            time_series[name][file].u = time_series[name][file].u/time_series[name][file].wtref
            time_series[name][file].v = time_series[name][file].v/time_series[name][file].wtref
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
        wtref_factor = 0.731    
    scale = wt_scale
    # data_nd = 1
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
                # ts.nondimensionalise()
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
    for name in namelist:
        files = wt.get_files(path,name)
        for file in files:    
            time_series[name][file].u = time_series[name][file].u/time_series[name][file].wtref
            time_series[name][file].v = time_series[name][file].v/time_series[name][file].wtref
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
                total_var = np.concatenate([total_var, var/palm_ref])
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
            fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
            # plot PALM masked output
            ax.errorbar(var_vars, wall_dists, xerr=PALM_flux_err, 
                        label= r'PALM', fmt='o', c='darkmagenta')
            ax.plot(1./3.*var_e[time_show,:-1]/palm_ref**2., z_e[:-1],
                    label = r'PALM: $1/3$ $e_{SGS}$', 
                    color = 'plum')
            
            #plot wt_data
            wt_z_plot = wt_z_SB[:5] +  wt_z_SB[7:]
            if var_name == 'u':
                wt_var_plot = wt_var1[:5] +  wt_var1[7:]
                ax.errorbar(wt_var_plot, wt_z_plot, xerr = flux_err,
                            label='reference profile', 
                            fmt='^', 
                            c='orangered')
                ax.hlines(45.,0., 0.022, 
                    linestyle='--', 
                    colors = 'cornflowerblue', 
                    label = r'$\delta_{cf}$')
            elif var_name == 'v':
                wt_var_plot = wt_var2[:5] +  wt_var2[7:]
                ax.errorbar(wt_var_plot, wt_z_plot, xerr = flux_err,
                            label='reference profile', 
                            fmt='^', 
                            c='orangered')
                ax.hlines(45.,0., 0.011, 
                    linestyle='--', 
                    colors = 'cornflowerblue', 
                    label = r'$\delta_{cf}$')
            elif var_name == 'w':
                wt_z_plot = wt_z_BL[:5] +  wt_z_BL[7:]
                wt_var_plot = wt_var3[:5] +  wt_var3[7:]
                ax.errorbar(wt_var_plot, wt_z_plot, xerr = flux_err,
                            label='reference profile', 
                            fmt='^', 
                            c='orangered')         
                ax.hlines(45.,0., 0.008, 
                    linestyle='--', 
                    colors = 'cornflowerblue', 
                    label = r'$\delta_{cf}$')
            ax.set_ylim(4.,140.)
            # ax.grid()
            ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                        borderaxespad = 0.,  
                        numpoints = 1)
            ax.set_ylabel(r'$z$ (m)')
            ax.set_xlabel(r'${}^\prime {}^\prime$ '.format(var_name, var_name) + r'(m$^2$ s$^{-2}$)')
            # fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_variance_{}_mask.{}'.format(papy.globals.run_name,
            #             papy.globals.run_number[-3:], 'BL',var_name, file_type), bbox_inches='tight', dpi=300)
            ax.set_yscale('log')
            ax.set_ylim(4.,140.)
            fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_variance_{}_mask_log.{}'.format(papy.globals.run_name,
                        papy.globals.run_number[-3:], 'BL',var_name, file_type), bbox_inches='tight', dpi=300)
            plt.close(12)
            print('         plotted variance of {}'.format(var_name))


######################################################
# compute BL covar in front of building
######################################################
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
        wtref_factor = 0.731    
    scale = wt_scale
    # data_nd = 1
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
                # ts.nondimensionalise()
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
    for name in namelist:
        files = wt.get_files(path,name)
        for file in files:    
            time_series[name][file].u = time_series[name][file].u/time_series[name][file].wtref
            time_series[name][file].v = time_series[name][file].v/time_series[name][file].wtref
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
            total_var1 = np.concatenate([total_var1, var1/palm_ref])
            total_var2 = np.concatenate([total_var2, var2/palm_ref])            
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
    var_e = var_e/palm_ref**2.
    z_e, z_unit_e = papy.read_nc_grid(ABL_file_path, ABL_file, 'zw*u*')
    ABL_time, ABL_time_unit = papy.read_nc_time(ABL_file_path,ABL_file)
    time_show = time.nonzero()[0][0]


    #plot profiles
    plot_wn_profiles = True
    if plot_wn_profiles:
        fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
        # plot PALM masked output
        ax.errorbar(var_vars, wall_dists, xerr=PALM_flux_err, 
                    label= r'PALM', fmt='o', c='darkmagenta')
        # plot SGS-Fluxes
        ax.plot(var_e[time_show,:-1], z_e[:-1],
                label = r'$\overline{u^\prime w^\prime}_{SGS}$', 
                color = 'plum')
        # plot wt_data
        ax.errorbar(wt_flux, wt_z, xerr = flux_err/2.,
                    label='reference profile',
                    fmt='^', 
                    c='orangered')
        ax.hlines(45.,-0.007, 0., 
                    linestyle='--', 
                    colors = 'cornflowerblue', 
                    label = r'$\delta_{cf}$')
        ax.set_ylim(4.,140.)
        # ax.grid()
        ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                    borderaxespad = 0.,  
                    numpoints = 1)
        ax.set_ylabel(r'$z$ (m)')
        ax.set_xlabel(r'$\overline{u^\prime w^\prime}$ ' + r'(m$^2$ s$^{-2}$)')
        ax.set_yscale('log')
        ax.set_ylim(4.,140.)
        fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_covariance_{}_mask_log.{}'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], 'BL','uw', file_type), bbox_inches='tight', dpi=300)
        plt.close(12)


######################################################
# compute BL turbints in front of building
######################################################
if compute_BL_turbint:
    experiment = 'single_building'
    wt_filename = 'SB_BL_UV_001'    
    namelist = [wt_filename]
    wt_path = '../../Documents/phd/experiments/{}/{}'.format(experiment, wt_filename[3:5])
    wt_file = '{}/coincidence/timeseries/{}.txt'.format(wt_path, wt_filename)
    path = '{}/coincidence/timeseries/'.format(wt_path) # path to timeseries folder
    wtref_path = '{}/wtref/'.format(wt_path)
    if wt_filename == 'SB_BL_UV_001':
        wtref_factor = 0.738
    elif wt_filename == 'BA_BL_UW_001':
        wtref_factor = 0.731    
    scale = wt_scale
    # data_nd = 1
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
                # ts.nondimensionalise()
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
    for name in namelist:
        files = wt.get_files(path,name)
        for file in files:    
            time_series[name][file].u = time_series[name][file].u/time_series[name][file].wtref
            time_series[name][file].v = time_series[name][file].v/time_series[name][file].wtref
            wt_var1.append(wt.calc_turb_data_wght(time_series[name][file].t_transit, 
                                            time_series[name][file].u,
                                            time_series[name][file].v)[0])
            wt_var2.append(wt.calc_turb_data_wght(time_series[name][file].t_transit, 
                                            time_series[name][file].u,
                                            time_series[name][file].v)[1])
            # M = np.mean(np.sqrt(time_series[name][file].u**2 + time_series[name][file].v**2))
            # wt_var1.append(np.sqrt(time_series[name][file].weighted_component_variance[0])/M)
            # wt_var2.append(np.sqrt(time_series[name][file].weighted_component_variance[1])/M)
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
        wtref_factor = 0.731    
    scale = wt_scale
    # data_nd = 1
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
                # ts.nondimensionalise()
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
    for name in namelist:
        files = wt.get_files(path,name)
        for file in files:    
            time_series[name][file].u = time_series[name][file].u/time_series[name][file].wtref
            time_series[name][file].v = time_series[name][file].v/time_series[name][file].wtref
            wt_var3.append(wt.calc_turb_data_wght(time_series[name][file].t_transit, 
                                            time_series[name][file].u,
                                            time_series[name][file].v)[1])
            # M = np.mean(np.sqrt(time_series[name][file].u**2 + time_series[name][file].v**2))
            # wt_var3.append(np.sqrt(time_series[name][file].weighted_component_variance[1])/M)
            wt_z_BL.append(time_series[name][file].z)

    var_name_list = ['u', 'v', 'w']
    print('     compute variances')
    for var_name in var_name_list:
        var_vars = np.array([])
        wall_dists = np.array([])
        for mask in mask_name_list:
            total_var = np.array([])
            total_var_2 = np.array([])
            total_time = np.array([])
            for run_no in papy.globals.run_numbers:
                nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask, run_no)
                # var_name = 'u'
                time, time_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'time')
                if var_name == 'u':
                    var, var_unit = papy.read_nc_var_ms(nc_file_path, nc_file, var_name)
                    var_2 , var_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'v')
                    y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'zu_3d')
                elif var_name == 'v':
                    var, var_unit = papy.read_nc_var_ms(nc_file_path, nc_file, var_name)
                    var_2 , var_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'u')
                    y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'zu_3d')
                elif var_name == 'w':
                    var, var_unit = papy.read_nc_var_ms(nc_file_path, nc_file, var_name)
                    var_2 , var_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'u')
                    y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'zw_3d')

                total_time = np.concatenate([total_time, time])
                total_var = np.concatenate([total_var, var/palm_ref])
                total_var_2 = np.concatenate([total_var_2, var_2/palm_ref])
            # gather values
            # if var_name == 'u':
            var_variance = np.asarray([wt.calc_turb_data(total_var,total_var_2)[0]])
            # else: 
            #     var_variance = np.asarray([wt.calc_turb_data(total_var,total_var_2)[1]])
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
            fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
            # plot PALM masked output            
            #plot wt_data
            wt_z_plot = wt_z_SB[:5] +  wt_z_SB[7:]
            if var_name == 'u':
                ax.errorbar(var_vars, wall_dists, xerr=PALM_I_u_err, 
                        label= r'PALM', fmt='o', c='darkmagenta')                
                wt_var_plot = wt_var1[:5] +  wt_var1[7:]
                ax.errorbar(wt_var_plot, wt_z_plot, xerr = I_u_err,
                            label='reference profile', 
                            fmt='^', 
                            c='orangered')
                ax.hlines(45., 0., 0.25,
                    linestyle='--', 
                    colors = 'cornflowerblue', 
                    label = r'$\delta_{cf}$')
                ax.set_xlim(0.,0.25)
            elif var_name == 'v':
                ax.errorbar(var_vars, wall_dists, xerr=PALM_I_v_err, 
                        label= r'PALM', fmt='o', c='darkmagenta')     
                wt_var_plot = wt_var2[:5] +  wt_var2[7:]
                ax.errorbar(wt_var_plot, wt_z_plot, xerr = I_v_err,
                            label='reference profile', 
                            fmt='^', 
                            c='orangered')
                ax.hlines(45., 0., 0.25,
                    linestyle='--', 
                    colors = 'cornflowerblue', 
                    label = r'$\delta_{cf}$')
                ax.set_xlim(0.,0.2)
            elif var_name == 'w':
                ax.errorbar(var_vars, wall_dists, xerr=PALM_I_w_err, 
                        label= r'PALM', fmt='o', c='darkmagenta')     
                wt_z_plot = wt_z_BL[:5] +  wt_z_BL[7:]
                wt_var_plot = wt_var3[:5] +  wt_var3[7:]
                ax.errorbar(wt_var_plot, wt_z_plot, xerr = I_v_err,
                            label='reference profile', 
                            fmt='^', 
                            c='orangered')         
                ax.hlines(45., 0., 0.25, 
                    linestyle='--', 
                    colors = 'cornflowerblue', 
                    label = r'$\delta_{cf}$')
                ax.set_xlim(0.,0.15)
            ax.set_ylim(4.,140.)

            # ax.grid()
            # if var_name == 'w':
            #     ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
            #             borderaxespad = 0.,  
            #             numpoints = 1, ncol=3)
            ax.set_ylabel(r'$z$ (m)')
            ax.set_xlabel(r'$I_{}$ '.format(var_name) + r'(-)')
            fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_turbint_I{}_mask.{}'.format(papy.globals.run_name,
                        papy.globals.run_number[-3:], 'BL',var_name, file_type), bbox_inches='tight', dpi=300)
            ax.set_yscale('log')
            ax.set_ylim(4.,140.)
            fig.savefig('../palm_results/{}/run_{}/maskprofiles/{}_turbint_I{}_mask_log.{}'.format(papy.globals.run_name,
                        papy.globals.run_number[-3:], 'BL',var_name, file_type), bbox_inches='tight', dpi=300)
            plt.close(12)
            print('         plotted turbints of {}'.format(var_name))




######################################################
# Intergral length scale Lux
######################################################
if compute_BL_lux:
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
            y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'zu_3d')
            total_time = np.concatenate([total_time, time])
            total_var = np.concatenate([total_var, var])
        # gather values
        wall_dist = np.asarray([abs(y[0])])
        wall_dists = np.concatenate([wall_dists, wall_dist])
        lux[i] = papy.calc_lux(np.abs(total_time[1]-total_time[0]), total_var)
    print('    calculated palm-LUX for {}'.format(nc_file))

    # plotting wt and PALM data
    fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
    err = 0.1 * lux
    ref_path = None
    Lux_10,Lux_1,Lux_01,Lux_001,Lux_obs_smooth,Lux_obs_rough = \
    papy.get_lux_referencedata(ref_path)
    h1 = ax.errorbar(lux, wall_dists, xerr=err, fmt='o',
                label=r'PALM - $u$', color='darkviolet')
    ref1 = ax.plot(Lux_10[1,:], Lux_10[0,:], 'k-', 
            label=r'$z_0=10\ m$ (theory)')
    ref2 = ax.plot(Lux_1[1,:], Lux_1[0,:], 'k--', 
            label=r'$z_0=1\ m$ (theory)')
    ref3 = ax.plot(Lux_01[1,:], Lux_01[0,:], 'k-.', 
            label=r'$z_0=0.1\ m$ (theory)')
    ref4 = ax.plot(Lux_001[1,:], Lux_001[0,:], 'k:', 
            label=r'$z_0=0.01\ m$ (theory)')
    ref5 = ax.plot(Lux_obs_smooth[1,:], Lux_obs_smooth[0,:], 'k+',
            label='observations smooth surface')
    ref6 = ax.plot(Lux_obs_rough[1,:], Lux_obs_rough[0,:], 'kx',
            label='observations rough surface')
    ax.set_xlim(10.,1000.)
    # ax.set_ylim([4.,1000.])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r"$L _{u}^x$ (m)")
    ax.set_ylabel(r"$z$ (m)" )
    ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
            borderaxespad = 0.,  
            numpoints = 1)
    # ax.grid(True,'both','both')
    plt.savefig('../palm_results/{}/run_{}/maskprofiles/{}_lux.{}'.format(papy.globals.run_name,papy.globals.run_number[-3:],
                'BL', file_type), bbox_inches='tight')
    print(' SAVED TO: ../palm_results/{}/run_{}/maskprofiles/{}_lux.{}'.format(papy.globals.run_name,papy.globals.run_number[-3:],
                'BL', file_type))                
    print('\n plotted integral length scale profiles')


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
            varv, varv_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'w')
            y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'zu_3d')
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

        wall_dist = np.asarray([abs(y[0])])
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

        plot_QA_PALM = False
        if plot_QA_PALM:
            # PLOT SINGLE Quadrant-scatterplots
            fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
            fig.gca().set_aspect('equal', adjustable='box')
            ax.plot(varu_fluc[q1_ind], varv_fluc[q1_ind] ,'o', color='blue',
                    label='Q1')
            ax.plot(varu_fluc[q2_ind], varv_fluc[q2_ind] ,'o', color='darkorange',
                     label='Q2')
            ax.plot(varu_fluc[q3_ind], varv_fluc[q3_ind] ,'o', color='cyan',
                     label='Q3')
            ax.plot(varu_fluc[q4_ind], varv_fluc[q4_ind] ,'o', color='red',
                     label='Q4')
            # ax.grid(True, 'both', 'both')
            ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                        borderaxespad = 0.,  
                        numpoints = 1)
            ax.set_xlabel(r'$u^\prime$ $u_{ref}^{-1}$ (-)')
            ax.set_ylabel(r'$v^\prime$ $u_{ref}^{-1}$ (-)')
            # save plots
            fig.savefig('../palm_results/{}/run_{}/quadrant_analysis/scatter/{}_QA_scatter_mask_{}.{}'.format(papy.globals.run_name,
                        papy.globals.run_number[-3:],
                        'BL', mask, file_type), bbox_inches='tight', dpi=300)
            print('     SAVED TO: ' 
                        + '../palm_results/{}/run_{}/quadrant_analysis/scatter/{}_QA_scatter_mask_{}.{}'.format(papy.globals.run_name,
                        papy.globals.run_number[-3:],
                        'BL', mask, file_type))
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
            fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
            fig.gca().set_aspect('equal', adjustable='box')        
            im1 = ax.contourf(jpdf.T, cmap='YlGnBu',
                    extent=[umin, umax, vmin, vmax], levels = 15)
            im2 = ax.contour(jpdf.T, extent=[umin, umax, vmin, vmax], levels = 15,
                    colors='gray')

            ax.vlines(0., vmin, vmax, colors='darkgray', 
                    linestyles='dashed')
            ax.hlines(0., umin, umax, colors='darkgray', 
                    linestyles='dashed')
            # ax.grid(True, 'both', 'both')
            plt.colorbar(im1, label=r'$\rho (u^\prime_{q_i},  w^\prime_{q_i})$ (-)')
            ax.set_xlabel(r'$u^\prime$ $u_{ref}^{-1}$ (-)')
            ax.set_ylabel(r'$w^\prime$ $u_{ref}^{-1}$ (-)')
            # save plots
            fig.savefig('../palm_results/{}/run_{}/quadrant_analysis/jpdf/{}_QA_jpdf_mask_{}.{}'.format(papy.globals.run_name,
                        papy.globals.run_number[-3:], 'BL', mask, file_type), bbox_inches='tight', dpi=300)
            print('     SAVED TO: ' 
                        + '../palm_results/{}/run_{}/quadrant_analysis/jpdf/{}_QA_jpdf_mask_{}.{}'.format(papy.globals.run_name,
                        papy.globals.run_number[-3:], 'BL', mask, file_type))
            plt.close()

    # plot wind tunnel data
    include_wt_data = True
    if include_wt_data:
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

                wt_wall_dist = np.asarray([abs(time_series[name][file].z)])
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
                    fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
                    fig.gca().set_aspect('equal', adjustable='box')
                    ax.plot(wt_varu_fluc[wt_q1_ind], wt_varv_fluc[wt_q1_ind] ,'o', color='blue',
                            label='Q1')
                    ax.plot(wt_varu_fluc[wt_q2_ind], wt_varv_fluc[wt_q2_ind] ,'o', color='darkorange',
                             label='Q2')
                    ax.plot(wt_varu_fluc[wt_q3_ind], wt_varv_fluc[wt_q3_ind] ,'o', color='cyan',
                             label='Q3')
                    ax.plot(wt_varu_fluc[wt_q4_ind], wt_varv_fluc[wt_q4_ind] ,'o', color='red',
                             label='Q4')
                    # ax.grid(True, 'both', 'both')
                    ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                                borderaxespad = 0.,  
                                numpoints = 1)
                    ax.set_xlabel(r'$u^\prime$ $u_{ref}^{-1}$ (-)')
                    ax.set_ylabel(r'$w^\prime$ $u_{ref}^{-1}$ (-)')
                    # save plots
                    fig.savefig('../palm_results/{}/run_{}/quadrant_analysis/scatter/{}_QA_scatter_WT_{}.{}'.format(papy.globals.run_name,
                                papy.globals.run_number[-3:], 'BL', file, file_type), bbox_inches='tight', dpi=300)
                    print('     SAVED TO: ' 
                                + '../palm_results/{}/run_{}/quadrant_analysis/scatter/{}_QA_scatter_WT_{}.{}'.format(papy.globals.run_name,
                                papy.globals.run_number[-3:], 'BL', file, file_type))
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
                    fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
                    fig.gca().set_aspect('equal', adjustable='box')
                    im1 = ax.contourf(jpdf.T, cmap='YlGnBu',
                            extent=[umin, umax, vmin, vmax], levels = 15)
                    im2 = ax.contour(jpdf.T, colors='gray',
                            extent=[umin, umax, vmin, vmax], levels = 15)

                    ax.vlines(0., vmin, vmax, colors='darkgray', 
                            linestyles='dashed')
                    ax.hlines(0., umin, umax, colors='darkgray', 
                            linestyles='dashed')
                    # ax.grid(True, 'both', 'both')
                    plt.colorbar(im1, 
                                label=r'$\rho (u^\prime_{q_i},  w^\prime_{q_i})$ (-)')
                    ax.set_xlabel(r'$u^\prime$ $u_{ref}^{-1}$ (-)')
                    ax.set_ylabel(r'$w^\prime$ $u_{ref}^{-1}$ (-)')
                    # save plots
                    fig.savefig('../palm_results/{}/run_{}/quadrant_analysis/jpdf/{}_QA_jpdf_WT_{}.{}'.format(papy.globals.run_name,
                                papy.globals.run_number[-3:], 'BL', file, file_type), bbox_inches='tight', dpi=300)
                    print('     SAVED TO: ' 
                                + '../palm_results/{}/run_{}/quadrant_analysis/jpdf/{}_QA_jpdf_WT_{}.{}'.format(papy.globals.run_name,
                                papy.globals.run_number[-3:], 'BL', file, file_type))
                    plt.close()

    # quadrant contributions
    fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
    ax.errorbar(s1_all, wall_dists, xerr=0.05,
            label = 'Q1 - PALM', fmt='d', c='navy')
    ax.errorbar(s2_all, wall_dists,xerr=0.05,
            label = 'Q2 - PALM', fmt='d', c='orange')
    ax.errorbar(s3_all, wall_dists, xerr=0.05,
            label = 'Q3 - PALM', fmt='d', c='seagreen')
    ax.errorbar(s4_all, wall_dists, xerr=0.05,
            label = 'Q4 - PALM', fmt='d', c='firebrick')
    ax.errorbar(wt_s1_all, wt_wall_dists, xerr=0.05,
            label = 'Q1 - Wind tunnel', fmt='o', c='cornflowerblue')
    ax.errorbar(wt_s2_all, wt_wall_dists, xerr=0.05,
            label = 'Q2 - Wind tunnel', fmt='o', c='navajowhite')
    ax.errorbar(wt_s3_all, wt_wall_dists, xerr=0.05,
            label = 'Q3 - Wind tunnel', fmt='o', c='springgreen')
    ax.errorbar(wt_s4_all, wt_wall_dists, xerr=0.05,
            label = 'Q4 - Wind tunnel', fmt='o', c='salmon')
    # ax.vlines(0.0066*150.*5., -5., 5., colors='tab:red', 
    #             linestyles='dashed', 
    #             label=r'$5 \cdot h_{r}$')
    ax.vlines(0., 0.4, 140, colors='black', 
                linestyles='dashed')
    # ax.grid(True, 'both', 'both')
    ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                borderaxespad = 0.,  ncol=2,
                numpoints = 1)
    ax.set_ylim(4.,140.)
    ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel(r'$\overline{u^\prime w^\prime_{q_i}}$ $\overline{u^\prime w^\prime}^{-1}$ (-)')
    ax.set_ylabel(r'$z$ (m)')
    ax.set_yscale('log')
    # save plots
    fig.savefig('../palm_results/{}/run_{}/quadrant_analysis/{}_quadrantcontribution_profile_both.{}'.format(papy.globals.run_name,
                papy.globals.run_number[-3:], 'BL', file_type), bbox_inches='tight', dpi=300)
    print('     SAVED TO: ' 
                + '../palm_results/{}/run_{}/quadrant_analysis/{}_quadrantcontribution_profile_both.{}'.format(papy.globals.run_name,
                papy.globals.run_number[-3:], 'BL', file_type))


######################################################
# compute BL timeseries correlations
######################################################
if compute_BL_correlation:
    var_name_list = ['u', 'v', 'w']
    print(' compute Correlations')   
    for var_name in var_name_list:
        mask_name_list = ['M09', 'M13', 'M14', 'M15']
        total_var = {}
        total_var = total_var.fromkeys(mask_name_list)             
        var_vars = np.array([])
        for mask in mask_name_list:
            total_var[mask] = np.array([])
            total_time = np.array([])
            for run_no in papy.globals.run_numbers:
                nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask, run_no)
                # var_name = 'u'
                time, time_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'time')
                var, var_unit = papy.read_nc_var_ms(nc_file_path, nc_file, var_name)
                total_time = np.concatenate([total_time, time])
                total_var[mask] = np.concatenate([total_var[mask], var])
            # gather values
        print('\n     VARIABLE = ', var_name)
        # corr1_onepoint = np.corrcoef(total_var[mask_name_list[0]], total_var[mask_name_list[0]])[0][1]
        timelags = np.arange(0.,round(len(total_var[mask_name_list[0]])/2), 1)
        corr1 = np.asarray([1. if x == 0 else np.corrcoef(total_var[mask_name_list[0]][x:], 
                                total_var[mask_name_list[0]][:-x])[0][1] for x in range(round(len(total_var[mask_name_list[0]])/2))])        
        corr2 = np.asarray([1. if x == 0 else np.corrcoef(total_var[mask_name_list[0]][x:], 
                                total_var[mask_name_list[1]][:-x])[0][1] for x in range(round(len(total_var[mask_name_list[0]])/2))])
        corr3 = np.asarray([1. if x == 0 else np.corrcoef(total_var[mask_name_list[0]][x:], 
                                total_var[mask_name_list[2]][:-x])[0][1] for x in range(round(len(total_var[mask_name_list[0]])/2))])
        corr4 = np.asarray([1. if x == 0 else np.corrcoef(total_var[mask_name_list[0]][x:], 
                                total_var[mask_name_list[3]][:-x])[0][1] for x in range(round(len(total_var[mask_name_list[0]])/2))])
        fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
        ax.plot(timelags*max(total_time)*0.5/max(timelags), 
                corr1,
                label=r'$R(BL, BL)$')
        ax.plot(timelags*max(total_time)*0.5/max(timelags), 
                corr2,
                label=r'$R(BL, B1)$')
        ax.plot(timelags*max(total_time)*0.5/max(timelags), 
                corr3,
                label=r'$R(BL, B2)$')
        ax.plot(timelags*max(total_time)*0.5/max(timelags),     
                corr4,
                label=r'$R(BL, S1)$')
        # ax.grid()
        if var_name == 'u':
            ax.legend(bbox_to_anchor = (-0.15,1.2), loc = 'lower left', 
                        borderaxespad = 0., ncol=4,
                        numpoints = 1)
        ax.set_xlabel(r'timelag (s)')
        ax.set_ylabel(r'Correlation for ${}$'.format(var_name))
        # ax.set_yscale('log')
        
        fig.savefig('../palm_results/{}/run_{}/{}_correlations_{}.{}'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], 'BL', var_name, file_type), bbox_inches='tight', dpi=300)
        print('     SAVED TO: ' + '../palm_results/{}/run_{}/{}_correlations_{}.{}'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], 'BL', var_name, file_type))
        plt.close(12)

print('\n Finished processing of: {}{}'.format(papy.globals.run_name, papy.globals.run_number))