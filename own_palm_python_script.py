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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import palm_py as papy

sys.path.append('/home/bene/Documents/phd/windtunnel_py/windtunnel/')    
import windtunnel as wt

import warnings
warnings.simplefilter("ignore")

plotformat = 'pgf'
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
        'xtick.labelsize' : 11,
        'ytick.labelsize' : 11,
        'legend.fontsize' : 11,
        'lines.linewidth' : 0.75,
        'lines.markersize' : 2.5,
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

def testing_spec():
    # test-cases: 
    for test_case in test_case_list:
        if test_case == 'frequency_peak':
            # phi(t) = sin(2*pi*f1*t) + sin(2*pi*f2*t) + sin(2*pi*f3*t)
            var_name = 'phi1'

            time = np.linspace(0,1023,num=1024)
            f1 = 16.**-1.
            f2 = 32.**-1.
            f3 = 128.**-1.
            var_sum1 = np.sin(2.* np.pi * f1 * time)
            var_sum2 = np.sin(2.* np.pi * f2 * time)
            var_sum3 = np.sin(2.* np.pi * f3 * time)
            var = var_sum1 + var_sum2 + var_sum3
        if test_case == 'fft_random':
            # phi(t) = whitegauss(t)
            var_name = 'phi2'

            time = np.linspace(0,1023,num=1024)
            mean = 0
            std = 1 
            num_samples = 1024
            var = np.random.normal(mean, std, num_samples)
        if test_case == 'smoothing':
            # phi(t) = 0.95*phi(t-1)+eps_w(t)
            var_name = 'phi3'         
            
            var = np.zeros(1024)
            eps_w = np.random.uniform(0,1,1024)
            time = np.linspace(0,1023,num=1024)
            for i in range(1,1024):
                var[i] = 0.95 * var[i-1] + eps_w[i]

        f_sm, S_uu_sm, u_aliasing = papy.calc_spectra(var,time,1.)
        print('\n calculated spectra for {} \n'.format(var_name))
        papy.plot_spectra(f_sm, S_uu_sm, u_aliasing, 1., 1., var_name, '', '', '')
        print('\n plotted spectra for {} \n'.format(var_name))
        papy.plot_timeseries(var, '-', var_name, time,'s')
        print('\n plotted timeseries for {} \n'.format(var_name))
        print('\n test-case: {} done \n'.format(test_case))   

################
"""
GLOBAL VARIABLES
"""
################
# PALM input files
papy.globals.run_name = 'Germano_BA_BL_z0_021_wtprof'
papy.globals.run_numbers = ['.001']
# papy.globals.run_name = 'BA_BL_z0_021_wtprof'
# papy.globals.run_numbers = ['.000', '.001', '.002', 
#                             '.003', '.004', '.005']
# papy.globals.run_name = 'BA_BL_z0_021'
# papy.globals.run_numbers = ['.000', '.001', '.002', 
#                             '.003', '.004', '.005']
papy.globals.run_name = 'SB_SI_back_yshift'
papy.globals.run_numbers = ['.008', '.009', 
                            '.010', '.011', '.012', '.013', '.014', '.015', '.016', '.017', '.018', '.019',
                            '.020', '.021', '.022', '.023', '.024', '.025', '.026', '.027', '.028', '.029',
                            '.030', '.031', '.032', '.033', '.034', '.035', '.036', '.037', '.038', '.039',
                            '.040', '.041', '.042', '.043', '.044', '.045', '.046', '.047', '.048', '.049',
                            '.050', '.051', '.052', '.053', '.054', '.055', '.056', '.057', '.058', '.059',
                            '.060', '.061', '.062', '.063', '.064', '.065', '.066', '.067', '.068', '.069',
                            '.070', '.071', '.072', '.073', '.074', '.075', '.076', '.077', '.078', '.079',
                            '.080', '.081', '.082', '.083', '.084', '.085', '.086', '.087', '.088', '.089',
                            '.090', '.091', '.092', '.093', '.094', '.095', '.096', '.097', '.098', '.099',                            
                            '.100', '.101', '.102']
# papy.globals.run_name = 'SB_SI_back'
# papy.globals.run_numbers = ['.007', '.008', '.009', '.010', '.011', '.012', 
#                         '.013', '.014', '.015', '.016', '.017', '.018',
#                         '.019', '.020', '.021', '.022', '.023', '.024',
#                         '.025', '.026', '.027', '.028', '.029', '.030', 
#                         '.031', '.032', '.033', '.034', '.035', '.036',
#                         '.037', '.038', '.039', '.040', '.041', '.042',
#                         '.043', '.044', '.045', '.046']
mask_name_list = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 
                    'M10','M11', 'M12']
# papy.globals.run_name = 'single_building_ABL_2m'
# papy.globals.run_numbers = ['.010']
papy.globals.run_number = papy.globals.run_numbers[-1]
# papy.globals.run_number = '.002'
nc_file_grid = '{}_pr{}.nc'.format(papy.globals.run_name,papy.globals.run_number)
nc_file_path = '../palm/current_version/JOBS/{}/OUTPUT/'.format(papy.globals.run_name)
# mask_name_list = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 
#                     'M10','M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19']
height_list = [2., 4., 5., 7.5, 10., 15.,  20., 25., 30., 35., 40., 45., 50., 60.,
                     70., 80., 90., 100., 125.]

# WIND TUNNEL INPIUT FILES
# experiment = 'single_building'
# wt_filename = 'SB_BL_UV_001'
experiment = 'balcony'
wt_filename = 'BA_BL_UW_001'
wt_path = '../../Documents/phd/experiments/{}/{}'.format(experiment, wt_filename[3:5])
wt_file = '{}/coincidence/timeseries/{}.txt'.format(wt_path, wt_filename)
wt_file_pr = '{}/coincidence/mean/{}.000001.txt'.format(wt_path, wt_filename)
wt_file_ref = '{}/wtref/{}_wtref.txt'.format(wt_path, wt_filename)
wt_scale = 150.
# wt_scale = 100.

# PHYSICS
papy.globals.z0 = 0.03
papy.globals.z0_wt = 0.021
# papy.globals.z0_wt = 0.071 
papy.globals.alpha = 0.18
papy.globals.ka = 0.41
papy.globals.d0 = 0.
balcony_run_list = ['single_building_ABL_2m', 'BA_BL_z0_02', 'BA_BL_z0_021', 'BA_BL_z0_06',
                    'BA_BL_z0_021_wtprof', 'BA_BL_z0_02_wtprof', 'BA_BL_z0_06_wtprof', 'Germano_BA_BL_z0_021',]
if papy.globals.run_name in balcony_run_list:
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
compute_lux = False
compute_timeseries = False
compute_turbint_masked = False
compute_turbint = False
compute_vertprof = False
compute_vertprof_flux = False
compute_spectra = False
compute_crosssections = False
compute_pure_fluxes = False
compute_simrange = False
compute_modelinput = False

compute_convergence_test = True

################
"""
MAIN
"""
################

# prepare the outputfolders
papy.prepare_plotfolder(papy.globals.run_name,papy.globals.run_number)

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
# Timeseries of several measures
if compute_timeseries:
    nc_file = '{}_ts{}.nc'.format(papy.globals.run_name, papy.globals.run_number)
    var_name_list = ['umax', 'w"u"0', 'E', 'E*', 'div_old', 'div_new', 'dt', 'us*']

    # read variables for plot and call plot-function
    time, time_unit = papy.read_nc_time(nc_file_path,nc_file)

    for var_name in var_name_list:
        var, var_unit = papy.read_nc_var_ts(nc_file_path,nc_file,var_name)
        print('\n READ {} from {}{} \n'.format(var_name, nc_file_path, nc_file))
        papy.plot_timeseries(var, var_unit, var_name, time, time_unit)
        print('\n plotted {} \n'.format(var_name))
    print(' Finished Timeseries')


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

################
# Copmute turbint profiles
if compute_turbint:
    nc_file = '{}_pr{}.nc'.format(papy.globals.run_name,papy.globals.run_number)

    # read wind tunnel timeseries
    # namelist = ['SB_BL_UV_001']
    # experiment = 'single_building'
    namelist = [wt_filename]
    config = namelist[0][3:5]
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
        wt_var = []
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
            wt_var.append(time_series[name][file].weighted_component_variance[0])
            wt_z.append(time_series[name][file].z)

    # get variance data from PALM
    var_name = 'u*2'
    grid_name = 'z{}'.format(var_name)
    var, var_max, var_unit = papy.read_nc_var_ver_pr(nc_file_path, nc_file, var_name)
    z, z_unit = papy.read_nc_grid(nc_file_path, nc_file, grid_name)
    time, time_unit = papy.read_nc_time(nc_file_path, nc_file)    

    var_e, var_e_max, var_e_unit = papy.read_nc_var_ver_pr(nc_file_path, nc_file, 'e')
    z_e, z_unit_e = papy.read_nc_grid(nc_file_path, nc_file, 'ze')

    var_full = var + 1/3. * var_e

    # plot quantities
    plt.figure(11)   
    fig, ax = plt.subplots()
    for i in range(len(time)-1,len(time)):
        try:
            print(time)
            ax.plot(var_full[i,:-1], z[:-1], 
                    label = r'PALM: $\overline{u^\prime u^\prime}_{RES} + 1/3$ $e_{SGS}$', 
                    color = 'darkviolet',
                    linewidth = 2)
            ax.plot(1./3.*var_e[i,:-1], z[:-1],
                    label = r'PALM: $1/3$ $e_{SGS}$', 
                    color = 'plum',
                    linewidth = 2)
            ax.plot(var[i,:-1], z[:-1], 
                    label = r'PALM: $\overline{u^\prime u^\prime}_{RES}$', 
                    color = 'violet',
                    linewidth = 2)
        except:
            print('Exception has occurred: StopIteration - plot_ver_profile')
    wt_var_plot = wt_var[:5] +  wt_var[7:]
    wt_z_plot = wt_z[:5] +  wt_z[7:]
    ax.errorbar(wt_var_plot, wt_z_plot, xerr = 0.025,
                label='wind tunnel', 
                fmt='^', 
                c='orangered')

    ax.set_xlabel(r'$\overline{u^\prime u^\prime}$' + r' (m$^2$ s$^{-2}$)', 
                )
    ax.set_ylabel(r'$z$ (m)', )

    ax.set_ylim(papy.globals.dx/2., 140.)
    ax.set_xlim(0., 0.6)    
    ax.legend(bbox_to_anchor=(0.5, 1.04),
                loc=8, numpoints=1, ncol=2, )
    ax.grid(True)
    # save plot
    fig.savefig('../palm_results/{}/run_{}/profiles/{}_{}_{}_verpr.png'.format(
                papy.globals.run_name, papy.globals.run_number[-3:],
                papy.globals.run_name, papy.globals.run_number[-3:], 'variance'), 
                bbox_inches='tight')
    print('saved image to :' + '../palm_results/{}/run_{}/profiles/{}_{}_{}_verpr.png'.format(
                papy.globals.run_name, papy.globals.run_number[-3:],
                papy.globals.run_name, papy.globals.run_number[-3:], 'variance'))
    #save logplot
    ax.set_yscale('log')    
    fig.savefig('../palm_results/{}/run_{}/profiles/{}_{}_{}_verpr_log.png'.format(
                papy.globals.run_name, papy.globals.run_number[-3:],
                papy.globals.run_name, papy.globals.run_number[-3:], 'variance'), 
                bbox_inches='tight')
    print('saved image to :' + '../palm_results/{}/run_{}/profiles/{}_{}_{}_verpr_log.png'.format(
                papy.globals.run_name, papy.globals.run_number[-3:],
                papy.globals.run_name, papy.globals.run_number[-3:], 'variance'))    
    plt.close(11)
    print(' Finished variance comparison')

###########################
# Copmute vertical profiles
if compute_vertprof:
    nc_file = '{}_pr{}.nc'.format(papy.globals.run_name,papy.globals.run_number)
    # read variables for plot
    time, time_unit = papy.read_nc_time(nc_file_path, nc_file)
    time_show = time.nonzero()[0][0]
    print('     Show time-slice {} of {}'.format(time_show, len(time)))    
    # read wind tunnel profile
    wt_pr, wt_u_ref, wt_z = papy.read_wt_ver_pr(wt_file_pr, wt_file_ref ,wt_scale)
    print('\n wind tunnel profile loaded \n') 

    # call plot-functions
    var_name_list = ['w*u*', 'w"u"', 'e', 'u']
    # var_name_list = ['e', 'u']    
    for i,var_name in enumerate(var_name_list):
        if var_name == 'u':
            # velocity profile
            grid_name = 'z{}'.format(var_name)
            var, var_max, var_unit = papy.read_nc_var_ver_pr(nc_file_path,nc_file,var_name)
            print('\n       u_max = {} \n'.format(var_max))
            z, z_unit = papy.read_nc_grid(nc_file_path,nc_file,grid_name)
            print('\n       wt_u_ref = {} \n'.format(wt_u_ref))
            plt.figure(i)
            papy.plot_ver_profile(var, var_unit, var_name, z, z_unit, wt_pr, wt_z, wt_u_ref, time)
            plt.close(i)
            plt.figure(i+3)
            papy.plot_semilog_u(var, var_name, z, z_unit, wt_pr, wt_z, wt_u_ref, time)
            plt.close(i+3)
            print('\n --> plottet {} \n'.format(var_name))
        elif var_name == 'e': 
            # velocity profile
            grid_name = 'z{}'.format(var_name)
            var, var_max, var_unit = papy.read_nc_var_ver_pr(nc_file_path,nc_file,
                                                            var_name)
            var_e, var_e_max, var_e_unit = papy.read_nc_var_ver_pr(nc_file_path,nc_file,
                                                            'e*')
            print('\n       u_max = {} \n'.format(var_max))
            z, z_unit = papy.read_nc_grid(nc_file_path,nc_file,grid_name)
            print('\n       wt_u_ref = {} \n'.format(wt_u_ref))
            plt.figure(8)
            fig, ax = plt.subplots()
            jet= plt.get_cmap('viridis')
            colors = iter(jet(np.linspace(0,1,10)))
            # try:
            ax.plot(var_e[time_show,:-1], z[:-1],
                label = r'$e_{RES}$',
                color='darkviolet')
            ax.plot(var[time_show,:-1], z[:-1],
                label = r'$e_{SGS}$',
                color = 'plum')
            ax.plot(var[time_show,:-1]+var_e[0,:-1], z[:-1],
                label = r'$e$',
                color = 'darkmagenta')                
            ax.grid(True, 'both')
            ax.set(xlabel=r'$e_{SGS}$ and $e_{RES}$' + r' (m$^2$/s$^2$)',
                ylabel=r'$z$ (m)')
            ax.xaxis.set_major_locator(plt.MaxNLocator(7))
            ax.set_ylim(0., 140)
            ax.legend(bbox_to_anchor=(0.5, 1.04),
                loc=8, numpoints=1, ncol=2, )
            fig.savefig('../palm_results/{}/run_{}/profiles/{}_{}_{}_verpr.png'.format(
                                                    papy.globals.run_name,
                                                    papy.globals.run_number[-3:],
                                                    papy.globals.run_name,
                                                    papy.globals.run_number[-3:],
                                                    var_name), 
                                                    bbox_inches='tight')
            plt.close(8)
            print('\n --> plottet {} \n'.format(var_name))
        else:
            #other profiles
            grid_name = 'z{}'.format(var_name)
            var, var_max, var_unit = papy.read_nc_var_ver_pr(nc_file_path,nc_file,var_name)
            z, z_unit = papy.read_nc_grid(nc_file_path,nc_file,grid_name)
            plt.figure(i)
            papy.plot_ver_profile(var, var_unit, var_name, z, z_unit, wt_pr, wt_z, wt_u_ref, time)
            plt.close(i)
            print('\n --> plottet {} \n'.format(var_name))

###########################
# Copmute vertical flux profiles
if compute_vertprof_flux:
    ###########################
    # SGS+resolved flux profile
    ###########################
    experiment = 'balcony'
    wt_filename = 'BA_BL_UW_001'    
    if wt_filename == 'BA_BL_UW_001':
        namelist = [wt_filename]
        config = namelist[0][3:5]
        wt_path = '../../Documents/phd/experiments/{}/{}'.format(experiment, wt_filename[3:5])
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

    grid_name = 'zw*u*'
    nc_file = '{}_pr{}.nc'.format(papy.globals.run_name,papy.globals.run_number)
    time, time_unit = papy.read_nc_time(nc_file_path,nc_file)
    time_show = time.nonzero()[0][0]
    print('     Show time-slice {} of {}'.format(time_show, len(time)))
    var1, var_max1, var_unit1 = papy.read_nc_var_ver_pr(nc_file_path, nc_file, 'w*u*')
    var2, var_max2, var_unit2 = papy.read_nc_var_ver_pr(nc_file_path, nc_file, 'w"u"')
    var = var1 + var2
    z, z_unit = papy.read_nc_grid(nc_file_path,nc_file,grid_name)
    plt.figure(7)
    # papy.plot_ver_profile(var, var_unit1, 'fluxes', z, z_unit, wt_pr, wt_z, wt_u_ref, time)
    fig, ax = plt.subplots()
    ax.plot(var1[time_show,:-1], z[:-1],
        label = r'$\overline{u^\prime w^\prime}_{RES}$',
        color='darkviolet')
    ax.plot(var2[time_show,:-1], z[:-1],
        label = r'$\overline{u^\prime w^\prime}_{SGS}$',
        color = 'plum')
    ax.plot(var[time_show,:-1], z[:-1],
        label = r'$\overline{u^\prime w^\prime}$',
        color = 'darkmagenta')
    if wt_filename == 'BA_BL_UW_001':
        ax.errorbar(wt_flux, wt_z, xerr = 0.005,
            # label='wind tunnel: ' + wt_filename[0:2], 
            label='wind tunnel',
            fmt='^', 
            c='orangered')
    ax.hlines(45., -0.105, 0., colors = 'tab:blue', 
            linestyles = 'dashed',
            label = r'$\delta_{cfl}$')
    ax.grid(True, 'both')
    ax.set_xlabel(r'$\overline{u^\prime w^\prime}$' + r' (m$^2$ s$^{-2}$)', 
                )
    ax.set_ylabel(r'$z$ (m)', )
    ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    #non-log plot
    ax.set_ylim(0., 140)
    ax.legend(bbox_to_anchor=(0.5, 1.04),
                loc=8, numpoints=1, ncol=2, )
    fig.savefig('../palm_results/{}/run_{}/profiles/{}_{}_{}_verpr.png'.format(
                papy.globals.run_name,
                papy.globals.run_number[-3:],
                papy.globals.run_name,
                papy.globals.run_number[-3:],
                'fluxes'), 
                bbox_inches='tight')
    # log plot
    ax.set_ylim(5, 140)
    ax.set_yscale('log')
    fig.savefig('../palm_results/{}/run_{}/profiles/{}_{}_{}_verpr_log.png'.format(
                papy.globals.run_name,
                papy.globals.run_number[-3:],
                papy.globals.run_name,
                papy.globals.run_number[-3:],
                'fluxes'), 
                bbox_inches='tight')
    plt.close(7)
    print('plotted total fluxes')
    
################
# Copmute spectra
if compute_spectra:
    var_name_list = ['u', 'v', 'w']

    if mode == mode_list[0]: 
        print('\n Testing: \n')
        testing_spec()
    elif mode == mode_list[1]:
        # heights mode
        print('\n Compute at different heights: \n')
        grid_name = 'zu'
        z, z_unit = papy.read_nc_grid(nc_file_path,nc_file_grid,grid_name)

        for i,mask_name in enumerate(mask_name_list):

            nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask_name, papy.globals.run_number)
            try:
                time, time_unit = papy.read_nc_var_ms(nc_file_path,nc_file,'time')   
                height = height_list[i]
            except: 
                print('\n Mask {} not in dataset. \n Check {} and the corresponding heights in the *_p3d-file'.format(mask_name, nc_file_path))

            print('\n HEIGHT = {} m'.format(height))
            for var_name in var_name_list:
                var, var_unit = papy.read_nc_var_ms(nc_file_path,nc_file,var_name)
                if var_name == 'u':
                    u_mean  = np.mean(var)      
                f_sm, S_uu_sm, u_aliasing = papy.calc_spectra(var,time,height,u_mean)
                print('    calculated spectra for {}'.format(var_name))
                papy.plot_spectra(f_sm, S_uu_sm, u_aliasing, u_mean, height, var_name, mask_name)
                print('    plotted spectra for {} \n'.format(var_name))

        # wind-tunnel spectrum
        compute_wt_spectra = False
        if compute_wt_spectra:
            height = 8.0
            mask_name = ''
            var_name = 'u_wt'
            wt_u, wt_v, wt_t = papy.read_wt_ts(wt_file)
            u_mean = np.mean(wt_u*3.4555)
            f_sm, S_uu_sm, u_aliasing = papy.calc_spectra(wt_u, wt_t, height,u_mean)
            print('\n calculated spectra for {}'.format(var_name))
            papy.plot_spectra(f_sm, S_uu_sm, u_aliasing, u_mean, height, var_name, mask_name)
            print(' plotted spectra for {} \n'.format(var_name))
    elif mode == mode_list[2]:
        print('\n Compute for comparison: \n')
        # plot wind tunnel spectrum together with PALM spectrum
        grid_name = 'zu'
        z, z_unit = papy.read_nc_grid(nc_file_path,nc_file_grid,grid_name)
        # read variables for plot
        f_refspecs = np.logspace(-4, 3, num=100, base = 10) 

        # 
        height = 8.0
        mask_name = ''
        var_name = 'u_wt'
        wt_u, wt_v, wt_t = papy.read_wt_ts(wt_file)
        u_mean = np.mean(wt_u*3.4555)
        f_sm_wt, S_wt_sm, wt_aliasing = papy.calc_spectra(wt_u, wt_t, height,u_mean)

        height = 5.
        var_name = 'u'
        mask_name = 'M01'
        nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask_name, papy.globals.run_number)
        time, time_unit = papy.read_nc_var_ms(nc_file_path,nc_file,'time')   
        var, var_unit = papy.read_nc_var_ms(nc_file_path,nc_file,var_name)
        if var_name == 'u':
            u_mean  = np.mean(var)            
        f_sm, S_uu_sm, comp1_aliasing = papy.calc_spectra(var,time,height,u_mean)
        print('    calculated spectra for {}'.format(var_name))
        ref_specs = papy.get_reference_spectra(height,None)

        E_min, E_max = papy.calc_ref_spectra(f_refspecs, ref_specs, var_name)

        # plot
        f_sm = [f_sm][np.argmin([np.nanmax(f_sm)])]
        f_sm = f_sm[:len(S_uu_sm)]

        f_sm_wt = [f_sm_wt][np.argmin([np.nanmax(f_sm_wt)])]
        f_sm_wt = f_sm_wt[:len(S_wt_sm)]

        fig, ax = plt.subplots()

        h1 = ax.loglog(f_sm[:comp1_aliasing], S_uu_sm[:comp1_aliasing], 'r', markersize=3,
                    label=r'PALM - $u$ at ${}$ m with ${}$ m/s'.format(height, str(u_mean)[:4]))
        h2 = ax.loglog(f_sm[comp1_aliasing:], S_uu_sm[comp1_aliasing:], 'b', markersize=3,
                    fillstyle='none')
        h3 = ax.loglog(f_sm_wt[:wt_aliasing+1], S_wt_sm[:wt_aliasing+1], 'c', markersize=3,
                    label=r'Windtunnel $u$ at $8$ m')
        # h4 = ax.loglog(f_sm_wt[wt_aliasing:], S_wt_sm[wt_aliasing:], 'b', markersize=3,
        #             fillstyle='none')
        try:
            h5 = ax.fill_between(f_refspecs, E_min, E_max,
                            facecolor=(1.,0.6,0.6),edgecolor='none',alpha=0.2,
                            label=r'VDI-range $S _{uu}$')
        except:
            print('\n There are no reference-spectra available for this flow \n')

        ax.set_xlim([10**-4,250])
        ax.set_ylim([10 ** -4, 1])

        ax.set_xlabel(r"$f\cdot z\cdot U^{-1}$", fontsize=18)
        ax.set_ylabel(r"$f\cdot S_{ij}\cdot (\sigma_i\sigma_j)^{-1}$", fontsize=18)
        ax.legend(bbox_to_anchor=(0.5, 1.04),
                loc=8, numpoints=1, ncol=2, )
        # ax.grid()

        plt.savefig('../palm_results/{}/run_{}/spectra/{}_{}_spectra{}_all.png'.format(papy.globals.run_name, papy.globals.run_number[-3:],
                    papy.globals.run_name, var_name, mask_name), bbox_inches='tight')
    elif mode == mode_list[3]:
        print('\n Gaussian Filter: \n')
        # gaussian filter demonstration

        from scipy.ndimage import gaussian_filter1d

        height = 8.0
        var_name = 'u_wt'
        mask_name = ''

        filter_sigs = [ 1., 10, 100.,0.00001]

        fig, ax = plt.subplots()


        for filter_sig in filter_sigs:
            wt_u, wt_v, wt_t = papy.read_wt_ts(wt_file)
            wt_u = gaussian_filter1d(wt_u,filter_sig)
            u_mean = np.mean(wt_u*3.4555)
            f_sm, S_uu_sm, u_aliasing = papy.calc_spectra(wt_u, wt_t, height,u_mean)
            print('\n calculated spectra for {}'.format(var_name))
            # ref-spectra
            ref_specs = papy.get_reference_spectra(height,None)
            E_min, E_max = papy.calc_ref_spectra(f_refspecs, ref_specs, var_name)

            #plot
            f_sm = [f_sm][np.argmin([np.nanmax(f_sm)])]
            f_sm = f_sm[:len(S_uu_sm)]

            if filter_sig == filter_sigs[3]:
                h1 = ax.loglog(f_sm[:u_aliasing], S_uu_sm[:u_aliasing], 'c:', markersize=3,
                            label=r'original signal'.format(filter_sig))
            elif filter_sig == filter_sigs[0]:
                h2 = ax.loglog(f_sm[:u_aliasing], S_uu_sm[:u_aliasing], 'y-.', markersize=3,
                            label=r'$\Delta _1$')
            if filter_sig == filter_sigs[1]:
                h3 = ax.loglog(f_sm[:u_aliasing], S_uu_sm[:u_aliasing], 'b--', markersize=3,
                            label=r'$\Delta _2$')
            elif filter_sig == filter_sigs[2]:
                h4 = ax.loglog(f_sm[:u_aliasing-37], S_uu_sm[:u_aliasing-37], 'r-', markersize=3,
                            label=r'$\Delta _3$')

        try:
            href = ax.fill_between(f_refspecs,E_min, E_max, facecolor=(1.,0.6,0.6),edgecolor='none',alpha=0.2,
                                    label=r'VDI-range $S _{uu}$')
        except:
            print('\n There are no reference-spectra available for this flow \n')        

        ax.set_xlim([10**-4,250])
        ax.set_ylim([10 ** -6, 1])    
        ax.set_xlabel(r"$f\cdot z\cdot U^{-1}$", fontsize=18)
        ax.set_ylabel(r"$f\cdot S_{ij}\cdot (\sigma_i\sigma_j)^{-1}$", fontsize=18)
        ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
            borderaxespad = 0., ncol = 2, 
            numpoints = 1, )
        # ax.grid()

        plt.savefig('../palm_results/testing/spectra/filter_tests/spectra_{}_{}.png'.format(
                    var_name,'filter'), bbox_inches='tight')    

        print(' plotted spectra for {} \n'.format(var_name))    

        colors = ['c:', 'y-.', 'b--', 'r-']
        filter_sigs = [0.00001, 1., 10, 100.]

        fig, ax = plt.subplots()

        i = 0
        for filter_sig in filter_sigs:
            wt_u, wt_v, wt_t = papy.read_wt_ts(wt_file)
            wt_u = gaussian_filter1d(wt_u,filter_sig)
            u_mean = np.mean(wt_u*3.4555)
            if filter_sig == filter_sigs[0]:
                h1 = ax.plot(wt_t, wt_u, colors[i],label=r'original signal')
            elif filter_sig == filter_sigs[1]:
                h2 = ax.plot(wt_t, wt_u, colors[i],label=r'$\Delta _{}$'.format(i))
            elif filter_sig == filter_sigs[2]:
                h3 = ax.plot(wt_t, wt_u, colors[i],label=r'$\Delta _{}$'.format(i))
            elif filter_sig == filter_sigs[3]:
                h4 = ax.plot(wt_t, wt_u, colors[i],label=r'$\Delta _{}$'.format(i))
            i = i + 1
            print(filter_sig)

        ax.set(xlabel=r'$t$ $[s]$', ylabel=r'$u$ $[-]$')

        # ax.set_ylim(-5.,5.)
        # ax.grid()
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
        plt.xlim(0,50)
        plt.ylim(0.4,1.)
        ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
            borderaxespad = 0., ncol = 2, 
            numpoints = 1, )
        fig.savefig('../palm_results/testing/spectra/filter_tests/testing_{}_ts.png'.format('filter'), bbox_inches='tight')
    print(' Finished Spectra')

################
# plot crosssections
if compute_crosssections:
    mode = 'averaged'
    # mode = 'snapshot'
    if mode =='averaged':
        nc_file = '{}_av_3d{}.nc'.format(papy.globals.run_name, papy.globals.run_number)
    elif 'snapshot':
        nc_file = '{}_3d{}.nc'.format(papy.globals.run_name, papy.globals.run_number)
  
    nc_file_path = '../palm/current_version/JOBS/{}/OUTPUT/'.format(papy.globals.run_name)

    # get reference velocity
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
    print('\n    PALM REFERENCE VELOCITY: {} m/s'.format(palm_ref))

    # read variables for plot
    x_grid_name = 'x'
    y_grid_name = 'y'
    z_grid_name = 'zw_3d'
    x_grid, x_unit = papy.read_nc_grid(nc_file_path, nc_file, x_grid_name)
    y_grid, y_unit = papy.read_nc_grid(nc_file_path, nc_file, y_grid_name)
    z_grid, z_unit = papy.read_nc_grid(nc_file_path, nc_file, z_grid_name)

    time, time_unit = papy.read_nc_time(nc_file_path,nc_file)
    time_show = time.nonzero()[0][0]
    print('     Show time-slice {} of {}'.format(time_show, len(time)))
    print('     READ {}{}\n'.format(nc_file_path,nc_file))
    var_name_list = ['u', 'v', 'w']

    for var_name in var_name_list:
        print('\n --> plot {}:'.format(var_name))
        # vertical crossection
        crosssection = 'xz'
        y_level = int(len(y_grid)/2)
        print('     y={}    level={}'.format(round(y_grid[y_level],2), y_level))
        vert_gridname = 'z'
        cut_gridname = x_grid_name
        var, var_unit = papy.read_nc_var_ver_3d(nc_file_path, nc_file, 
                        var_name, y_level, time_show)
        plt.figure(8)
        papy.plot_contour_crosssection(x_grid, z_grid, var/palm_ref, var_name, y_grid, 
                        y_level, vert_gridname, cut_gridname, crosssection)
        plt.close(8)
        # horizontal crosssection
        crosssection = 'xy'
        z_level = int(len(z_grid)/6)+2
        vert_gridname = y_grid_name
        cut_gridname = x_grid_name
        print('     z={}    level={}'.format(round(z_grid[z_level],2), z_level))
        var, var_unit = papy.read_nc_var_hor_3d(nc_file_path, nc_file, 
                        var_name, z_level, time_show)
        plt.figure(9)
        papy.plot_contour_crosssection(x_grid, y_grid, var/palm_ref, var_name, z_grid, 
                        z_level, vert_gridname, cut_gridname, crosssection)
        plt.close(9)
    print(' Finished Crosssections')

################
# compute model input data
if compute_modelinput:
    wind_profile = False
    topo_file_building = False
    topo_file_roughness = False
    topo_file_roughness_building = True
    if wind_profile:
        # read wind tunnel profile
        wt_u_pr, wt_u_ref, wt_z = papy.read_wt_ver_pr(wt_file_pr, wt_file_ref, wt_scale)
        print('\n wind tunnel profile loaded \n')
        # calculate z
        domain_upper = 140.
        z = np.linspace(0., domain_upper, 17)
        reference_height = [7., 30.]

        # calculate theoretical profile
        u_pr, u_fric = papy.calc_input_profile(wt_u_pr, wt_z, z, reference_height)

        print(u_pr)
        print(z)

        plt.semilogy(u_pr, z, color='darkorange', linestyle='--', 
                label=r'$z_0 = {}$'.format(papy.globals.z0))
        plt.errorbar(wt_u_pr, wt_z,xerr=0.03*wt_u_pr, fmt='x', 
                c='cornflowerblue', label='wind tunnel')
        plt.xlabel(r'$u$ (m/s)')
        plt.ylabel(r'$z$ (m)')
        plt.legend(bbox_to_anchor=(0.5, 1.04),
                loc=8, numpoints=1, ncol=2, )
        plt.grid(True,'both')
        plt.show()
    if topo_file_building:
        print('     Start constructing Topo-file')
        building_height = 50.
        building_x_length = 36.
        building_y_length = 76.
        papy.calc_topofile_building(building_height, building_x_length, building_y_length)
        print('     Finished constructing Topo-file')
    elif topo_file_roughness:
        print('     Start constructing Topo-file')
        rough_height = 3.
        rough_dist_x = 60.
        rough_dist_y = 30.
        papy.calc_topofile_roughness(rough_dist_x, rough_dist_y, rough_height)
        print('     Finished constructing Topo-file')
    elif topo_file_roughness_building:
        print('     Start constructing Topo-file')
        building_height = 50.
        building_x_length = 76.
        building_y_length = 36.
        rough_height = 3.
        rough_dist_x = 60.
        rough_dist_y = 30.
        papy.calc_topofile_roughness_building(rough_dist_x, rough_dist_y, rough_height, building_height, building_x_length, building_y_length)
        print('     Finished constructing Topo-file')        

################
# compute fluxes based on timeseries
if compute_pure_fluxes:
    nc_file = '{}_masked_M02{}.nc'.format(papy.globals.run_name, papy.globals.run_number)
    nc_file = '{}_pr{}.nc'.format(papy.globals.run_name,papy.globals.run_number)
    # palm_wtref = 5.51057969
    # palm_wtref = 0.
    flux13 = np.zeros(len(height_list))
    grid_name = 'zu'
    z, z_unit = papy.read_nc_grid(nc_file_path, nc_file_grid, grid_name)
    time, time_unit = papy.read_nc_time(nc_file_path,nc_file)
    time_show = time.nonzero()[0][0]
    print('     Show time-slice {} of {}'.format(time_show, len(time)))
    grid_name = 'zw*u*'
    var1, var_max1, var_unit1 = papy.read_nc_var_ver_pr(nc_file_path, nc_file, 'w*u*')
    var2, var_max2, var_unit2 = papy.read_nc_var_ver_pr(nc_file_path, nc_file, 'w"u"')
    var = var1 + var2
    z, z_unit = papy.read_nc_grid(nc_file_path,nc_file,grid_name)
    var_u, var_umax, var_u_unit = papy.read_nc_var_ver_pr(nc_file_path, nc_file, 'u')
    palm_wtref = var_umax
    print(palm_wtref)

    var = var/palm_wtref**2.
    var1 = var1/palm_wtref**2.
    var2 = var2/palm_wtref**2.

    # plot
    plt.figure(12)
    fig, ax = plt.subplots()
    ax.plot(var[time_show,:-1], z[:-1], label=r'$\overline{u^\prime w^\prime}$', color='darkviolet')
    ax.plot(var1[time_show,:-1], z[:-1], label=r'$\widetilde{u^\prime w^\prime}$', color='plum')
    ax.plot(var2[time_show,:-1], z[:-1], label=r'$(u^\prime w^\prime)^s$', color='magenta')
    ax.set(xlabel=r'$\overline{u^\prime w^\prime} \cdot u_{ref}^2$' + ' $(-)$', 
            ylabel=r'$z$ $(m)$'.format(z_unit))
    ax.set_yscale('log')
    plt.ylim(1.,300.)
    ax.legend(bbox_to_anchor=(0.5, 1.04),
                loc=8, numpoints=1, ncol=2, )
    plt.grid(True, 'both', 'both')
    # plt.show()
    fig.savefig('../palm_results/{}/run_{}/{}_{}.png'.format(papy.globals.run_name, papy.globals.run_number[-3:],
                papy.globals.run_name, 'fluxes'), bbox_inches='tight')
    plt.close(12)
    print('plotted total fluxes')

################
# compare simulations
if compute_simrange:

    palm_data = {}
    palm_data.fromkeys(papy.globals.run_numbers)
    var_name_list = ['flux', 'u']
    #read palm-data and init 
    for run in papy.globals.run_numbers:
        print('     Start processing palm-run #{}'.format(run[-3:]))
        papy.globals.run_number = run
        palm_data[papy.globals.run_number] = {}
        palm_data[papy.globals.run_number].fromkeys(var_name_list)
        nc_file = '{}_pr{}.nc'.format(papy.globals.run_name,papy.globals.run_number)
        
        # read variables for plot
        time, time_unit = papy.read_nc_time(nc_file_path,nc_file)
        # read wind tunnel profile
        wt_pr, wt_u_ref, wt_z = papy.read_wt_ver_pr(wt_file_pr, wt_file_ref ,wt_scale)        
        
        for i,var_name in enumerate(var_name_list):
            if var_name == 'u':
                grid_name = 'z{}'.format(var_name)        
                var, var_max, var_unit = papy.read_nc_var_ver_pr(nc_file_path,nc_file,var_name)
                z, z_unit = papy.read_nc_grid(nc_file_path,nc_file,grid_name)
                palm_data[run][var_name] = var
            elif var_name == 'flux':
                grid_name = 'zw*u*'
                var1, var_max1, var_unit1 = papy.read_nc_var_ver_pr(nc_file_path, nc_file, 'w*u*')
                var2, var_max2, var_unit2 = papy.read_nc_var_ver_pr(nc_file_path, nc_file, 'w"u"')
                var = var1 + var2
                palm_data[run][var_name] = var
            else:
                grid_name = 'z{}'.format(var_name)        
                var, var_max, var_unit = papy.read_nc_var_ver_pr(nc_file_path,nc_file,var_name)
                z, z_unit = papy.read_nc_grid(nc_file_path,nc_file,grid_name)
                palm_data[run][var_name] = var                
        print('     End processing palm-run #{}'.format(run[-3:]))

    print(' Start plotting of palm-runs of {}'.format(papy.globals.run_name))
    for i,var_name in enumerate(var_name_list):
        fig, ax = plt.subplots()
        ax.grid(True)
        plt.ylim(1.,max(z[:-1]))
        ax.set_yscale('log')
        ax.set(xlabel= var_name, 
                ylabel=r'$z$ (m)'.format(z_unit), title= r'Height profile of ${}$'.format(var_name))
        for i in range(len(time)-1,len(time)):
            try:
                for run in papy.globals.run_numbers:
                    ax.plot(palm_data[run][var_name][i,:-1], z[:-1], 
                            label='PALM - {}'.format(run[-3:]))
            except:
                print('Exception has occurred: StopIteration - plot_ver_profile')
            ax.fill_betweenx(z[:-1], palm_data[papy.globals.run_numbers[0]][var_name][i,:-1], 
                    palm_data[papy.globals.run_numbers[1]][var_name][i,:-1], color ='thistle')
        plt.legend(numpoints=1, ncol =2, )
        plt.show()
    print(' End plotting of palm-runs of {}'.format(papy.globals.run_name))

################
# compute convergence test
if compute_convergence_test:
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
    palm_ref = np.mean(total_palm_u)
    print('     PALM REFERENCE VELOCITY: {} m/s \n'.format(palm_ref))    

    print('     Compute Convergence Test for: {}'.format(papy.globals.run_name))
    var_vars = np.array([])
    wall_dists = np.array([])

    error_comp_list = ['M02', 'M07', 'M12']
    error_comp_list = ['M02']
    for mask in mask_name_list:
        if mask in error_comp_list:
            print('MASK: {}'.format(mask))
            total_var1 = np.array([])
            total_var2 = np.array([])
            total_var3 = np.array([])
            total_time = np.array([])
            for run_no in papy.globals.run_numbers:
                nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask, run_no)
                # var_name = 'u'
                time, time_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'time')
                var1, var1_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'u')
                var2, var2_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'v')
                var3, var3_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'w')
                y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'y')
                total_time = np.concatenate([total_time, time])
                total_var1 = np.concatenate([total_var1, var1/palm_ref])
                total_var2 = np.concatenate([total_var2, var2/palm_ref])
                total_var3 = np.concatenate([total_var3, var3/palm_ref])
            # gather values
            var1_fluc = np.asarray([np.mean(total_var1)]-total_var1)
            var2_fluc = np.asarray([np.mean(total_var2)]-total_var2)
            var_varu = np.asarray(([np.mean(total_var1)]-total_var1)**2.)
            var_varv = np.asarray(([np.mean(total_var2)]-total_var2)**2.)
            var_varw = np.asarray(([np.mean(total_var3)]-total_var3)**2.)
            var_flux = np.asarray(var1_fluc*var2_fluc)
            print('     T_max = {}'.format(max(total_time)))

            # Convergence-test for u
            convergence_dict_u = wt.convergence_test(total_var1)
            time_scale = max(total_time)/(max(convergence_dict_u.keys())*3600.)
            fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
            handles = wt.plot_convergence_test(convergence_dict_u, scale = time_scale, 
                                                ylabel=r'$\overline{u}$ $u_{ref}^{-1}$ (-)', calc_overlap = True)
            plt.hlines(np.mean(total_var1), 0., max(total_time/3600.), 
                        label = r'$\overline{u}_{total}$ $u_{ref}^{-1}$', linestyles = '--', color = 'black')

            key_list = []
            for key in convergence_dict_u.keys():
                key_list.append(key)        
            key = key_list[2]
            plt.hlines(max(convergence_dict_u[key]), 0., max(total_time/3600.), 
                    linestyle='-.', color='gray', 
                    label=r'$\Delta \overline{u}$ $u_{ref}^{-1}=$ ' + r'${}$'.format(str(abs(max(convergence_dict_u[key])-min(convergence_dict_u[key])))[:8]))
            plt.hlines(min(convergence_dict_u[key]), 0., max(total_time/3600.), 
                    linestyle='-.', color='gray')
            all_vals = list(convergence_dict_u.values())       
            all_vals_max = max([max(sublist) for sublist in all_vals[1:]])
            all_vals_min = min([min(sublist) for sublist in all_vals[1:]])
            plt.vlines(27000/3600., all_vals_min, all_vals_max, 
                        label = r'$T_{sim} = 7.5$ h', linestyles = '--', color = 'tab:red')
            ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                        borderaxespad = 0.,  
                        numpoints = 1, ncol = 1)
            plt.savefig('../palm_results/{}/run_{}/{}_convergence_u.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], mask), bbox_inches='tight', dpi=300)
            print('     SAVED TO: ' 
                    + '../palm_results/{}/run_{}/{}_convergence_u.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], mask))
            plt.close()

            # Convergence-test for v
            convergence_dict_v = wt.convergence_test(total_var2)
            time_scale = max(total_time)/(max(convergence_dict_v.keys())*3600.)    
            fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
            handles = wt.plot_convergence_test(convergence_dict_v, scale = time_scale, 
                                                ylabel=r'$\overline{v}$ $u_{ref}^{-1}$ (-)', calc_overlap = True)
            plt.hlines(np.mean(total_var2), 0., max(total_time/3600.), 
                        label = r'$\overline{v}_{total}$ $u_{ref}^{-1}$', linestyles = '--', color = 'black')

            key_list = []
            for key in convergence_dict_v.keys():
                key_list.append(key)        
            key = key_list[2]
            plt.hlines(max(convergence_dict_v[key]), 0., max(total_time/3600.), 
                    linestyle='-.', color='gray', 
                    label=r'$\Delta \overline{v}$ $u_{ref}^{-1}=$ ' + r'${}$'.format(str(abs(max(convergence_dict_v[key])-min(convergence_dict_v[key])))[:8]))
            plt.hlines(min(convergence_dict_v[key]), 0., max(total_time/3600.), 
                    linestyle='-.', color='gray')

            all_vals = list(convergence_dict_v.values())       
            all_vals_max = max([max(sublist) for sublist in all_vals[1:]])
            all_vals_min = min([min(sublist) for sublist in all_vals[1:]])
            plt.vlines(27000/3600., all_vals_min, all_vals_max, 
                        label = r'$T_{sim} = 7.5$ h', linestyles = '--', color = 'tab:red')
            ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                        borderaxespad = 0.,  
                        numpoints = 1, ncol = 1)
            plt.savefig('../palm_results/{}/run_{}/{}_convergence_v.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], mask), bbox_inches='tight', dpi=500)
            print('     SAVED TO: ' 
                    + '../palm_results/{}/run_{}/{}_convergence_v.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], mask))
            plt.close()

            # Convergence-test for w
            convergence_dict_w = wt.convergence_test(total_var3)
            time_scale = max(total_time)/(max(convergence_dict_w.keys())*3600.)    
            fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
            handles = wt.plot_convergence_test(convergence_dict_w, scale = time_scale, 
                                                ylabel=r'$\overline{w}$ $u_{ref}^{-1}$ (-)', calc_overlap = True)
            plt.hlines(np.mean(total_var3), 0., max(total_time/3600.), 
                        label = r'$\overline{w}_{total}$ $u_{ref}^{-1}$', linestyles = '--', color = 'black')

            key_list = []
            for key in convergence_dict_w.keys():
                key_list.append(key)        
            key = key_list[2]
            plt.hlines(max(convergence_dict_w[key]), 0., max(total_time/3600.), 
                    linestyle='-.', color='gray', 
                    label=r'$\Delta \overline{w}$ $u_{ref}^{-1}=$ ' + r'${}$'.format(str(abs(max(convergence_dict_w[key])-min(convergence_dict_w[key])))[:8]))
            plt.hlines(min(convergence_dict_w[key]), 0., max(total_time/3600.), 
                    linestyle='-.', color='gray')

            all_vals = list(convergence_dict_w.values())       
            all_vals_max = max([max(sublist) for sublist in all_vals[1:]])
            all_vals_min = min([min(sublist) for sublist in all_vals[1:]])
            plt.vlines(27000/3600., all_vals_min, all_vals_max, 
                        label = r'$T_{sim} = 7.5$ h', linestyles = '--', color = 'tab:red')
            ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                        borderaxespad = 0.,  
                        numpoints = 1, ncol = 1)
            plt.savefig('../palm_results/{}/run_{}/{}_convergence_w.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], mask), bbox_inches='tight', dpi=500)
            print('     SAVED TO: ' 
                    + '../palm_results/{}/run_{}/{}_convergence_w.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], mask))
            plt.close()

            # Convergence-test for variance u
            convergence_dict_varu = wt.convergence_test(var_varu)
            time_scale = max(total_time)/(max(convergence_dict_varu.keys())*3600.)
            fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
            handles = wt.plot_convergence_test(convergence_dict_varu, scale = time_scale, 
                                                ylabel=r'$\overline{u^\prime u^\prime}$ $u_{ref}^{-2}$ (-)', calc_overlap = True)
            plt.hlines(np.mean(var_varu), 0., max(total_time/3600.), 
                        label = r'$\overline{u^\prime u^\prime}_{total}$ $u_{ref}^{-2}$', linestyles = '--', color = 'black')

            key_list = []
            for key in convergence_dict_varu.keys():
                key_list.append(key)        
            key = key_list[2]
            plt.hlines(max(convergence_dict_varu[key]), 0., max(total_time/3600.), 
                    linestyle='-.', color='gray', 
                    label=r'$\Delta \overline{u^\prime u^\prime}$ $u_{ref}^{-2}=$ ' + r'${}$'.format(str(abs(max(convergence_dict_varu[key])-min(convergence_dict_varu[key])))[:8]))
            plt.hlines(min(convergence_dict_varu[key]), 0., max(total_time/3600.), 
                    linestyle='-.', color='gray')

            all_vals = list(convergence_dict_varu.values())       
            all_vals_max = max([max(sublist) for sublist in all_vals[1:]])
            all_vals_min = min([min(sublist) for sublist in all_vals[1:]])
            plt.vlines(27000/3600., all_vals_min, all_vals_max, 
                        label = r'$T_{total} = 7.5$ h', linestyles = '--', color = 'tab:red')
            ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                        borderaxespad = 0.,  
                        numpoints = 1, ncol = 1)
            plt.savefig('../palm_results/{}/run_{}/{}_convergence_variance_u.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], mask), bbox_inches='tight', dpi=500)
            print('     SAVED TO: ' 
                    + '../palm_results/{}/run_{}/{}_convergence_variance_u.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], mask))
            plt.close()

            # Convergence-test for variance v
            convergence_dict_varv = wt.convergence_test(var_varv)
            time_scale = max(total_time)/(max(convergence_dict_varv.keys())*3600.)
            fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
            handles = wt.plot_convergence_test(convergence_dict_varv, scale = time_scale, 
                                                ylabel=r'$\overline{v^\prime v^\prime}$ $u_{ref}^{-2}$ (-)', calc_overlap = True)
            plt.hlines(np.mean(var_varv), 0., max(total_time/3600.), 
                        label = r'$\overline{v^\prime v^\prime}_{total}$ $u_{ref}^{-2}$', linestyles = '--', color = 'black')

            key_list = []
            for key in convergence_dict_varv.keys():
                key_list.append(key)        
            key = key_list[2]
            plt.hlines(max(convergence_dict_varv[key]), 0., max(total_time/3600.), 
                    linestyle='-.', color='gray', 
                    label=r'$\Delta \overline{v^\prime v^\prime}$ $u_{ref}^{-2}=$ ' + r'${}$'.format(str(abs(max(convergence_dict_varv[key])-min(convergence_dict_varv[key])))[:8]))
            plt.hlines(min(convergence_dict_varv[key]), 0., max(total_time/3600.), 
                    linestyle='-.', color='gray')

            all_vals = list(convergence_dict_varv.values())       
            all_vals_max = max([max(sublist) for sublist in all_vals[1:]])
            all_vals_min = min([min(sublist) for sublist in all_vals[1:]])
            plt.vlines(27000/3600., all_vals_min, all_vals_max, 
                        label = r'$T_{sim} = 7.5$ h', linestyles = '--', color = 'tab:red')
            ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                        borderaxespad = 0.,  
                        numpoints = 1, ncol = 1)
            plt.savefig('../palm_results/{}/run_{}/{}_convergence_variance_v.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], mask), bbox_inches='tight', dpi=500)
            print('     SAVED TO: ' 
                    + '../palm_results/{}/run_{}/{}_convergence_variance_v.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], mask))
            plt.close()

            # Convergence-test for variance w
            convergence_dict_varw = wt.convergence_test(var_varw)
            time_scale = max(total_time)/(max(convergence_dict_varw.keys())*3600.)
            fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
            handles = wt.plot_convergence_test(convergence_dict_varw, scale = time_scale, 
                                                ylabel=r'$\overline{w^\prime w^\prime}$ $u_{ref}^{-2}$ (-)', calc_overlap = True)
            plt.hlines(np.mean(var_varw), 0., max(total_time/3600.), 
                        label = r'$\overline{w^\prime w^\prime}_{total}$ $u_{ref}^{-2}$', linestyles = '--', color = 'black')

            key_list = []
            for key in convergence_dict_varw.keys():
                key_list.append(key)        
            key = key_list[2]
            plt.hlines(max(convergence_dict_varw[key]), 0., max(total_time/3600.), 
                    linestyle='-.', color='gray', 
                    label=r'$\Delta \overline{w^\prime w^\prime}$ $u_{ref}^{-2}=$ ' + r'${}$'.format(str(abs(max(convergence_dict_varw[key])-min(convergence_dict_varw[key])))[:8]))
            plt.hlines(min(convergence_dict_varw[key]), 0., max(total_time/3600.), 
                    linestyle='-.', color='gray')

            all_vals = list(convergence_dict_varw.values())       
            all_vals_max = max([max(sublist) for sublist in all_vals[1:]])
            all_vals_min = min([min(sublist) for sublist in all_vals[1:]])
            plt.vlines(27000/3600., all_vals_min, all_vals_max, 
                        label = r'$T_{sim} = 7.5$ h', linestyles = '--', color = 'tab:red')
            ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                        borderaxespad = 0.,  
                        numpoints = 1, ncol = 1)
            plt.savefig('../palm_results/{}/run_{}/{}_convergence_variance_w.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], mask), bbox_inches='tight', dpi=500)
            print('     SAVED TO: ' 
                    + '../palm_results/{}/run_{}/{}_convergence_variance_w.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], mask))
            plt.close()


            # Convergence-test for flux
            convergence_dict_flux = wt.convergence_test(var_flux)
            time_scale = max(total_time)/(max(convergence_dict_flux.keys())*3600.)
            fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
            handles = wt.plot_convergence_test(convergence_dict_flux, scale = time_scale, 
                                                ylabel=r'$\overline{u^\prime v^\prime}$ $u_{ref}^{-2}$ (-)', calc_overlap = True)
            plt.hlines(np.mean(var_flux), 0., max(total_time/3600.), 
                        label = r'$\overline{u^\prime v^\prime}_{total}$ $u_{ref}^{-2}$', linestyles = '--', color = 'black')

            key_list = []
            for key in convergence_dict_flux.keys():
                key_list.append(key)        
            key = key_list[2]
            plt.hlines(max(convergence_dict_flux[key]), 0., max(total_time/3600.), 
                    linestyle='-.', color='gray', 
                    label=r'$\Delta \overline{u^\prime v^\prime}$ $u_{ref}^{-2}=$ ' + r'${}$'.format(str(abs(max(convergence_dict_flux[key])-min(convergence_dict_flux[key])))[:8]))
            plt.hlines(min(convergence_dict_flux[key]), 0., max(total_time/3600.), 
                    linestyle='-.', color='gray')

            all_vals = list(convergence_dict_flux.values())       
            all_vals_max = max([max(sublist) for sublist in all_vals[1:]])
            all_vals_min = min([min(sublist) for sublist in all_vals[1:]])
            plt.vlines(27000/3600., all_vals_min, all_vals_max, 
                        label = r'$T_{sim} = 7.5$ h', linestyles = '--', color = 'tab:red')
            ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                        borderaxespad = 0.,  
                        numpoints = 1, ncol = 1)
            plt.savefig('../palm_results/{}/run_{}/{}_convergence_flux.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], mask), bbox_inches='tight', dpi=500)
            print('     SAVED TO: ' 
                    + '../palm_results/{}/run_{}/{}_convergence_flux.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], mask))
            plt.close()

print('')
print('Finished processing of: {}{}'.format(papy.globals.run_name, papy.globals.run_number))