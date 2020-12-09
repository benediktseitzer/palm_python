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

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import palm_py as papy

import warnings
warnings.simplefilter("ignore")

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
papy.globals.run_name = 'BA_BL_UW_001'
papy.globals.run_number = '.013'
nc_file_grid = '{}_pr{}.nc'.format(papy.globals.run_name,papy.globals.run_number)
nc_file_path = '../palm/current_version/JOBS/{}/OUTPUT/'.format(papy.globals.run_name)
mask_name_list = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 
                    'M10','M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20']
height_list = [2., 4., 5., 7.5, 10., 15.,  20., 25., 30., 35., 40., 45., 50., 60.,
                     70., 80., 90., 100., 125., 150.]

# wind tunnel input files
wt_filename = 'BA_BL_UW_001'
wt_path = '../../Documents/phd/experiments/balcony/{}'.format(wt_filename[3:5], wt_filename)
wt_file = '{}/coincidence/timeseries/{}.txt'.format(wt_path, wt_filename)
wt_file_pr = '{}/coincidence/mean/{}.000001.txt'.format(wt_path, wt_filename)
wt_file_ref = '{}/wtref/{}_wtref.txt'.format(wt_path, wt_filename)
wt_scale = 100.

# PHYSICS
papy.globals.z0 = 0.07
# papy.globals.z0 = 0.021
papy.globals.alpha = 0.17
papy.globals.ka = 0.41
papy.globals.d0 = 0.

# test-cases for spectral analysis testing
test_case_list = ['frequency_peak']
# spectra mode to run scrupt in
mode_list = ['testing', 'heights', 'compare', 'filtercheck'] 
mode = mode_list[1]

# Steeringflags
# compute_lux = True
# compute_timeseries = True
# compute_turbint = True
# compute_vertprof = True
# compute_spectra = True
# compute_crosssections = True
# compute_pure_fluxes = False
# compute_modelinput = False

compute_lux = False
compute_timeseries = False
compute_turbint = False
compute_vertprof = False
compute_spectra = False
compute_crosssections = False
compute_pure_fluxes = False
compute_modelinput = True



################
"""
MAIN
"""
################

# prepare the outputfolders
papy.prepare_plotfolder(papy.globals.run_name,papy.globals.run_number)
plt.style.use('classic')

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

################
# Compute turbulence intensities
if compute_turbint:
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
# Copmute vertical profiles
if compute_vertprof:
    nc_file = '{}_pr{}.nc'.format(papy.globals.run_name,papy.globals.run_number)
    # read variables for plot
    time, time_unit = papy.read_nc_time(nc_file_path,nc_file)
    # read wind tunnel profile
    wt_pr, wt_u_ref, wt_z = papy.read_wt_ver_pr(wt_file_pr, wt_file_ref ,wt_scale)
    print('\n wind tunnel profile loaded \n') 

    # call plot-functions
    var_name_list = ['w*u*', 'w"u"', 'e', 'e*', 'u*2', 'u']
    for i,var_name in enumerate(var_name_list):
        if var_name == 'u':
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
        else:
            grid_name = 'z{}'.format(var_name)
            var, var_max, var_unit = papy.read_nc_var_ver_pr(nc_file_path,nc_file,var_name)
            z, z_unit = papy.read_nc_grid(nc_file_path,nc_file,grid_name)
            plt.figure(i)
            papy.plot_ver_profile(var, var_unit, var_name, z, z_unit, wt_pr, wt_z, wt_u_ref, time)
            plt.close(i)
            print('\n --> plottet {} \n'.format(var_name))

    grid_name = 'zw*u*'
    var1, var_max1, var_unit1 = papy.read_nc_var_ver_pr(nc_file_path, nc_file, 'w*u*')
    var2, var_max2, var_unit2 = papy.read_nc_var_ver_pr(nc_file_path, nc_file, 'w"u"')
    var = var1 + var2
    z, z_unit = papy.read_nc_grid(nc_file_path,nc_file,grid_name)
    plt.figure(7)
    papy.plot_ver_profile(var, var_unit1, 'fluxes', z, z_unit, wt_pr, wt_z, wt_u_ref, time)
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
        i = 0
        for mask_name in mask_name_list:

            nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask_name, papy.globals.run_number)
            try:
                time, time_unit = papy.read_nc_var_ms(nc_file_path,nc_file,'time')   
                height = height_list[i]
            except: 
                print('\n Mask {} not in dataset. \n Check {} and the corresponding heights in the *_p3d-file'.format(mask_name, nc_file_path))
            i = i + 1

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

        plt.style.use('classic')
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

        ax.set_xlabel(r"$f\cdot z\cdot U^{-1}$")
        ax.set_ylabel(r"$f\cdot S_{ij}\cdot (\sigma_i\sigma_j)^{-1}$")
        ax.legend(loc='lower left', fontsize=11)
        ax.grid()

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


        plt.style.use('classic')
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
        ax.set_xlabel(r"$f\cdot z\cdot U^{-1}$")
        ax.set_ylabel(r"$f\cdot S_{ij}\cdot (\sigma_i\sigma_j)^{-1}$")
        ax.legend(loc='lower left', fontsize=11)
        ax.grid()

        plt.savefig('../palm_results/testing/spectra/filter_tests/spectra_{}_{}.png'.format(
                    var_name,'filter'), bbox_inches='tight')    

        print(' plotted spectra for {} \n'.format(var_name))    

        colors = ['c:', 'y-.', 'b--', 'r-']
        filter_sigs = [0.00001, 1., 10, 100.]

        plt.style.use('classic')
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
        ax.grid()
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
        plt.xlim(0,50)
        plt.ylim(0.4,1.)
        ax.legend(loc='lower left', fontsize=11)
        fig.savefig('../palm_results/testing/spectra/filter_tests/testing_{}_ts.png'.format('filter'), bbox_inches='tight')

################
# plot crosssections
if compute_crosssections:
    nc_file = '{}_3d{}.nc'.format(papy.globals.run_name, papy.globals.run_number)
    nc_file_path = '../palm/current_version/JOBS/{}/OUTPUT/'.format(papy.globals.run_name)

    # read variables for plot
    x_grid_name = 'x'
    y_grid_name = 'y'
    z_grid_name = 'zw_3d'
    x_grid, x_unit = papy.read_nc_grid(nc_file_path, nc_file, x_grid_name)
    y_grid, y_unit = papy.read_nc_grid(nc_file_path, nc_file, y_grid_name)
    z_grid, z_unit = papy.read_nc_grid(nc_file_path, nc_file, z_grid_name)

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
        plt.figure(8)
        papy.plot_contour_crosssection(x_grid, z_grid, var, var_name, y_grid, y_level, vert_gridname, cut_gridname, crosssection)
        plt.close(8)
        # elif crosssection == 'xy':
        crosssection = 'xy'
        z_level = int(len(z_grid)/6)
        vert_gridname = y_grid_name
        cut_gridname = 'z'
        print('     z={}    level={}'.format(round(z_grid[z_level],2), z_level))
        var, var_unit = papy.read_nc_var_hor_3d(nc_file_path,nc_file,var_name, z_level, time_show)
        plt.figure(9)
        papy.plot_contour_crosssection(x_grid, y_grid, var, var_name, z_grid, z_level, vert_gridname, cut_gridname, crosssection)
        plt.close(9)
################
# compute model input data
if compute_modelinput:
    wt_scale = 100.
    # read wind tunnel profile
    wt_u_pr, wt_u_ref, wt_z = papy.read_wt_ver_pr(wt_file_pr, wt_file_ref, wt_scale)
    print('\n wind tunnel profile loaded \n')
    # calculate z
    z = np.linspace(0.,256.,65)

    # calculate theoretical profile
    u_pr, u_fric = papy.calc_input_profile(wt_u_pr, wt_z, z)

    print(u_pr)
    print(z)

    plt.plot(u_pr, z, color='darkorange', linestyle='--', 
            label=r'$z_0 = {}$'.format(papy.globals.z0))
    plt.errorbar(wt_u_pr, wt_z,xerr=0.03*wt_u_pr, fmt='x', 
            c='cornflowerblue', label='wind tunnel')
    plt.xlabel(r'$u$ (m/s)')
    plt.ylabel(r'$z$ (m)')
    plt.legend(loc='best', numpoints=1)
    plt.grid(True,'both')
    plt.show()

################
# compute fluxes based on timeseries
if compute_pure_fluxes:
    nc_file = '{}_masked_M02{}.nc'.format(papy.globals.run_name, papy.globals.run_number)

    palm_wtref = 5.51057969
    flux13 = np.zeros(len(height_list))
    grid_name = 'zu'
    z, z_unit = papy.read_nc_grid(nc_file_path, nc_file_grid, grid_name)

    for i,mask_name in enumerate(mask_name_list): 
        nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask_name, papy.globals.run_number)
        height = height_list[i]
        
        var_u, var_unit_u = papy.read_nc_var_ms(nc_file_path, nc_file, 'u')
        var_v, var_unit_v = papy.read_nc_var_ms(nc_file_path, nc_file, 'v')
        var_w, var_unit_w = papy.read_nc_var_ms(nc_file_path, nc_file, 'w')

        flux11, flux22, flux33, flux12, flux13[i], flux23 = papy.calc_fluxes(var_u, var_v, var_w)

    # read variables for plot
    time, time_unit = papy.read_nc_time(nc_file_path,nc_file)

    nc_file = '{}_pr{}.nc'.format(papy.globals.run_name,papy.globals.run_number)
    grid_name = 'zw*u*'
    var1, var_max1, var_unit1 = papy.read_nc_var_ver_pr(nc_file_path, nc_file, 'w*u*')
    var2, var_max2, var_unit2 = papy.read_nc_var_ver_pr(nc_file_path, nc_file, 'w"u"')
    var = var1 + var2
    z, z_unit = papy.read_nc_grid(nc_file_path,nc_file,grid_name)

    flux13 = flux13/palm_wtref**2.
    var = var/palm_wtref**2.
    var1 = var1/palm_wtref**2.
    var2 = var2/palm_wtref**2.
    # plot
    plt.figure(12)
    fig, ax = plt.subplots()
    plt.style.use('classic')
    ax.plot(var[5,:-1], z[:-1], label='PALM - total', color='darkviolet')
    ax.plot(var1[5,:-1], z[:-1], label='PALM - $>\Delta$', color='plum')
    ax.plot(var2[5,:-1], z[:-1], label='PALM - $<\Delta$', color='magenta')
    ax.errorbar(flux13, height_list, xerr=0.05*flux13, fmt='o', color='darkviolet', label= 'PALM single')
    ax.set(xlabel=r'$\tau_{ges}$' + ' $(m^2/s^2)$', 
            ylabel=r'$z$  (m)'.format(z_unit))
    ax.set_yscale('log', nonposy='clip')
    plt.ylim(1.,300.)
    plt.legend(loc='best', numpoints=1)
    plt.grid(True, 'both', 'both')
    plt.show()
    plt.close(12)
    print('plotted total fluxes')
