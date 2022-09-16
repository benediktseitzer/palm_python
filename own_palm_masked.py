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
papy.globals.run_name = 'SB_SI_masktest'
papy.globals.run_number = '.035'
papy.globals.run_numbers = ['.005', '.006', '.007', '.008', '.009', '.010',
                            '.011', '.012', '.013', '.014', '.015', '.016', '.017', '.018', '.019', '.020',
                            '.021', '.022', '.023', '.024', '.025', '.026', '.027', '.028', '.029', '.030',
                            '.031', '.032', '.033', '.034', '.035']
nc_file_grid = '{}_pr{}.nc'.format(papy.globals.run_name,papy.globals.run_number)
nc_file_path = '../palm/current_version/JOBS/{}/OUTPUT/'.format(papy.globals.run_name)
mask_name_list = ['M02']
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
compute_mean = True
compute_lux = False
compute_turbint_masked = False
compute_spectra = False

################
"""
MAIN
"""
################

# prepare the outputfolders
papy.prepare_plotfolder(papy.globals.run_name,papy.globals.run_number)
plt.style.use('classic')

################
# mean velocity
if compute_mean:
    total_var = np.array([])
    total_time = np.array([])
    for mask in mask_name_list:

        for run_no in papy.globals.run_numbers:
            nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask, run_no)
            var_name = 'u'
            time, time_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'time')
            var, var_unit = papy.read_nc_var_ms(nc_file_path, nc_file, var_name)
            total_time = np.concatenate([total_time, time])
            total_var = np.concatenate([total_var, var])
        print('acquired data for run: ' + papy.globals.run_name + ' up to # ' + run_no)
        var_mean = np.mean(total_var)

    plt.figure(11)
    plt.plot(total_time, total_var, 
                label = r'$\overline{u}$' + r'= {}'.format(str(var_mean)[:6]))
    plt.legend()
    plt.show()
    plt.close(11)
print('end')

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

        ax.set_xlabel(r"$f\cdot z\cdot U^{-1}$", fontsize=18)
        ax.set_ylabel(r"$f\cdot S_{ij}\cdot (\sigma_i\sigma_j)^{-1}$", fontsize=18)
        ax.legend(bbox_to_anchor=(0.5, 1.04),
                loc=8, numpoints=1, ncol=2, fontsize = 18)
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
        ax.set_xlabel(r"$f\cdot z\cdot U^{-1}$", fontsize=18)
        ax.set_ylabel(r"$f\cdot S_{ij}\cdot (\sigma_i\sigma_j)^{-1}$", fontsize=18)
        ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
            borderaxespad = 0., ncol = 2, 
            numpoints = 1, fontsize = 18)
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
        ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
            borderaxespad = 0., ncol = 2, 
            numpoints = 1, fontsize = 18)
        fig.savefig('../palm_results/testing/spectra/filter_tests/testing_{}_ts.png'.format('filter'), bbox_inches='tight')
    print(' Finished Spectra')

print('')
print('Finished processing of: {}{}'.format(papy.globals.run_name, papy.globals.run_number))