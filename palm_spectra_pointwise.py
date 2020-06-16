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

import os
import numpy as np
import math as m
import netCDF4
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


def plot_spectra(f_comp1_sm, S_comp1_sm,
                 comp1_aliasing, height,var_name):
    """
    Plots spectra using INPUT with reference data.
    @parameter ax: axis passed to function
    """


    f_sm = [f_comp1_sm][np.argmin([np.nanmax(f_comp1_sm)])]
    f_sm = f_sm[:len(S_comp1_sm)]

    plt.style.use('classic')
    fig, ax = plt.subplots()

    if var_name == 'u':
        h1 = ax.loglog(f_sm[:comp1_aliasing], S_comp1_sm[:comp1_aliasing], 'ro', markersize=3,
                    label=r'PALM - $u$ at ${}$ m with ${}$ m/s'.format(height, str(u_mean)[:4]))
        h2 = ax.loglog(f_sm[comp1_aliasing:], S_comp1_sm[comp1_aliasing:], 'bo', markersize=3,
                    fillstyle='none')
        try:
            if not calc_kai_sim:
                h3 = ax.fill_between(f_refspecs, E_min, E_max,
                                facecolor=(1.,0.6,0.6),edgecolor='none',alpha=0.2,
                                label=r'VDI-range $S _{uu}$')
            else:
                h3 = ax.loglog(f_refspecs, E_kai, color='grey', linestyle='--', label=r'Kaimal et al. (1972)')
                h4 = ax.loglog(f_refspecs, E_sim, color='grey', linestyle='-.', label=r'Simiu and Scanlan (1986)')
        except:
            print('\n There are no reference-spectra available for this flow \n')
    elif var_name == 'v':
        h1 = ax.loglog(f_sm[:comp1_aliasing], S_comp1_sm[:comp1_aliasing], 'ro', markersize=3,
                    label=r'PALM - $v$ at ${}$ m with ${}$ m/s'.format(height, str(u_mean)[:4]))
        h2 = ax.loglog(f_sm[comp1_aliasing:], S_comp1_sm[comp1_aliasing:], 'bo', markersize=3,
                    fillstyle='none')
        try:
            if not calc_kai_sim:
                h3 = ax.fill_between(f_refspecs, E_min, E_max,
                                facecolor=(1.,0.6,0.6),edgecolor='none',alpha=0.2,
                                label=r'VDI-range $S _{vv}$')
            else:
                h3 = ax.loglog(f_refspecs, E_kai, color='grey', linestyle='--', label=r'Kaimal et al. (1972)')
                h4 = ax.loglog(f_refspecs, E_sim, color='grey', linestyle='-.', label=r'Simiu and Scanlan (1986)')
        except:
            print('\n There are no reference-spectra available for this flow \n')            
    elif var_name == 'w':
        h1 = ax.loglog(f_sm[:comp1_aliasing], S_comp1_sm[:comp1_aliasing], 'ro', markersize=3,
                    label=r'PALM - $w$ at ${}$ m with ${}$ m/s'.format(height, str(u_mean)[:4]))
        h2 = ax.loglog(f_sm[comp1_aliasing:], S_comp1_sm[comp1_aliasing:], 'bo', markersize=3,
                    fillstyle='none')
        try:
            if not calc_kai_sim:
                h3 = ax.fill_between(f_refspecs, E_min, E_max,
                                facecolor=(1.,0.6,0.6),edgecolor='none',alpha=0.2,
                                label=r'VDI-range $S _{ww}$')
            else:
                h3 = ax.loglog(f_refspecs, E_kai, color='grey', linestyle='--', label=r'Kaimal et al. (1972)')
                h4 = ax.loglog(f_refspecs, E_sim, color='grey', linestyle='-.', label=r'Simiu and Scanlan (1986)')
        except:
            print('\n There are no reference-spectra available for this flow \n')    
    elif var_name == 'phi1':
        h1 = ax.plot(f_sm[:comp1_aliasing], S_comp1_sm[:comp1_aliasing], 'r', markersize=3,
                    label=r'$\phi _1$ - normal')
        h2 = ax.plot(f_sm[:comp1_aliasing], S_comp1_sm[:comp1_aliasing], 'ro', markersize=3,
                    label=r'$\phi _1$ - normal')                    
        # h3 = ax.plot(f_sm[comp1_aliasing:], S_comp1_sm[comp1_aliasing:], 'b', markersize=3,
        #             fillstyle='none')
        ax.plot([128.**-1.,128.**-1.], [0., 800.],'grey')
        ax.plot([16.**-1.,16.**-1.], [0., 800.],'grey',)
        ax.plot([32.**-1.,32.**-1.], [0., 800.],'grey',label=r'$f _1$, $f _2$, $f _3$ ')    
    elif var_name == 'phi2':
        h1 = ax.loglog(f_sm[:comp1_aliasing], S_comp1_sm[:comp1_aliasing], 'r', markersize=3)
        h2 = ax.loglog(f_sm[:comp1_aliasing], S_comp1_sm[:comp1_aliasing], 'ro', markersize=3,
                    label=r'$\phi _2$')                    
        h3 = ax.loglog(f_sm[comp1_aliasing:], S_comp1_sm[comp1_aliasing:], 'bo', markersize=3,
                    fillstyle='none')
        ax.plot([0,1],[1,1],'grey')
    elif var_name == 'phi3':
        h1 = ax.loglog(f_sm[:comp1_aliasing], S_comp1_sm[:comp1_aliasing], 'r', markersize=3)
        h2 = ax.loglog(f_sm[:comp1_aliasing], S_comp1_sm[:comp1_aliasing], 'ro', markersize=3,
                    label=r'$\phi _3$')                    
        h3 = ax.loglog(f_sm[comp1_aliasing:], S_comp1_sm[comp1_aliasing:], 'bo', markersize=3,
                    fillstyle='none')
    else:
        h1 = ax.loglog(f_sm[:comp1_aliasing], S_comp1_sm[:comp1_aliasing], 'co', markersize=3,
                    label=r'Windtunnel $u$ at ${}$ m'.format(height))
        h2 = ax.loglog(f_sm[comp1_aliasing:], S_comp1_sm[comp1_aliasing:], 'bo', markersize=3,
                    fillstyle='none')
        try:
            if not calc_kai_sim:
                h3 = ax.fill_between(f_refspecs,E_min, E_max,
                                facecolor=(1.,0.6,0.6),edgecolor='none',alpha=0.2,
                                label=r'VDI-range $S _{uu}$')
            else:
                h3 = ax.loglog(f_refspecs, E_kai, color='grey', linestyle='--', label=r'Kaimal et al. (1972)')
                h4 = ax.loglog(f_refspecs, E_sim, color='grey', linestyle='-.', label=r'Simiu and Scanlan (1986)')
        except:
            print('\n There are no reference-spectra available for this flow \n')        

    set_limits = True

    if set_limits:
        if testing:
            if var_name == 'phi1':
                ax.set_xlim(0.,0.1)
            if var_name == 'phi2':
                ax.set_xlim(10**-4,1)
                ax.set_ylim(10**-3,10**2)
            if var_name == 'phi3':
                ax.set_xlim(10**-4,1)
                ax.set_ylim(10**-4,10**6)
        else:
            ax.set_xlim([10**-4,250])
            ax.set_ylim([10 ** -4, 1])
    else:
        xsmin = np.nanmin(f_sm[np.where(f_sm > 0)])
        xsmax = np.nanmax(f_sm[np.where(f_sm > 0)])
        ax.set_xlim(xsmin,xsmax)
        ysmin = np.nanmin(S_comp1_sm)
        ysmax = np.nanmax(S_comp1_sm)
        ax.set_ylim(ysmin,ysmax)

    ax.set_xlabel(r"$f\cdot z\cdot U^{-1}$")
    ax.set_ylabel(r"$f\cdot S_{ij}\cdot (\sigma_i\sigma_j)^{-1}$")
    ax.legend(loc='lower left', fontsize=11)
    ax.grid()

    if testing:
        fig.savefig('../palm_results/testing/spectra/testing_{}_spectra.png'.format(var_name), bbox_inches='tight')
    else:
        plt.savefig('../palm_results/{}/run_{}/spectra/{}_{}_spectra{}.png'.format(run_name, run_number[-3:],
                    run_name, var_name, mask_name), bbox_inches='tight')


def plot_timeseries(var, var_unit, var_name, time, time_unit,run_number):
    """
    plot height profile for all available times
    """

    if run_number == '':
        run_number = '.000'

    plt.style.use('classic')
    fig, ax = plt.subplots()
    ax.plot(time, var, color='green')

    ax.set(xlabel=r'$t$ $[{}]$'.format(time_unit), ylabel=r'{} $[{}]$'.format(var_name, var_unit),
            title= 'Timeseries {}'.format(var_name))

    # ax.set_ylim(-5.,5.)

    ax.grid()
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

    plt.xlim(min(time),max(time))
    if testing:
        fig.savefig('../../palm_results/testing/spectra/testing_{}_ts.png'.format(var_name), bbox_inches='tight')
    else:
        fig.savefig('../../palm_results/{}/run_{}/spectra/{}_{}_ts.png'.format(run_name,run_number[-3:],
                    run_name,var_name), bbox_inches='tight')


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
        plot_spectra(f_sm, S_uu_sm, u_aliasing, 1.,var_name)
        print('\n plotted spectra for {} \n'.format(var_name))
        plot_timeseries(var, '-', var_name, time,'s',run_number)
        print('\n plotted timeseries for {} \n'.format(var_name))
        print('\n test-case: {} done \n'.format(test_case))   


################
"""
GLOBAL VARIABLES
"""
################

run_name = 'thunder_balcony_resstudy_precursor'
run_number = '.013'
nc_file = '{}_masked_M01{}.nc'.format(run_name,run_number)
nc_file_grid = '{}_pr{}.nc'.format(run_name,run_number)
nc_file_path = '../current_version/JOBS/{}/OUTPUT/'.format(run_name)

var_name_list = ['u', 'v', 'w']
mask_name_list = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 
                    'M10','M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20']
height_list = [5., 10., 12.5, 15., 17.5, 20., 25., 30., 35., 40., 45., 50., 60.,
                     70., 80., 90., 100., 110., 120., 130.]

wt_file = '../../Documents/phd/palm/input_data/windtunnel_data/HG_BL_MR_DOK_UV_014_000001_timeseries_test.txt'

# testing parameters
mode_list = ['testing', 'heights', 'compare', 'filtercheck'] 
mode = modelist[1]
test_case_list = ['frequency_peak']

# reference spectra
calc_kai_sim = False

################
"""
MAIN
"""
################

# prepare the outputfolders
papy.prepare_plotfolder(run_name,run_number)

if mode == modelist[0]: 
    print('\n Testing: \n')
    testing_spec()
elif mode == modelist[1]:
    # heights mode
    print('\n Compute at different heights: \n')
    grid_name = 'zu'
    z, z_unit = papy.read_nc_grid(nc_file_path,nc_file_grid,grid_name)
    # read variables for plot
    f_refspecs = np.logspace(-4, 3, num=100, base = 10) 

    i = 0
    for mask_name in mask_name_list:

        nc_file = '{}_masked_{}{}.nc'.format(run_name, mask_name, run_number)
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
            ref_specs = papy.get_reference_spectra(height,None)

            E_min, E_max = papy.calc_ref_spectra(f_refspecs, ref_specs, var_name)

            plot_spectra(f_sm, S_uu_sm, u_aliasing, height,var_name)
            print('    plotted spectra for {} \n'.format(var_name))

    # wind-tunnel spectrum
    height = 8.0
    mask_name = ''
    var_name = 'u_wt'
    wt_u, wt_v, wt_t = papy.read_wt_ts(wt_file)
    u_mean = np.mean(wt_u*3.4555)
    f_sm, S_uu_sm, u_aliasing = papy.calc_spectra(wt_u, wt_t, height,u_mean)
    print('\n calculated spectra for {}'.format(var_name))
    # ref-spectra
    ref_specs = papy.get_reference_spectra(height,None)
    E_min, E_max = papy.calc_ref_spectra(f_refspecs, ref_specs, var_name)
    plot_spectra(f_sm, S_uu_sm, u_aliasing, height, var_name)
    print(' plotted spectra for {} \n'.format(var_name))


elif mode == modelist[2]:
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
    nc_file = '{}_masked_{}{}.nc'.format(run_name, mask_name, run_number)
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
    h4 = ax.loglog(f_sm_wt[wt_aliasing:], S_wt_sm[wt_aliasing:], 'b', markersize=3,
                fillstyle='none')
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

    plt.savefig('../palm_results/{}/run_{}/spectra/{}_{}_spectra{}_all.png'.format(run_name, run_number[-3:],
                run_name, var_name, mask_name), bbox_inches='tight')

elif mode == modelist[3]:
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

