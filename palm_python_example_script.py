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



def plot_contour_crosssection(x,y,var,z_level,vert_gridname, cut_gridname, run_number):
    """
    plot cross sections
    """    

    if run_number == '':
        run_number = '.000'

    plt.style.use('classic')
    fig, ax = plt.subplots()


    # estimate bounds of colorbar
    if abs(np.min(var)) > abs(np.max(var)):
        v_bound = np.min(var)
    elif abs(np.min(var)) < abs(np.max(var)):
        v_bound = np.max(var)

    # set colorbar and mark masked buildings in grey
    current_cmap = plt.cm.seismic
    current_cmap.set_bad(color='gray')
    # plot the 2D-array var
    im = ax.imshow(var, interpolation='bilinear', cmap=current_cmap, 
                    extent=(np.min(x), np.max(x), np.min(y), np.max(y)), 
                    vmin=-v_bound, vmax=v_bound, origin='lower')

    # labeling 
    fig.colorbar(im, label=r'${}$ $[{}]$'.format(var_name,var_unit),orientation="horizontal")
    ax.set_title(r'Crosssection of ${}$ at ${}={}$ ${}$'.format(var_name,cut_gridname, 
                    round(z_grid[z_level],2), z_unit))
    ax.set(xlabel=r'${}$ $[{}]$'.format(x_grid_name,x_unit), 
            ylabel=r'${}$ $[{}]$'.format(vert_gridname,y_unit))

    # file output
    fig.savefig('../palm_results/{}/run_{}/crosssections/{}_{}_cs_{}_{}.png'.format(run_name,run_number[-3:],
                run_name,var_name,str(round(z_grid[z_level],2)),crosssection), bbox_inches='tight')
    # plt.show()
    plt.close()


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


def plot_semilog_u(var, var_unit, z, z_unit, time, time_unit, run_number):
    """
    semilog-plot u-profile for all available times
    """    

    xerror = 0.03*wt_pr
    if run_number == '':
        run_number = '.000'

    fig2, ax2 = plt.subplots()    
    ax2.set_yscale("log", nonposy='clip')    
    jet= plt.get_cmap('viridis')
    colors = iter(jet(np.linspace(0,1,10)))
    for i in range(len(time)-1,len(time)):
        try:
            ax2.plot(var[i,1:-1], z[1:-1], label='PALM', color=next(colors))
            ax2.plot(u_pw[1:-1], z[1:-1], label='power law', color='red',linestyle='--')
            ax2.plot(u_pr[1:-1], z[1:-1], label='prandtls law', color='blue',linestyle='--')
            ax2.errorbar(wt_pr[:],wt_z[:],xerr=xerror[:],label='wind tunnel',fmt='x',color='grey')
        except:
            print('Exception has occurred: StopIteration - plot_semilog-plot_u')

    ax2.xaxis.set_major_locator(plt.MaxNLocator(7))

    plt.style.use('classic')

    ax2.legend(loc='best')
    ax2.set(xlabel=r'$u/u_{ref}$ $[-]$', ylabel=r'$z$ $[{}]$'.format(z_unit), 
            title= r'Logarithmic profile of $u/u_{ref}$')
    ax2.grid()
    ax2.grid(which='minor', alpha=0.2)
    ax2.grid(which='major', alpha=0.2)
    fig2.savefig('../palm_results/{}/run_{}/profiles/{}_{}_pr_verpr_log.png'.format(run_name,run_number[-3:],
                    run_name,var_name), bbox_inches='tight')
    # plt.show()


def plot_ver_profile(var_plt, var_unit, z, z_unit, time, time_unit, run_number):
    """
    plot height profile for all available times
    """    

    xerror = 0.03*wt_pr
    if run_number == '':
        run_number = '.000'

    plt.style.use('classic')
    fig, ax = plt.subplots()
    jet= plt.get_cmap('viridis')
    colors = iter(jet(np.linspace(0,1,10)))
    ax.grid()
    ax.xaxis.set_major_locator(plt.MaxNLocator(7))

    for i in range(len(time)-1,len(time)):
        try:
            ax.plot(var_plt[i,:-1], z[:-1], label='PALM', 
                    color=next(colors))
        except:
            print('Exception has occurred: StopIteration - plot_ver_profile')
    if var_name == 'u':
        try:
            ax.plot(u_pw[:-1], z[:-1], label='power law', color='red',linestyle='--')
            ax.plot(u_pr[:-1], z[:-1], label='prandtls law', color='blue',linestyle='--')
            ax.errorbar(wt_pr[:-1],wt_z[:-1],xerr=xerror[:-1],label='wind tunnel',fmt='x',c='gray')
        except:
            print('Exception has occurred: Stop wt-plotting - plot_ver_profile')

    if var_name == 'w"u"':
        ax.set(xlabel=r'$\tau _{31}$'+'$[m^2/s^2]$', 
                ylabel=r'$z$ $[m]$', title= r'Height profile of $\tau _{31}$')
    elif var_name == 'w*u*':
        ax.set(xlabel=r'$\tau^* _{31}$'+'$[m^2/s^2]$', 
                ylabel=r'$z$ $[m]$', title= r'Height profile of $\tau^* _{31}$')
    elif var_name == 'u':
        ax.set(xlabel=r'$u/u_{ref}$'+'$[-]$', 
                ylabel=r'$z$ $[m]$', title= r'Height profile of $u/u_{ref}$')
    elif var_name == 'e*':
        ax.set(xlabel=r'$e^*$'+'$[m^2/s^2]$', 
                ylabel=r'$z$ $[m]$', title= r'Height profile of $e^*$')
    elif var_name == 'u*2':
        ax.set(xlabel=r'$\tau _{11}$'+'$[m^2/s^2]$', 
                ylabel=r'$z$ $[m]$', title= r'Height profile of $\tau _{11}$')                
    elif var_name == 'e':
        ax.set(xlabel=r'$e$'+'$[m^2/s^2]$', 
                ylabel=r'$z$ $[m]$', title= r'Height profile of $e$')                
    else:     
        ax.set(xlabel=r'${}$ $[{}]$'.format(var_name,var_unit), 
                ylabel=r'$z$ $[{}]$'.format(z_unit), title= r'Height profile of ${}$'.format(var_name))
     
    ax.legend(loc='best')
    plt.ylim(min(z),max(z[:-1]))
    fig.savefig('../palm_results/{}/run_{}/profiles/{}_{}_verpr.png'.format(run_name,run_number[-3:],
                run_name,var_name), bbox_inches='tight')
    # plt.show()


def plot_turbint_profile(turbint, height, var_name, run_name, run_number):
    """
    Plot turbulence intensities Iu or Iv.
    @parameter ax: axis passed to function
    """

    ref_path = None
    I_slight, I_moderate, I_rough, I_very = papy.get_turbint_referencedata(var_name, ref_path)

    plt.style.use('classic')
    fig, ax = plt.subplots()

    err = 0.1 * turbint
    if run_number == '':
        run_number = '.000'

    # plot data
    h1 = ax.errorbar(turbint, height_list, xerr=err, fmt='o', markersize=3,
                label=r'PALM - $I _{}$'.format(var_name))
    # plot ref-data
    r1 = ax.plot(I_slight[1,:],I_slight[0,:],'k-', linewidth=0.5,
                label= 'VDI slightly rough (lower bound)')
    r2 = ax.plot(I_moderate[1,:],I_moderate[0,:],'k-.', linewidth=0.5,
                label= 'VDI moderately rough (lower bound)')
    r3 = ax.plot(I_rough[1,:],I_rough[0,:],'k--', linewidth=0.5,
                label= 'VDI rough (lower bound)')
    r4 = ax.plot(I_very[1,:],I_very[0,:],'k:', linewidth=0.5,
                label= 'VDI very rough (lower bound)')                
    set_limits = True
    if set_limits:
        ax.set_xlim(0,0.3)
        ax.set_ylim(0,300)

    ax.set_xlabel(r"$I _{}$ [-]".format(var_name))
    ax.set_ylabel(r"$z$ [m]" )
    ax.legend(loc='upper left', fontsize=11)
    ax.grid()

    if testing:
        fig.savefig('../palm_results/testing/turbint/testing_{}_turbint.png'.format(var_name), 
                    bbox_inches='tight')
    else:
        plt.savefig('../palm_results/{}/run_{}/turbint/{}_{}_turbint.png'.format(run_name,
                    run_number[-3:], run_name, var_name), bbox_inches='tight')


def plot_timeseries(var, var_unit, time, time_unit, run_number):
    """
    plot height profile for all available times
    """    

    if run_number == '':
        run_number = '.000'

    plt.style.use('classic')
    fig, ax = plt.subplots()
    ax.plot(time, var, color='green')
    

    ax.set(xlabel=r'$t$ $[{}]$'.format(time_unit), ylabel=r'{} $[{}]$'.format(var_name,var_unit), 
            title= 'Timeseries {}'.format(var_name))

    ax.grid()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

    plt.xlim(min(time),max(time))
    fig.savefig('../palm_results/{}/run_{}/timeseries/{}_{}_ts.png'.format(run_name,run_number[-3:],
                run_name,var_name), bbox_inches='tight')
    # plt.show()


def plot_lux_profile(lux, height, var_name, run_name, run_number):
    """
    Plot Lux-profiles.
    @parameter ax: axis passed to function
    """

    ref_path = None
    Lux_10,Lux_1,Lux_01,Lux_001,Lux_obs_smooth,Lux_obs_rough = \
    papy.get_lux_referencedata(ref_path)

    plt.style.use('classic')
    fig, ax = plt.subplots()

    err = 0.1 * lux
    if run_number == '':
        run_number = '.000'

    h1 = ax.errorbar(lux, height_list, xerr=err, fmt='o', markersize=3,
                label=r'PALM - $u$')
    ref1 = ax.plot(Lux_10[1,:],Lux_10[0,:],'k-',linewidth=1,label=r'$z_0=10\ m$ (theory)')
    ref2 = ax.plot(Lux_1[1,:],Lux_1[0,:],'k--',linewidth=1,label=r'$z_0=1\ m$ (theory)')
    ref3 = ax.plot(Lux_01[1,:],Lux_01[0,:],'k-.',linewidth=1,label=r'$z_0=0.1\ m$ (theory)')
    ref4 = ax.plot(Lux_001[1,:],Lux_001[0,:],'k:',linewidth=1,label=r'$z_0=0.01\ m$ (theory)')
    ref5 = ax.plot(Lux_obs_smooth[1,:],Lux_obs_smooth[0,:],'k+',
                    linewidth=1,label='observations smooth surface')
    ref6 = ax.plot(Lux_obs_rough[1,:],Lux_obs_rough[0,:],'kx',linewidth=1,label='observations rough surface')
    
    set_limits = True
    if set_limits:
        ax.set_xlim(10,1000)
        ax.set_ylim(10,1000)


    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r"$L _{ux}$ [m]")
    ax.set_ylabel(r"$z$ [m]" )
    ax.legend(loc='upper left', fontsize=11)
    ax.grid()

    if testing:
        fig.savefig('../palm_results/testing/lux/testing_{}_lux.png'.format(var_name), bbox_inches='tight')
    else:
        plt.savefig('../palm_results/{}/run_{}/lux/{}_{}_lux.png'.format(run_name,run_number[-3:],
                    run_name,var_name), bbox_inches='tight')


################
"""
GLOBAL VARIABLES
"""
################

run_name = 'thunder_balcony_resstudy_precursor'
run_number = '.014'

nc_file_grid = '{}_pr{}.nc'.format(run_name,run_number)
nc_file_path = '../current_version/JOBS/{}/OUTPUT/'.format(run_name)

mask_name_list = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 
                    'M10','M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20']
height_list = [5., 10., 12.5, 15., 17.5, 20., 25., 30., 35., 40., 45., 50., 60.,
                     70., 80., 90., 100., 110., 120., 130.]

# wind tunnel input files
wt_file = '../../Documents/phd/palm/input_data/windtunnel_data/HG_BL_MR_DOK_UV_014_000001_timeseries_test.txt'
wt_file_pr = '../../Documents/phd/palm/input_data/windtunnel_data/HG_BL_MR_DOK_UV_015_means.txt'

# testing parameters
testing = False
test_case_list = ['frequency_peak']

# spectra mode to run scrupt in
mode_list = ['testing', 'heights', 'compare', 'filtercheck'] 
mode = mode_list[1]

# PHYSICS
# roughness length
z0 = 0.066
# exponent for powerlaw-fit
alpha = 0.17
# von Karman constant
ka = 0.41
# displacement height
d0 = 0.
# save physical parameters to list
phys_params = [z0,alpha,ka,d0]


################
"""
MAIN
"""
################

# prepare the outputfolders
papy.prepare_plotfolder(run_name,run_number)

################
# Intergral length scale Lux
nc_file = '{}_masked_M02{}.nc'.format(run_name,run_number)

lux = np.zeros(len(height_list))

grid_name = 'zu'
z, z_unit = papy.read_nc_grid(nc_file_path,nc_file_grid,grid_name)
var_name = 'u'
i = 0 

for mask_name in mask_name_list: 
    nc_file = '{}_masked_{}{}.nc'.format(run_name,mask_name,run_number)
    height = height_list[i]
        

    time, time_unit = papy.read_nc_var_ms(nc_file_path,nc_file,'time')        
    var, var_unit = papy.read_nc_var_ms(nc_file_path,nc_file,var_name)
    
    lux[i] = papy.calc_lux(np.abs(time[1]-time[0]),var)
    
    i = i + 1
    print('\n calculated integral length scale for {}'.format(str(height)))

plot_lux_profile(lux, height_list, var_name, run_name, run_number)
print('\n plotted integral length scale profiles')

################
# Timeseries of several measures
nc_file = '{}_ts{}.nc'.format(run_name,run_number)
var_name_list = ['umax', 'w"u"0', 'E', 'E*', 'div_old', 'div_new', 'dt', 'us*']

# read variables for plot and call plot-function
time, time_unit = papy.read_nc_time(nc_file_path,nc_file)

for var_name in var_name_list:
    var, var_unit = papy.read_nc_var_ts(nc_file_path,nc_file,var_name)
    print('\n READ {} from {}{} \n'.format(var_name, nc_file_path, nc_file))
    plot_timeseries(var, var_unit, time, time_unit, run_number)
    print('\n plotted {} \n'.format(var_name))

################
# Copmute turbulence intensities
nc_file = '{}_masked_M02{}.nc'.format(run_name,run_number)

Iu = np.zeros(len(height_list))
Iv = np.zeros(len(height_list))
Iw = np.zeros(len(height_list))

grid_name = 'zu'
z, z_unit = papy.read_nc_grid(nc_file_path,nc_file_grid,grid_name)
i = 0 

for mask_name in mask_name_list: 
    nc_file = '{}_masked_{}{}.nc'.format(run_name,mask_name,run_number)
    height = height_list[i]
    
    var_u, var_unit_u = papy.read_nc_var_ms(nc_file_path, nc_file, 'u')
    var_v, var_unit_v = papy.read_nc_var_ms(nc_file_path, nc_file, 'v')
    var_w, var_unit_w = papy.read_nc_var_ms(nc_file_path, nc_file, 'w')

    turbint_dat = papy.calc_turbint(var_u, var_v, var_w)

    Iu[i] = turbint_dat[0]
    Iv[i] = turbint_dat[1]
    Iw[i] = turbint_dat[2]
    i = i + 1
    print('\n calculated turbulence intensities scale for {}'.format(str(height)))

plot_turbint_profile(Iu, height_list, 'u', run_name, run_number)
print('\n plotted turbulence intensity profiles for u-component')

plot_turbint_profile(Iv, height_list, 'v', run_name, run_number)
print('\n plotted turbulence intensity profiles for v-component')

plot_turbint_profile(Iw, height_list, 'w', run_name, run_number)
print('\n plotted turbulence intensity profiles for w-component')

################
# Copmute vertical profiles
nc_file = '{}_pr{}.nc'.format(run_name,run_number)
# read variables for plot
time, time_unit = papy.read_nc_time(nc_file_path,nc_file)
# read wind tunnel profile
wt_pr, wt_u_ref, wt_z = papy.read_wt_ver_pr(wt_file_pr)
print('\n wind tunnel profile loaded \n') 

# call plot-functions
var_name_list = ['w*u*', 'w"u"', 'e', 'e*', 'u*2', 'u']
for var_name in var_name_list:
    if var_name == 'u':
        grid_name = 'z{}'.format(var_name)
        var, var_max, var_unit = papy.read_nc_var_ver_pr(nc_file_path,nc_file,var_name)
        print('\n       u_max = {} \n'.format(var_max))
        z, z_unit = papy.read_nc_grid(nc_file_path,nc_file,grid_name)
        # calculate theoretical wind profile
        u_pr, u_pw, u_fric = papy.calc_theoretical_profile(wt_pr, wt_u_ref, wt_z,z,phys_params)
        print('\n       wt_u_ref = {} \n'.format(wt_u_ref))
        print('\n       th_u_fric = {} \n'.format(u_fric))
        plot_ver_profile(var/wt_u_ref, var_unit, z, z_unit, time, time_unit,run_number)
        plot_semilog_u(var/wt_u_ref, var_unit, z, z_unit, time, time_unit,run_number)
        print('\n --> plottet {} \n'.format(var_name))
    else:
        grid_name = 'z{}'.format(var_name)
        var, var_max, var_unit = papy.read_nc_var_ver_pr(nc_file_path,nc_file,var_name)
        z, z_unit = papy.read_nc_grid(nc_file_path,nc_file,grid_name)
        plot_ver_profile(var, var_unit, z, z_unit, time, time_unit,run_number)
        print('\n --> plottet {} \n'.format(var_name))


################
# Copmute spectra
var_name_list = ['u', 'v', 'w']
# reference spectra
calc_kai_sim = False

if mode == mode_list[0]: 
    print('\n Testing: \n')
    testing_spec()
elif mode == mode_list[1]:
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
nc_file = '{}_3d{}.nc'.format(run_name, run_number)
nc_file_path = '../current_version/JOBS/{}/OUTPUT/'.format(run_name)

# read variables for plot
x_grid_name = 'x'
y_grid_name = 'y'
z_grid_name = 'zw_3d'
x_grid, x_unit = papy.read_nc_grid(nc_file_path,nc_file,x_grid_name)
y_grid, y_unit = papy.read_nc_grid(nc_file_path,nc_file,y_grid_name)
z_grid, z_unit = papy.read_nc_grid(nc_file_path,nc_file,z_grid_name)

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
    plot_contour_crosssection(x_grid, z_grid, var, y_level, vert_gridname, cut_gridname, run_number)
    
    # elif crosssection == 'xy':
    crosssection = 'xy'
    z_level = int(len(z_grid)/6)
    vert_gridname = y_grid_name
    cut_gridname = 'z'
    print('     z={}    level={}'.format(round(z_grid[z_level],2), z_level))
    var, var_unit = papy.read_nc_var_hor_3d(nc_file_path,nc_file,var_name, z_level, time_show)
    plot_contour_crosssection(x_grid, y_grid, var, z_level, vert_gridname, cut_gridname, run_number)
