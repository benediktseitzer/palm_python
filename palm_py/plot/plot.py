# -*- coding: utf-8 -*-
################
""" 
author: benedikt.seitzer
name: palm_py.plot
purpose: plotting functions for processed PALM-Data
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

################
"""
FUNCTIONS
"""
################


def plot_lux_profile(lux, height_list, var_name, run_name, run_number, testing):
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


def plot_timeseries(var, var_unit, var_name, time, time_unit, run_name, run_number):
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


def plot_turbint_profile(turbint, height_list, var_name, run_name, run_number, testing):
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


def plot_semilog_u(var, var_unit, var_name, z, z_unit, wt_pr, wt_z, wt_u_ref, time, time_unit, run_name, run_number, phys_params):
    """
    semilog-plot u-profile for all available times
    """    

    # calculate theoretical wind profile
    u_pr, u_pw, u_fric = papy.calc_theoretical_profile(wt_pr, wt_u_ref, wt_z,z,phys_params)

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


def plot_ver_profile(var_plt, var_unit, var_name, z, z_unit, wt_pr, wt_z, wt_u_ref, time, time_unit, run_name, run_number, phys_params):
    """
    plot height profile for all available times
    """    

    # calculate theoretical wind profile
    u_pr, u_pw, u_fric = papy.calc_theoretical_profile(wt_pr, wt_u_ref, wt_z, z, phys_params)
    
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


def plot_spectra(f_comp1_sm, S_comp1_sm,
                 comp1_aliasing, u_mean, height, var_name, run_name, run_number, mask_name, testing):
    """
    Plots spectra using INPUT with reference data.
    @parameter ax: axis passed to function
    """


    # reference spectra
    f_refspecs = np.logspace(-4, 3, num=100, base = 10) 
    ref_specs = papy.get_reference_spectra(height,None)
    E_min, E_max = papy.calc_ref_spectra(f_refspecs, ref_specs, var_name)

    # global calc_kai_sim
    calc_kai_sim = False

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


def plot_contour_crosssection(x, y, var, var_name, o_grid, o_level, vert_gridname, x_grid_name, run_name, run_number, crosssection):
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
    fig.colorbar(im, label=r'${}$ [m/s]'.format(var_name),orientation="horizontal")
    ax.set(xlabel=r'${}$ [m]'.format(x_grid_name), 
            ylabel=r'${}$ [m]'.format(vert_gridname))

    # file output
    fig.savefig('../palm_results/{}/run_{}/crosssections/{}_{}_cs_{}_{}.png'.format(run_name,run_number[-3:],
                run_name,var_name,str(round(o_grid[o_level],2)),crosssection), bbox_inches='tight')
    # plt.show()
    plt.close()
