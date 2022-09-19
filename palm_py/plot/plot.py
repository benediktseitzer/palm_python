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

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import palm_py as papy

################
"""
FUNCTIONS
"""
################

__all__ = [
    'plot_lux_profile',
    'plot_timeseries',
    'plot_turbint_profile',
    'plot_semilog_u',
    'plot_ver_profile',
    'plot_spectra',
    'plot_contour_crosssection'
]

def plot_lux_profile(lux, height_list):
    """
    Plot Lux-profiles.

    ----------
    Parameters:

    lux: array-like
    height_list: array-like

    """

    ref_path = None
    Lux_10,Lux_1,Lux_01,Lux_001,Lux_obs_smooth,Lux_obs_rough = \
    papy.get_lux_referencedata(ref_path)

    plt.style.use('classic')
    fig, ax = plt.subplots()

    err = 0.1 * lux
    if papy.globals.run_number == '':
        papy.globals.run_number = '.000'

    h1 = ax.errorbar(lux, height_list, xerr=err, fmt='o', markersize=3,
                label=r'PALM - $u$', color='darkviolet')
    ref1 = ax.plot(Lux_10[1,:], Lux_10[0,:], 'k-', 
            linewidth=1, label=r'$z_0=10\ m$ (theory)')
    ref2 = ax.plot(Lux_1[1,:], Lux_1[0,:], 'k--', 
            linewidth=1, label=r'$z_0=1\ m$ (theory)')
    ref3 = ax.plot(Lux_01[1,:], Lux_01[0,:], 'k-.', 
            linewidth=1, label=r'$z_0=0.1\ m$ (theory)')
    ref4 = ax.plot(Lux_001[1,:], Lux_001[0,:], 'k:', 
            linewidth=1, label=r'$z_0=0.01\ m$ (theory)')
    ref5 = ax.plot(Lux_obs_smooth[1,:], Lux_obs_smooth[0,:], 'k+',
            linewidth=1, label='observations smooth surface')
    ref6 = ax.plot(Lux_obs_rough[1,:], Lux_obs_rough[0,:], 'kx',
            linewidth=1, label='observations rough surface')
    
    set_limits = True
    if set_limits:
        ax.set_xlim(10.,1000.)
        ax.set_ylim([4.,1000.])

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r"$L _{ux}$ (m)")
    ax.set_ylabel(r"$z$ (m)" )
    ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
            borderaxespad = 0., ncol = 3, 
            numpoints = 1, fontsize = 18)
    ax.grid(True,'both','both')

    if papy.globals.testing:
        fig.savefig('../palm_results/testing/lux/testing_lux.png', bbox_inches='tight')
    else:
        plt.savefig('../palm_results/{}/run_{}/lux/{}_lux.png'.format(papy.globals.run_name,papy.globals.run_number[-3:],
                    papy.globals.run_name), bbox_inches='tight')

def plot_timeseries(var, var_unit, var_name, time, time_unit):
    """
    Plot the height profile for all available times.

    ----------
    Parameters:

    var: array-like
    var_unit: str
    var_name: str
    time: array-like
    time_unit: str
    
    """    

    if papy.globals.run_number == '':
        papy.globals.run_number = '.000'

    plt.style.use('classic')
    fig, ax = plt.subplots()
    ax.plot(time, var, color='darkviolet')
    ax.set_xlabel(r'$t$ $({})$'.format('s'), fontsize = 18)
    if var_name == 'dt':
        ax.set_ylabel(r'$\Delta t$  $(s)$', fontsize = 18)
        ax.set_ylim()
    elif var_name == 'E':
        ax.set_ylabel(r'$E$  $(m^2/s^2)$', fontsize = 18)
    elif var_name == 'E*':
        ax.set_ylabel(r'$E^*$  $(m^2/s^2)$', fontsize = 18)
    elif var_name == 'umax':
        ax.set_ylabel(r'$u_{max}$  $(m/s)$')
    elif var_name == 'div_new':
        ax.set_ylabel(r'$(\nabla \cdot \vec u)_{new}$  $(1/s)$', fontsize = 18)
        ax.set_ylim(10**-6., 10**-3.)
        # ax.set_yscale('log', nonposy='clip')
    elif var_name == 'div_old':
        ax.set_ylabel(r'$(\nabla \cdot \vec u)_{old}$  $(1/s)$', fontsize = 18) 
        ax.set_ylim(10**-6., 10**-3.)
        # ax.set_yscale('log', nonposy='clip')
    elif var_name == 'us*':
        ax.set_ylabel(r'$u_s^*$  $(m/s)$', fontsize = 18) 
    elif var_name == 'w"u"0':
        ax.set_ylabel(r'$w^\prime u^\prime_0$  $(m^2/s^2)$', fontsize = 18) 
    else:
        ax.set_ylabel(r'{} $({})$'.format(var_name,var_unit), fontsize = 18)

    ax.grid()
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    plt.xlim(250., max(time))
    fig.savefig('../palm_results/{}/run_{}/timeseries/{}_{}_ts.png'.format(papy.globals.run_name,papy.globals.run_number[-3:],
                papy.globals.run_name,var_name), bbox_inches='tight', dpi=500)

def plot_turbint_profile(turbint, height_list, var_name):
    """
    Plot turbulence intensities Iu or Iv.

    ----------
    Parameters:

    turbint: array-like
    height_list: array-like 
    var_name: str
    
    """

    ref_path = None
    I_slight, I_moderate, I_rough, I_very = papy.get_turbint_referencedata(var_name, ref_path)

    plt.style.use('classic')
    fig, ax = plt.subplots()

    err = 0.1 * turbint
    if papy.globals.run_number == '':
        papy.globals.run_number = '.000'

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

    ax.set_xlabel(r"$I _{}$ (-)".format(var_name), fontsize = 18)
    ax.set_ylabel(r"$z$ (m)", fontsize = 18)
    ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
            borderaxespad = 0., ncol = 3, 
            numpoints = 1, fontsize = 18)
    ax.grid()

    if papy.globals.testing:
        fig.savefig('../palm_results/testing/turbint/testing_{}_turbint.png'.format(var_name), 
                    bbox_inches='tight')
    else:
        plt.savefig('../palm_results/{}/run_{}/turbint/{}_{}_turbint.png'.format(papy.globals.run_name,
                    papy.globals.run_number[-3:], papy.globals.run_name, var_name), bbox_inches='tight')

def plot_semilog_u(var, var_name, z, z_unit, wt_pr, wt_z, wt_u_ref, time):
    """
    Semilog-plot u-profile for all available times.

    ----------
    Parameters:

    var: array-like
    var_name: str
    z: array-like
    z_unit: str
    wt_pr: array-like
    wt_z: array-like
    wt_u_ref: float
    time: array-like
    """    

    plt.style.use('classic')

    # calculate theoretical wind profile
    u_pr, u_pw, u_fric = papy.calc_theoretical_profile(wt_pr, wt_u_ref, wt_z,z)

    xerror = 0.03*wt_pr
    if papy.globals.run_number == '':
        papy.globals.run_number = '.000'

    fig2, ax2 = plt.subplots()    
    ax2.set_yscale("log", nonposy='clip')
    jet= plt.get_cmap('viridis')
    colors = iter(jet(np.linspace(0,1,10)))
    for i in range(len(time)-1,len(time)):
        try:
            ax2.plot(var[i,1:-1], z[1:-1], 
                    label=r'PALM: $z={}$m'.format(papy.globals.z0), 
                    color=next(colors), 
                    linewidth = 2)
            # ax2.plot(u_pw[1:-1], z[1:-1], label='power law', color='red',linestyle='--')
            ax2.plot(u_pr[1:-1], z[1:-1], 
                    label=r'fit: $z_0=({} \pm 0.003)$m'.format(papy.globals.z0_wt), 
                    color='darkorange', 
                    linestyle='--',
                    linewidth = 2)
            ax2.errorbar(wt_pr[:],wt_z[:],xerr=xerror[:],
                        label=r'wind tunnel',fmt='^',color='orangered')
        except:
            print('Exception has occurred: StopIteration - plot_semilog-plot_u')

    ax2.xaxis.set_major_locator(plt.MaxNLocator(7))
    ax2.set_ylim(1., max(z))
    ax2.set_xlim(0.8, 5.6)
    ax2.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
            borderaxespad = 0., ncol = 2, 
            numpoints = 1, fontsize = 18)
    ax2.set_xlabel(r'$\overline{u}$ (m s$^{-1}$)', fontsize = 18)
    ax2.set_ylabel(r'$z$ (m)', fontsize = 18)
    ax2.grid()
    ax2.grid(which='minor', alpha=0.2)
    ax2.grid(which='major', alpha=0.2)
    fig2.savefig('../palm_results/{}/run_{}/profiles/{}_{}_{}_pr_verpr_log.png'.format(
                                            papy.globals.run_name,
                                            papy.globals.run_number[-3:],
                                            papy.globals.run_name,
                                            papy.globals.run_number[-3:],
                                            var_name), 
                                            bbox_inches='tight')
    # plt.show()

def plot_ver_profile(var_plt, var_unit, var_name, z, z_unit, wt_pr, wt_z, wt_u_ref, time):
    """
    plot height profile for all available times

    ----------
    Parameters:

    var_plt: array-like
    var_unit: str
    var_name: str
    z: array-like
    z_unit: str
    wt_pr: array-like
    wt_z: array-like
    wt_u_ref: float
    time: array-like

    """    

    # calculate theoretical wind profile
    u_pr, u_pw, u_fric = papy.calc_theoretical_profile(wt_pr, wt_u_ref, wt_z, z)
    
    xerror = 0.03*wt_pr
    if papy.globals.run_number == '':
        papy.globals.run_number = '.000'

    plt.style.use('classic')
    fig, ax = plt.subplots()
    jet= plt.get_cmap('viridis')
    colors = iter(jet(np.linspace(0,1,10)))
    ax.grid()
    ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    plt.ylim(0.5, max(z[:-1]))

    for i in range(len(time)-1,len(time)):
        try:
            ax.plot(var_plt[i,:-1], z[:-1], 
                    label=r'PALM: $z={}$m'.format(papy.globals.z0), 
                    color = next(colors),
                    linewidth = 2)
        except:
            print('Exception has occurred: StopIteration - plot_ver_profile')
    if var_name == 'u':
        try:
            # ax.plot(u_pw[:-1], z[:-1], label='power law', color='red',linestyle='--')
            ax.plot(u_pr[:-1], z[:-1], label=r'fit: $z_0=({} \pm 0.003)$m'.format(papy.globals.z0_wt), 
                    color='darkorange', linestyle='--', linewidth = 2)
            ax.errorbar(wt_pr[:-1], wt_z[:-1], xerr=xerror[:-1], 
                    label=r'wind tunnel', fmt='^', c='orangered')
        except:
            print('Exception has occurred: Stop wt-plotting - plot_ver_profile')

    ax.set_ylabel(r'$z$ (m)', fontsize = 18)
    if var_name == 'w"u"':
        ax.set_xlabel(r'$u^\prime w^\prime_{SGS}$' + r' (m$^2$/s$^2$)',
                fontsize = 18)
        ax.set_yscale('log', nonposy='clip')
    elif var_name == 'w*u*':
        ax.set_xlabel(r'$u^\prime w^\prime_{RES}$'+ r' (m$^2$/s$^2$)', 
                fontsize = 18)
        ax.set_yscale('log', nonposy='clip')
    elif var_name == 'u':
        ax.set_xlabel(r'$\overline{u}$' + r' (m s$^{-1}$)', 
                fontsize = 18)
    elif var_name == 'e*':
        ax.set_xlabel(r'$e_{RES}$' + r' (m$^2$/s$^2$)', 
                fontsize = 18)
    elif var_name == 'u*2':
        ax.set_xlabel(r'$u^\prime u^\prime_{RES}$' + r' (m$^2$/s$^2$)', 
                fontsize = 18)
    elif var_name == 'e':
        ax.set_xlabel(r'$e_{SGS}$' + r' (m$^2$/s$^2$)', 
                fontsize = 18)
    elif var_name == 'fluxes':     
        ax.set_xlabel(r'$u^\prime w^\prime$' + r' (m$^2$/s$^2$)', 
                fontsize = 18)
        ax.set_yscale('log', nonposy='clip')
        plt.ylim(0.5, 256.)
        plt.xlim(right=0.)
    else:     
        ax.set_xlabel(r'${}$ ({})'.format(var_name,var_unit), 
                fontsize = 18) 
        ax.set_ylabel(r'$z$ (m)'.format(z_unit), 
                fontsize = 18)
        # ax.set_yscale('log', nonposy='clip')
        plt.ylim(min(z),80.)

    ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
            borderaxespad = 0., ncol = 2, 
            numpoints = 1, fontsize = 18)
    fig.savefig('../palm_results/{}/run_{}/profiles/{}_{}_{}_verpr.png'.format(
                                            papy.globals.run_name,
                                            papy.globals.run_number[-3:],
                                            papy.globals.run_name,
                                            papy.globals.run_number[-3:],
                                            var_name), 
                                            bbox_inches='tight')
    # plt.show()

def plot_spectra(f_comp1_sm, S_comp1_sm,
                 comp1_aliasing, u_mean, height, var_name, mask_name):
    """
    Plots spectra using INPUT with reference data.

    -----------
    Parameters:

    f_comp1_sm: array-like
    S_comp1_sm: array-like
    comp1_aliasing: array-like
    u_mean: float
    height: float
    var_name: str
    mask_name: str

    """

    # reference spectra
    f_refspecs = np.logspace(-4, 3, num=100, base = 10) 
    ref_specs = papy.get_reference_spectra(height,None)
    E_min, E_max, E_kai, E_sim = papy.calc_ref_spectra(f_refspecs, ref_specs, var_name)

    f_sm = [f_comp1_sm][np.argmin([np.nanmax(f_comp1_sm)])]
    f_sm = f_sm[:len(S_comp1_sm)]

    plt.style.use('classic')
    fig, ax = plt.subplots()

    if var_name == 'u':
        h1 = ax.loglog(f_sm[:comp1_aliasing], S_comp1_sm[:comp1_aliasing], 'ro', markersize=3,
                    label=r'PALM - $u$ at ${}$ $m$ with ${}$ $m/s$'.format(height, str(u_mean)[:4]))
        h2 = ax.loglog(f_sm[comp1_aliasing:], S_comp1_sm[comp1_aliasing:], 'bo', markersize=3,
                    fillstyle='none')
        try:
            if not papy.globals.calc_kai_sim:
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
                    label=r'PALM - $v$ at ${}$ $m$ with ${}$ $m/s$'.format(height, str(u_mean)[:4]))
        h2 = ax.loglog(f_sm[comp1_aliasing:], S_comp1_sm[comp1_aliasing:], 'bo', markersize=3,
                    fillstyle='none')
        try:
            if not papy.globals.calc_kai_sim:
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
                    label=r'PALM - $w$ at ${}$ $m$ with ${}$ $m/s$'.format(height, str(u_mean)[:4]))
        h2 = ax.loglog(f_sm[comp1_aliasing:], S_comp1_sm[comp1_aliasing:], 'bo', markersize=3,
                    fillstyle='none')
        try:
            if not papy.globals.calc_kai_sim:
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
            if not papy.globals.calc_kai_sim:
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
        if papy.globals.testing:
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

    ax.set_xlabel(r"$f\cdot z\cdot u_{ref}^{-1}$", fontsize = 18)
    ax.set_ylabel(r"$f\cdot S_{ij}\cdot (\sigma_i\sigma_j)^{-1}$", fontsize = 18)
    ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
            borderaxespad = 0., ncol = 2, 
            numpoints = 1, fontsize = 18)
    ax.grid()

    if papy.globals.testing:
        fig.savefig('../palm_results/testing/spectra/testing_{}_spectra.png'.format(var_name), bbox_inches='tight')
    else:
        plt.savefig('../palm_results/{}/run_{}/spectra/{}_{}_spectra{}.png'.format(papy.globals.run_name, papy.globals.run_number[-3:],
                    papy.globals.run_name, var_name, mask_name), bbox_inches='tight')

def plot_contour_crosssection(x, y, var, var_name, o_grid, o_level, vert_gridname, x_grid_name, crosssection):
    """
    Plot cross sections of all three velocity components.

    -----------
    Parameters:
    
    x: array-like 
    y: array-like 
    var: array-like 
    var_name: str 
    o_grid: array-like
    o_level: integer
    vert_gridname: str
    x_grid_name: str
    crosssection: str
    """    

    if papy.globals.run_number == '':
        papy.globals.run_number = '.000'

    plt.style.use('classic')
    fig, ax = plt.subplots()

    # estimate bounds of colorbar
    if abs(np.min(var)) >= abs(np.max(var)):
        v_bound = abs(np.min(var))
    elif abs(np.min(var)) < abs(np.max(var)):
        v_bound = abs(np.max(var))

    # set colorbar and mark masked buildings in grey
    current_cmap = plt.cm.seismic
    current_cmap.set_bad(color='gray')
    # plot the 2D-array var
    im = ax.imshow(var, interpolation='bilinear', cmap=current_cmap,
                    extent=(np.min(x), np.max(x), np.min(y), np.max(y)),
                    vmin = -v_bound, vmax = v_bound, origin='lower')

    # labeling 
    if crosssection=='xz':
        ax.set_yticks([0,40,70,100,140])
        fig.colorbar(im, label=r'${}$  (m/s)'.format(var_name), 
                    orientation = 'horizontal')
    else:
        fig.colorbar(im, label=r'${}$  (m/s)'.format(var_name), 
                    orientation = 'vertical')        
    ax.set_xlabel(r'${}$  (m)'.format(x_grid_name), 
                    fontsize = 18)
    ax.set_ylabel(r'${}$  (m)'.format(vert_gridname), 
                    fontsize = 18)

    # file output
    fig.savefig('../palm_results/{}/run_{}/crosssections/{}_{}_cs_{}_{}.png'.format(
                papy.globals.run_name,papy.globals.run_number[-3:],
                papy.globals.run_name,var_name,str(round(o_grid[o_level],2)),
                crosssection), 
                bbox_inches='tight', 
                dpi=600)
