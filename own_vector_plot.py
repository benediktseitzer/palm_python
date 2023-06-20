# -*- coding: utf-8 -*-

################
'''
IMPORTS
'''
################

import numpy as np
import pandas as pd
from scipy import stats
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
import os
import sys

import palm_py as papy

sys.path.append('/home/bene/Documents/phd/windtunnel_py/windtunnel/')    
import windtunnel as wt

# supress SOURCE ID warnings by matplotlib backend
warnings.simplefilter("ignore")
# Create logger
logger = logging.getLogger()

plotformat = 'pgf'
# plotformat = 'png'
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
        'text.usetex': False,
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
'''
MAIN
'''
################

#%%#

run_list = ['SB_SI_front', 'SB_SI_2', 'SB_SI_back']
papy.globals.run_name = run_list[0]
papy.globals.run_numbers = ['.007', '.008', '.009', '.010', '.011', '.012', 
                        '.013', '.014', '.015', '.016', '.017', '.018',
                        '.019', '.020', '.021', '.022', '.023', '.024',
                        '.025', '.026', '.027', '.028', '.029', '.030', 
                        '.031', '.032', '.033', '.034', '.035', '.036',
                        '.037', '.038', '.039', '.040', '.041', '.042',
                        '.043', '.044', '.045', '.046']
# papy.globals.run_numbers = ['.007', '.008', '.009', '.010', '.011', '.012', 
#                             '.013', '.014', '.015', '.016', '.017']
papy.globals.run_number = papy.globals.run_numbers[-1]
mask_list_middle = ['M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20',
                    'M21', 'M22', 'M23', 'M24']
mask_list = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08',
                    'M09', 'M10', 'M11', 'M12']



vectorplot_xy = True
vectorplot_xz = False

if vectorplot_xy:
    # FL
    namelist = ['SB_FL_SI_UV_023', 
                'SB_FL_SI_UV_022', 
                'SB_FL_SI_UV_001', 
                'SB_FL_SI_UV_006']
    # BR
    # namelist = ['SB_BR_SI_UV_012',
    #             'SB_BR_SI_UV_011']
    # # WB
    # namelist = ['SB_WB_SI_UV_013',
    #             'SB_WB_SI_UV_012']
if vectorplot_xz:
    # FL
    namelist = ['SB_FL_LU_UW_001', 
                'SB_FL_LU_UW_002',
                'SB_FL_LE_UW_001',
                'SB_FL_LE_UW_002',
                'SB_FL_LE_UW_004',
                'SB_FL_LE_UW_005',
                'SB_FL_LE_UW_007',
                'SB_FL_LE_UW_008',
                'SB_FL_LU_UW_003',
                'SB_FL_LE_UW_006']
    # # BR
    # namelist = ['SB_BR_SI_UV_012',
    #             'SB_BR_SI_UV_011']
    # # WB
    # namelist = ['SB_WB_SI_UV_013',
    #             'SB_WB_SI_UV_012']

# set paths and files
experiment = 'single_building'
if len(namelist) == 1:
    config = namelist[0][3:5]
else:
    if namelist[1][-4:] == 'long':
        config = 'LONG'    
    else:
        config = namelist[0][3:5]
        print('config = {}'.format(config))
config = 'CO_REF'
wt_path = '../../Documents/phd/experiments/{}/{}'.format(experiment, 'CO_REF')
path = '{}/coincidence/timeseries/'.format(wt_path) # path to timeseries folder
wtref_path = '{}/wtref/'.format(wt_path)
mean_path = '../../experiments/{}/{}/coincidence/mean/'.format(experiment, config)
txt_path = './postprocessed/'
ref_path = '../wt_ref_path/'
file_type = plotformat
plot_path_0 = '../palm_results/{}/run_{}/'.format(papy.globals.run_name,
                                                papy.globals.run_number[-3:])


# create file paths
if os.path.exists('{}'.format(plot_path_0)):
    print('\n {} already exists \n'.format(plot_path_0))
else:
    os.mkdir('{}'.format(plot_path_0))

wt_err = {}
wt_err.fromkeys(namelist)
for name in namelist:
    files = wt.get_files(path,name)
    var_names = ['umean', 'vmean', 'u_var', 'v_var', 'covar', 'lux']    
    wt_err[name] = {}
    wt_err[name].fromkeys(var_names)
    if name[3:5] == 'FL':
        wt_err[name]['umean'] = 0.0192
        wt_err[name]['vmean'] = 0.0085
        wt_err[name]['u_var'] = 0.0085
        wt_err[name]['v_var'] = 0.0030
        wt_err[name]['covar'] = 0.0021
        wt_err[name]['lux'] =   3.6480
    if name[3:5] == 'BR':
        wt_err[name]['umean'] = 0.0165
        wt_err[name]['vmean'] = 0.0076
        wt_err[name]['u_var'] = 0.0051
        wt_err[name]['v_var'] = 0.0034
        wt_err[name]['covar'] = 0.0018
        wt_err[name]['lux'] =   4.4744
    if name[3:5] == 'WB':
        wt_err[name]['umean'] = 0.0195
        wt_err[name]['vmean'] = 0.0069
        wt_err[name]['u_var'] = 0.0052
        wt_err[name]['v_var'] = 0.0029
        wt_err[name]['covar'] = 0.0021
        wt_err[name]['lux'] =   3.5338


outdata_path = '../wt_outdata/'# format in npz

data_nd = 0
# scale of the model. 
scale = 1./0.007
# boundary layer reference
wtref_factor = 0.738 # SI-building experiment

# shift coordinate system
x_val_shift = 0.115*scale
y_val_shift = 0.115*scale
x_val_shift = 0.

# Check if all necessary output directories exist
wt.check_directory(plot_path_0)
wt.check_directory(txt_path)
time_series = {}
time_series.fromkeys(namelist)

# Gather all files into Timeseries objects, save raw timeseries
for name in namelist:
    # prepare environment
    if os.path.exists('{}{}'.format(plot_path_0,name)):
        print('\n {}{} already exists \n'.format(plot_path_0,name))
    else:
        os.mkdir('{}{}'.format(plot_path_0,name))
        os.mkdir('{}{}/spectra/'.format(plot_path_0,name))
        os.mkdir('{}{}/scatter/'.format(plot_path_0,name))
        os.mkdir('{}{}/hist/'.format(plot_path_0,name))

    files = wt.get_files(path,name)
    time_series[name] = {}
    time_series[name].fromkeys(files)  
    for i,file in enumerate(files):
        ts = wt.Timeseries.from_file(path+file)            
        ts.get_wind_comps(path+file)
        ts.get_wtref(wtref_path,name,index=i)
        # edit 6/20/19: Assume that input data is dimensional, not non-dimensional
        if data_nd == 0:
            print('Warning: Assuming that data is dimensional. If using non-dimensional input data, set variable data_nd to 1')
            ts.wtref = ts.wtref * wtref_factor
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
        ts.mean_magnitude
        ts.mean_direction
        ts.calc_direction()
        ts.calc_magnitude()
        # ts.save2file(file)     
        time_series[name][file] = ts

    print('\n created Timeseries object for: {} \n'.format(name))

#check if files-list got created
if files==[]:
   raise Exception('No Matching File Names Found. Please check namelist and/or path!')

##########################
##### vector plot XY #####
##########################
if vectorplot_xy:
    plot_palm_vectors = True
    print('\n     compute vector plot in XY')
    c_list = ['firebrick', 'seagreen', 'orange']
    label_list = ['SB_SI_front', 'SB_SI_back', 'SB_SI_middle']
    if plot_palm_vectors:
        fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
        y_PALM_shift = 530.
        x_PALM_shift = 512.
        for i,run in enumerate(run_list):
            nc_file_path = '../palm/current_version/JOBS/{}/OUTPUT/'.format(run)
            x = []
            y = []
            u_mean = []
            v_mean = []
            mean_uvars = np.array([])
            mean_vvars = np.array([])
            mask_list_cur = mask_list
            if run == 'SB_SI_2':
                mask_list_cur = mask_list_middle
            for j, mask in enumerate(mask_list_cur):
                total_var_u = np.array([])
                total_var_v = np.array([])
                total_time = np.array([])
                for run_no in papy.globals.run_numbers: 
                    nc_file = '{}_masked_{}{}.nc'.format(run, mask, run_no)
                    time, time_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'time')
                    var_u, var_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'u')
                    var_v, var_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'v')
                    yu, yu_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'y')
                    xu, xu_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'x')
                    total_time = np.concatenate([total_time, time])
                    total_var_u = np.concatenate([total_var_u, var_u])
                    total_var_v = np.concatenate([total_var_v, var_v])
                # gather values
                u_mean = np.asarray([np.mean(total_var_u)])
                v_mean = np.asarray([np.mean(total_var_v)])
                mean_uvars = np.concatenate([mean_uvars, u_mean])
                mean_vvars = np.concatenate([mean_vvars, v_mean])
                x.append(xu-x_PALM_shift)
                y.append(yu-y_PALM_shift)

        
            if run == 'SB_SI_front':
                ax.quiver(x, y, mean_uvars, mean_vvars, 
                        color= 'firebrick', 
                        label = 'SB_SI_front',
                        scale = 25.,
                        width=0.003*textwidth_half,)
                        # headlength = 3., headaxislength=3.-0.5)
            elif run == 'SB_SI_back':
                ax.quiver(x, y, mean_uvars, mean_vvars, 
                        color= 'seagreen', 
                        label = 'SB_SI_back',
                        scale = 25.,
                        width=0.003*textwidth_half,)
                        # headlength = 3., headaxislength=3.-0.5)
            else:
                ax.quiver(x, y, mean_uvars, mean_vvars, 
                        color= 'orange', 
                        label = '',
                        scale = 25.,
                        width=0.001*textwidth_half,)
                        # headlength = 6., headaxislength=6.-0.5, 
                        # headwidth=4.)    


        ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                    borderaxespad = 0.,  
                    numpoints = 1)    
        
        ax.set_xlabel(r'$x$ (m)')
        ax.set_ylabel(r'$y$ (m)')

        building = [(-76.5/2., -34.5/2.-y_val_shift), (-76.5/2., 0.), (76.5/2., 0.), (76.5/2, -34.5/2.-y_val_shift)]
        ax.add_patch(patches.Polygon(building,
                    facecolor = 'grey'))
        # ax.hlines(0.0066*150.*5., -76.5/2., 76.5/2., colors='tab:red', 
        #                         linestyles='dashed', linewidth=0.5)
        #                         # , label=r'$5 \cdot h_{r}$')
        ax.set_ylim(-10, 30.)
        ax.set_xlim(-60., 60.)
        plt.savefig(plot_path_0 + 'vectorplot_xy_PALM_' + namelist[0][3:8] + '.' + file_type,
                    dpi=300,bbox_inches='tight')
        plt.savefig(plot_path_0 + 'vectorplot_xy_PALM_' + namelist[0][3:8] + '.' + 'png',
                    dpi=300,bbox_inches='tight')
        print(' saved image to ' + plot_path_0 + 'vectorplot_xy_PALM_' + namelist[0][3:8] + '.' + file_type)
        plt.close()

    # wind tunnel
    fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
    for i,name in enumerate(namelist):
        files = wt.get_files(path,name)
        x = []
        y = []
        u_mean = []
        v_mean = []
        for j, file in enumerate(files):
            # if (j%2) == 0:            
            #     x.append(time_series[name][file].x)
            #     y.append(time_series[name][file].y-y_val_shift)
            #     u_mean.append(time_series[name][file].weighted_component_mean[0])
            #     v_mean.append(time_series[name][file].weighted_component_mean[1])
            x.append(time_series[name][file].x)
            y.append(time_series[name][file].y-y_val_shift)
            u_mean.append(time_series[name][file].weighted_component_mean[0])
            v_mean.append(time_series[name][file].weighted_component_mean[1])
        # ax.quiver(x, y, u_mean, v_mean, 
        #             color=c_list[i], 
        #             label = label_list[i])
    
        if name == 'SB_FL_SI_UV_023' or name == 'SB_BR_SI_UV_012' or name == 'SB_WB_SI_UV_013':
            ax.quiver(x, y, u_mean, v_mean, 
                    color= 'firebrick', 
                    label = 'SB_SI_front',
                    scale = 1.5,)
                    # width=0.003*textwidth_half,
                    # headlength = 3., headaxislength=3.-0.5)
        elif name == 'SB_FL_SI_UV_022' or name == 'SB_BR_SI_UV_011' or name == 'SB_WB_SI_UV_012':
            ax.quiver(x, y, u_mean, v_mean, 
                    color= 'seagreen', 
                    label = 'SB_SI_back',
                    scale = 7.5,)
                    # width=0.003*textwidth_half,
                    # headlength = 3., headaxislength=3.-0.5)
        else:
            ax.quiver(x, y, u_mean, v_mean, 
                    color= 'orange', 
                    label = '',
                    scale = 2.5,)
                    # width=0.001*textwidth_half,
                    # headlength = 6., headaxislength=6.-0.5, 
                    # headwidth=4.)    


    ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                borderaxespad = 0.,  
                numpoints = 1)    
    
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')

    building = [(-76.5/2., -34.5/2.-y_val_shift), (-76.5/2., 0.), (76.5/2., 0.), (76.5/2, -34.5/2.-y_val_shift)]
    ax.add_patch(patches.Polygon(building,
                facecolor = 'grey'))
    ax.hlines(0.0066*150.*5., -76.5/2., 76.5/2., colors='tab:red', 
                            linestyles='dashed', linewidth=0.5)
                            # , label=r'$5 \cdot h_{r}$')
    ax.set_ylim(-10, 30.)
    ax.set_xlim(-60., 60.)
    plt.savefig(plot_path_0 + 'vectorplot_xy_' + namelist[0][3:8] + '.' + file_type,
                dpi=300,bbox_inches='tight')
    print(' saved image to ' + plot_path_0 + 'vectorplot_xy_' + namelist[0][3:8] + '.' + file_type)



##########################
##### vector plot XZ #####
##########################
if vectorplot_xz:
    print('\n     compute vector plot in XZ')    
    c_list = ['firebrick', 'orange', 'orange', 'seagreen', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange']
    label_list = ['SB_LU', '', '', 'SB_LE', '', '', '', '', '', '', '', '']
    fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
    for i,name in enumerate(namelist):
        files = wt.get_files(path,name)
        x = []
        z = []
        u_mean = []
        v_mean = []
        for j, file in enumerate(files):
            # if name == 'SB_FL_LU_UW_003' or name == 'SB_FL_LE_UW_006':
            #     x.append(time_series[name][file].x)
            #     z.append(time_series[name][file].z)
            #     u_mean.append(time_series[name][file].weighted_component_mean[0])
            #     v_mean.append(time_series[name][file].weighted_component_mean[1])
            # else:
            #     if (j%2) == 0:            
            #         x.append(time_series[name][file].x)
            #         z.append(time_series[name][file].z)
            #         u_mean.append(time_series[name][file].weighted_component_mean[0])
            #         v_mean.append(time_series[name][file].weighted_component_mean[1])
            x.append(time_series[name][file].x)
            z.append(time_series[name][file].z)
            u_mean.append(time_series[name][file].weighted_component_mean[0])
            v_mean.append(time_series[name][file].weighted_component_mean[1])

        if name == 'SB_FL_LU_UW_003':
            ax.quiver(x, z, u_mean, v_mean, 
                    color= 'firebrick', 
                    label = 'SB_LU',
                    scale = 7.5,
                    width=0.003*textwidth_half,
                    headlength = 3., headaxislength=3.-0.5)
        elif name == 'SB_FL_LE_UW_006':
            ax.quiver(x, z, u_mean, v_mean, 
                    color= 'seagreen', 
                    label = 'SB_LE',
                    scale = 7.5,
                    width=0.003*textwidth_half,
                    headlength = 3., headaxislength=3.-0.5)
        else:
            ax.quiver(x, z, u_mean, v_mean, 
                    color= 'orange', 
                    label = '',
                    scale = 7.5,
                    width=0.001*textwidth_half,
                    headlength = 6., headaxislength=6.-0.5, 
                    headwidth=4.)    

    ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
                borderaxespad = 0.,  
                numpoints = 1)    
    
    ax.vlines(-34.5/2.-0.0066*150.*5., 0., 50.4, colors='tab:red', 
                            linestyles='dashed', linewidth=0.5)
    ax.vlines(34.5/2.+0.0066*150.*5., 0., 50.4, colors='tab:red', 
                            linestyles='dashed', linewidth=0.5)

    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$z$ (m)')

    building = [(-34./2., 0.), (-34./2., 50.4), (34.5/2., 50.4), (34.5/2., 0.)]
    ax.add_patch(patches.Polygon(building,
                facecolor = 'grey'))
    # ax.hlines(0.0066*150.*5., -76.5/2., 76.5/2., colors='tab:red', 
    #                         linestyles='dashed')
                            # , label=r'$5 \cdot h_{r}$')
    ax.set_ylim(0, 80.)
    ax.set_xlim(-70., 70.)


    plt.savefig(plot_path_0 + 'vectorplot_xz_' + namelist[0][3:8] + '.' + file_type,
                dpi=300,bbox_inches='tight')
    print('     saved image to ' + plot_path_0 + 'vectorplot_xz_' + namelist[0][3:8] + '.' + file_type)

# %%
