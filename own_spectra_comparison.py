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

# from hashlib import shake_128
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
        'lines.markersize' : 1.,
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
        'lines.markersize' : 1.,
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
Steeringflags
"""
################
compute_SI_back_spectra = True
compute_SI_front_spectra = False
compute_LE_up_spectra = False
compute_LE_mid_spectra = False
compute_LU_spectra = False
compute_BL_spectra = False
compute_BA_spectra = False

################
"""
GLOBAL VARIABLES
"""
################
# PALM input files
experiment = 'single_building'
scale = 150.
wtref_factor = 0.738
if compute_SI_back_spectra:
    mask_name_list = ['M02', 'M03', 'M04', 
                        'M06', 'M07', 'M08',
                        'M10', 'M11', 'M12']
    namelist = ['SB_FL_SI_UV_022',
                'SB_BR_SI_UV_011',
                'SB_WB_SI_UV_012']
    FL_file_list = ['SB_FL_SI_UV_022.000009.txt', 'SB_FL_SI_UV_022.000012.txt', 'SB_FL_SI_UV_022.000014.txt',
                    'SB_FL_SI_UV_022.000015.txt', 'SB_FL_SI_UV_022.000016.txt', 'SB_FL_SI_UV_022.000017.txt',
                    'SB_FL_SI_UV_022.000019.txt', 'SB_FL_SI_UV_022.000020.txt', 'SB_FL_SI_UV_022.000022.txt']
    BR_file_list = ['SB_BR_SI_UV_011.000004.txt', 'SB_BR_SI_UV_011.000007.txt', 'SB_BR_SI_UV_011.000009.txt',
                    'SB_BR_SI_UV_011.000010.txt', 'SB_BR_SI_UV_011.000011.txt', 'SB_BR_SI_UV_011.000012.txt',
                    'SB_BR_SI_UV_011.000014.txt', 'SB_BR_SI_UV_011.000015.txt', 'SB_BR_SI_UV_011.000017.txt']
    WB_file_list = ['SB_WB_SI_UV_012.000004.txt', 'SB_WB_SI_UV_012.000007.txt', 'SB_WB_SI_UV_012.000009.txt',
                    'SB_WB_SI_UV_012.000010.txt', 'SB_WB_SI_UV_012.000011.txt', 'SB_WB_SI_UV_012.000012.txt',
                    'SB_WB_SI_UV_012.000014.txt', 'SB_WB_SI_UV_012.000015.txt', 'SB_WB_SI_UV_012.000017.txt']
    papy.globals.run_name = 'SB_SI_back'
    papy.globals.run_numbers = ['.007', '.008', '.009', '.010', '.011', '.012', 
                                '.013', '.014', '.015', '.016', '.017', '.018',
                                '.019', '.020', '.021', '.022', '.023', '.024',
                                '.025', '.026', '.027', '.028', '.029', '.030', 
                                '.031', '.032', '.033', '.034', '.035', '.036',
                                '.037', '.038', '.039', '.040', '.041', '.042',
                                '.043', '.044', '.045', '.046']                    
if compute_SI_front_spectra:
    mask_name_list = ['M02', 'M03', 'M04', 
                    'M06', 'M07', 'M08',
                    'M10', 'M11', 'M12']                     
    namelist = ['SB_FL_SI_UV_023',
                'SB_BR_SI_UV_012',
                'SB_WB_SI_UV_013']
    FL_file_list = ['SB_FL_SI_UV_023.000009.txt', 'SB_FL_SI_UV_023.000012.txt', 'SB_FL_SI_UV_023.000014.txt',
                    'SB_FL_SI_UV_023.000015.txt', 'SB_FL_SI_UV_023.000016.txt', 'SB_FL_SI_UV_023.000017.txt',
                    'SB_FL_SI_UV_023.000019.txt', 'SB_FL_SI_UV_023.000020.txt', 'SB_FL_SI_UV_023.000022.txt']
    BR_file_list = ['SB_BR_SI_UV_012.000004.txt', 'SB_BR_SI_UV_012.000007.txt', 'SB_BR_SI_UV_012.000009.txt',
                    'SB_BR_SI_UV_012.000010.txt', 'SB_BR_SI_UV_012.000011.txt', 'SB_BR_SI_UV_012.000012.txt',
                    'SB_BR_SI_UV_012.000014.txt', 'SB_BR_SI_UV_012.000015.txt', 'SB_BR_SI_UV_012.000017.txt']
    WB_file_list = ['SB_WB_SI_UV_013.000004.txt', 'SB_WB_SI_UV_013.000007.txt', 'SB_WB_SI_UV_013.000009.txt',
                    'SB_WB_SI_UV_013.000010.txt', 'SB_WB_SI_UV_013.000011.txt', 'SB_WB_SI_UV_013.000012.txt',
                    'SB_WB_SI_UV_013.000014.txt', 'SB_WB_SI_UV_013.000015.txt', 'SB_WB_SI_UV_013.000017.txt']
    papy.globals.run_name = 'SB_SI_front'
    papy.globals.run_numbers = ['.007', '.008', '.009', '.010', '.011', '.012', 
                                '.013', '.014', '.015', '.016', '.017', '.018',
                                '.019', '.020', '.021', '.022', '.023', '.024',
                                '.025', '.026', '.027', '.028', '.029', '.030', 
                                '.031', '.032', '.033', '.034', '.035', '.036',
                                '.037', '.038', '.039', '.040', '.041', '.042',
                                '.043', '.044', '.045', '.046']
if compute_LE_mid_spectra:
    mask_name_list = ['M02', 'M03', 'M05', 'M06', 
                    'M08', 'M10', 'M11', 'M12']
    namelist = ['SB_FL_LE_UW_006',
                'SB_BR_LE_UW_006',
                'SB_WB_LE_UW_005'] 
    FL_file_list = ['SB_FL_LE_UW_006.000011.txt', 'SB_FL_LE_UW_006.000014.txt', 'SB_FL_LE_UW_006.000015.txt',
                    'SB_FL_LE_UW_006.000016.txt', 'SB_FL_LE_UW_006.000017.txt', 'SB_FL_LE_UW_006.000018.txt',
                    'SB_FL_LE_UW_006.000019.txt', 'SB_FL_LE_UW_006.000020.txt']
    BR_file_list = ['SB_BR_LE_UW_006.000006.txt', 'SB_BR_LE_UW_006.000009.txt', 'SB_BR_LE_UW_006.000010.txt',
                    'SB_BR_LE_UW_006.000011.txt', 'SB_BR_LE_UW_006.000012.txt', 'SB_BR_LE_UW_006.000013.txt',
                    'SB_BR_LE_UW_006.000014.txt', 'SB_BR_LE_UW_006.000015.txt']
    WB_file_list = ['SB_WB_LE_UW_005.000007.txt', 'SB_WB_LE_UW_005.000010.txt', 'SB_WB_LE_UW_005.000011.txt',
                    'SB_WB_LE_UW_005.000012.txt', 'SB_WB_LE_UW_005.000013.txt', 'SB_WB_LE_UW_005.000014.txt',
                    'SB_WB_LE_UW_005.000015.txt', 'SB_WB_LE_UW_005.000016.txt']
    papy.globals.run_name = 'SB_LE'
    papy.globals.run_numbers = ['.008', '.009', '.010', '.011', '.012', 
                                '.013', '.014', '.015', '.016', '.017', '.018',
                                '.019', '.020', '.021', '.022', '.023', '.024',
                                '.025', '.026', '.027', '.028', '.029', '.030',
                                '.031', '.032', '.033', '.034', '.035', '.036',
                                '.037', '.038', '.039', '.040', '.041', '.041',
                                '.042', '.043', '.044', '.045', '.046', '.047',
                                '.048', '.049', '.050']
if compute_LE_up_spectra: 
    mask_name_list = ['M14', 'M15', 'M17', 'M18', 
                    'M20', 'M22', 'M23', 'M24']
    namelist = ['SB_FL_LE_UW_007',
                'SB_BR_LE_UW_007',
                'SB_WB_LE_UW_006'] 
    FL_file_list = ['SB_FL_LE_UW_007.000011.txt', 'SB_FL_LE_UW_007.000014.txt', 'SB_FL_LE_UW_007.000015.txt',
                    'SB_FL_LE_UW_007.000016.txt', 'SB_FL_LE_UW_007.000017.txt', 'SB_FL_LE_UW_007.000018.txt',
                    'SB_FL_LE_UW_007.000019.txt', 'SB_FL_LE_UW_007.000020.txt']
    BR_file_list = ['SB_BR_LE_UW_007.000006.txt', 'SB_BR_LE_UW_007.000009.txt', 'SB_BR_LE_UW_007.000010.txt',
                    'SB_BR_LE_UW_007.000011.txt', 'SB_BR_LE_UW_007.000012.txt', 'SB_BR_LE_UW_007.000013.txt',
                    'SB_BR_LE_UW_007.000014.txt', 'SB_BR_LE_UW_007.000015.txt']
    WB_file_list = ['SB_WB_LE_UW_006.000007.txt', 'SB_WB_LE_UW_006.000010.txt', 'SB_WB_LE_UW_006.000011.txt',
                    'SB_WB_LE_UW_006.000012.txt', 'SB_WB_LE_UW_006.000013.txt', 'SB_WB_LE_UW_006.000014.txt',
                    'SB_WB_LE_UW_006.000015.txt', 'SB_WB_LE_UW_006.000016.txt']
    papy.globals.run_name = 'SB_LE'
    papy.globals.run_numbers = ['.008', '.009', '.010', '.011', '.012', 
                                '.013', '.014', '.015', '.016', '.017', '.018',
                                '.019', '.020', '.021', '.022', '.023', '.024',
                                '.025', '.026', '.027', '.028', '.029', '.030',
                                '.031', '.032', '.033', '.034', '.035', '.036',
                                '.037', '.038', '.039', '.040', '.041', '.041',
                                '.042', '.043', '.044', '.045', '.046', '.047',
                                '.048', '.049', '.050']                    
if compute_LU_spectra:
    mask_name_list = ['M03', 'M04', 'M05', 'M07', 
                    'M09', 'M10', 'M11', 'M12']
    namelist = ['SB_FL_LU_UW_003',
                'SB_BR_LU_UW_001',
                'SB_WB_LU_UW_001']
    FL_file_list = ['SB_FL_LU_UW_003.000008.txt', 'SB_FL_LU_UW_003.000010.txt', 'SB_FL_LU_UW_003.000011.txt',
                    'SB_FL_LU_UW_003.000012.txt', 'SB_FL_LU_UW_003.000013.txt', 'SB_FL_LU_UW_003.000014.txt',
                    'SB_FL_LU_UW_003.000016.txt', 'SB_FL_LU_UW_003.000018.txt']
    BR_file_list = ['SB_BR_LU_UW_001.000003.txt', 'SB_BR_LU_UW_001.000005.txt', 'SB_BR_LU_UW_001.000006.txt',
                    'SB_BR_LU_UW_001.000007.txt', 'SB_BR_LU_UW_001.000008.txt', 'SB_BR_LU_UW_001.000009.txt',
                    'SB_BR_LU_UW_001.000011.txt', 'SB_BR_LU_UW_001.000013.txt']
    WB_file_list = ['SB_WB_LU_UW_001.000003.txt', 'SB_WB_LU_UW_001.000005.txt', 'SB_WB_LU_UW_001.000006.txt',
                    'SB_WB_LU_UW_001.000007.txt', 'SB_WB_LU_UW_001.000008.txt', 'SB_WB_LU_UW_001.000009.txt',
                    'SB_WB_LU_UW_001.000011.txt', 'SB_WB_LU_UW_001.000013.txt']
    papy.globals.run_name = 'SB_LU'
    papy.globals.run_numbers = ['.008', '.009', '.010', '.011', '.012', 
                                '.013', '.014', '.015', '.016', '.017', '.018',
                                '.019', '.020', '.021', '.022', '.023', '.024',
                                '.025', '.026', '.027', '.028', '.029', '.030',
                                '.031', '.032', '.033', '.034', '.035', '.036',
                                '.037', '.038', '.039', '.040', '.041', '.041',
                                '.042', '.043', '.044', '.045', '.046', '.047']
if compute_BL_spectra:
    mask_name_list = ['M04', 'M05', 'M06', 
                        'M07', 'M08', 'M09', 
                        'M10', 'M11', 'M12']
    experiment = 'balcony'
    namelist = ['BA_BL_UW_001']
    BL_file_list = ['BA_BL_UW_001.000001.txt', 'BA_BL_UW_001.000005.txt', 'BA_BL_UW_001.000008.txt', 
                    'BA_BL_UW_001.000010.txt', 'BA_BL_UW_001.000011.txt', 'BA_BL_UW_001.000014.txt', 
                    'BA_BL_UW_001.000016.txt', 'BA_BL_UW_001.000018.txt', 'BA_BL_UW_001.000023.txt']
    papy.globals.run_name = 'SB_SI_BL'
    papy.globals.run_numbers = ['.007', '.008', '.009', '.010', '.011', '.012', 
                                '.013', '.014', '.015', '.016', '.017', '.018',
                                '.019', '.020', '.021', '.022', '.023', '.024',
                                '.025', '.026', '.027', '.028', '.029', '.030', 
                                '.031', '.032', '.033', '.034', '.035', '.036',
                                '.037', '.038', '.039', '.040', '.041', '.042',
                                '.043', '.044', '.045', '.046', '.047']
if compute_BA_spectra:
    # balcony
    experiment = 'balcony'
    namelist = ['BA_BL_UW_001',
                'BA_S5_L_UW_002',
                'BA_S8_L_UW_007',
                'BA_BL_UW_010']
    BL_file_list = ['BA_BL_UW_001.000001.txt', 'BA_BL_UW_001.000006.txt', 'BA_BL_UW_001.000010.txt',
                    'BA_BL_UW_001.000016.txt', 'BA_BL_UW_001.000023.txt']
    S5_file_list = ['BA_S5_L_UW_002.000006.txt', 'BA_S5_L_UW_002.000011.txt', 'BA_S5_L_UW_002.000015.txt',
                    'BA_S5_L_UW_002.000021.txt', 'BA_S5_L_UW_002.000025.txt']
    S8_file_list = ['BA_S8_L_UW_007.000005.txt', 'BA_S8_L_UW_007.000012.txt', 'BA_S8_L_UW_007.000016.txt',
                    'BA_S8_L_UW_007.000017.txt', 'BA_S8_L_UW_007.000019.txt']
    FL_file_list = ['BA_BL_UW_010.000006.txt', 'BA_BL_UW_010.000011.txt', 'BA_BL_UW_010.000015.txt',
                    'BA_BL_UW_010.000021.txt', 'BA_BL_UW_010.000025.txt']
    papy.globals.run_name = 'Germano_BA_BL_z0_021_wtprof'
    papy.globals.run_numbers = ['.000', '.001']
    papy.globals.run_name = 'BA_BL_z0_021_wtprof'
    papy.globals.run_numbers = ['.000', '.002', '.003', '.004', '.005']
    # 5, 7.5, 10.,  30, 70
    mask_name_list = ['M03', 'M04', 'M05', 
                      'M09', 'M15']
    scale = 100.
    wtref_factor = 0.731

papy.globals.run_number = papy.globals.run_numbers[-1]
print('Analyze PALM-run up to: ' + papy.globals.run_number)
nc_file_grid = '{}_pr{}.nc'.format(papy.globals.run_name,papy.globals.run_number)
nc_file_path = '../palm/current_version/JOBS/{}/OUTPUT/'.format(papy.globals.run_name)

# WIND TUNNEL INPIUT FILES
# wt_filename = 'BA_BL_UW_001'
wt_path = '../../Documents/phd/experiments/{}/{}'.format(experiment, 'CO_REF')

# PHYSICS and NUMERICS
papy.globals.z0 = 0.03
papy.globals.z0_wt = 0.071
papy.globals.alpha = 0.18
papy.globals.ka = 0.41
papy.globals.d0 = 0.
papy.globals.nx = 1024
papy.globals.ny = 1024
papy.globals.dx = 1.

################
"""
MAIN
"""
################

# prepare the outputfolders
papy.prepare_plotfolder(papy.globals.run_name,papy.globals.run_number)

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
if data_nd == 0:
    palm_ref = np.mean(total_palm_u)
elif data_nd == 1:
    palm_ref = 1.
print('     PALM REFERENCE VELOCITY: {} m/s \n'.format(palm_ref))

config = 'CO_REF'
path = '{}/coincidence/timeseries/'.format(wt_path) # path to timeseries folder
wtref_path = '{}/wtref/'.format(wt_path)

wt_err = {}
wt_err.fromkeys(namelist)
for name in namelist:
    files = wt.get_files(path,name)
    var_names = ['umean', 'vmean', 'u_var', 'v_var', 'covar', 'lux']    
    wt_err[name] = {}
    wt_err[name].fromkeys(var_names)
    if name[3:5] == 'FL':
        wt_err[name]['umean'] = [0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 0.0395, 
                                0.0395, 0.0395, 0.0395, 0.0217, 0.0217, 0.0217, 0.0167, 0.0167, 0.0229, 0.0229, 0.0229, 0.0173]
        wt_err[name]['vmean'] = [0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 0.0107, 
                                0.0107, 0.0107, 0.0107, 0.0101, 0.0101, 0.0101, 0.0152, 0.0152, 0.0081, 0.0081, 0.0081, 0.008]
        wt_err[name]['u_var'] = [0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 
                                0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0039, 0.0039, 0.0047, 0.0047, 0.0047, 0.0006]
        wt_err[name]['v_var'] = [0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 
                                0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0039, 0.0039, 0.0047, 0.0047, 0.0047, 0.0006]
        wt_err[name]['covar'] = [0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 
                                0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0039, 0.0039, 0.0047, 0.0047, 0.0047, 0.0006]
        wt_err[name]['lux'] =   [3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 3.1814, 
                                3.1814, 3.1814, 3.1814, 1.5144, 1.5144, 1.5144, 2.9411, 2.9411, 2.2647, 2.2647, 2.2647, 26.5786]
    if name[3:5] == 'BR':
        wt_err[name]['umean'] = [0.0255, 0.0255, 0.0255, 0.0255, 0.0255, 0.0255, 0.0255, 0.0255, 0.0465, 0.0465, 
                                0.0465, 0.0292, 0.0292, 0.0179, 0.0179, 0.0179, 0.0202]
        wt_err[name]['vmean'] = [0.0156, 0.0156, 0.0156, 0.0156, 0.0156, 0.0156, 0.0156, 0.0156, 0.0116, 0.0116, 
                                0.0116, 0.0101, 0.0101, 0.0114, 0.0114, 0.0114, 0.0073]
        wt_err[name]['u_var'] = [0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 
                                0.0029, 0.0048, 0.0048, 0.0037, 0.0037, 0.0037, 0.0007]
        wt_err[name]['v_var'] = [0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 
                                0.0029, 0.0048, 0.0048, 0.0037, 0.0037, 0.0037, 0.0007]
        wt_err[name]['covar'] = [0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 
                                0.0029, 0.0048, 0.0048, 0.0037, 0.0037, 0.0037, 0.0007]
        wt_err[name]['lux'] =   [2.6852, 2.6852, 2.6852, 2.6852, 2.6852, 2.6852, 2.6852, 2.6852, 3.3587, 3.3587, 
                                3.3587, 1.9594, 1.9594, 4.7631, 4.7631, 4.7631, 22.5726]
    if name[3:5] == 'WB':
        wt_err[name]['umean'] = [0.0171, 0.0171, 0.0171, 0.0171, 0.0171, 0.0171, 0.0171, 0.0171, 0.0245, 0.0245, 
                                0.0245, 0.0335, 0.0335, 0.0175, 0.0175, 0.0175, 0.0202]
        wt_err[name]['vmean'] = [0.0133, 0.0133, 0.0133, 0.0133, 0.0133, 0.0133, 0.0133, 0.0133, 0.016, 0.016, 
                                0.016, 0.0106, 0.0106, 0.007, 0.007, 0.007, 0.0006]
        wt_err[name]['u_var'] = [0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0028, 0.0028, 
                                0.0028, 0.004, 0.004, 0.0029, 0.0029, 0.0029, 0.0008]
        wt_err[name]['v_var'] = [0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0028, 0.0028, 
                                0.0028, 0.004, 0.004, 0.0029, 0.0029, 0.0029, 0.0008]
        wt_err[name]['covar'] = [0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0028, 0.0028, 
                                0.0028, 0.004, 0.004, 0.0029, 0.0029, 0.0029, 0.0008]
        wt_err[name]['lux'] =   [1.9007, 1.9007, 1.9007, 1.9007, 1.9007, 1.9007, 1.9007, 1.9007, 2.2369, 2.2369, 
                                2.2369, 4.4863, 4.4863, 2.6004, 2.6004, 2.6004, 33.5205]

# data_nd = 1
print('\n   READ WT-Data')
time_series = {}
time_series.fromkeys(namelist)
time_series_eq = {}
time_series_eq.fromkeys(namelist)
# # Gather all files into Timeseries objects
for name in namelist:
    files = wt.get_files(path,name)
    time_series[name] = {}
    time_series[name].fromkeys(files)
    time_series_eq[name] = {}
    time_series_eq[name].fromkeys(files)    
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
        ts_eq = ts
        ts_eq.calc_equidistant_timesteps()  
        ts.index = ts.t_arr         
        ts.weighted_component_mean
        ts_eq.weighted_component_mean
        ts.weighted_component_variance
        ts_eq.weighted_component_variance
        ts.mean_magnitude
        ts_eq.mean_magnitude
        ts.mean_direction
        ts_eq.mean_direction
        # ts.save2file(file)     
        time_series[name][file] = ts
        time_series_eq[name][file] = ts_eq

# plotting colors and markers
c_list = ['forestgreen', 'darkorange', 'navy', 'tab:red', 'tab:olive', 'cyan']
marker_list = ['^', 'o', 'd', 'x', '8', '<']
label_list = ['flat facade', 'rough facade', 'medium rough facade']
# label_list = namelist

######################################################
# Compute spectra
######################################################
if compute_SI_back_spectra or compute_SI_front_spectra:
    # heights mode
    print('\n Compute at different walldists: \n')
    y_val_shift = 0.115*scale
    var_name_list = ['u', 'v', 'uv']
    for i,mask in enumerate(mask_name_list):
        total_var_u = np.array([])
        total_var_v = np.array([])        
        total_time = np.array([])
        for run_no in papy.globals.run_numbers:
            nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask, run_no)
            time, time_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'time')
            var_u, var_u_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'u')
            var_v, var_v_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'v')                
            y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'y')
            total_time = np.concatenate([total_time, time])
            total_var_u = np.concatenate([total_var_u, var_u])
            total_var_v = np.concatenate([total_var_v, var_v])                
        # gather values 
        wall_dist = np.asarray([abs(y[0]-530.)])
        print('\n HEIGHT = {} m'.format(wall_dist[0]))
        # equidistant timestepping
        time_eq = np.linspace(total_time[0], total_time[-1], len(total_time))
        var_u_eq = wt.equ_dist_ts(total_time, time_eq, total_var_u/palm_ref)
        var_v_eq = wt.equ_dist_ts(total_time, time_eq, total_var_v/palm_ref)
        scaling = 'u_hor' # u_hor, u_mean or v_mean
        f_sm, S_uu_sm, S_vv_sm, S_uv_sm, u_aliasing, v_aliasing, uv_aliasing = wt.calc_spectra(
                            var_u_eq,
                            var_v_eq,
                            total_time,
                            wall_dist[0], scaling)
                            
        FL_name = namelist[0]
        fl_f_sm, fl_S_uu_sm, fl_S_vv_sm, fl_S_uv_sm, fl_u_aliasing, fl_v_aliasing, fl_uv_aliasing = wt.calc_spectra(
                            time_series_eq[FL_name][FL_file_list[i]].u_eq.dropna(),
                            time_series_eq[FL_name][FL_file_list[i]].v_eq.dropna(),
                            time_series_eq[FL_name][FL_file_list[i]].t_eq[~np.isnan(time_series_eq[FL_name][FL_file_list[i]].t_eq)],
                            # wall_dist[0], scaling)
                            time_series_eq[FL_name][FL_file_list[i]].y-y_val_shift, scaling)
        BR_name = namelist[1]
        br_f_sm, br_S_uu_sm, br_S_vv_sm, br_S_uv_sm, br_u_aliasing, br_v_aliasing, br_uv_aliasing = wt.calc_spectra(
                            time_series_eq[BR_name][BR_file_list[i]].u_eq.dropna(),
                            time_series_eq[BR_name][BR_file_list[i]].v_eq.dropna(),
                            time_series_eq[BR_name][BR_file_list[i]].t_eq[~np.isnan(time_series_eq[BR_name][BR_file_list[i]].t_eq)],
                            # wall_dist[0], scaling)
                            time_series_eq[BR_name][BR_file_list[i]].y-y_val_shift, scaling)
        WB_name = namelist[2]
        wb_f_sm, wb_S_uu_sm, wb_S_vv_sm, wb_S_uv_sm, wb_u_aliasing, wb_v_aliasing, wb_uv_aliasing = wt.calc_spectra(
                            time_series_eq[WB_name][WB_file_list[i]].u_eq.dropna(),
                            time_series_eq[WB_name][WB_file_list[i]].v_eq.dropna(),
                            time_series_eq[WB_name][WB_file_list[i]].t_eq[~np.isnan(time_series_eq[WB_name][WB_file_list[i]].t_eq)],
                            # wall_dist[0], scaling)
                            time_series_eq[WB_name][WB_file_list[i]].y-y_val_shift, scaling)

        print('    calculated spectra for {}'.format(mask))

        for var_name in var_name_list:
            fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))         
            if var_name == 'u':
                # palm
                h1 = ax.loglog(f_sm[:u_aliasing], S_uu_sm[:u_aliasing], 'x', color='darkmagenta', 
                            label=r'PALM'.format(wall_dist[0]))
                h2 = ax.loglog(f_sm[u_aliasing:], S_uu_sm[u_aliasing:], 'x', color='darkmagenta', fillstyle='none'
                            )
                # FL
                fl_h1 = ax.loglog(fl_f_sm[:fl_u_aliasing], fl_S_uu_sm[:fl_u_aliasing], '^', color='forestgreen', 
                            label=r'FL'.format(str(time_series_eq[FL_name][FL_file_list[i]].y-y_val_shift)[0:4]))
                fl_h2 = ax.loglog(fl_f_sm[fl_u_aliasing:], fl_S_uu_sm[fl_u_aliasing:], '^', color='forestgreen', fillstyle='none' 
                            )
                # BR
                br_h1 = ax.loglog(br_f_sm[:br_u_aliasing], br_S_uu_sm[:br_u_aliasing], 'o', color='darkorange', 
                            label=r'BR'.format(str(time_series_eq[BR_name][BR_file_list[i]].y-y_val_shift)[0:4]))
                br_h2 = ax.loglog(br_f_sm[br_u_aliasing:], br_S_uu_sm[br_u_aliasing:], 'o', color='darkorange', fillstyle='none' 
                            )
                # WB
                wb_h1 = ax.loglog(wb_f_sm[:wb_u_aliasing], wb_S_uu_sm[:wb_u_aliasing], 'd', color='navy', 
                            label=r'WB'.format(str(time_series_eq[WB_name][WB_file_list[i]].y-y_val_shift)[0:4]))
                wb_h2 = ax.loglog(wb_f_sm[wb_u_aliasing:], wb_S_uu_sm[wb_u_aliasing:], 'd', color='navy', fillstyle='none'
                            )
                ax.set_ylabel(r"$f\cdot S_{uu}\cdot (\sigma_u \sigma_u)^{-1}$")
            elif var_name == 'v':
                h1 = ax.loglog(f_sm[:v_aliasing], S_vv_sm[:v_aliasing], 'x', color='darkmagenta', 
                            label=r'PALM'.format(wall_dist[0]))
                h2 = ax.loglog(f_sm[v_aliasing:], S_vv_sm[v_aliasing:], 'x', color='darkmagenta', fillstyle='none' 
                            )
                # FL
                fl_h1 = ax.loglog(fl_f_sm[:fl_v_aliasing], fl_S_vv_sm[:fl_v_aliasing], '^', color='forestgreen', 
                            label=r'FL'.format(str(time_series_eq[FL_name][FL_file_list[i]].y-y_val_shift)[0:4]))
                fl_h2 = ax.loglog(fl_f_sm[fl_v_aliasing:], fl_S_vv_sm[fl_v_aliasing:], '^', color='forestgreen', fillstyle='none' 
                            )
                # BR
                br_h1 = ax.loglog(br_f_sm[:br_v_aliasing], br_S_vv_sm[:br_v_aliasing], 'o', color='darkorange', 
                            label=r'BR'.format(str(time_series_eq[BR_name][BR_file_list[i]].y-y_val_shift)[0:4]))
                br_h2 = ax.loglog(br_f_sm[br_v_aliasing:], br_S_vv_sm[br_v_aliasing:], 'o', color='darkorange', fillstyle='none' 
                            )
                # WB
                wb_h1 = ax.loglog(wb_f_sm[:wb_v_aliasing], wb_S_vv_sm[:wb_v_aliasing], 'd', color='navy', 
                            label=r'WB'.format(str(time_series_eq[WB_name][WB_file_list[i]].y-y_val_shift)[0:4]))
                wb_h2 = ax.loglog(wb_f_sm[wb_v_aliasing:], wb_S_vv_sm[wb_v_aliasing:], 'd', color='navy', fillstyle='none' 
                            )
                ax.set_ylabel(r"$f\cdot S_{vv}\cdot (\sigma_v \sigma_v)^{-1}$")
            elif var_name == 'uv':
                h1 = ax.loglog(f_sm[:uv_aliasing], S_uv_sm[:uv_aliasing], 'x', color='darkmagenta', 
                            label=r'PALM'.format(wall_dist[0]))
                h2 = ax.loglog(f_sm[uv_aliasing:], S_uv_sm[uv_aliasing:], 'x', color='darkmagenta', fillstyle='none')
                # FL
                fl_h1 = ax.loglog(fl_f_sm[:fl_uv_aliasing], fl_S_uv_sm[:fl_uv_aliasing], '^', color='forestgreen', 
                            label=r'FL'.format(str(time_series_eq[FL_name][FL_file_list[i]].y-y_val_shift)[0:4]))
                fl_h2 = ax.loglog(fl_f_sm[fl_uv_aliasing:], fl_S_uv_sm[fl_uv_aliasing:], '^', color='lightgreen', fillstyle='none')
                # BR
                br_h1 = ax.loglog(br_f_sm[:br_uv_aliasing], br_S_uv_sm[:br_uv_aliasing], 'o', color='darkorange', 
                            label=r'BR'.format(str(time_series_eq[BR_name][BR_file_list[i]].y-y_val_shift)[0:4]))
                br_h2 = ax.loglog(br_f_sm[br_uv_aliasing:], br_S_uv_sm[br_uv_aliasing:], 'o', color='moccasin', fillstyle='none')
                # WB
                wb_h1 = ax.loglog(wb_f_sm[:wb_uv_aliasing], wb_S_uv_sm[:wb_uv_aliasing], 'd', color='navy', 
                            label=r'WB'.format(str(time_series_eq[WB_name][WB_file_list[i]].y-y_val_shift)[0:4]))
                wb_h2 = ax.loglog(wb_f_sm[wb_uv_aliasing:], wb_S_uv_sm[wb_uv_aliasing:], 'd', color='lightblue', fillstyle='none')
                # if mask == 'M02' or mask == 'M07':
                ax.set_ylabel(r"$f\cdot S_{uv}\cdot (\sigma_u \sigma_v)^{-1}$")
            
            # ax.vlines(1., 10**(-3.), 10**0., 
            #           label=r'$\Delta_i$', color='darkorchid', linestyle='-.')
            # ax.vlines(np.mean(var_u_eq)**(-1) * 1/(total_time[1]-total_time[0])/5., 10**(-3.), 10**0., 
            #           label=r'$5 \cdot \Delta_i$', color='violet', linestyle=':')
            # slopey
            freq_slopey = np.logspace(np.log10(5*10.**(-2.)), np.log10(10.), num=20)
            twothird_slopey = 0.01*(freq_slopey)**(-2./3.)
            ax.plot(freq_slopey, twothird_slopey , label=r'K41: $-2/3$', color='black', linestyle='dashed')
            ax.set_title(r'$\Delta y = {}$ m'.format(wall_dist[0]))
            ax.set_xlim(10.**-3., 10.**2.)
            ax.set_ylim(10.**-3., 1.)
            # if mask == 'M07' or mask == 'M10':
            if scaling == 'u_mean':
                ax.set_xlabel(r"$f\cdot \Delta y\cdot \overline{u}^{-1}$")
            elif scaling == 'u_hor':
                ax.set_xlabel(r"$f\cdot \Delta y\cdot \overline{u_{h}}^{-1}$")
            elif scaling == 'v_mean':
                ax.set_xlabel(r"$f\cdot \Delta y\cdot \overline{v}^{-1}$")
            else:
                ax.set_xlabel(r"$f\cdot \Delta y\cdot \overline{u}^{-1}$")

            if mask == mask_name_list[0]:
                ax.legend(bbox_to_anchor = (-0.1, 1.22), loc = 'lower left', 
                    borderaxespad = 0., ncol = 5, 
                    numpoints = 1)
            ax.grid(False)

            plt.savefig('../palm_results/{}/run_{}/spectra/{}_{}_spectra{}_{}.{}'.format(papy.globals.run_name, papy.globals.run_number[-3:],
                        papy.globals.run_name, var_name, mask, scaling, plotformat), bbox_inches='tight')
            print('     SAVED TO: ' 
            + '../palm_results/{}/run_{}/spectra/{}_{}_spectra_{}_{}.{}'.format(papy.globals.run_name, papy.globals.run_number[-3:],
                        papy.globals.run_name, var_name, mask, scaling, plotformat))


if compute_LE_mid_spectra or compute_LE_up_spectra:
    # heights mode
    print('\n Compute at different walldists: \n')
    y_val_shift = 0.115*scale
    var_name_list = ['u', 'v', 'uv']
    for i,mask in enumerate(mask_name_list):
        total_var_u = np.array([])
        total_var_v = np.array([])        
        total_time = np.array([])
        for run_no in papy.globals.run_numbers:
            nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask, run_no)
            time, time_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'time')
            var_u, var_u_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'u')
            var_v, var_v_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'w')                
            y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'x')
            total_time = np.concatenate([total_time, time])
            total_var_u = np.concatenate([total_var_u, var_u])
            total_var_v = np.concatenate([total_var_v, var_v])                
        # gather values         
        wall_dist = np.asarray([abs(y[0]-530.)])
        print('\n HEIGHT = {} m'.format(wall_dist[0]))
        # equidistant timestepping
        time_eq = np.linspace(total_time[0], total_time[-1], len(total_time))
        var_u_eq = wt.equ_dist_ts(total_time, time_eq, total_var_u/palm_ref)
        var_v_eq = wt.equ_dist_ts(total_time, time_eq, total_var_v/palm_ref)        
        scaling = 'u_hor' # u_hor, u_mean or v_mean
        f_sm, S_uu_sm, S_vv_sm, S_uv_sm, u_aliasing, v_aliasing, uv_aliasing = wt.calc_spectra(var_u_eq,
                            var_v_eq,
                            total_time,
                            wall_dist[0], scaling)
                            
        FL_name = namelist[0]
        fl_f_sm, fl_S_uu_sm, fl_S_vv_sm, fl_S_uv_sm, fl_u_aliasing, fl_v_aliasing, fl_uv_aliasing = wt.calc_spectra(
                            time_series_eq[FL_name][FL_file_list[i]].u_eq.dropna(),
                            time_series_eq[FL_name][FL_file_list[i]].v_eq.dropna(),
                            time_series_eq[FL_name][FL_file_list[i]].t_eq[~np.isnan(time_series_eq[FL_name][FL_file_list[i]].t_eq)],
                            time_series_eq[FL_name][FL_file_list[i]].x-y_val_shift, scaling)
        BR_name = namelist[1]
        br_f_sm, br_S_uu_sm, br_S_vv_sm, br_S_uv_sm, br_u_aliasing, br_v_aliasing, br_uv_aliasing = wt.calc_spectra(
                            time_series_eq[BR_name][BR_file_list[i]].u_eq.dropna(),
                            time_series_eq[BR_name][BR_file_list[i]].v_eq.dropna(),
                            time_series_eq[BR_name][BR_file_list[i]].t_eq[~np.isnan(time_series_eq[BR_name][BR_file_list[i]].t_eq)],
                            time_series_eq[BR_name][BR_file_list[i]].x-y_val_shift, scaling)
        WB_name = namelist[2]
        wb_f_sm, wb_S_uu_sm, wb_S_vv_sm, wb_S_uv_sm, wb_u_aliasing, wb_v_aliasing, wb_uv_aliasing = wt.calc_spectra(
                            time_series_eq[WB_name][WB_file_list[i]].u_eq.dropna(),
                            time_series_eq[WB_name][WB_file_list[i]].v_eq.dropna(),
                            time_series_eq[WB_name][WB_file_list[i]].t_eq[~np.isnan(time_series_eq[WB_name][WB_file_list[i]].t_eq)],
                            time_series_eq[WB_name][WB_file_list[i]].x-y_val_shift, scaling)

        print('    calculated spectra for {}'.format(mask))

        for var_name in var_name_list:
            fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))
            if var_name == 'u':
                # palm
                h1 = ax.loglog(f_sm[:u_aliasing], S_uu_sm[:u_aliasing], 'x', color='darkmagenta', 
                            label=r'PALM - $\Delta x = {}$ m'.format(wall_dist[0]))
                h2 = ax.loglog(f_sm[u_aliasing:], S_uu_sm[u_aliasing:], 'x',
                            )
                # FL
                fl_h1 = ax.loglog(fl_f_sm[:fl_u_aliasing], fl_S_uu_sm[:fl_u_aliasing], '^', color='forestgreen', 
                            label=r'FL: $\Delta x = {}$ m'.format(str(time_series_eq[FL_name][FL_file_list[i]].x-y_val_shift)[0:4]))
                fl_h2 = ax.loglog(fl_f_sm[fl_u_aliasing:], fl_S_uu_sm[fl_u_aliasing:], '^', color='forestgreen', 
                            )
                # BR
                br_h1 = ax.loglog(br_f_sm[:br_u_aliasing], br_S_uu_sm[:br_u_aliasing], 'o', color='darkorange', 
                            label=r'BR: $\Delta x = {}$ m'.format(str(time_series_eq[BR_name][BR_file_list[i]].x-y_val_shift)[0:4]))
                br_h2 = ax.loglog(br_f_sm[br_u_aliasing:], br_S_uu_sm[br_u_aliasing:], 'o', color='darkorange', 
                            )
                # WB
                wb_h1 = ax.loglog(wb_f_sm[:wb_u_aliasing], wb_S_uu_sm[:wb_u_aliasing], 'd', color='navy', 
                            label=r'WB: $\Delta x = {}$ m'.format(str(time_series_eq[WB_name][WB_file_list[i]].x-y_val_shift)[0:4]))
                wb_h2 = ax.loglog(wb_f_sm[wb_u_aliasing:], wb_S_uu_sm[wb_u_aliasing:], 'd', color='navy', 
                            )
                ax.set_ylabel(r"$f\cdot S_{uu}\cdot (\sigma_u \sigma_u)^{-1}$")
            elif var_name == 'v':
                h1 = ax.loglog(f_sm[:v_aliasing], S_vv_sm[:v_aliasing], 'x', color='darkmagenta', 
                            label=r'PALM - $u$ $\Delta x = {}$ m'.format(wall_dist[0]))
                h2 = ax.loglog(f_sm[v_aliasing:], S_vv_sm[v_aliasing:], 'x', 
                            )
                # FL
                fl_h1 = ax.loglog(fl_f_sm[:fl_v_aliasing], fl_S_vv_sm[:fl_v_aliasing], '^', color='forestgreen', 
                            label=r'FL: $\Delta x = {}$ m'.format(str(time_series_eq[FL_name][FL_file_list[i]].x-y_val_shift)[0:4]))
                fl_h2 = ax.loglog(fl_f_sm[fl_v_aliasing:], fl_S_vv_sm[fl_v_aliasing:], '^', color='forestgreen', 
                            )
                # BR
                br_h1 = ax.loglog(br_f_sm[:br_v_aliasing], br_S_vv_sm[:br_v_aliasing], 'o', color='darkorange', 
                            label=r'BR: $\Delta x = {}$ m'.format(str(time_series_eq[BR_name][BR_file_list[i]].x-y_val_shift)[0:4]))
                br_h2 = ax.loglog(br_f_sm[br_v_aliasing:], br_S_vv_sm[br_v_aliasing:], 'o', color='darkorange', 
                            )
                # WB
                wb_h1 = ax.loglog(wb_f_sm[:wb_v_aliasing], wb_S_vv_sm[:wb_v_aliasing], 'd', color='navy', 
                            label=r'WB: $\Delta x = {}$ m'.format(str(time_series_eq[WB_name][WB_file_list[i]].x-y_val_shift)[0:4]))
                wb_h2 = ax.loglog(wb_f_sm[wb_v_aliasing:], wb_S_vv_sm[wb_v_aliasing:], 'd', color='navy', 
                            )
                ax.set_ylabel(r"$f\cdot S_{ww}\cdot (\sigma_w \sigma_w)^{-1}$")
            elif var_name == 'uv':
                h1 = ax.loglog(f_sm[:uv_aliasing], S_uv_sm[:uv_aliasing], 'x', color='darkmagenta', 
                            label=r'PALM - $\Delta x = {}$ m'.format(wall_dist[0]))
                h2 = ax.loglog(f_sm[uv_aliasing:], S_uv_sm[uv_aliasing:], 'x', 
                            )
                # FL
                fl_h1 = ax.loglog(fl_f_sm[:fl_uv_aliasing], fl_S_uv_sm[:fl_uv_aliasing], '^', color='forestgreen', 
                            label=r'FL: $\Delta x = {}$ m'.format(str(time_series_eq[FL_name][FL_file_list[i]].x-y_val_shift)[0:4]))
                fl_h2 = ax.loglog(fl_f_sm[fl_uv_aliasing:], fl_S_uv_sm[fl_uv_aliasing:], '^', color='forestgreen', 
                            )
                # BR
                br_h1 = ax.loglog(br_f_sm[:br_uv_aliasing], br_S_uv_sm[:br_uv_aliasing], 'o', color='darkorange', 
                            label=r'BR: $\Delta x = {}$ m'.format(str(time_series_eq[BR_name][BR_file_list[i]].x-y_val_shift)[0:4]))
                br_h2 = ax.loglog(br_f_sm[br_uv_aliasing:], br_S_uv_sm[br_uv_aliasing:], 'o', color='darkorange', 
                            )
                # WB
                wb_h1 = ax.loglog(wb_f_sm[:wb_uv_aliasing], wb_S_uv_sm[:wb_uv_aliasing], 'd', color='navy', 
                            label=r'WB: $\Delta x = {}$ m'.format(str(time_series_eq[WB_name][WB_file_list[i]].x-y_val_shift)[0:4]))
                wb_h2 = ax.loglog(wb_f_sm[wb_uv_aliasing:], wb_S_uv_sm[wb_uv_aliasing:], 'd', color='navy', 
                            )
                ax.set_ylabel(r"$f\cdot S_{uw}\cdot (\sigma_u \sigma_w)^{-1}$")
            
            # slopey
            freq_slopey = np.logspace(np.log10(5*10.**(-2.)), np.log10(10.), num=20)
            twothird_slopey = 0.01*(freq_slopey)**(-2./3.)
            ax.plot(freq_slopey, twothird_slopey , label=r'K41: $-2/3$', color='black', linestyle='dashed')

            ax.set_xlim(10.**-3., 10.**2.)
            ax.set_ylim(10.**-3., 1.)
            if scaling == 'u_mean':
                ax.set_xlabel(r"$f\cdot \Delta x\cdot \overline{u}^{-1}$")
            elif scaling == 'u_hor':
                ax.set_xlabel(r"$f\cdot \Delta x\cdot \overline{u_{h}}^{-1}$")
            elif scaling == 'v_mean':
                ax.set_xlabel(r"$f\cdot \Delta x\cdot \overline{w}^{-1}$")
            else:
                ax.set_xlabel(r"$f\cdot \Delta x\cdot \overline{u}^{-1}$")
            ax.legend(bbox_to_anchor = (0.5, 1.15), loc = 'lower center', 
                    borderaxespad = 0., ncol = 2, 
                    numpoints = 1)
            ax.grid(False)
            if compute_LE_mid_spectra:
                plt.savefig('../palm_results/{}/run_{}/spectra/mid_{}_{}_spectra{}_{}.png'.format(papy.globals.run_name, papy.globals.run_number[-3:],
                            papy.globals.run_name, var_name, mask, scaling), bbox_inches='tight')
                print('     SAVED TO: ' 
                + '../palm_results/{}/run_{}/spectra/mid_{}_{}_spectra_{}_{}.png'.format(papy.globals.run_name, papy.globals.run_number[-3:],
                            papy.globals.run_name, var_name, mask, scaling))
            if compute_LE_up_spectra:
                plt.savefig('../palm_results/{}/run_{}/spectra/up_{}_{}_spectra{}_{}.png'.format(papy.globals.run_name, papy.globals.run_number[-3:],
                            papy.globals.run_name, var_name, mask, scaling), bbox_inches='tight')
                print('     SAVED TO: ' 
                + '../palm_results/{}/run_{}/spectra/up_{}_{}_spectra_{}_{}.png'.format(papy.globals.run_name, papy.globals.run_number[-3:],
                            papy.globals.run_name, var_name, mask, scaling))


if compute_LU_spectra:
    # heights mode
    print('\n Compute at different walldists: \n')
    y_val_shift = 0.115*scale
    var_name_list = ['u', 'v', 'uv']
    for i,mask in enumerate(mask_name_list):
        total_var_u = np.array([])
        total_var_v = np.array([])        
        total_time = np.array([])
        for run_no in papy.globals.run_numbers:
            nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask, run_no)
            time, time_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'time')
            var_u, var_u_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'u')
            var_v, var_v_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'w')                
            y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'x')
            total_time = np.concatenate([total_time, time])
            total_var_u = np.concatenate([total_var_u, var_u])
            total_var_v = np.concatenate([total_var_v, var_v])
        # gather values
        wall_dist = np.asarray([abs(y[0]-494.)])
        print('\n HEIGHT = {} m'.format(wall_dist[0]))
        # equidistant timestepping
        time_eq = np.linspace(total_time[0], total_time[-1], len(total_time))
        var_u_eq = wt.equ_dist_ts(total_time, time_eq, total_var_u/palm_ref)
        var_v_eq = wt.equ_dist_ts(total_time, time_eq, total_var_v/palm_ref)
        scaling = 'u_hor' # u_hor, u_mean or v_mean
        f_sm, S_uu_sm, S_vv_sm, S_uv_sm, u_aliasing, v_aliasing, uv_aliasing = wt.calc_spectra(var_u_eq,
                            var_v_eq,
                            total_time,
                            wall_dist[0], scaling)
                            
        FL_name = namelist[0]
        fl_f_sm, fl_S_uu_sm, fl_S_vv_sm, fl_S_uv_sm, fl_u_aliasing, fl_v_aliasing, fl_uv_aliasing = wt.calc_spectra(
                            time_series_eq[FL_name][FL_file_list[i]].u_eq.dropna(),
                            time_series_eq[FL_name][FL_file_list[i]].v_eq.dropna(),
                            time_series_eq[FL_name][FL_file_list[i]].t_eq[~np.isnan(time_series_eq[FL_name][FL_file_list[i]].t_eq)],
                            abs(time_series_eq[FL_name][FL_file_list[i]].x)-y_val_shift, scaling)
        BR_name = namelist[1]
        br_f_sm, br_S_uu_sm, br_S_vv_sm, br_S_uv_sm, br_u_aliasing, br_v_aliasing, br_uv_aliasing = wt.calc_spectra(
                            time_series_eq[BR_name][BR_file_list[i]].u_eq.dropna(),
                            time_series_eq[BR_name][BR_file_list[i]].v_eq.dropna(),
                            time_series_eq[BR_name][BR_file_list[i]].t_eq[~np.isnan(time_series_eq[BR_name][BR_file_list[i]].t_eq)],
                            abs(time_series_eq[BR_name][BR_file_list[i]].x)-y_val_shift, scaling)
        WB_name = namelist[2]
        wb_f_sm, wb_S_uu_sm, wb_S_vv_sm, wb_S_uv_sm, wb_u_aliasing, wb_v_aliasing, wb_uv_aliasing = wt.calc_spectra(
                            time_series_eq[WB_name][WB_file_list[i]].u_eq.dropna(),
                            time_series_eq[WB_name][WB_file_list[i]].v_eq.dropna(),
                            time_series_eq[WB_name][WB_file_list[i]].t_eq[~np.isnan(time_series_eq[WB_name][WB_file_list[i]].t_eq)],
                            abs(time_series_eq[WB_name][WB_file_list[i]].x)-y_val_shift, scaling)

        print('    calculated spectra for {}'.format(mask))

        for var_name in var_name_list:
            fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))            
            if var_name == 'u':
                # palm
                h1 = ax.loglog(f_sm[:u_aliasing], S_uu_sm[:u_aliasing], 'x', color='darkmagenta', 
                            label=r'PALM - $\Delta x = {}$ m'.format(wall_dist[0]))
                h2 = ax.loglog(f_sm[u_aliasing:], S_uu_sm[u_aliasing:], 'x',
                            )
                # FL
                fl_h1 = ax.loglog(fl_f_sm[:fl_u_aliasing], fl_S_uu_sm[:fl_u_aliasing], '^', color='forestgreen', 
                            label=r'FL: $\Delta x = {}$ m'.format(str(abs(time_series_eq[FL_name][FL_file_list[i]].x)-y_val_shift)[0:4]))
                fl_h2 = ax.loglog(fl_f_sm[fl_u_aliasing:], fl_S_uu_sm[fl_u_aliasing:], '^', color='forestgreen', 
                            )
                # BR
                br_h1 = ax.loglog(br_f_sm[:br_u_aliasing], br_S_uu_sm[:br_u_aliasing], 'o', color='darkorange', 
                            label=r'BR: $\Delta x = {}$ m'.format(str(abs(time_series_eq[BR_name][BR_file_list[i]].x)-y_val_shift)[0:4]))
                br_h2 = ax.loglog(br_f_sm[br_u_aliasing:], br_S_uu_sm[br_u_aliasing:], 'o', color='darkorange', 
                            )
                # WB
                wb_h1 = ax.loglog(wb_f_sm[:wb_u_aliasing], wb_S_uu_sm[:wb_u_aliasing], 'd', color='navy', 
                            label=r'WB: $\Delta x = {}$ m'.format(str(abs(time_series_eq[WB_name][WB_file_list[i]].x)-y_val_shift)[0:4]))
                wb_h2 = ax.loglog(wb_f_sm[wb_u_aliasing:], wb_S_uu_sm[wb_u_aliasing:], 'd', color='navy', 
                            )
                ax.set_ylabel(r"$f\cdot S_{uu}\cdot (\sigma_u \sigma_u)^{-1}$")
            elif var_name == 'v':
                h1 = ax.loglog(f_sm[:v_aliasing], S_vv_sm[:v_aliasing], 'x', color='darkmagenta', 
                            label=r'PALM - $u$ $\Delta x = {}$ m'.format(wall_dist[0]))
                h2 = ax.loglog(f_sm[v_aliasing:], S_vv_sm[v_aliasing:], 'x', 
                            )
                # FL
                fl_h1 = ax.loglog(fl_f_sm[:fl_v_aliasing], fl_S_vv_sm[:fl_v_aliasing], '^', color='forestgreen', 
                            label=r'FL: $\Delta x = {}$ m'.format(str(abs(time_series_eq[FL_name][FL_file_list[i]].x)-y_val_shift)[0:4]))
                fl_h2 = ax.loglog(fl_f_sm[fl_v_aliasing:], fl_S_vv_sm[fl_v_aliasing:], '^', color='forestgreen', 
                            )
                # BR
                br_h1 = ax.loglog(br_f_sm[:br_v_aliasing], br_S_vv_sm[:br_v_aliasing], 'o', color='darkorange', 
                            label=r'BR: $\Delta x = {}$ m'.format(str(abs(time_series_eq[BR_name][BR_file_list[i]].x)-y_val_shift)[0:4]))
                br_h2 = ax.loglog(br_f_sm[br_v_aliasing:], br_S_vv_sm[br_v_aliasing:], 'o', color='darkorange', 
                            )
                # WB
                wb_h1 = ax.loglog(wb_f_sm[:wb_v_aliasing], wb_S_vv_sm[:wb_v_aliasing], 'd', color='navy', 
                            label=r'WB: $\Delta x = {}$ m'.format(str(abs(time_series_eq[WB_name][WB_file_list[i]].x)-y_val_shift)[0:4]))
                wb_h2 = ax.loglog(wb_f_sm[wb_v_aliasing:], wb_S_vv_sm[wb_v_aliasing:], 'd', color='navy', 
                            )
                ax.set_ylabel(r"$f\cdot S_{ww}\cdot (\sigma_w \sigma_w)^{-1}$")
            elif var_name == 'uv':
                h1 = ax.loglog(f_sm[:uv_aliasing], S_uv_sm[:uv_aliasing], 'x', color='darkmagenta', 
                            label=r'PALM - $\Delta x = {}$ m'.format(wall_dist[0]))
                h2 = ax.loglog(f_sm[uv_aliasing:], S_uv_sm[uv_aliasing:], 'x', 
                            )
                # FL
                fl_h1 = ax.loglog(fl_f_sm[:fl_uv_aliasing], fl_S_uv_sm[:fl_uv_aliasing], '^', color='forestgreen', 
                            label=r'FL: $\Delta x = {}$ m'.format(str(abs(time_series_eq[FL_name][FL_file_list[i]].x)-y_val_shift)[0:4]))
                fl_h2 = ax.loglog(fl_f_sm[fl_uv_aliasing:], fl_S_uv_sm[fl_uv_aliasing:], '^', color='forestgreen', 
                            )
                # BR
                br_h1 = ax.loglog(br_f_sm[:br_uv_aliasing], br_S_uv_sm[:br_uv_aliasing], 'o', color='darkorange', 
                            label=r'BR: $\Delta x = {}$ m'.format(str(abs(time_series_eq[BR_name][BR_file_list[i]].x)-y_val_shift)[0:4]))
                br_h2 = ax.loglog(br_f_sm[br_uv_aliasing:], br_S_uv_sm[br_uv_aliasing:], 'o', color='darkorange', 
                            )
                # WB
                wb_h1 = ax.loglog(wb_f_sm[:wb_uv_aliasing], wb_S_uv_sm[:wb_uv_aliasing], 'd', color='navy', 
                            label=r'WB: $\Delta x = {}$ m'.format(str(abs(time_series_eq[WB_name][WB_file_list[i]].x)-y_val_shift)[0:4]))
                wb_h2 = ax.loglog(wb_f_sm[wb_uv_aliasing:], wb_S_uv_sm[wb_uv_aliasing:], 'd', color='navy', 
                            )
                ax.set_ylabel(r"$f\cdot S_{uw}\cdot (\sigma_u \sigma_w)^{-1}$")
            
            # slopey
            freq_slopey = np.logspace(np.log10(5*10.**(-2.)), np.log10(10.), num=20)
            twothird_slopey = 0.01*(freq_slopey)**(-2./3.)
            ax.plot(freq_slopey, twothird_slopey , label=r'K41: $-2/3$', color='black', linestyle='dashed')

            ax.set_xlim(10.**-3., 10.**2.)
            ax.set_ylim(10.**-3., 1.)
            if scaling == 'u_mean':
                ax.set_xlabel(r"$f\cdot \Delta x\cdot \overline{u}^{-1}$")
            elif scaling == 'u_hor':
                ax.set_xlabel(r"$f\cdot \Delta x\cdot \overline{u_{h}}^{-1}$")
            elif scaling == 'v_mean':
                ax.set_xlabel(r"$f\cdot \Delta x\cdot \overline{w}^{-1}$")
            else:
                ax.set_xlabel(r"$f\cdot \Delta x\cdot \overline{u}^{-1}$")
            ax.legend(bbox_to_anchor = (0.5, 1.15), loc = 'lower center', 
                    borderaxespad = 0., ncol = 2, 
                    numpoints = 1)
            ax.grid(False)
            plt.savefig('../palm_results/{}/run_{}/spectra/{}_{}_spectra{}_{}.{}'.format(papy.globals.run_name, papy.globals.run_number[-3:],
                        papy.globals.run_name, var_name, mask, scaling, plotformat), bbox_inches='tight')
            print('     SAVED TO: ' 
            + '../palm_results/{}/run_{}/spectra/{}_{}_spectra_{}_{}.{}'.format(papy.globals.run_name, papy.globals.run_number[-3:],
                        papy.globals.run_name, var_name, mask, scaling, plotformat))


if compute_BL_spectra:
    # heights mode
    print('\n Compute at different walldists: \n')
    var_name_list = ['u', 'v', 'uv']
    for i,mask in enumerate(mask_name_list):
        total_var_u = np.array([])
        total_var_v = np.array([])        
        total_time = np.array([])
        for run_no in papy.globals.run_numbers:
            nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask, run_no)
            time, time_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'time')
            var_u, var_u_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'u')
            var_v, var_v_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'w')                
            y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'zu_3d')
            total_time = np.concatenate([total_time, time])
            total_var_u = np.concatenate([total_var_u, var_u])
            total_var_v = np.concatenate([total_var_v, var_v])
        # gather values
        wall_dist = np.asarray([abs(y[0])])
        print('\n HEIGHT = {} m'.format(wall_dist[0]))
        # equidistant timestepping
        time_eq = np.linspace(total_time[0], total_time[-1], len(total_time))
        var_u_eq = wt.equ_dist_ts(total_time, time_eq, total_var_u/palm_ref)
        var_v_eq = wt.equ_dist_ts(total_time, time_eq, total_var_v/palm_ref)
        scaling = 'u_hor' # u_hor, u_mean or v_mean
        f_sm, S_uu_sm, S_vv_sm, S_uv_sm, u_aliasing, v_aliasing, uv_aliasing = wt.calc_spectra(var_u_eq,
                            var_v_eq,
                            total_time,
                            wall_dist[0], scaling)
                            
        BL_name = namelist[0]
        fl_f_sm, fl_S_uu_sm, fl_S_vv_sm, fl_S_uv_sm, fl_u_aliasing, fl_v_aliasing, fl_uv_aliasing = wt.calc_spectra(
                            time_series_eq[BL_name][BL_file_list[i]].u_eq.dropna(),
                            time_series_eq[BL_name][BL_file_list[i]].v_eq.dropna(),
                            time_series_eq[BL_name][BL_file_list[i]].t_eq[~np.isnan(time_series_eq[BL_name][BL_file_list[i]].t_eq)],
                            abs(time_series_eq[BL_name][BL_file_list[i]].z), scaling)

        print('    calculated spectra for {}'.format(mask))

        for var_name in var_name_list:
            fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))          
            if var_name == 'u':
                # palm
                h1 = ax.loglog(f_sm[:u_aliasing], S_uu_sm[:u_aliasing], 'x', color='darkmagenta', 
                            label=r'PALM - $z = {}$ m'.format(wall_dist[0]))
                h2 = ax.loglog(f_sm[u_aliasing:], S_uu_sm[u_aliasing:], 'x',
                            )
                # BL
                fl_h1 = ax.loglog(fl_f_sm[:fl_u_aliasing], fl_S_uu_sm[:fl_u_aliasing], '^', color='orangered', 
                            label=r'Boundary Layer - $z = {}$ m'.format(str(abs(time_series_eq[BL_name][BL_file_list[i]].z))[0:4]))
                fl_h2 = ax.loglog(fl_f_sm[fl_u_aliasing:], fl_S_uu_sm[fl_u_aliasing:], '^', color='orangered', 
                            )

                ax.set_ylabel(r"$f\cdot S_{uu}\cdot (\sigma_u \sigma_u)^{-1}$")
            elif var_name == 'v':
                h1 = ax.loglog(f_sm[:v_aliasing], S_vv_sm[:v_aliasing], 'x', color='darkmagenta', 
                            label=r'PALM - $u$ $z = {}$ m'.format(wall_dist[0]))
                h2 = ax.loglog(f_sm[v_aliasing:], S_vv_sm[v_aliasing:], 'x', 
                            )
                # BL
                fl_h1 = ax.loglog(fl_f_sm[:fl_v_aliasing], fl_S_vv_sm[:fl_v_aliasing], '^', color='orangered', 
                            label=r'Boundary Layer - $z = {}$ m'.format(str(abs(time_series_eq[BL_name][BL_file_list[i]].z))[0:4]))
                fl_h2 = ax.loglog(fl_f_sm[fl_v_aliasing:], fl_S_vv_sm[fl_v_aliasing:], '^', color='orangered', 
                            )

                ax.set_ylabel(r"$f\cdot S_{ww}\cdot (\sigma_w \sigma_w)^{-1}$")
            elif var_name == 'uv':
                h1 = ax.loglog(f_sm[:uv_aliasing], S_uv_sm[:uv_aliasing], 'x', color='darkmagenta', 
                            label=r'PALM - $z = {}$ m'.format(wall_dist[0]))
                h2 = ax.loglog(f_sm[uv_aliasing:], S_uv_sm[uv_aliasing:], 'x', 
                            )
                # BL
                fl_h1 = ax.loglog(fl_f_sm[:fl_uv_aliasing], fl_S_uv_sm[:fl_uv_aliasing], '^', color='orangered', 
                            label=r'Boundary Layer - $z = {}$ m'.format(str(abs(time_series_eq[BL_name][BL_file_list[i]].z))[0:4]))
                fl_h2 = ax.loglog(fl_f_sm[fl_uv_aliasing:], fl_S_uv_sm[fl_uv_aliasing:], '^', color='orangered', 
                            )
                ax.set_ylabel(r"$f\cdot S_{uw}\cdot (\sigma_u \sigma_w)^{-1}$")
            
            # slopey
            freq_slopey = np.logspace(np.log10(5*10.**(-2.)), np.log10(10.), num=20)
            twothird_slopey = 0.01*(freq_slopey)**(-2./3.)
            ax.plot(freq_slopey, twothird_slopey , label=r'K41: $-2/3$', color='black', linestyle='dashed')

            ax.set_xlim(10.**-3., 10.**2.)
            ax.set_ylim(10.**-3., 1.)
            if scaling == 'u_mean':
                ax.set_xlabel(r"$f\cdot z\cdot \overline{u}^{-1}$")
            elif scaling == 'u_hor':
                ax.set_xlabel(r"$f\cdot z\cdot \overline{u_{h}}^{-1}$")
            elif scaling == 'v_mean':
                ax.set_xlabel(r"$f\cdot z\cdot \overline{w}^{-1}$")
            else:
                ax.set_xlabel(r"$f\cdot z\cdot \overline{u}^{-1}$")
            ax.legend(bbox_to_anchor = (0.5, 1.15), loc = 'lower center', 
                    borderaxespad = 0., ncol = 2, 
                    numpoints = 1)
            ax.grid(False)
            plt.savefig('../palm_results/{}/run_{}/spectra/{}_{}_spectra{}_{}.{}'.format(papy.globals.run_name, papy.globals.run_number[-3:],
                        papy.globals.run_name, var_name, mask, scaling, plotformat), bbox_inches='tight')
            print('     SAVED TO: ' 
            + '../palm_results/{}/run_{}/spectra/{}_{}_spectra_{}_{}.{}'.format(papy.globals.run_name, papy.globals.run_number[-3:],
                        papy.globals.run_name, var_name, mask, scaling, plotformat))


#############################
########## balcony ##########
#############################
if compute_BA_spectra:
    # heights mode
    print('\n Compute at different walldists: \n')
    var_name_list = ['u', 'v', 'uv']
    for i,mask in enumerate(mask_name_list):
        total_var_u = np.array([])
        total_var_v = np.array([])        
        total_time = np.array([])
        for run_no in papy.globals.run_numbers:
            nc_file = '{}_masked_{}{}.nc'.format(papy.globals.run_name, mask, run_no)
            time, time_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'time')
            var_u, var_u_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'u')
            var_v, var_v_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'w')                
            y, y_unit = papy.read_nc_var_ms(nc_file_path, nc_file, 'zu_3d')
            total_time = np.concatenate([total_time, time])
            total_var_u = np.concatenate([total_var_u, var_u])
            total_var_v = np.concatenate([total_var_v, var_v])                
        # gather values 
        wall_dist = y
        print('\n HEIGHT = {} m'.format(wall_dist[0]))
        # equidistant timestepping
        time_eq = np.linspace(total_time[0], total_time[-1], len(total_time))
        var_u_eq = wt.equ_dist_ts(total_time, time_eq, total_var_u/palm_ref)
        var_v_eq = wt.equ_dist_ts(total_time, time_eq, total_var_v/palm_ref)        
        scaling = 'u_mean' # u_hor, u_mean or v_mean
        f_sm, S_uu_sm, S_vv_sm, S_uv_sm, u_aliasing, v_aliasing, uv_aliasing = wt.calc_spectra(var_u_eq,
                            var_v_eq,
                            total_time,
                            wall_dist[0], scaling)
                            
        FL_name = namelist[3]
        fl_f_sm, fl_S_uu_sm, fl_S_vv_sm, fl_S_uv_sm, fl_u_aliasing, fl_v_aliasing, fl_uv_aliasing = wt.calc_spectra(
                            time_series_eq[FL_name][FL_file_list[i]].u_eq.dropna(),
                            time_series_eq[FL_name][FL_file_list[i]].v_eq.dropna(),
                            time_series_eq[FL_name][FL_file_list[i]].t_eq[~np.isnan(time_series_eq[FL_name][FL_file_list[i]].t_eq)],
                            time_series_eq[FL_name][FL_file_list[i]].z, scaling)
        S5_name = namelist[1]
        br_f_sm, br_S_uu_sm, br_S_vv_sm, br_S_uv_sm, br_u_aliasing, br_v_aliasing, br_uv_aliasing = wt.calc_spectra(
                            time_series_eq[S5_name][S5_file_list[i]].u_eq.dropna(),
                            time_series_eq[S5_name][S5_file_list[i]].v_eq.dropna(),
                            time_series_eq[S5_name][S5_file_list[i]].t_eq[~np.isnan(time_series_eq[S5_name][S5_file_list[i]].t_eq)],
                            time_series_eq[S5_name][S5_file_list[i]].z, scaling)
        S8_name = namelist[2]
        wb_f_sm, wb_S_uu_sm, wb_S_vv_sm, wb_S_uv_sm, wb_u_aliasing, wb_v_aliasing, wb_uv_aliasing = wt.calc_spectra(
                            time_series_eq[S8_name][S8_file_list[i]].u_eq.dropna(),
                            time_series_eq[S8_name][S8_file_list[i]].v_eq.dropna(),
                            time_series_eq[S8_name][S8_file_list[i]].t_eq[~np.isnan(time_series_eq[S8_name][S8_file_list[i]].t_eq)],
                            time_series_eq[S8_name][S8_file_list[i]].z, scaling)

        print('    calculated spectra for {}'.format(mask))

        for var_name in var_name_list:
            fig, ax = plt.subplots(figsize=(textwidth_half,textwidth_half*0.75))        
            if var_name == 'u':
                # palm
                h1 = ax.loglog(f_sm[:u_aliasing], S_uu_sm[:u_aliasing], 'x', color='darkmagenta', 
                            label=r'PALM - $z = {}$ m'.format(wall_dist[0]))
                h2 = ax.loglog(f_sm[u_aliasing:], S_uu_sm[u_aliasing:], 'x',
                            )
                # FL
                fl_h1 = ax.loglog(fl_f_sm[:fl_u_aliasing], fl_S_uu_sm[:fl_u_aliasing], '^', color='forestgreen', 
                            label=r'flat: $z = {}$ m'.format(str(time_series_eq[FL_name][FL_file_list[i]].z)[0:4]))
                fl_h2 = ax.loglog(fl_f_sm[fl_u_aliasing:], fl_S_uu_sm[fl_u_aliasing:], '^', color='forestgreen', 
                            )
                # BR
                br_h1 = ax.loglog(br_f_sm[:br_u_aliasing], br_S_uu_sm[:br_u_aliasing], 'o', color='navy', 
                            label=r'medium rough: $z = {}$ m'.format(str(time_series_eq[S5_name][S5_file_list[i]].z)[0:4]))
                br_h2 = ax.loglog(br_f_sm[br_u_aliasing:], br_S_uu_sm[br_u_aliasing:], 'o', color='navy', 
                            )
                # WB
                wb_h1 = ax.loglog(wb_f_sm[:wb_u_aliasing], wb_S_uu_sm[:wb_u_aliasing], 'd', color='darkorange', 
                            label=r'rough: $z = {}$ m'.format(str(time_series_eq[S8_name][S8_file_list[i]].z)[0:4]))
                wb_h2 = ax.loglog(wb_f_sm[wb_u_aliasing:], wb_S_uu_sm[wb_u_aliasing:], 'd', color='darkorange', 
                            )
                ax.set_ylabel(r"$f\cdot S_{uu}\cdot (\sigma_u \sigma_u)^{-1}$")
            elif var_name == 'v':
                h1 = ax.loglog(f_sm[:v_aliasing], S_vv_sm[:v_aliasing], 'x', color='darkmagenta', 
                            label=r'PALM - $u$ $z = {}$ m'.format(wall_dist[0]))
                h2 = ax.loglog(f_sm[v_aliasing:], S_vv_sm[v_aliasing:], 'x', 
                            )
                # FL
                fl_h1 = ax.loglog(fl_f_sm[:fl_v_aliasing], fl_S_vv_sm[:fl_v_aliasing], '^', color='forestgreen', 
                            label=r'flat: $z = {}$ m'.format(str(time_series_eq[FL_name][FL_file_list[i]].z)[0:4]))
                fl_h2 = ax.loglog(fl_f_sm[fl_v_aliasing:], fl_S_vv_sm[fl_v_aliasing:], '^', color='forestgreen', 
                            )
                # BR
                br_h1 = ax.loglog(br_f_sm[:br_v_aliasing], br_S_vv_sm[:br_v_aliasing], 'o', color='navy', 
                            label=r'medium rough: $z = {}$ m'.format(str(time_series_eq[S5_name][S5_file_list[i]].z)[0:4]))
                br_h2 = ax.loglog(br_f_sm[br_v_aliasing:], br_S_vv_sm[br_v_aliasing:], 'o', color='navy', 
                            )
                # WB
                wb_h1 = ax.loglog(wb_f_sm[:wb_v_aliasing], wb_S_vv_sm[:wb_v_aliasing], 'd', color='darkorange', 
                            label=r'rough: $z = {}$ m'.format(str(time_series_eq[S8_name][S8_file_list[i]].z)[0:4]))
                wb_h2 = ax.loglog(wb_f_sm[wb_v_aliasing:], wb_S_vv_sm[wb_v_aliasing:], 'd', color='darkorange', 
                            )
                ax.set_ylabel(r"$f\cdot S_{vv}\cdot (\sigma_v \sigma_v)^{-1}$")
            elif var_name == 'uv':
                h1 = ax.loglog(f_sm[:uv_aliasing], S_uv_sm[:uv_aliasing], 'x', color='darkmagenta', 
                            label=r'PALM - $z = {}$ m'.format(wall_dist[0]))
                h2 = ax.loglog(f_sm[uv_aliasing:], S_uv_sm[uv_aliasing:], 'x', 
                            )
                # FL
                fl_h1 = ax.loglog(fl_f_sm[:fl_uv_aliasing], fl_S_uv_sm[:fl_uv_aliasing], '^', color='forestgreen', 
                            label=r'flat: $z = {}$ m'.format(str(time_series_eq[FL_name][FL_file_list[i]].z)[0:4]))
                fl_h2 = ax.loglog(fl_f_sm[fl_uv_aliasing:], fl_S_uv_sm[fl_uv_aliasing:], '^', color='forestgreen', 
                            )
                # BR
                br_h1 = ax.loglog(br_f_sm[:br_uv_aliasing], br_S_uv_sm[:br_uv_aliasing], 'o', color='navy', 
                            label=r'medium rough: $z = {}$ m'.format(str(time_series_eq[S5_name][S5_file_list[i]].z)[0:4]))
                br_h2 = ax.loglog(br_f_sm[br_uv_aliasing:], br_S_uv_sm[br_uv_aliasing:], 'o', color='navy', 
                            )
                # WB
                wb_h1 = ax.loglog(wb_f_sm[:wb_uv_aliasing], wb_S_uv_sm[:wb_uv_aliasing], 'd', color='darkorange', 
                            label=r'rough: $z = {}$ m'.format(str(time_series_eq[S8_name][S8_file_list[i]].z)[0:4]))
                wb_h2 = ax.loglog(wb_f_sm[wb_uv_aliasing:], wb_S_uv_sm[wb_uv_aliasing:], 'd', color='darkorange', 
                            )
                ax.set_ylabel(r"$f\cdot S_{uv}\cdot (\sigma_u \sigma_v)^{-1}$")
            
            # ax.vlines(np.mean(var_u_eq) * (total_time[23]-total_time[22])/wall_dist[0], 10**(-3.), 10**0., 
            #           label=r'$\Delta_i$', color='darkorchid', linestyle='-.')
            # ax.vlines(np.mean(var_u_eq)**(-1) * 1/(total_time[23]-total_time[22])/5., 10**(-3.), 10**0., 
            #           label=r'$5 \cdot \Delta_i$', color='violet', linestyle=':')
            # slopey
            freq_slopey = np.logspace(np.log10(5*10.**(-2.)), np.log10(10.), num=20)
            twothird_slopey = 0.01*(freq_slopey)**(-2./3.)
            ax.plot(freq_slopey, twothird_slopey , label=r'K41: $-2/3$', color='black', linestyle='dashed')

            ax.set_xlim(10.**-3., 10.**2.)
            ax.set_ylim(10.**-3., 1.)
            if scaling == 'u_mean':
                ax.set_xlabel(r"$f\cdot z\cdot \overline{u}^{-1}$")
            elif scaling == 'u_hor':
                ax.set_xlabel(r"$f\cdot z\cdot \overline{u_{h}}^{-1}$")
            elif scaling == 'v_mean':
                ax.set_xlabel(r"$f\cdot z\cdot \overline{v}^{-1}$")
            else:
                ax.set_xlabel(r"$f\cdot z\cdot \overline{u}^{-1}$")
            ax.legend(bbox_to_anchor = (0.5, 1.15), loc = 'lower center', 
                    borderaxespad = 0., ncol = 2, 
                    numpoints = 1)
            ax.grid(False)

            plt.savefig('../palm_results/{}/run_{}/spectra/{}_{}_spectra{}_{}.{}'.format(papy.globals.run_name, papy.globals.run_number[-3:],
                        papy.globals.run_name, var_name, mask, scaling, plotformat), bbox_inches='tight')
            print('     SAVED TO: ' 
            + '../palm_results/{}/run_{}/spectra/{}_{}_spectra_{}_{}.{}'.format(papy.globals.run_name, papy.globals.run_number[-3:],
                        papy.globals.run_name, var_name, mask, scaling, plotformat))




print('\n Finished processing of: {}'.format(papy.globals.run_name))