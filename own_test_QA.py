################
"""
IMPORTS
"""
################

from hashlib import shake_128
import numpy as np
import math as m
import pandas as pd
import sys
import os
import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.style.use('classic')

import palm_py as papy

sys.path.append('/home/bene/Documents/phd/windtunnel_py/windtunnel/')    
import windtunnel as wt

import warnings
warnings.simplefilter("ignore")

################
"""
PARS
"""
################


run_name = 'SB_SI_back'
run_number = '.032'

################
"""
MAIN
"""
################
q1_fluxes = np.array([])
q2_fluxes = np.array([])
q3_fluxes = np.array([])
q4_fluxes = np.array([])
s1_all = np.array([])
s2_all = np.array([])
s3_all = np.array([])
s4_all = np.array([])
total_varu = np.array([])
total_varv = np.array([])

# produce test timeseries
num_list = [10, 100, 1000, 10000]
num_samples = 10000

# calc total flux timeseries and fluctuations
varu_fluc = np.random.normal(-1., 1., size=num_samples)
varv_fluc = np.random.normal(0, 1., size=num_samples)
total_flux = np.asarray(varu_fluc * varv_fluc)

# conditional sampling
q1_ind = np.where(np.logical_and(varu_fluc>=0, varv_fluc>0))
q2_ind = np.where(np.logical_and(varu_fluc<=0, varv_fluc>0))
q3_ind = np.where(np.logical_and(varu_fluc<=0, varv_fluc<0))
q4_ind = np.where(np.logical_and(varu_fluc>=0, varv_fluc<0))

# mean fluxes for each quadrant
q1_flux = np.asarray([np.mean(total_flux[q1_ind])])
q2_flux = np.asarray([np.mean(total_flux[q2_ind])])
q3_flux = np.asarray([np.mean(total_flux[q3_ind])])
q4_flux = np.asarray([np.mean(total_flux[q4_ind])])

# calc relative quadrant contributions
s1 = np.asarray([q1_flux[0]/np.mean(total_flux)])
s2 = np.asarray([q2_flux[0]/np.mean(total_flux)])
s3 = np.asarray([q3_flux[0]/np.mean(total_flux)])
s4 = np.asarray([q4_flux[0]/np.mean(total_flux)])

print('\n S1 = {}'.format(str(s1[0])[:7]) + '   N1 = {}'.format(len(q1_ind[0])))
print(' S2 = {}'.format(str(s2[0])[:7]) + '   N2 = {}'.format(len(q2_ind[0])))
print(' S3 = {}'.format(str(s3[0])[:7]) + '   N3 = {}'.format(len(q3_ind[0])))
print(' S4 = {}'.format(str(s4[0])[:7]) + '   N4 = {}'.format(len(q4_ind[0])))
print(' Flux = {}'.format(str(np.mean(total_flux))[:7]) + '   N = {}'.format(len(total_flux)))        
print(' SUM = {} \n'.format(str(s1[0] * len(q1_ind[0])/len(total_flux) + 
                                s2[0] * len(q2_ind[0])/len(total_flux) + 
                                s3[0] * len(q3_ind[0])/len(total_flux) + 
                                s4[0] * len(q4_ind[0])/len(total_flux))))

# CALC JOINT PROBABILITY DENSITY FUNCTIONS
umin = varu_fluc.min()
umax = varu_fluc.max()
vmin = varv_fluc.min()
vmax = varv_fluc.max()
u_jpdf, v_jpdf = np.mgrid[umin:umax:100j, vmin:vmax:100j]
positions = np.vstack([u_jpdf.ravel(), v_jpdf.ravel()])
values = np.vstack([varu_fluc, varv_fluc])
kernel = stats.gaussian_kde(values)
jpdf = np.reshape(kernel.evaluate(positions), u_jpdf.shape)

# PLOT Quadrant-scatterplots
fig, ax = plt.subplots()
fig.gca().set_aspect('equal', adjustable='box')
ax.plot(varu_fluc[q1_ind], varv_fluc[q1_ind] ,'o', color='blue',
        markersize=2,label='Q1')
ax.plot(varu_fluc[q2_ind], varv_fluc[q2_ind] ,'o', color='darkorange',
        markersize=2, label='Q2')
ax.plot(varu_fluc[q3_ind], varv_fluc[q3_ind] ,'o', color='cyan',
        markersize=2, label='Q3')
ax.plot(varu_fluc[q4_ind], varv_fluc[q4_ind] ,'o', color='red',
        markersize=2, label='Q4')
ax.grid(True, 'both', 'both')
ax.legend(bbox_to_anchor = (0.5,1.05), loc = 'lower center', 
            borderaxespad = 0., ncol = 2, 
            numpoints = 1, fontsize = 18)
ax.set_xlabel(r'$u^\prime$ $u_{ref}^{-1}$ (-)', fontsize = 18)
ax.set_ylabel(r'$v^\prime$ $u_{ref}^{-1}$ (-)', fontsize = 18)
# save plots
fig.savefig('../palm_results/{}/run_{}/quadrant_analysis/{}_QA_scatter_mask_{}.png'.format(run_name,
            run_number[-3:],
            'back', str(num_samples)), bbox_inches='tight', dpi=500)
plt.close()

# PLOT QUADRANT JPDFs
fig, ax = plt.subplots()
fig.gca().set_aspect('equal', adjustable='box')        
im1 = ax.contourf(jpdf.T, cmap='YlGnBu',
        extent=[umin, umax, vmin, vmax], levels = 15)
im2 = ax.contour(jpdf.T, colors='gray',
        extent=[umin, umax, vmin, vmax], levels = 15)

ax.vlines(0., vmin, vmax, colors='black', 
        linestyles='dashed')
ax.hlines(0., umin, umax, colors='black', 
        linestyles='dashed')
ax.grid(True, 'both', 'both')
plt.colorbar(im1, label=r'$\rho (u^\prime_{q_i},  v^\prime{q_i})$ (-)')
ax.set_xlabel(r'$u^\prime$ $u_{ref}^{-1}$ (-)', fontsize = 18)
ax.set_ylabel(r'$v^\prime$ $u_{ref}^{-1}$ (-)', fontsize = 18)
# save plots
fig.savefig('../palm_results/{}/run_{}/quadrant_analysis/{}_QA_jpdf_mask_{}.png'.format(run_name,
            run_number[-3:],
            'back', str(num_samples)), bbox_inches='tight', dpi=500)
plt.close()

