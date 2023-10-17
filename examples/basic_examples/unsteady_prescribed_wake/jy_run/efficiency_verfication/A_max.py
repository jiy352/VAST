'''Example 3 : verification of prescibed vlm with Theodorsen solution'''
from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver import UVLMSolver
from VAST.core.submodels.actuation_submodels.anderson_actuation import PitchingModel
import time
import numpy as np
from VAST.core.submodels.output_submodels.vlm_post_processing.efficiency import EfficiencyModel
from VAST.utils.visualization import run_visualization
import os


import python_csdl_backend
import csdl
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt


'''
This is a test case to check the prescribed wake solver against the Theodorsen solution
with a wing pitching sinusoidally from 1 to -1 deg and the reduced frequency k = [0.2, 0.6, 1, 3]
'''
########################################
# 1. define geometry
########################################
chord = 1
# k_list = [1.06, 2.1]
# k_list = [0.6, 1.06, 1.3, 1.45, 1.64, 1.8, 2.1]
# k_list = [0.1, 0.2, 0.39, 0.55, 0.73, 0.90, 1.03]
k_list = [0.1, 0.2, 0.39, 0.55, 0.73, 0.90, 1.05]
# k_list = [0.2, 0.39, 0.5, 0.69, 0.8, 0.98]
num_nodes = 150
h_0_star = 0.75
# h_0_star = 0.25

St_list=np.array(k_list) *  2 * h_0_star / (chord*np.pi)
St = St_list
num_nodes = 1000
N_period=1
alpha_0_deg = 15
# alpha_0_deg = 5

save_vtk=False

save_results=False
path = "verfication_data/theodorsen"
nx = 11; ny = 7         # num_pts in chordwise and spanwise direction
chord = 1; span = 6   # chord and span of the wing
b = chord/2

omega = 1
h_0 = h_0_star * chord
v_inf = omega*h_0 / (np.pi * St)
k = omega * b/v_inf

print('k is', k)
T = 2*np.pi/(omega)

theta_0 = np.rad2deg(np.arctan2(omega*h_0, v_inf) - np.deg2rad(alpha_0_deg))
print('theta_0 is', theta_0)

t_vec = np.linspace(0, N_period*T, num_nodes)
h_max = np.zeros(len(k))
St_TE = np.zeros(len(k))
for i in range(len(theta_0)):
    theta_0_current = np.deg2rad(theta_0[i])
    theta = theta_0_current * np.cos(omega*t_vec)
    h = h_0 * np.sin(omega*t_vec)
    # delta_h = 2/3 * chord * np.sin(theta) + h
    delta_h = 2/3 * chord * theta + h
    h_max[i] = np.max(delta_h)
    St_TE[i]=np.array(k_list)[i] *  2 * h_max[i] / (chord*np.pi)
    plt.plot(t_vec/T, delta_h, label = 'k = ' + str(k[i]))
print('St is', St_list)
print('St_TE is', St_TE)

plt.xlabel(r'$t/T$')
plt.ylabel(r'$\delta h$')
plt.legend()
plt.show()




