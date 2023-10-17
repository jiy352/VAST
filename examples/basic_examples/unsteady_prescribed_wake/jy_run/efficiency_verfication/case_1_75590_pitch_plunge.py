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

########################################
# This is a test case to check the prescribed wake solver
# pitching and plunging efficiency with Anderson paper
# "Oscillating foils of high propulsive efficiency"
# kinematic parameters are 
# h_0/c = 0.75, alpha_0 = 5 deg, Psi = 90 deg (pitch leading heave)
########################################

'''
This is a test case to check the prescribed wake solver against the Theodorsen solution
with a wing pitching sinusoidally from 1 to -1 deg and the reduced frequency k = [0.2, 0.6, 1, 3]
'''
########################################
# 1. define geometry
########################################

def anderson_pitch_plunge(St,h_0_star):
    num_nodes = 150
    N_period=3
    # alpha_0_deg = 15
    alpha_0_deg = 5

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

    A = theta_0

    t_vec = np.linspace(0, N_period*T, num_nodes) 

    u_val = np.ones(num_nodes) * v_inf
    w_vel = np.zeros((num_nodes,1)) *np.tan(np.deg2rad(0))

    states_dict = {
        'u': u_val, 'v': np.zeros((num_nodes, 1)), 'w': w_vel,
        'p': np.zeros((num_nodes, 1)), 'q': np.zeros((num_nodes, 1)), 'r': np.zeros((num_nodes, 1)),
        'theta': np.zeros((num_nodes,1))*np.deg2rad(-5), 'psi': np.zeros((num_nodes, 1)),
        'x': np.zeros((num_nodes, 1)), 'y': np.zeros((num_nodes, 1)), 'z': np.zeros((num_nodes, 1)),
        'phiw': np.zeros((num_nodes, 1)), 'gamma': np.zeros((num_nodes, 1)),'psiw': np.zeros((num_nodes, 1)),
    }

    surface_properties_dict = {'surface_names':['wing'],
                                'surface_shapes':[(nx, ny, 3)],
                            'frame':'wing_fixed',}

    h_stepsize = delta_t = t_vec[1] 

    model = csdl.Model()
    ode_surface_shapes = [(num_nodes, ) + item for item in surface_properties_dict['surface_shapes']]

    Pitching = PitchingModel(surface_names=['wing'], surface_shapes=[(nx,ny)], num_nodes=num_nodes,A=A, k=k,
                            v_inf=v_inf, c_0=chord, N_period=N_period, AR=span/chord, h_0=h_0)

    model.add(Pitching, 'pitching')
    model.add(UVLMSolver(num_times=num_nodes, h_stepsize=h_stepsize,states_dict=states_dict,
                                        surface_properties_dict=surface_properties_dict,mesh_val=None), 'uvlm_solver')

    model.add(EfficiencyModel(surface_names=['wing'],surface_shapes=ode_surface_shapes,n_ignore=int(num_nodes/N_period)),name='EfficiencyModel')

    sim = python_csdl_backend.Simulator(model)
        
    t_start = time.time()
    sim.run()

    print('simulation time is', time.time() - t_start)
    print('st is -----------------------', St)  
    print('k is -----------------------', k)  
    print('thrust coefficient is -----------------------', np.average(-sim['wing_C_D_i'][int(num_nodes/N_period):]))  
    print('power coefficient is -----------------------', np.average(-sim['wing_C_D_i'][int(num_nodes/N_period):])/sim['efficiency'])  
    print('efficiency is -----------------------', sim['efficiency'])  
    alpha = A * np.cos(omega*t_vec)

    return sim, t_vec, np.average(-sim['wing_C_D_i'][int(num_nodes/N_period):]), np.average(-sim['wing_C_D_i'][int(num_nodes/N_period):])/sim['efficiency']


chord = 1

# k_list = [0.6, 1.06, 1.3, 1.45, 1.64, 1.8, 2.1] # 0.25
# k_list = [0.1, 0.2, 0.39, 0.55, 0.73, 0.90, 1.05] # 0.7515
k_list = [0.2, 0.39, 0.5, 0.69, 0.8, 0.98] # 0.75590

num_nodes = 150
N_period=3
h_0_star = 0.75
# h_0_star = 0.25
# St_list=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# St_list=np.array([0.1, 0.6])
# St_list=np.array([0.3])
St_list=np.array(k_list) *  2 * h_0_star / (chord*np.pi)

CT_list=[]
CP_list=[]
eta_list=[]
sim_list=[]
for St in St_list:
    sim,t_vec, CT, CP = anderson_pitch_plunge(St,h_0_star)
    CT_list.append(CT)
    CP_list.append(CP)
    eta_list.append(CT/CP)
    sim_list.append(sim)

# folder = '2515'
# folder = '7515'
folder = '7505'
file_name = "_CT"

analytical_CT = np.loadtxt(folder+'/Andersen_'+folder+'_analytical'+file_name+'.txt')
exp_CT = np.loadtxt(folder+'/Andersen_'+folder+'_exp'+file_name+'.txt')
nu_CT = np.loadtxt(folder+'/Andersen_'+folder+'_nu'+file_name+'.txt')

# [0.09579106 0.19687034 0.27005967 0.32226096 0.39451612 0.45994277 0.59181021]

# [0.09552578 0.1914031  0.28096821 0.39058197 0.50043635 0.58733838]
# [0.09705306 0.20007055 0.26631867 0.39040923 0.46650562 0.59564577]

# ST_TE = np.array([0.09579106, 0.19687034, 0.27005967, 0.32226096, 0.39451612, 0.45994277, 0.59181021])
# ST_TE = np.array([0.04798624, 0.09552578, 0.1914031,  0.28096821, 0.39058197, 0.50043635, 0.6]) # 0.7515
ST_TE = np.array([0.09705306, 0.20007055, 0.26631867, 0.39040923, 0.46650562, 0.59564577])
plt.figure(figsize=(5,5))
plt.plot(ST_TE, CT_list, '.-', label='VAST', color='blue')
plt.plot(analytical_CT[:,0], analytical_CT[:,1], '-',label='Andersen linear', color='black')
plt.plot(nu_CT[:,0], nu_CT[:,1], 'o', label='Andersen nonlinear', color='black')
plt.plot(exp_CT[:,0], exp_CT[:,1],'x',  label='Andersen experimental', color='black')
plt.xlabel(r'$St$')
plt.ylabel(r'$C_T$')
plt.legend()
# plt.savefig('7515CT.png',dpi=400,transparent=True)
# plt.savefig('2515CT.png',dpi=400,transparent=True)
plt.savefig('7515CT.png',dpi=400,transparent=True)

file_name = "_CP"

analytical_solution = np.loadtxt(folder+'/Andersen_'+folder+'_analytical'+file_name+'.txt')
exp = np.loadtxt(folder+'/Andersen_'+folder+'_exp'+file_name+'.txt')
nu = np.loadtxt(folder+'/Andersen_'+folder+'_nu'+file_name+'.txt')

plt.figure(figsize=(5,5))
plt.plot(ST_TE, CP_list,'.-',label='VAST', color='blue')
plt.plot(analytical_solution[:,0], analytical_solution[:,1], '-',label='Andersen linear', color='black')
plt.plot(nu[:,0], nu[:,1], 'o', label='Andersen nonlinear', color='black')
plt.plot(exp[:,0], exp[:,1],'x',  label='Andersen experimental', color='black')
plt.xlabel(r'$St$')
plt.ylabel(r'$C_P$')
plt.legend()
# plt.savefig('7515CP.png',dpi=400,transparent=True)
# plt.savefig('2515CP.png',dpi=400,transparent=True)
plt.savefig('7515CP.png',dpi=400,transparent=True)


file_name = "_eta"

analytical_solution = np.loadtxt(folder+'/Andersen_'+folder+'_analytical'+file_name+'.txt')

plt.figure(figsize=(5,5))
plt.plot(ST_TE, eta_list,'.-',label='VAST', color='blue')
plt.plot(analytical_solution[:,0], analytical_solution[:,1], '-',label='Andersen linear', color='black')
plt.plot(nu[:,0], nu_CT[:,1]/nu[:,1], 'o', label='Andersen nonlinear', color='black')
plt.plot(exp[:,0], exp_CT[:,1]/exp[:,1],'x',  label='Andersen experimental', color='black')
plt.xlabel(r'$St$')
plt.ylabel(r'$\eta$')
plt.legend()
# plt.savefig('7515eta.png',dpi=400,transparent=True)
# plt.savefig('2515eta.png',dpi=400,transparent=True)
plt.savefig('755eta.png',dpi=400,transparent=True)

plt.figure()
for i in range(len(sim_list)):
    plt.plot(t_vec[int(num_nodes/N_period):]/np.pi/2-1, sim_list[i]['wing_C_L'][int(num_nodes/N_period):])
plt.gca().invert_yaxis()

plt.figure()
for i in range(len(sim_list)):
    plt.plot(t_vec[int(num_nodes/N_period):]/np.pi/2-1, -sim_list[i]['wing_C_D_i'][int(num_nodes/N_period):])
    

plt.show()

print('St is', ST_TE)
print('h_0_star is', h_0_star)
print('the thrust coefficient is', np.average(-sim['wing_C_D_i'][int(num_nodes/N_period):]))
print('the power coefficient is', np.average(-sim['wing_C_D_i'][int(num_nodes/N_period):])/sim['efficiency'])
print('the efficiency is', sim['efficiency'])



