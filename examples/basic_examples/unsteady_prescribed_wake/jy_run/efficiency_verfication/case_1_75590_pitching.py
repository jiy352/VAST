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

def anderson_pitching(St,h_0_star):
    num_nodes = 150
    N_period=3
    alpha_0_deg = 15

    # h_0_star=0.1
    save_vtk=False

    save_results=False
    path = "verfication_data/theodorsen"
    nx = 11; ny = 7         # num_pts in chordwise and spanwise direction
    chord = 1; span = 6   # chord and span of the wing
    b = chord/2

    omega = 1
    h_0 = h_0_star * chord
    v_inf = 4* np.pi *omega*h_0 / ( St)
    k = omega * b/v_inf

    print('k is', k)
    # exit()
    T = 2*np.pi/(omega)

    theta_0 = np.rad2deg(np.arctan2(omega*h_0, v_inf) - np.deg2rad(alpha_0_deg))
    # array([-14.5440643 , -13.63242379, -12.72147527])

    A = 1

    t_vec = np.linspace(0, N_period*T, num_nodes) 

    u_val = np.ones(num_nodes) * v_inf
    w_vel = np.zeros((num_nodes,1)) *np.tan(np.deg2rad(0))

    states_dict = {
        'u': u_val, 'v': np.zeros((num_nodes, 1)), 'w': w_vel,
        'p': np.zeros((num_nodes, 1)), 'q': np.zeros((num_nodes, 1)), 'r': np.zeros((num_nodes, 1)),
        'theta': np.zeros((num_nodes,1)), 'psi': np.zeros((num_nodes, 1)),
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
                            v_inf=v_inf, c_0=chord, N_period=N_period, AR=span/chord, h_0=h_0*0)

    model.add(Pitching, 'pitching')
    model.add(UVLMSolver(num_times=num_nodes, h_stepsize=h_stepsize,states_dict=states_dict,
                                        surface_properties_dict=surface_properties_dict,mesh_val=None), 'uvlm_solver')

    model.add(EfficiencyModel(surface_names=['wing'],surface_shapes=ode_surface_shapes,n_ignore=int(num_nodes/N_period)),name='EfficiencyModel')

    sim = python_csdl_backend.Simulator(model)
        
    t_start = time.time()
    sim.run()

    print('simulation time is', time.time() - t_start)
    print('density is -----------------------',sim['density'][0,0])  
    alpha = A * np.cos(omega*t_vec)

    return sim, alpha, np.average(-sim['wing_C_D_i'][int(num_nodes/N_period):]), np.average(-sim['wing_C_D_i'][int(num_nodes/N_period):])/sim['efficiency']

   
alpha_file_name = '../../verfication_data/theodorsen/analytical_solution/alpha_1deg0.2.txt'
Cl_file_name = [
                '../../verfication_data/theodorsen/analytical_solution/Cl_1deg0.2.txt',
                '../../verfication_data/theodorsen/analytical_solution/Cl_1deg0.6.txt',
                '../../verfication_data/theodorsen/analytical_solution/Cl_1deg1.txt',
                '../../verfication_data/theodorsen/analytical_solution/Cl_1deg3.txt'
                ]

def plot_cl(alpha_file_name, Cl_file_name, L_file_name=None):
    for i in range(len(Cl_file_name)):
        alpha = np.loadtxt(alpha_file_name)
        if alpha.max()<0.2:
            alpha = np.rad2deg(alpha)
        Cl = np.loadtxt(Cl_file_name[i])
        print('i', Cl_file_name[i])

        if L_file_name is not None:
            L = np.loadtxt(L_file_name[i])
        else:
            L = None
        if np.loadtxt(alpha_file_name).max()>0.2:
            plt.plot(alpha, Cl,'--')
        else:
            plt.plot(alpha, Cl,'--')

    # St=0.5/np.pi

omega = chord = 1
k_list = [0.2, 0.6, 1, 3]
num_nodes = 150
N_period=3
# h_0_star = 0.1
h_0_star = 0.75
St_list=np.array(k_list) * np.pi * 8 * h_0_star / (chord)
# St_list=np.array([0.1, 0.3, 0.5])
CT_list=[]
CP_list=[]
eta_list=[]
sim_list=[]
for St in St_list:
    sim,alpha, CT, CP = anderson_pitching(St,h_0_star)
    CT_list.append(CT)
    CP_list.append(CP)
    eta_list.append(CT/CP)
    sim_list.append(sim)

plt.figure()
plt.plot(St_list, CT_list, '.-')
plt.xlabel(r'$St$')
plt.ylabel(r'$C_T$')

plt.figure()
plt.plot(St_list, CP_list,'.-')
plt.xlabel(r'$St$')
plt.ylabel(r'$C_P$')

plt.figure()
plt.plot(St_list, eta_list,'.-')
plt.xlabel(r'$St$')
plt.ylabel(r'$\eta$')

plt.figure()
for i in range(len(sim_list)):
    plt.plot(alpha[int(num_nodes/N_period):], sim_list[i]['wing_C_L'][int(num_nodes/N_period):])
    plot_cl(alpha_file_name, Cl_file_name)


plt.figure()
for i in range(len(sim_list)):
    plt.plot(alpha[int(num_nodes/N_period):], -sim_list[i]['wing_C_D_i'][int(num_nodes/N_period):])
    

plt.show()




# print('the thrust coefficient is', np.average(-sim['wing_C_D_i'][int(num_nodes/N_period):]))

# np.average(-sim['thrust'][int(num_nodes/N_period):])/(0.5*sim['density'][0,0]*v_inf**2*chord*span)

# alpha = A * np.cos(omega*t_vec)


# plt.figure()

# plt.plot(alpha[int(num_nodes/N_period):], -sim['wing_C_D_i'][int(num_nodes/N_period):])
# plt.xlabel(r'$\alpha$')
# plt.ylabel(r'$C_T$')

# plt.figure()
# plt.plot(t_vec[int(num_nodes/N_period):], -sim['wing_C_D_i'][int(num_nodes/N_period):])

# plt.figure()
# plt.plot(alpha[int(num_nodes/N_period):], sim['wing_C_L'][int(num_nodes/N_period):])
# plt.xlabel(r'$\alpha$')
# plt.ylabel(r'$C_L$')

# plot_cl(alpha_file_name, Cl_file_name)
# plt.legend(['VAST k = 0.2', 'VAST k = 0.6', 'VAST k = 1', 'VAST k = 3',
#             'Theodorsen k = 0.2', 'Theodorsen k = 0.6', 'Theodorsen k = 1', 'Theodorsen k = 3'])
# plt.xlabel(r'$\alpha$')
# plt.ylabel(r'$C_l$')
# plt.savefig('verfication_data/theodorsen/C_L_theodorsen_vs_analytical.png',dpi=400,transparent=True)
# plt.show()