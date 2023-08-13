'''Example 3 : verification of prescibed vlm with Katz and Plotkin 1991'''
from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver import UVLMSolver
from VAST.core.submodels.actuation_submodels.pitching_wing_actuation import PitchingModel
from VAST.utils.generate_mesh import *
from VAST.utils.make_video_vedo import make_video as make_video_vedo
import time
import numpy as np
from visualization import run_visualization
# Script to create optimization problem

be = 'python_csdl_backend'
make_video = 0
plot_cl = 1

# This is a test case to check the prescribed wake solver
########################################
# 1. define geometry
########################################
nx = 5; ny = 15
chord = 1; span = 6
num_nodes = 40;  nt = num_nodes

# this is the same geometry as the dynamic_simple.ji

########################################
# 2. define kinematics
########################################
A = 5
v_inf = np.pi*8
c_0 = 1
k = [0.1]
N_period = 2

omega = 2*v_inf*k[0]/c_0
T = 2*np.pi/(omega)
t_vec = np.linspace(0, N_period*T, num_nodes) 

u_val = np.ones(num_nodes)
w_vel = np.zeros((num_nodes,1)) *np.tan(np.deg2rad(5))

states_dict = {
    'u': u_val, 'v': np.zeros((num_nodes, 1)), 'w': w_vel,
    'p': np.zeros((num_nodes, 1)), 'q': np.zeros((num_nodes, 1)), 'r': np.zeros((num_nodes, 1)),
    'theta': np.zeros((num_nodes,1)), 'psi': np.zeros((num_nodes, 1)),
    'x': np.zeros((num_nodes, 1)), 'y': np.zeros((num_nodes, 1)), 'z': np.zeros((num_nodes, 1)),
    'phiw': np.zeros((num_nodes, 1)), 'gamma': np.zeros((num_nodes, 1)),'psiw': np.zeros((num_nodes, 1)),
}

surface_properties_dict = {'surface_names':['wing'],
                            'surface_shapes':[(nx, ny, 3)],
                           'frame':'inertia',}


h_stepsize = delta_t = t_vec[1] 

import python_csdl_backend
import csdl

model = csdl.Model()

Pitching = PitchingModel(surface_names=['wing'], surface_shapes=[(nx,ny)], num_nodes=num_nodes,A=A, k=k[0],
                         v_inf=v_inf, c_0=c_0, N_period=N_period, AR=span/chord)

model.add(Pitching, 'pitching')
model.add(UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                    surface_properties_dict=surface_properties_dict,mesh_val=None), 'uvlm_solver')

sim = python_csdl_backend.Simulator(model)
    
t_start = time.time()
sim.run()

print('simulation time is', time.time() - t_start)
# print('theta',sim['theta'])
######################################################
# make video
######################################################
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt

plt.plot(t_vec, sim['wing_C_L'])
plt.ylim([0,0.6])
plt.xlim([0,t_vec.max()+1])
plt.show()
######################################################
# end make video
######################################################

# plot geometry

# plot the velocity of the surface
surface_vel_z = -sim['wing_kinematic_vel'][:,0,2]
plt.plot(sim['wing'][:,0,0,2],'.-')
plt.plot(surface_vel_z,'.-')
# plot the acceleration of the surface
acc = (surface_vel_z[1:] - surface_vel_z[:-1])/h_stepsize
plt.plot(acc,'.-' )
plt.legend(['z','z_vel','z_acc'])

# plot force properties

run_visualization(sim,h_stepsize)