'''Example 3 : verification of prescibed vlm with Katz and Plotkin 1991'''
from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver import UVLMSolver

from VAST.utils.generate_mesh import *
from VAST.utils.make_video_vedo import make_video as make_video_vedo
import time
import numpy as np
# Script to create optimization problem

be = 'python_csdl_backend'
make_video = 0
plot_cl = 1

# This is a test case to check the prescribed wake solver
########################################
# 1. define geometry
########################################
nx = 5; ny = 7
chord = 1; span = 4
num_nodes = 42;  nt = num_nodes

mesh_dict = {"num_y": ny, "num_x": nx, "wing_type": "rect",  "symmetry": False, "span": span, "root_chord": chord,"span_cos_spacing": False, "chord_cos_spacing": False}
mesh = generate_mesh(mesh_dict)
# this is the same geometry as the dynamic_simple.ji

########################################
# 2. define kinematics
########################################
n_period = 3 
omg=1 
h=0.1 * chord
alpha = - np.deg2rad(5) 
t_vec = np.linspace(0, n_period*np.pi*2, num_nodes) 

u_val = (np.ones(num_nodes) * np.cos(alpha)).reshape((num_nodes,1)) 
w_vel = np.ones((num_nodes,1)) * np.sin(alpha) - h * np.cos(omg*t_vec).reshape((num_nodes,1))
# In dynamic_simple.ji there are only the first five elements of the vector, last one is missing
# signs are the same
# TODO: check wake geometry and wake velocity

alpha_equ = np.arctan2(w_vel, u_val)

states_dict = {
    'u': u_val, 'v': np.zeros((num_nodes, 1)), 'w': w_vel,
    'p': np.zeros((num_nodes, 1)), 'q': np.zeros((num_nodes, 1)), 'r': np.zeros((num_nodes, 1)),
    'theta': alpha, 'psi': np.zeros((num_nodes, 1)),
    'x': np.zeros((num_nodes, 1)), 'y': np.zeros((num_nodes, 1)), 'z': np.zeros((num_nodes, 1)),
    'phiw': np.zeros((num_nodes, 1)), 'gamma': np.zeros((num_nodes, 1)),'psiw': np.zeros((num_nodes, 1)),
}

surface_properties_dict = {'wing':(nx,ny,3)}

# mesh_val = generate_simple_mesh(nx, ny, num_nodes)
mesh_val = np.zeros((num_nodes, nx, ny, 3))
z_offset = omg*h*sin(omg*t_vec) #* 0
# z_offset = omg*h*sin(omg*t_vec) 

for i in range(num_nodes):
    mesh_val[i, :, :, :] = mesh
    mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0] 
    mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] 
    mesh_val[i, :, :, 2] += z_offset[i]

h_stepsize = delta_t = t_vec[1] 

import python_csdl_backend
sim = python_csdl_backend.Simulator(UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                    surface_properties_dict=surface_properties_dict,mesh_val=mesh_val), mode='rev')
    
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

######################################################
# end make video
######################################################
