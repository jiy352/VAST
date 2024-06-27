'''Example 3 : verification of prescibed vlm with Katz and Plotkin 1991'''
from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver import UVLMSolver

from VAST.utils.generate_mesh import *
import time
import numpy as np
import csdl


# TODO: fix this inertia and wing_fixed frame
# for now, the inertia frame is a frame that zeros out the freestream z directional velocity
# and the wing_fixed frame is a frame that does not zero out the freestream z directional velocity

k = 0.5
frame = 'wing_fixed'
########################################
# This is a test case to check the prescribed wake solver
########################################
# 1. define geometry
########################################
v_inf = 1
alpha = - np.deg2rad(5) 
AR = 4
chord = 1
span = AR * chord
ns = 3
nc = 2

mesh_dict = {"num_y": ns, "num_x": nc, "wing_type": "rect",  "symmetry": False,
                "span": span, "root_chord": chord,"span_cos_spacing": False, "chord_cos_spacing": False}
mesh = generate_mesh(mesh_dict)

num_ts = 4

# this is the same geometry as the dynamic_simple.ji

########################################
# 2. define kinematics
########################################
chord = 1
v_inf = 1
omg = 2*v_inf*k/chord
t_vec = np.linspace(0, np.pi*9/omg, num_ts)
'''figure out these dt'''
# heaving amplitude
h = 0.1 * chord


u_val = (np.ones(num_ts) * np.cos(alpha)).reshape((num_ts,1)) * v_inf
# w_vel = (np.ones(num_ts) * np.sin(alpha)).reshape((num_ts,1)) * v_inf
w_vel = np.ones((num_ts,1)) * np.sin(alpha) * v_inf#- h* np.cos(omg*t_vec).reshape((num_ts,1))

# TODO: check wake geometry and wake velocity

alpha_equ = np.arctan2(w_vel, u_val)

states_dict = {
    'u': u_val, 'v': np.zeros((num_ts, 1)), 'w': w_vel,
    'p': np.zeros((num_ts, 1)), 'q': np.zeros((num_ts, 1)), 'r': np.zeros((num_ts, 1)),
    'theta': alpha*np.ones((num_ts, 1)), 'psi': np.zeros((num_ts, 1)),
    'x': np.zeros((num_ts, 1)), 'y': np.zeros((num_ts, 1)), 'z': np.zeros((num_ts, 1)),
    'phiw': np.zeros((num_ts, 1)), 'gamma': np.zeros((num_ts, 1)),'psiw': np.zeros((num_ts, 1)),
}

surface_properties_dict = {'surface_names':['wing'],
                            'surface_shapes':[(nc, ns, 3)],
                        'frame':frame,}

# mesh_val = generate_simple_mesh(nc, ns, num_nodes)
mesh_val = np.zeros((num_ts, nc, ns, 3))
# z_offset = h*sin(omg*t_vec)
z_offset = np.zeros(num_ts) 

for i in range(num_ts):
    mesh_val[i, :, :, :] = mesh
    mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0] 
    mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] 
    mesh_val[i, :, :, 2] += z_offset[i]

h_stepsize = delta_t = t_vec[1] 


model_1 = csdl.Model()
wing = model_1.create_input('wing', val=mesh_val)
wing_coll_val = np.einsum('i,jkl->ijkl',np.array([1,-1,1,-1])*0.1, np.ones((nc-1, ns-1, 3)))
wing_coll_vel = model_1.create_input('wing_coll_vel', val=wing_coll_val)
rho = model_1.create_input('density', val=np.ones((num_ts,1)))  
z_vel = h * np.cos(omg*t_vec)


import python_csdl_backend
submodel = UVLMSolver(num_times=num_ts,h_stepsize=h_stepsize,states_dict=states_dict,
                    surface_properties_dict=surface_properties_dict,mesh_val=mesh_val)
model_1.add(submodel, 'VLMSolverModel')
sim = python_csdl_backend.Simulator(model_1) # add simulator
    
t_start = time.time()
sim.run()

print('simulation time is', time.time() - t_start)

print(sim['wing_C_L'])
k_list = [k]
CL = sim['wing_C_L']
import matplotlib.pyplot as plt
# turn off tex
plt.rc('text', usetex=False)
plt.plot(t_vec/np.pi/2-2, CL)

plt.xlabel('t/T')
plt.ylabel("$C_L$")
plt.xlim([0,1])
plt.ylim([-1,0.1])

a = np.loadtxt('verfication_data/katz_plunging/katz_0.5.txt',delimiter=',')

plt.plot(a[:,0]/np.pi/2,a[:,1],'.')

plt.gca().invert_yaxis()

plt.legend(['VAST k = '+str(k) for k in k_list ]+[ 'Katz&Plotkin k = '+str(k) for k in k_list])
plt.savefig('vast_heaving.png',dpi=400,transparent=True)
plt.show()