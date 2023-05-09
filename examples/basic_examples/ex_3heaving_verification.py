'''Example 2 : verification of prescibed vlm with Katz and Plotkin 1991'''

from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver import RunModel

from VAST.utils.generate_mesh import *
from VAST.utils.make_video_vedo import make_video as make_video_vedo
import time
import numpy as np
# Script to create optimization problem

be = 'python_csdl_backend'
make_video = 0
plot_cl = 1
########################################
# define mesh here
########################################
nx = 15; ny = 5
chord = 1; span = 4
num_nodes = 99;  nt = num_nodes
n_period = 4
omg=1
h=0.1
alpha = - np.deg2rad(5)

t_vec = np.linspace(0, n_period*np.pi*2, num_nodes)

u_val = (np.ones(num_nodes) * np.cos(alpha)).reshape((num_nodes,1))
w_vel = np.ones((num_nodes,1)) * np.sin(alpha) - h * np.cos(omg*t_vec).reshape((num_nodes,1))

alpha_equ = np.arctan2(w_vel, u_val)

states_dict = {
    'u': u_val, 'v': np.zeros((num_nodes, 1)), 'w': w_vel,
    'p': np.zeros((num_nodes, 1)), 'q': np.zeros((num_nodes, 1)), 'r': np.zeros((num_nodes, 1)),
    'theta': alpha_equ, 'psi': np.zeros((num_nodes, 1)),
    'x': np.zeros((num_nodes, 1)), 'y': np.zeros((num_nodes, 1)), 'z': np.zeros((num_nodes, 1)),
    'phiw': np.zeros((num_nodes, 1)), 'gamma': np.zeros((num_nodes, 1)),'psiw': np.zeros((num_nodes, 1)),
}

mesh_dict = {"num_y": ny, "num_x": nx, "wing_type": "rect",  "symmetry": False, "span": span, "root_chord": chord,"span_cos_spacing": False, "chord_cos_spacing": False}
mesh = generate_mesh(mesh_dict)

surface_properties_dict = {'wing':(nx,ny,3)}

# mesh_val = generate_simple_mesh(nx, ny, num_nodes)
mesh_val = np.zeros((num_nodes, nx, ny, 3))
z_offset = h*sin(omg*t_vec)

for i in range(num_nodes):
    mesh_val[i, :, :, :] = mesh
    mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0] 
    mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] 
    mesh_val[i, :, :, 2] += z_offset[i]

h_stepsize = delta_t = 1 

if be == 'csdl_om':
    import csdl_om
    sim = csdl_om.Simulator(RunModel(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                        surface_properties_dict=surface_properties_dict,mesh_val=mesh_val), mode='rev')
if be == 'python_csdl_backend':
    import python_csdl_backend
    sim = python_csdl_backend.Simulator(RunModel(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                        surface_properties_dict=surface_properties_dict,mesh_val=mesh_val), mode='rev')
    
t_start = time.time()
sim.run()
print('simulation time is', time.time() - t_start)
# print('theta',sim['theta'])

######################################################
# make video
######################################################

if make_video == 1:
    make_video_vedo(surface_properties_dict,num_nodes,sim)
if plot_cl == 1:
    import matplotlib.pyplot as plt
    plt.plot(t_vec,sim['wing_C_L'],'.-')
    plt.gca().invert_yaxis()
    plt.show()
    cl = sim['wing_C_L'][-int(num_nodes/4)*2:-int(num_nodes/4)-1]
    cl_ref = np.loadtxt('data.txt')
    plt.plot(np.linspace(0, np.pi*2,cl.shape[0]),cl,'.-')
    plt.plot(np.linspace(0, np.pi*2,cl_ref.shape[0]),cl_ref,'.-')
    plt.legend(['VAST','BYU_UVLM'])
    plt.gca().invert_yaxis()
    plt.show()

cl_ref = np.loadtxt('/Users/jyan/Documents/packages/VAST/tests/verifications/uvlm_plunging.txt').flatten()
cl = sim['wing_C_L'][-int(num_nodes/4)*2:-int(num_nodes/4)-1].flatten()
print('the error is', np.linalg.norm(cl-cl_ref)/np.linalg.norm(cl_ref)*100,'%')
# sim.compute_totals(of='',wrt='*')
######################################################
# end make video
######################################################

# sim.visualize_implementation()
# partials = sim.check_partials(compact_print=True)
# sim.prob.check_totals(compact_print=True)

# sim.check_totals(compact_print=True)