'''Example 2 : verification of prescibed vlm with Katz and Plotkin 1991'''

from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver_eel import RunModel

from VAST.utils.generate_mesh import *
from VAST.utils.make_video_vedo import make_video as make_video_vedo
import time
import numpy as np
# Script to create optimization problem

be = 'python_csdl_backend'
make_video = 0
plot_ct = 1


########################################
# define actuatons here
########################################
nx = 15; ny = 5
# nx = 5; ny = 3
# chord = 1; span = 4
num_nodes = 99;  nt = num_nodes
# n_period = 2
# omg=1
# h=0.1
# alpha = - np.deg2rad(0)
v_inf = 0.3
lambda_ = 1
N_period= 4         
st = 0.15
A = 0.125
# f = st*v_inf/A 
f = 0.48
surface_properties_dict = {'wing':(nx,ny,3)}
# t_vec = np.linspace(0, n_period*np.pi*2, num_nodes)

u_val = (np.ones(num_nodes)).reshape((num_nodes,1)) * v_inf
w_vel = np.zeros((num_nodes, 1))

alpha_equ = np.arctan2(w_vel, u_val)

states_dict = {
    'u': u_val, 'v': np.zeros((num_nodes, 1)), 'w': w_vel,
    'p': np.zeros((num_nodes, 1)), 'q': np.zeros((num_nodes, 1)), 'r': np.zeros((num_nodes, 1)),
    'theta': alpha_equ, 'psi': np.zeros((num_nodes, 1)),
    'x': np.zeros((num_nodes, 1)), 'y': np.zeros((num_nodes, 1)), 'z': np.zeros((num_nodes, 1)),
    'phiw': np.zeros((num_nodes, 1)), 'gamma': np.zeros((num_nodes, 1)),'psiw': np.zeros((num_nodes, 1)),
}
t_vec = np.linspace(0,N_period/f,num_nodes)
h_stepsize = t_vec[1]

########################################
# generate mesh here
########################################

if be == 'csdl_om':
    import csdl_om
    sim = csdl_om.Simulator(RunModel(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                        surface_properties_dict=surface_properties_dict), mode='rev')
if be == 'python_csdl_backend':
    import python_csdl_backend
    sim = python_csdl_backend.Simulator(RunModel(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                        surface_properties_dict=surface_properties_dict), mode='rev')
    
t_start = time.time()
sim.run()
thrust = sim['thrust']
thrust_coeff = thrust/(0.5*sim['density']*sim['u'][0]**2*0.13826040386294708)
thrust_coeff_avr = np.average(thrust_coeff)
exit()

from modopt.csdl_library import CSDLProblem

from modopt.scipy_library import SLSQP
# Define problem for the optimization
prob = CSDLProblem(
    problem_name='eel_kinematic_opt',
    simulator=sim,
)
optimizer = SLSQP(prob, maxiter=5)


optimizer.solve()
optimizer.print_results(summary_table=True)

print('simulation time is', time.time() - t_start)
# np.savetxt('cl12full',sim['wing_C_L'])
######################################################
# make video
######################################################

if make_video == 1:
    make_video_vedo(surface_properties_dict,num_nodes,sim)
if plot_ct == 1:
    import matplotlib.pyplot as plt
    plt.plot(t_vec,thrust_coeff,'.-')
    # plt.gca().invert_yaxis()
    plt.show()
    # cl = sim['wing_C_L'][-int(num_nodes/4)*2:-int(num_nodes/4)-1]
    # cl_ref = np.loadtxt('data.txt')
    # plt.plot(np.linspace(0, np.pi*2,cl.shape[0]),cl,'.-')
    # plt.plot(np.linspace(0, np.pi*2,cl_ref.shape[0]),cl_ref,'.-')
    # plt.legend(['VAST','BYU_UVLM'])
    # plt.gca().invert_yaxis()
    # plt.show()


# sim.compute_totals(of='',wrt='*')
######################################################
# end make video
######################################################

# sim.visualize_implementation()
# partials = sim.check_partials(compact_print=True)
# sim.prob.check_totals(compact_print=True)

# sim.check_totals(compact_print=True)


import matplotlib.pyplot as plt
for i in range(num_nodes):
    plt.plot(sim['wing'][i,:,0,0],sim['wing'][i,:,0,1])


plt.show()

import pyvista as pv

# for i in range(num_nodes-1):
#     wake_mesh = sim["op_wing_wake_coords"][i][:i,:,:].reshape(-1,3,3)
#     grid = pv.StructuredGrid(wake_mesh[:,:,0], wake_mesh[:,:,1], wake_mesh[:,:,2])
#     grid.save(f"wake_mesh_{i}.vtk")


import pyvista as pv

# for i in range(num_nodes):
#     wake_mesh = sim["op_wing_wake_coords"][i].reshape(-1,3,3)
#     grid = pv.StructuredGrid(wake_mesh[:,:,0], wake_mesh[:,:,1], wake_mesh[:,:,2])
#     bd_vtx_mesh = sim["wing_bd_vtx_coords"][i]
#     grid_mesh = pv.StructuredGrid(bd_vtx_mesh[:,:,0], bd_vtx_mesh[:,:,1], bd_vtx_mesh[:,:,2])
#     grid.save(f"wake_mesh_{i}.vtk")
#     grid_mesh.save(f"grid_mesh_{i}.vtk")