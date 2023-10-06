'''Example 3 : verification of prescibed vlm with Katz and Plotkin 1991'''
from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver import UVLMSolver

from VAST.utils.generate_mesh import *
import time
import numpy as np
import csdl


# TODO: fix this inertia and wing_fixed frame
# for now, the inertia frame is a frame that zeros out the freestream z directional velocity
# and the wing_fixed frame is a frame that does not zero out the freestream z directional velocity

def run_vl(k=0.5, frame='inertia'):
    ########################################
    # This is a test case to check the prescribed wake solver
    ########################################
    # 1. define geometry
    ########################################
    nx = 5; ny = 15
    chord = 1; span = 4
    num_nodes = 100;  nt = num_nodes

    mesh_dict = {"num_y": ny, "num_x": nx, "wing_type": "rect",  "symmetry": False,
                    "span": span, "root_chord": chord,"span_cos_spacing": False, "chord_cos_spacing": False}
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
    v_inf = omg * chord / 2/ k

    u_val = (np.ones(num_nodes) * np.cos(alpha)).reshape((num_nodes,1)) * v_inf
    w_vel = (np.ones(num_nodes) * np.sin(alpha)).reshape((num_nodes,1)) * v_inf
    # w_vel = np.ones((num_nodes,1)) * np.sin(alpha) - h* omg* np.cos(omg*t_vec).reshape((num_nodes,1))
    # In dynamic_simple.ji there are only the first five elements of the vector, last one is missing
    # signs are the same
    # TODO: check wake geometry and wake velocity

    alpha_equ = np.arctan2(w_vel, u_val)

    states_dict = {
        'u': u_val, 'v': np.zeros((num_nodes, 1)), 'w': w_vel,
        'p': np.zeros((num_nodes, 1)), 'q': np.zeros((num_nodes, 1)), 'r': np.zeros((num_nodes, 1)),
        'theta': alpha*np.ones((num_nodes, 1)), 'psi': np.zeros((num_nodes, 1)),
        'x': np.zeros((num_nodes, 1)), 'y': np.zeros((num_nodes, 1)), 'z': np.zeros((num_nodes, 1)),
        'phiw': np.zeros((num_nodes, 1)), 'gamma': np.zeros((num_nodes, 1)),'psiw': np.zeros((num_nodes, 1)),
    }

    surface_properties_dict = {'surface_names':['wing'],
                                'surface_shapes':[(nx, ny, 3)],
                            'frame':frame,}

    # mesh_val = generate_simple_mesh(nx, ny, num_nodes)
    mesh_val = np.zeros((num_nodes, nx, ny, 3))
    z_offset = h*sin(omg*t_vec)

    for i in range(num_nodes):
        mesh_val[i, :, :, :] = mesh
        mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0] 
        mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] 
        mesh_val[i, :, :, 2] += z_offset[i]

    h_stepsize = delta_t = t_vec[1] 
    model_1 = csdl.Model()
    wing = model_1.create_input('wing', val=mesh_val)
    z_vel = h * np.cos(omg*t_vec)
    z_vel_exp = np.zeros((num_nodes, nx-1, ny-1, 3))
    if frame == 'inertia':
        z_vel_exp[:,:,:,2] = np.einsum('i,jk->ijk',-w_vel.flatten(), np.ones((nx-1,ny-1))) + np.einsum('i,jk->ijk',z_vel, np.ones((nx-1,ny-1)))
    else:
        z_vel_exp[:,:,:,2] = np.einsum('i,jk->ijk',z_vel, np.ones((nx-1,ny-1)))
    wing_coll_vel = model_1.create_input('wing_coll_vel', val=z_vel_exp) 
    import python_csdl_backend
    submodel = UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                        surface_properties_dict=surface_properties_dict,mesh_val=mesh_val)
    model_1.add(submodel, 'VLMSolverModel')
    sim = python_csdl_backend.Simulator(model_1) # add simulator
        
    t_start = time.time()
    sim.run()

    print('simulation time is', time.time() - t_start)
# print('theta',sim['theta'])
######################################################
# make video
######################################################


######################################################
# end make video
######################################################

# plot geometry

# # plot the velocity of the surface
# surface_vel_z = -sim['wing_kinematic_vel'][:,0,2]
# plt.plot(sim['wing'][:,0,0,2],'.-')
# plt.plot(surface_vel_z,'.-')
# # plot the acceleration of the surface
# acc = (surface_vel_z[1:] - surface_vel_z[:-1])/h_stepsize
# plt.plot(acc,'.-' )
# plt.legend(['z','z_vel','z_acc'])
# plt.show()
# # plot force properties
    return t_vec, sim['wing_C_L']
    
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt

k_list = [0.1, 0.3, 0.5]
CL_list = []

for k in k_list:
    t_vec, CL = run_vl(k=k,frame='wing_fixed')
    CL_list.append(CL)
    plt.plot(t_vec/np.pi/2-1, CL)

plt.xlabel('t/T')
plt.ylabel("$C_L$")
plt.xlim([0,1])
plt.ylim([-1,0.1])

a = np.loadtxt('verfication_data/katz_plunging/katz_0.5.txt',delimiter=',')
b = np.loadtxt('verfication_data/katz_plunging/katz_0.3.txt',delimiter=',')
c = np.loadtxt('verfication_data/katz_plunging/katz_0.1.txt',delimiter=',')
plt.plot(a[:,0]/np.pi/2,a[:,1],'.')
plt.plot(b[:,0]/np.pi/2,b[:,1],'^')
plt.plot(c[:,0]/np.pi/2,c[:,1],'x')
plt.gca().invert_yaxis()

plt.legend(['VAST k = '+str(k) for k in k_list ]+[ 'Katz&Plotkin k = '+str(k) for k in k_list])
plt.savefig('vast_heaving.png',dpi=400,transparent=True)
plt.show()