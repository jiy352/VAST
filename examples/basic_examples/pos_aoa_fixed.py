'''Example 3 : verification of prescibed vlm with Katz and Plotkin 1991'''
from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver import UVLMSolver

from VAST.utils.generate_mesh import *
from VAST.utils.make_video_vedo import make_video as make_video_vedo
import time
import numpy as np
# Script to create optimization problem


# This is a test case to check the prescribed wake solver
def run_fixed(span,num_nodes):
    ########################################
    # 1. define geometry
    ########################################
    nx = 5; ny = 13
    chord = 1; 
    nt = num_nodes

    mesh_dict = {"num_y": ny, "num_x": nx, "wing_type": "rect",  "symmetry": False,
                    "span": span, "root_chord": chord,"span_cos_spacing": False, "chord_cos_spacing": False}
    mesh = generate_mesh(mesh_dict)
    # this is the same geometry as the dynamic_simple.ji

    ########################################
    # 2. define kinematics
    ########################################
    # n_period = 3 #/ 7 
    # omg = 1
    # h = 0.1 * chord
    alpha = np.deg2rad(5) 
    # t_vec = np.linspace(0, n_period*np.pi*2, num_nodes) *0.5
    t_vec = np.linspace(0, 10, num_nodes) 
    print('delta_t',np.cos(alpha)*t_vec[1])

    u_val = (np.ones(num_nodes) * np.cos(alpha)).reshape((num_nodes,1)) 
    w_vel = np.ones((num_nodes,1)) *np.tan(np.deg2rad(5))
    # In dynamic_simple.ji there are only the first five elements of the vector, last one is missing
    # signs are the same
    # TODO: check wake geometry and wake velocity


    states_dict = {
        'u': u_val, 'v': np.zeros((num_nodes, 1)), 'w': w_vel,
        'p': np.zeros((num_nodes, 1)), 'q': np.zeros((num_nodes, 1)), 'r': np.zeros((num_nodes, 1)),
        'theta': alpha* np.ones((num_nodes,1)), 'psi': np.zeros((num_nodes, 1)),
        'x': np.zeros((num_nodes, 1)), 'y': np.zeros((num_nodes, 1)), 'z': np.zeros((num_nodes, 1)),
        'phiw': np.zeros((num_nodes, 1)), 'gamma': np.zeros((num_nodes, 1)),'psiw': np.zeros((num_nodes, 1)),
    }

    surface_properties_dict = {'surface_names':['wing'],
                                'surface_shapes':[(nx, ny, 3)],
                            'frame':'wing_fixed',}

    # mesh_val = generate_simple_mesh(nx, ny, num_nodes)
    mesh_val = np.zeros((num_nodes, nx, ny, 3))
    z_offset = -w_vel.flatten()*t_vec*0

    for i in range(num_nodes):
        mesh_val[i, :, :, :] = mesh
        mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0] 
        mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] 
        mesh_val[i, :, :, 2] += z_offset[i]


    h_stepsize = delta_t = t_vec[1] 

    import python_csdl_backend
    import csdl

    model = csdl.Model()

    vz = -np.zeros((num_nodes,nx-1,ny-1,3))*np.tan(np.deg2rad(5)).copy()

    model.create_input('wing_coll_vel', val = vz)

    model.create_input('wing', val = mesh_val)

    model.add(UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                        surface_properties_dict=surface_properties_dict,mesh_val=mesh_val), 'uvlm_solver')

    sim = python_csdl_backend.Simulator(model)
        
    t_start = time.time()
    sim.run()
    print('simulation time is', time.time() - t_start)

    return sim['wing_C_L'], t_vec, sim


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt

be = 'python_csdl_backend'
make_video = 0
plot_cl = 1
# span = [4, 8, 12, 20, 1000]
# span = [4, ]
# span = [8, ]
# span = [12, ]
# span = [20, ]
span = [10000, ]

# span = [4]
# span = [1000]*4
# num_nodes = [40,70, 100, 120]
# num_nodes = [160]*5
num_nodes = [160]
# num_nodes = [100]
# num_nodes = [42]
# print('theta',sim['theta'])
######################################################
# make video
######################################################
wing_C_L_list = []
t_vec_list = []
for (i,j) in zip(span,num_nodes):
    wing_C_L, t_vec, sim = run_fixed(i,j)
    # wing_C_L = wing_C_L_list[0]
    # t_vec = t_vec_list[0]
    plt.plot(t_vec, wing_C_L,'.-')
    plt.ylim([0,0.6])
    plt.xlim([0,t_vec.max()+1])
    np.savetxt('wing_C_L_'+str(i)+'_'+str(j)+'.txt',wing_C_L)
    np.savetxt('t_vec_'+str(i)+'_'+str(j)+'.txt',t_vec)
    wing_C_L_list.append(wing_C_L)
    t_vec_list.append(t_vec)

plt.legend(['AR = '+str(i) for i in span])
plt.xlabel('$U_{\inf}t/c$')
plt.ylabel('C_L')
plt.savefig('C_L.png',dpi=300,transparent=True)
plt.show()
######################################################
# end make video
######################################################

# # plot geometry

# # plot the velocity of the surface
# surface_vel_z = -sim['wing_kinematic_vel'][:,0,2]
# plt.plot(sim['wing'][:,0,0,2],'.-')
# plt.plot(surface_vel_z,'.-')
# # plot the acceleration of the surface
# acc = (surface_vel_z[1:] - surface_vel_z[:-1])/h_stepsize
# plt.plot(acc,'.-' )
# plt.legend(['z','z_vel','z_acc'])

# plot force properties

from visualization import run_visualization
run_visualization(sim,t_vec[1],'fixed')