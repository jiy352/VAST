'''Example 5 : verification of prescibed vlm with Katz and Plotkin 1991'''
from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver import UVLMSolver

from VAST.utils.generate_mesh import *
from VAST.utils.make_video_vedo import make_video as make_video_vedo
import time
import numpy as np

import python_csdl_backend
import csdl

# there are two options for the frame: 'inertia' and 'wing_fixed
# This might subject to change in the future
# 'inertia' means the frame is moving with a constant x direction, 'wing_fixed' means the frame is fixed
# for the "wing_fixed" frame, the wing is not moving, therefore z_offset is not needed for the wing mesh
# for the "inertia" frame, the wing is moving, therefore z_offset is needed for the wing mesh

# This is a test case to check the prescribed wake solver
def run_fixed(span,num_nodes,frame='wing_fixed'):
    ########################################
    # 1. define geometry
    ########################################
    nx = 5; ny = 13
    chord = 1; 
    nt = num_nodes

    mesh_dict = {"num_y": ny, "num_x": nx, "wing_type": "rect",  "symmetry": False,
                    "span": span, "root_chord": chord,"span_cos_spacing": False, "chord_cos_spacing": False}
    mesh = generate_mesh(mesh_dict)

    ########################################
    # 2. define kinematics
    ########################################

    alpha = np.deg2rad(5) 
    t_vec = np.linspace(0, 9, num_nodes) 

    u_val = (np.ones(num_nodes) * np.cos(alpha)).reshape((num_nodes,1)) 
    w_vel = np.ones((num_nodes,1)) *np.sin(alpha)

    states_dict = {
        'u': u_val, 'v': np.zeros((num_nodes, 1)), 'w': w_vel,
        'p': np.zeros((num_nodes, 1)), 'q': np.zeros((num_nodes, 1)), 'r': np.zeros((num_nodes, 1)),
        'theta': alpha* np.ones((num_nodes,1)), 'psi': np.zeros((num_nodes, 1)),
        'x': np.zeros((num_nodes, 1)), 'y': np.zeros((num_nodes, 1)), 'z': np.zeros((num_nodes, 1)),
        'phiw': np.zeros((num_nodes, 1)), 'gamma': np.zeros((num_nodes, 1)),'psiw': np.zeros((num_nodes, 1)),
    }

    surface_properties_dict = {'surface_names':['wing'],
                                'surface_shapes':[(nx, ny, 3)],
                            'frame':frame,}

    mesh_val = np.zeros((num_nodes, nx, ny, 3))

    if frame == 'wing_fixed':
        z_offset = -w_vel.flatten()*t_vec*0
        vz = -np.zeros((num_nodes,nx-1,ny-1,3))*np.tan(np.deg2rad(5)).copy()
    elif frame == 'inertia':
        z_offset = (-np.ones((num_nodes,1)) *np.tan(alpha)).flatten()*t_vec
        vz = -np.ones((num_nodes,nx-1,ny-1,3))*np.tan(np.deg2rad(5)).copy()
        vz[:,:,:,0] = 0
        vz[:,:,:,1] = 0
    for i in range(num_nodes):
        mesh_val[i, :, :, :] = mesh
        mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0] 
        mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] 
        mesh_val[i, :, :, 2] += z_offset[i]

    h_stepsize = delta_t = t_vec[1] 

    model = csdl.Model()

    model.create_input('wing_coll_vel', val = vz)

    model.create_input('wing', val = mesh_val)

    model.add(UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                        surface_properties_dict=surface_properties_dict,mesh_val=mesh_val,
                                        symmetry=True), 'uvlm_solver')
    sim = python_csdl_backend.Simulator(model)
        
    t_start = time.time()
    sim.run()
    print('simulation time is', time.time() - t_start)

    wing_C_L = sim['wing_C_L']
    del sim
    return wing_C_L, t_vec

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt

be = 'python_csdl_backend'
make_video = 0
plot_cl = 1
span = [4, 8, 12, 20, 1000]

num_nodes = [141] * len(span)

wing_C_L_list = []
t_vec_list = []
for (i,j) in zip(span,num_nodes):
    wing_C_L, t_vec = run_fixed(i,j)
    plt.plot(t_vec, wing_C_L,'.-')
    # wing_C_L_, t_vec_ = run_fixed(i,num_nodes=j,frame='inertia')
    # plt.plot(t_vec_, wing_C_L_,'.-')
    plt.ylim([0,0.6])
    plt.xlim([0,t_vec.max()+1])
    # np.savetxt('wing_C_L_'+str(i)+'_'+str(j)+'.txt',wing_C_L)
    # np.savetxt('t_vec_'+str(i)+'_'+str(j)+'.txt',t_vec)
    # wing_C_L_list.append(wing_C_L)
    # t_vec_list.append(t_vec)

plt.legend(['AR = '+str(i) for i in span])
plt.xlabel('$U_{\inf}t/c$')
plt.ylabel('C_L')
plt.savefig('verfication_data/sudden_acc/C_L.png',dpi=300,transparent=True)
plt.show()

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

# from visualization import run_visualization
# run_visualization(sim,t_vec[1],'fixed')