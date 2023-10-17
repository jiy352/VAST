import numpy as np

import csdl
import numpy as np
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL

from scipy.spatial.transform import Rotation as R
import python_csdl_backend

from VAST.utils.generate_mesh import *

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt

class PitchingModel(ModuleCSDL):
    """
    Compute the rotated mesh given axis, .

    parameters
    ----------
    def_mesh[num_nodes,num_pts_chord, num_pts_span, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface.

    Returns
    -------
    1. mesh
    2. coll_vel (rotational vel)
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('num_nodes')
        self.parameters.declare('A')
        self.parameters.declare('k')
        self.parameters.declare('v_inf', default=1)
        self.parameters.declare('c_0', default=1)
        self.parameters.declare('AR', default=6)
        self.parameters.declare('N_period', default=4)
        self.parameters.declare('h_0')

    def define(self):    
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        A = self.parameters['A'] 
        k_i = self.parameters['k']
        v_inf = self.parameters['v_inf']
        c_0 = self.parameters['c_0']
        N_period = self.parameters['N_period']
        AR = self.parameters['AR']
        num_nodes = self.parameters['num_nodes']
        h_0 = self.parameters['h_0']  


        omega = v_inf*k_i/(c_0/2)
        T = 2*np.pi/(omega)
        t = np.linspace(0,N_period*T,num_nodes)

        f = A * np.cos(omega*t) 

        h = h_0*np.sin(omega*t)

        f_dot  = np.deg2rad(-A*omega*np.sin(omega*t))
        h_dot = h_0*omega*np.cos(omega*t) 
        # print('h_0 is -----------', h_0)
        # print('h is -----------', h)
        # exit()
        # plt.plot(t,f)
        # plt.plot(t,-A*omega*np.sin(omega*t))
        # plt.show()
        # print('f----------------------------------',f)
        # print('f_dot----------------------------------',np.rad2deg(f_dot))
        # print('AR----------------------------------',AR)
        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            nx = surface_shape[0]; ny = surface_shape[1]
            chord = c_0; span = chord*AR


            mesh_dict = {"num_y": ny, "num_x": nx, "wing_type": "rect",  "symmetry": False,
                            "span": span, "root_chord": chord,"span_cos_spacing": False, "chord_cos_spacing": False}
            mesh = generate_mesh(mesh_dict)
            mesh[:,:,0]  = mesh[:,:,0] + 0.33*chord # 1/3 chord pitching axis
            r = R.from_euler('y', f, degrees=True).as_matrix() # num_nodes,3,3

            rotated_mesh = np.einsum('ijk,lmk->ilmj', r, mesh)


            plunged_dist = np.zeros((num_nodes, nx, ny, 3))
            delta_z = np.einsum('i,jk->ijk',h, np.ones((nx,ny)))
            plunged_dist[:,:,:,2] = delta_z
            plunged_dist_csdl = self.create_input(surface_name+'_plunged_dist', shape=(num_nodes, nx, ny, 3), val=plunged_dist)

            rotated_mesh_csdl_temp = self.create_input(surface_name+'_rotated_mesh', rotated_mesh )
            rotated_mesh_csdl = rotated_mesh_csdl_temp + plunged_dist_csdl
            self.register_output(surface_name, rotated_mesh_csdl)

            shift_vel = True

            if shift_vel:
                f_dot_exp = np.einsum('il,jk->ijkl',np.array([np.zeros(num_nodes), f_dot, np.zeros(num_nodes)]).T,
                                    np.ones((nx,ny)))

                rot_vel = self.create_input(surface_name+'_rot_velocity', f_dot_exp)

                coll_pts = (rotated_mesh_csdl[:,:-1,:-1,:]+rotated_mesh_csdl[:,:-1,1:,:])*0.125 +\
                    (rotated_mesh_csdl[:,1:,:-1,:]+rotated_mesh_csdl[:,1:,1:,:])*0.75
                # ref_axis
                # print('rot_vel',rot_vel.shape)
                # print('coll_pts',coll_pts.shape)

                plunging_z_vel = np.einsum('i,jk->ijk',h_dot, np.ones((nx-1,ny-1)))
                plunging_vel = np.zeros((num_nodes, nx-1, ny-1, 3))
                plunging_vel[:,:,:,2] = plunging_z_vel

                plunging_vel_csdl = self.create_input(surface_name+'_plunging_vel', plunging_vel)

                # coll_vel = csdl.cross(rot_vel, coll_pts, axis=3)
                mesh_vel = csdl.cross(rot_vel, rotated_mesh_csdl, axis=3)
                coll_vel = (mesh_vel[:,:-1,:-1,:]+mesh_vel[:,:-1,1:,:])*0.25 +\
                        (mesh_vel[:,1:,:-1,:]+mesh_vel[:,1:,1:,:])*0.25 +\
                            plunging_vel_csdl
            else:
                f_dot_exp = np.einsum('il,jk->ijkl',np.array([np.zeros(num_nodes), f_dot, np.zeros(num_nodes)]).T,
                    np.ones((nx-1,ny-1)))
                rot_vel = self.create_input(surface_name+'_rot_velocity', f_dot_exp)
                coll_pts = (rotated_mesh_csdl[:,:-1,:-1,:]+rotated_mesh_csdl[:,:-1,1:,:])*0.125 +\
                    (rotated_mesh_csdl[:,1:,:-1,:]+rotated_mesh_csdl[:,1:,1:,:])*0.75
                # ref_axis
                # print('rot_vel',rot_vel.shape)
                # print('coll_pts',coll_pts.shape)

                coll_vel = csdl.cross(rot_vel, coll_pts, axis=3) 
                # mesh_vel = csdl.cross(rot_vel, coll_pts, axis=3)
                # coll_vel = (mesh_vel[:,:-1,:-1,:]+mesh_vel[:,:-1,1:,:])*0.25 +\
                #         (mesh_vel[:,1:,:-1,:]+mesh_vel[:,1:,1:,:])*0.25

            self.register_output(surface_name+'_coll_vel', coll_vel)


# A = 5
# v_inf = 1
# c_0 = 1
# # k = [0.1,0.3,0.5]
# k = [0.1]
# N_period = 4
# num_nodes = 80



# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)

# import matplotlib.pyplot as plt
# for k_i in k:
#     omega = 2*v_inf*k_i/c_0
#     T = 2*np.pi/(omega)
#     t = np.linspace(0,N_period*T,num_nodes)
#     ###########################################
#     f = A * np.cos(omega*t)
#     ###########################################
#     f_dot  = -A*2*v_inf*k_i/c_0*np.sin(omega*t)
#     plt.plot(t,f)
#     plt.plot(t, f_dot)
#     plt.legend(['k=0.1','k=0.3','k=0.5'])
# plt.ylim([-A,A])
# plt.yticks(np.arange(-A,A,A/10))
# plt.xlabel('time')
# plt.ylabel('piching angle')
# plt.show()

# # exit()

# # generate initial mesh
# nx = 5; ny = 15
# chord = 1; span = 6

# mesh_dict = {"num_y": ny, "num_x": nx, "wing_type": "rect",  "symmetry": False,
#                  "span": span, "root_chord": chord,"span_cos_spacing": False, "chord_cos_spacing": False}
# mesh = generate_mesh(mesh_dict)
# mesh[:,:,0]  = mesh[:,:,0] + 0.5


# # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html

# r = R.from_euler('y', f, degrees=True).as_matrix() # num_nodes,3,3

# rotated_mesh = np.einsum('ijk,lmk->ilmj', r, mesh)
# model = csdl.Model()
# Pitching = PitchingModel(surface_names=['wing'], surface_shapes=[(nx,ny)], num_nodes=num_nodes,A=A, k=k[0], v_inf=v_inf, c_0=c_0, N_period=N_period, AR=span/chord)
# model.add(Pitching,'Pitching')
# sim = python_csdl_backend.Simulator(model)
# sim.run()
# # visualize mesh
# import pyvista as pv
# for i in range(num_nodes):
#     x = sim['wing'][i, :, :, 0]
#     y = sim['wing'][i, :, :, 1]
#     z = sim['wing'][i, :, :, 2]

#     grid = pv.StructuredGrid(x,y,z)
#     grid.save('test_pitching/pitching'+str(i)+'.vtk')
#     # grid.plot(show_edges=True, show_grid=True)