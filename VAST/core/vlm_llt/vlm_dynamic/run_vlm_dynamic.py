import numpy as np
import csdl
import matplotlib.pyplot as plt
import numpy as np
import time

import pyvista as pv

from panel_method.utils.generate_eel_vlm_mesh import (
    generate_eel_carling_vlm,)

from panel_method.mesh_preprocessing.generate_kinematics import eel_kinematics


from lsdo_uvlm.examples.simple_wing_constant_aoa_sls_outputs.plunging_system_free import ODESystemModel
from lsdo_uvlm.examples.simple_wing_constant_aoa_sls_outputs.plunging_profile_outputs import ProfileSystemModel
from ozone.api import ODEProblem
import csdl

from VAST.core.vlm_llt.vlm_system_dynamic import RunModel
def generate_simple_mesh_mesh(nx, ny, nt=None):
    if nt == None:
        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))/(nx-1)
        mesh[:, :, 1] = 0.
        mesh[:, :, 2] = np.outer(np.arange(ny), np.ones(nx)).T/(ny-1)-0.5
    else:
        mesh = np.zeros((nt, nx, ny, 3))
        for i in range(nt):
            mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
            mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            mesh[i, :, :, 2] = 0.
    return mesh

be = 'python_csdl_backend'
make_video = 1


# define the problem
num_nodes = 10
nt = num_nodes+1

alpha = np.deg2rad(0)

# define the direction of the flapping motion (hardcoding for now)

# u_val = np.concatenate((np.array([0.01, 0.5,1.]),np.ones(num_nodes-3))).reshape(num_nodes,1)


########################################
# define mesh here
########################################
num_pts_L = 30
num_pts_R = ny = 23
nx = num_pts_L-5

L = 1.
s_1_ind = 5
s_2_ind = 26
num_fish = 1
grid,h = generate_eel_carling_vlm(num_pts_L,num_pts_R,L,s_1_ind,s_2_ind)

mesh = generate_simple_mesh_mesh(num_pts_L,num_pts_R)

x= mesh[:, :, 0]
y= mesh[:, :, 1]
z= mesh[:, :, 2]*np.einsum('i,j->ji',np.ones(num_pts_R),h)

grid = pv.StructuredGrid(x,y,z)
grid.plot(show_edges=True, show_bounds=False, show_grid=False, show_axes=True)
# grid.save('vtk/fish.vtk')

num_time_steps = num_nodes

lambda_         = 1
f               = 0.2
n_period = 1

t_list          = np.linspace(0, 1/f*n_period, num_time_steps)

mid_line_points = x[:,5]
num_time_steps  = t_list.size

x = (mid_line_points-mid_line_points[0]) / (mid_line_points[-1]-mid_line_points[0])
L = 1           # L is 1 because we scaled x to [0, 1]

fig = plt.figure()
ax = fig.add_subplot()
legend_list = []
y = np.zeros((num_time_steps,x.size))
y_dot = y.copy()
y_lateral = np.zeros((num_time_steps,mid_line_points.size))
mesh_val = np.zeros((num_nodes, nx, ny, 3))
for i in range(num_time_steps):
    t = t_list[i]
    omg = 2*np.pi*f
    y_lateral[i,:] = 0.125*((mid_line_points+0.03125)/(1.03125))*np.sin(np.pi*2*mid_line_points/lambda_ - omg*t)
    y_lateral[i,:] =  0.125*((mid_line_points+0.03125)/(1.03125))*np.cos(np.pi*2*mid_line_points/lambda_ - omg*t)*(-omg)
    # plt.plot(x,y_dot[i],'.')
    # plt.plot(x,y[i])
    x= mesh[:, :, 0]
    y= mesh[:, :, 1]+np.einsum('i,j->ij', y_lateral[i],np.ones(num_pts_R))
    z= mesh[:, :, 2]*np.einsum('i,j->ji',np.ones(num_pts_R),h)
    grid= pv.StructuredGrid(x,y,z)
    # grid.save('vtks_fish/fish'+str(i)+'.vtk')
    # mesh_val[i,:,:,0] = x[-nx:,:]
    # mesh_val[i,:,:,1] = y[-nx:,:]
    # mesh_val[i,:,:,2] = z[-nx:,:]

    mesh_val[i,:,:,0] = mesh[-nx:,:,0]
    mesh_val[i,:,:,1] = mesh[-nx:,:,1]+np.einsum('i,j->ij', y_lateral[i],np.ones(num_pts_R))[-nx:,:]
    mesh_val[i,:,:,2] = mesh[-nx:,:,2]
plt.show()



surface_names=['eel']
surface_shapes=[(nx, ny, 3)]
h_stepsize = delta_t = t_list[1]-t_list[0]

import python_csdl_backend
sim = python_csdl_backend.Simulator(RunModel(
    num_times=nt - 1,
    h_stepsize=h_stepsize,
    surface_names=surface_names,
    surface_shapes=surface_shapes,
    mesh_val=mesh_val), mode='rev')

t_start = time.time()
sim.run()
print('simulation time is', time.time() - t_start)
######################################################
# make video
######################################################

if make_video == 1:
    axs = Axes(
        xrange=(0, 5),
        yrange=(-1, 1),
        zrange=(-1, 1),
    )
    video = Video("spider.gif", duration=10, backend='ffmpeg')
    for i in range(nt - 1):
        vp = Plotter(
            bg='beige',
            bg2='lb',
            # axes=0,
            #  pos=(0, 0),
            offscreen=False,
            interactive=1)
        # Any rendering loop goes here, e.g.:
        for surface_name in surface_names:
            vps = Points(np.reshape(sim[surface_name][i, :, :, :], (-1, 3)),
                        r=8,
                        c='red')
            vp += vps
            vp += __doc__
            vps = Points(np.reshape(sim[surface_name+'_wake_coords_int'][i, 0:i, :, :],
                                    (-1, 3)),
                        r=8,
                        c='blue')
            vp += vps
            vp += __doc__
        # cam1 = dict(focalPoint=(3.133, 1.506, -3.132))
        # video.action(cameras=[cam1, cam1])
        vp.show(axs, elevation=-60, azimuth=-0,
                axes=False)  # render the scene
        video.addFrame()  # add individual frame
        # time.sleep(0.1)
        # vp.interactive().close()
        vp.closeWindow()
    vp.closeWindow()
    video.close()  # merge all the recorded frames
