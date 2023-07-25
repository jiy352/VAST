import numpy as np

'''
list of variables to visualize are meshes and cell_data.set_vectors:
wakes
bound meshes
forces
kinematic velocities in body fixed frame ?
accelerations in body fixed frame ?
induced vel
gamma? 
'''

surface_names = ['eel']
surface_shapes = [(num_nodes, nx, ny, 3)]

# 'bd_vtx_coords', 
# 'wake_coords',

visualize_var_names = ['kinematic_vel']

import pyvista as pv

plotter = pv.Plotter()

vel = sim['eel_kinematic_vel'].copy()
panel_forces_all = sim['panel_forces_all'].copy()
panel_forces_dynamic = sim['panel_forces_dynamic'].copy()
panel_forces = sim['panel_forces'].copy()

acc = np.zeros(vel.shape)

acc[:-1,:,:] = (vel[1:,:,:] - vel[:-1,:,:])/h_stepsize
acc[-1,:,:] = (vel[-1,:,:] - vel[-2,:,:])/h_stepsize

# forces = sim['eel_forces'].copy()
vel[:,:,0] = 0
nframe = num_nodes
for i in range(len(surface_names)):
    surface_name = surface_names[i]
    bound_mesh = sim[surface_name+'_bd_vtx_coords']
    bound_grid = pv.StructuredGrid(bound_mesh[-1,:,:,0], bound_mesh[-1,:,:,1], bound_mesh[-1,:,:,2])
    wake_mesh = sim['op_'+surface_name+'_wake_coords']
    wake_grid = pv.StructuredGrid(wake_mesh[-1,:,:,0], wake_mesh[-1,:,:,1], wake_mesh[-1,:,:,2])
    plotter.add_mesh(
        bound_grid,
        # scalars=z.ravel(),
        lighting=False,
        show_edges=True,
        scalar_bar_args={"title": "Height"},
        clim=[-1, 1],
    )
    plotter.add_mesh(
        wake_grid,
        # scalars=z.ravel(),
        lighting=False,
        show_edges=True,
        scalar_bar_args={"title": "Height"},
        clim=[-1, 1],
    )
    for phase in np.linspace(0, 2 * np.pi, nframe + 1)[:nframe]:
        x_m = bound_mesh[i,:,:,0]
        y_m = bound_mesh[i,:,:,1]
        z_m = bound_mesh[i,:,:,2]
        x = wake_mesh[i,:,:,0]
        y = wake_mesh[i,:,:,1]
        z = wake_mesh[i,:,:,2]
        # grid = pv.StructuredGrid(x,y,z)
        grid_mesh = pv.StructuredGrid(x_m,y_m,z_m)
        grid = pv.StructuredGrid(x,y,z)
        for vars in visualize_var_names:
            var = sim[surface_name+'_'+vars]
            var[:,:,0] = 0
        grid_mesh.cell_data.set_vectors(np.swapaxes(vel[i].reshape(nx-1,ny-1,3), 0,1).reshape(-1,3),vars)
        grid_mesh.cell_data.set_vectors(np.swapaxes(panel_forces_all[i].reshape(nx-1,ny-1,3), 0,1).reshape(-1,3),'panel_forces_all')
        grid_mesh.cell_data.set_vectors(np.swapaxes(panel_forces_dynamic[i].reshape(nx-1,ny-1,3), 0,1).reshape(-1,3),'panel_forces_dynamic')
        grid_mesh.cell_data.set_vectors(np.swapaxes(panel_forces[i].reshape(nx-1,ny-1,3), 0,1).reshape(-1,3),'panel_forces')
        grid_mesh.cell_data.set_vectors(np.swapaxes(acc[i].reshape(nx-1,ny-1,3), 0,1).reshape(-1,3),'acc')
        # print(i,grid.points.shape)
        grid_mesh.save('d'+str(i)+'.vtk')
        grid.save('dmesh'+str(i)+'.vtk')
        i+=1



# grid_mesh = pv.StructuredGrid(cat_mesh[-1,:,:,0], cat_mesh[-1,:,:,1], cat_mesh[-1,:,:,2])
# grid = pv.StructuredGrid(mesh_val[-1,:,:,0], mesh_val[-1,:,:,1], mesh_val[-1,:,:,2])
# grid_mesh.save('d.vtk')
# # Create a plotter object and set the scalars to the Z height
# plotter.add_mesh(
#     grid_mesh,
#     # scalars=z.ravel(),
#     lighting=False,
#     show_edges=False,
#     scalar_bar_args={"title": "Height"},
#     clim=[-1, 1],
#     color='red',
#     opacity=0.0,
# )
# plotter.add_mesh(
#     grid,
#     # scalars=z.ravel(),
#     lighting=False,
#     show_edges=True,
#     scalar_bar_args={"title": "Height"},
#     clim=[-1, 1],
# )
# plotter.set_background("white")
# plotter.add_axes()
# plotter.bounds = (-0.5, 3.170018838495248, -1.052461965466036, 2.0872241742687794, -1.107604303006731, -2.4492935982947065e-17)
# # plotter.show()

# # Open a gif
# plotter.open_gif("wave.gif")

# pts = grid.points.copy()
# # Update Z and write a frame for each updated position
# nframe = num_nodes
# i=0
# plotter.write_frame()

# for phase in np.linspace(0, 2 * np.pi, nframe + 1)[:nframe]:
#     x_m = cat_mesh[i,:,:,0]
#     y_m = cat_mesh[i,:,:,1]
#     z_m = cat_mesh[i,:,:,2]
#     x = mesh_val[i,:,:,0]
#     y = mesh_val[i,:,:,1]
#     z = mesh_val[i,:,:,2]
#     # grid = pv.StructuredGrid(x,y,z)
#     grid_mesh = pv.StructuredGrid(x_m,y_m,z_m)
#     grid = pv.StructuredGrid(x,y,z)
#     # print(i,grid.points.shape)
#     grid_mesh.save('d'+str(i)+'.vtk')
#     grid.save('dmesh'+str(i)+'.vtk')
#     plotter.update_coordinates(grid_mesh.points.copy(),grid_mesh, render=False)
#     plotter.update_coordinates(grid.points.copy(),grid, render=False)
#     plotter.write_frame()
#     i+=1

# # Closes and finalizes movie
# plotter.close()