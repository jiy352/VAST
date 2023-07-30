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

def run_visualization(sim, h_stepsize):
    surface_names = ['wing']
    num_nodes = sim[surface_names[0]].shape[0]
    nx = sim[surface_names[0]].shape[1]
    ny = sim[surface_names[0]].shape[2]
    surface_shapes = [(num_nodes, nx, ny, 3)]
    # 'bd_vtx_coords', 
    # 'wake_coords',
    visualize_mesh_names = ['bd_vtx_coords', 'wake_coords']
    visualize_var_names = ['kinematic_vel']

    import pyvista as pv

    plotter = pv.Plotter()

    vel = sim['wing_eval_pts_coords_eval_pts_induced_vel']


    panel_forces = sim['panel_forces'].copy()

    # acc = np.zeros(vel.shape)
    # # we added a nagetive sign here to calculate the acceleration of the fish body
    # acc[:-1,:,:] = -(vel[1:,:,:] - vel[:-1,:,:])/h_stepsize
    # acc[-1,:,:] = -(vel[-1,:,:] - vel[-2,:,:])/h_stepsize

    # forces = sim['eel_forces'].copy()
    # vel[:,:,0] = 0
    nframe = num_nodes
    for i in range(len(surface_names)):
        surface_name = surface_names[i]
        mesh = sim[surface_name]
        mesh_grid = pv.StructuredGrid(mesh[-1,:,:,0], mesh[-1,:,:,1], mesh[-1,:,:,2])
        bound_mesh = sim[surface_name+'_bd_vtx_coords']
        bound_grid = pv.StructuredGrid(bound_mesh[-1,:,:,0], bound_mesh[-1,:,:,1], bound_mesh[-1,:,:,2])
        wake_mesh = sim[surface_name+'_wake_coords']
        wake_grid = pv.StructuredGrid(wake_mesh[-1,:,:,0], wake_mesh[-1,:,:,1], wake_mesh[-1,:,:,2])
        plotter.add_mesh(
            mesh_grid,
            # scalars=z.ravel(),
            lighting=False,
            show_edges=True,
            scalar_bar_args={"title": "Height"},
            clim=[-1, 1],
        )
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
            x_mesh = mesh[i,:,:,0]
            y_mesh = mesh[i,:,:,1]
            z_mesh = mesh[i,:,:,2]
            x_m = bound_mesh[i,:,:,0]
            y_m = bound_mesh[i,:,:,1]
            z_m = bound_mesh[i,:,:,2]
            x = wake_mesh[i,:,:,0]
            y = wake_mesh[i,:,:,1]
            z = wake_mesh[i,:,:,2]
            grid_mesh = pv.StructuredGrid(x_mesh,y_mesh,z_mesh)
            grid_bdmesh = pv.StructuredGrid(x_m,y_m,z_m)
            grid = pv.StructuredGrid(x,y,z)
            for vars in visualize_var_names:
                var = sim[surface_name+'_'+vars]
                var[:,:,0] = 0
            grid_bdmesh.cell_data.set_vectors(np.swapaxes(vel[i].reshape(nx-1,ny-1,3), 0,1).reshape(-1,3),vars)
            # grid_bdmesh.cell_data.set_vectors(np.swapaxes(panel_forces_all[i].reshape(nx-1,ny-1,3), 0,1).reshape(-1,3),'panel_forces_all')
            # grid_bdmesh.cell_data.set_vectors(np.swapaxes(panel_forces_dynamic[i].reshape(nx-1,ny-1,3), 0,1).reshape(-1,3),'panel_forces_dynamic')
            grid_bdmesh.cell_data.set_vectors(np.swapaxes(panel_forces[i].reshape(nx-1,ny-1,3), 0,1).reshape(-1,3),'panel_forces')
            # grid_bdmesh.cell_data.set_vectors(np.swapaxes(acc[i].reshape(nx-1,ny-1,3), 0,1).reshape(-1,3),'acc')
            # print(i,grid.points.shape)
            grid_mesh.save('fixedwk_vtks/mesh'+str(i)+'.vtk')
            grid_bdmesh.save('fixedwk_vtks/bd'+str(i)+'.vtk')
            grid.save('fixedwk_vtks/wake'+str(i)+'.vtk')
            i+=1



