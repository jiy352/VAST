import csdl
import numpy as np
from numpy.core.fromnumeric import size


class EfficiencyModel(csdl.Model):
    """
    Compute the mesh at each time step for the eel actuation model given the kinematic variables.
    parameters
    ----------
    tail_amplitude : csdl variable [1,]
    tail_frequency : csdl variable [1,]
    v_x : csdl variable [1,]
    Returns
    -------
    1. mesh[num_nodes,num_pts_chord, num_pts_span, 3] : csdl array
    bound vortices points
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        # self.parameters.declare('mesh_unit', default='m')
        # self.parameters.declare('n_period')


    def define(self):
        # surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = surface_shapes[0][0]
        # mesh_unit = self.parameters['mesh_unit']
        # N_period = self.parameters['n_period']

        panel_forces_all = self.declare_variable('panel_forces_all',shape=(surface_shapes[0][0],int(surface_shapes[0][1]-1)*int(surface_shapes[0][2]-1),3))
        velocities = self.declare_variable('eval_total_vel',shape=panel_forces_all.shape)
        v_x = self.declare_variable('v_x')
        thrust = self.declare_variable('thrust',shape=(num_nodes,1))
        panel_forces_all_x = panel_forces_all[:,:,0]
        velocities_x = velocities[:,:,0]
        panel_thrust_power = csdl.sum(panel_forces_all_x*velocities_x,axes=(1,2))

        # print('shapes')
        # print('thrust',thrust.shape)
        # print('panel_thrust_power',panel_thrust_power.shape)
        # print('csdl.expand(v_x,shape=(num_nodes,1))',csdl.expand(v_x,shape=(num_nodes,1)).shape)



        efficiency = csdl.sum(csdl.sum(thrust,axes=(1,))*csdl.expand(v_x,shape=(num_nodes,))/panel_thrust_power)/num_nodes
        self.register_output('efficiency',efficiency)
        # panel_thrust = csdl.sum(panel_forces_all*velocities,axis=2)

        # panel_forces_all_mag = csdl.sum(panel_forces_all**2,axes=(2,))**0.5
        # velocities_mag = csdl.sum(velocities**2,axes=(2,))**0.5
        # panel_power = csdl.dot(panel_forces_all,velocities,axis=2)
        # self.register_output('panel_power_new',panel_power)



        



if __name__ == "__main__":
    import python_csdl_backend
    import numpy as np
    import pyvista as pv
    # simulator_name = 'csdl_om'
    num_nodes = 30
    num_pts_chord = 13 # nx
    num_pts_span = 3

    model = csdl.Model()
    model.add(EelActuationModel(surface_names=['surface'],surface_shapes=[(num_nodes,num_pts_chord,num_pts_span)]),'actuation')
    

    sim = python_csdl_backend.Simulator(model)
    sim.run()

    surface = sim["surface"]
    print(surface.shape)

    for i in range(num_nodes):
        x = surface[i,:,:,0]
        y = surface[i,:,:,1]
        z = surface[i,:,:,2]

        grid = pv.StructuredGrid(x,y,z)
        grid.save(filename=f'eel_actuation_{i}.vtk')
