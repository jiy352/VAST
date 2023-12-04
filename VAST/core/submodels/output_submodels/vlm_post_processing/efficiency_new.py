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
        self.parameters.declare('n_ignore',default=50)


    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = surface_shapes[0][0]
        surface_name = surface_names[0]
        # mesh_unit = self.parameters['mesh_unit']
        # N_period = self.parameters['n_period']

        n_ignore = self.parameters['n_ignore']

        panel_forces_all = self.declare_variable('panel_forces_all',shape=(surface_shapes[0][0],int(surface_shapes[0][1]-1)*int(surface_shapes[0][2]-1),3))
        velocities = self.declare_variable(surface_name+'_kinematic_vel',shape=panel_forces_all.shape)
        v_x = self.declare_variable('v_x')
        thrust = self.declare_variable('thrust',shape=(num_nodes,1))
        eval_total_vel = self.declare_variable('eval_total_vel',shape=velocities.shape)  
        panel_forces_all_x = panel_forces_all[:,:,0]
        velocities_x = self.create_output('velocities_x',shape=velocities.shape,val=0)
        velocities_x[:,:,0] = (velocities[:,:,0] - csdl.expand(v_x,shape=velocities[:,:,0].shape)) 
        velocities_x[:,:,1] = velocities[:,:,1] 
        velocities_x[:,:,2] = velocities[:,:,2] 

        # panel_thrust_power = -csdl.sum(csdl.dot(panel_forces_all[n_ignore:,:,:],-velocities_x[n_ignore:,:,:],axis=2))

        # panel_thrust_power = -csdl.sum((panel_forces_all[n_ignore:,:,0]*0.5))
        # panel_thrust_power = -csdl.sum((panel_forces_all[n_ignore:,:,0]*velocities_x[n_ignore:,:,0]))
        # panel_thrust_power = csdl.sum(-panel_forces_all[n_ignore:,:,0]*velocities_x[n_ignore:,:,0] - panel_forces_all[n_ignore:,:,2]*velocities_x[n_ignore:,:,2])
        thrust_power = -csdl.sum(csdl.sum(thrust[n_ignore:,:],axes=(0,))*v_x)
        # panel_thrust_power = thrust_power + csdl.sum( panel_forces_all[n_ignore:,:,2]*velocities_x[n_ignore:,:,2])
        # omega = (0,1,0)
        omega_y = 1
        # moment_eval_pt = shifted quarter_chord_line_center
        # find the variable for total force on each panel
        # cross product of r_m and total force on each panel
        # take the y value of the cross product times omega_y
        
        ####################################################
        # computing the moment arm r_m relative to the evaluation_pt
        ####################################################
        # evaluation_pt = self.declare_variable('evaluation_pt',
        #                                         val=np.zeros(3, ))
        # evaluation_pt_exp = csdl.expand(
        #     evaluation_pt,
        #     (eval_pts_all.shape),
        #     'i->jki',
        # )          
        # r_M = eval_pts_all - evaluation_pt_exp

        # total_moments_surface_temp = csdl.cross(r_M[:,start:start+delta,:], total_forces_surface, axis=2)


        torque_power = torque * omega
        panel_thrust_power = csdl.sum( -panel_forces_all[n_ignore:,:,0]*-velocities_x[n_ignore:,:,0]) + csdl.sum( -panel_forces_all[n_ignore:,:,1]*-velocities_x[n_ignore:,:,1]) + csdl.sum( -panel_forces_all[n_ignore:,:,2]*-velocities_x[n_ignore:,:,2])

        # thrust is negative, -v_x is negative, so thrust_power is positive

        self.print_var(panel_thrust_power)
        self.print_var(thrust_power)
        self.print_var(csdl.sum(- panel_forces_all[n_ignore:,:,2]*velocities_x[n_ignore:,:,2]))
        # self.print_var(velocities_x[n_ignore:,:,0])
        self.print_var(v_x)
        # self.print_var(csdl.sum(thrust[n_ignore:,:],axes=(0,)))

        efficiency = thrust_power/(panel_thrust_power)
        self.print_var(efficiency)
        self.register_output('panel_thrust_power',panel_thrust_power)
        self.register_output('thrust_power',thrust_power)
        self.register_output('efficiency',efficiency)

# panel_forces_all = sim['panel_forces_all']
# velocities_panel = - sim['eel_kinematic_vel']
# panel_forces_dynamic = sim['panel_forces_dynamic']
# panel_acc = np.gradient(velocities_panel,axis=0)/h_stepsize

# mid_panel_vel = velocities_panel[:,int(velocities_panel.shape[1]/2),:]*0.5
# mid_panel_acc = panel_acc[:,int(panel_acc.shape[1]/2),:]*0.1
# mid_panel_forces = panel_forces_all[:,int(panel_forces_all.shape[1]/2),:]
# mid_panel_forces_dynamic = panel_forces_dynamic[:,int(panel_forces_dynamic.shape[1]/2),:]
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
# plt.plot(mid_panel_vel[:,1])
# plt.plot(mid_panel_acc[:,1])
# plt.plot(mid_panel_forces[:,1])
# plt.plot(mid_panel_forces_dynamic[:,1])
# plt.plot(mid_panel_forces[:,1]-mid_panel_forces_dynamic[:,1])
# plt.legend(['vel','acc','force','force_dynamic','force_static'])
# plt.show()


        # compute energy efficiency here

# thrust = sim['thrust']
# v_x = sim['v_x']
# panel_forces_all = sim['panel_forces_all']
# velocities = sim['eel_kinematic_vel']
# -np.sum(np.sum(thrust,axis=(1,))*v_x)        

# -np.sum(np.einsum('ijk,ijk->ij',panel_forces_all,velocities))
# velocities_x = velocities.copy()
# velocities_x[:,:,0] = 0
# -np.sum(np.einsum('ijk,ijk->ij',panel_forces_all,velocities_x))
        
# thrust = sim['thrust']
# v_x = -sim['v_x']

# thrust_power = np.sum(thrust*v_x)/t_vec[-1]

# panel_forces_all = sim['panel_forces_all']
# velocities = -sim['eval_total_vel']
# v_0 = velocities.copy()
# v_0[:,:,0]=0
# panel_power = np.sum(panel_forces_all * v_0)/t_vec[-1]
# # efficiency = sim['efficiency']



# if __name__ == "__main__":
#     import python_csdl_backend
#     import numpy as np
#     import pyvista as pv
#     # simulator_name = 'csdl_om'
#     num_nodes = 30
#     num_pts_chord = 13 # nx
#     num_pts_span = 3

#     model = csdl.Model()
#     model.add(EelActuationModel(surface_names=['surface'],surface_shapes=[(num_nodes,num_pts_chord,num_pts_span)]),'actuation')
    

#     sim = python_csdl_backend.Simulator(model)
#     sim.run()

#     surface = sim["surface"]
#     print(surface.shape)

#     for i in range(num_nodes):
#         x = surface[i,:,:,0]
#         y = surface[i,:,:,1]
#         z = surface[i,:,:,2]

#         grid = pv.StructuredGrid(x,y,z)
#         grid.save(filename=f'eel_actuation_{i}.vtk')