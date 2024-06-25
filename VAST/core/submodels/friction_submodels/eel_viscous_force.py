from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size


# class EelViscousModel(Model):
#     """
#     Compute the viscous force of an eel geometry with a simple surrogate model.

#     parameters
#     ----------
#     v_x : csdl variable [1,]
#         array defining the x velocity of the eel

#     Returns
#     -------
#     1. visous_force_coeff : csdl array [1,]
#     """
#     def initialize(self):
#         pass

#     def define(self):
#         v_x = self.declare_variable(name='v_x')
#         C_F = 3.66/1000*(v_x)**(-0.5) * 0.2944404050399099 /0.13826040386294708 #* 6

#         self.register_output('C_F', C_F)

class EelViscousModel(Model):
    """
    Compute the viscous force of an 'eel' geometry with a simple surrogate model.
    Compute the viscous force of an 'eel' geometry with a simple surrogate model.

    parameters
    ----------
    v_x : csdl variable [1,]
        array defining the x velocity of the eel

    Returns
    -------
    1. visous_force_coeff : csdl array [1,]
    """
    def initialize(self):
        self.parameters.declare('surface_shapes')
        self.parameters.declare('surface_shapes')

    def define(self):
        v_x = self.declare_variable(name='v_x')
        # C_F = 3.66/1000*(v_x)**(-0.5) * 0.2944404050399099 /0.13826040386294708 #* 6

        # C_F = 3.66/1000*(v_x)**(-0.5) * 0.2944404050399099 /0.13826040386294708 #* 6


        surface_shapes = self.parameters['surface_shapes']
        # compute the theta
        # theta_analytical_laminar = 0.664 * x_vals / np.sqrt(U_inf * x_vals / nu)
        # L = 1
        (num_nodes, nx, ny,_) = surface_shapes[0]
        # x_vals = np.linspace(1e-4, L, nx-1)
        eel_mesh = self.declare_variable('eel', shape=(num_nodes, nx, ny, 3))

        x_vals = csdl.reshape(eel_mesh[0,1:,0,0], (nx-1,))


        v_x_expand = csdl.expand(v_x, x_vals.shape)
        nu = 1.004e-6 # kinematic viscosity of water
        theta = 0.664 * x_vals / (v_x_expand * (x_vals/ nu) )**0.5
        l = 0.22 # for lamba = 0
        cf = 2 * nu * l / (theta * v_x_expand)


        panel_area = self.declare_variable('eel' + '_s_panel',shape=(num_nodes, nx-1,ny-1))
        panel_area_strip = csdl.reshape(csdl.sum(panel_area[0,:,:], axes=(2,)) , (nx-1,))
        panel_area_sum = csdl.sum(panel_area[0,:,:])
        # self.print_var(panel_area)
        # self.print_var(panel_area_sum)
        # self.print_var(cf)

        CF = csdl.sum(panel_area_strip * cf) / panel_area_sum * 2
        # self.register_output('panel_area_strip', panel_area_strip)
        # self.register_output('panel_area_sum', panel_area_sum)
        # self.register_output('cf', cf)
        # self.print_var(panel_area_strip)
        # self.print_var(panel_area_sum)
        # self.print_var(cf)
        self.register_output('C_F', CF)


