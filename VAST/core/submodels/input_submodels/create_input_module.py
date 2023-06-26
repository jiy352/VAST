from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL


class CreateACSatesModule(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('v_inf', types=np.ndarray)
        self.parameters.declare('theta', types=np.ndarray)
        self.parameters.declare('num_nodes')

    def define(self):
        v_inf = self.parameters['v_inf']
        theta = self.parameters['theta']
        num_nodes = self.parameters['num_nodes']

        u_all = self.register_module_input('u', shape=(num_nodes, 1), vectorized=True,val=v_inf)
        v_all = self.register_module_input('v', shape=(num_nodes, 1), vectorized=True)
        w_all = self.register_module_input('w', shape=(num_nodes, 1), vectorized=True)
        theta_all = self.register_module_input('theta', shape=(num_nodes, 1), vectorized=True,val=theta)
        gamma_all = self.register_module_input('gamma', shape=(num_nodes, 1), vectorized=True)
        psi_all = self.register_module_input('psi', shape=(num_nodes, 1), vectorized=True)
        p_all = self.register_module_input('p', shape=(num_nodes, 1), vectorized=True)
        q_all = self.register_module_input('q', shape=(num_nodes, 1), vectorized=True)
        r_all = self.register_module_input('r', shape=(num_nodes, 1), vectorized=True)
        x_all = self.register_module_input('x', shape=(num_nodes, 1), vectorized=True)
        y_all = self.register_module_input('y', shape=(num_nodes, 1), vectorized=True)
        z_all = self.register_module_input('z', shape=(num_nodes, 1), vectorized=True)
        rho_all = self.register_module_input('density', shape=(num_nodes, 1), vectorized=True)

        # print('theta',theta )

        # dummy = self.register_module_output('dummy', u_all*theta)
        # a = self.re

