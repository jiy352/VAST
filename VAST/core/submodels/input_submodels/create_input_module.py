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

        # u = self.create_input('u',val=v_inf)
        # v = self.create_input('v',val=np.zeros((num_nodes, 1)))
        # w = self.create_input('w',val=np.ones((num_nodes,1))*0)
        # p = self.create_input('p',val=np.zeros((num_nodes, 1)))
        # q = self.create_input('q',val=np.zeros((num_nodes, 1)))
        # r = self.create_input('r',val=np.zeros((num_nodes, 1)))
        # phi = self.create_input('phi',val=np.zeros((num_nodes, 1)))
        # theta = self.create_input('theta',val=theta)
        # psi = self.create_input('psi',val=np.zeros((num_nodes, 1)))
        # x = self.create_input('x',val=np.zeros((num_nodes, 1)))
        # y = self.create_input('y',val=np.zeros((num_nodes, 1)))
        # z = self.create_input('z',val=np.ones((num_nodes, 1))*1000)
        # phiw = self.create_input('phiw',val=np.zeros((num_nodes, 1)))
        # gamma = self.create_input('gamma',val=np.zeros((num_nodes, 1)))
        # psiw = self.create_input('psiw',val=np.zeros((num_nodes, 1)))


        u_all = self.register_module_input('u', shape=(num_nodes, 1), vectorized=True)
        v_all = self.register_module_input('v', shape=(num_nodes, 1), vectorized=True)
        w_all = self.register_module_input('w', shape=(num_nodes, 1), vectorized=True)
        theta_all = self.register_module_input('theta', shape=(num_nodes, 1), vectorized=True)
        gamma_all = self.register_module_input('gamma', shape=(num_nodes, 1), vectorized=True)
        psi_all = self.register_module_input('psi', shape=(num_nodes, 1), vectorized=True)
        p_all = self.register_module_input('p', shape=(num_nodes, 1), vectorized=True)
        q_all = self.register_module_input('q', shape=(num_nodes, 1), vectorized=True)
        r_all = self.register_module_input('r', shape=(num_nodes, 1), vectorized=True)
        x_all = self.register_module_input('x', shape=(num_nodes, 1), vectorized=True)
        y_all = self.register_module_input('y', shape=(num_nodes, 1), vectorized=True)
        z_all = self.register_module_input('z', shape=(num_nodes, 1), vectorized=True)
        rho_all = self.register_module_input('density', shape=(num_nodes, 1), vectorized=True)

        u = self.register_module_output('u_active_nodes', shape=(num_active_nodes, 1), val=0)
        v = self.register_module_output('v_active_nodes', shape=(num_active_nodes, 1), val=0)
        w = self.register_module_output('w_active_nodes', shape=(num_active_nodes, 1), val=0)
        theta = self.register_module_output('theta_active_nodes', shape=(num_active_nodes, 1), val=0)
        gamma = self.register_module_output('gamma_active_nodes', shape=(num_active_nodes, 1), val=0)
        psi  = self.register_module_output('psi_active_nodes', shape=(num_active_nodes, 1), val=0)
        p = self.register_module_output('p_active_nodes', shape=(num_active_nodes, 1), val=0)
        q = self.register_module_output('q_active_nodes', shape=(num_active_nodes, 1), val=0)
        r = self.register_module_output('r_active_nodes', shape=(num_active_nodes, 1), val=0)
        x = self.register_module_output('x_active_nodes', shape=(num_active_nodes, 1), val=0)
        y = self.register_module_output('y_active_nodes', shape=(num_active_nodes, 1), val=0)
        z = self.register_module_output('z_active_nodes', shape=(num_active_nodes, 1), val=0)
        rho = self.register_module_output('density_active_nodes', shape=(num_active_nodes, 1), val=0)

        for i in range(len(active_nodes)):
            index = int(active_nodes[i])
            u[i, 0] = u_all[index, 0]
            v[i, 0] = v_all[index, 0]
            w[i, 0] = w_all[index, 0]
            theta[i, 0] = theta_all[index, 0]
            gamma[i, 0] = gamma_all[index, 0]
            psi[i, 0] = psi_all[index, 0]
            p[i, 0] = p_all[index, 0]
            q[i, 0] = q_all[index, 0]
            r[i, 0] = r_all[index, 0]
            x[i, 0] = x_all[index, 0]
            y[i, 0] = y_all[index, 0]
            z[i, 0] = z_all[index, 0]
            rho[i, 0] = rho_all[index, 0]
