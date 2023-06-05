from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL


    # single lifting surface 
    # (nx: number of points in streamwise direction; ny:number of points in spanwise direction)
    surface_names = ['wing']
    surface_shapes = [(num_nodes, nx, ny, 3)]
    mesh_dict = {
        "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": 10.0,
        "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0,
    }
    # Generate mesh of a rectangular wing
    mesh = generate_mesh(mesh_dict) 
    wing = model_1.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))

class CreateMeshModule(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('v_inf', types=np.ndarray)
        self.parameters.declare('theta', types=np.ndarray)
        self.parameters.declare('num_nodes')

    def define(self):
        v_inf = self.parameters['v_inf']
        theta = self.parameters['theta']
        num_nodes = self.parameters['num_nodes']

        u = self.create_input('u',val=v_inf)
        v = self.create_input('v',val=np.zeros((num_nodes, 1)))
        w = self.create_input('w',val=np.ones((num_nodes,1))*0)
        p = self.create_input('p',val=np.zeros((num_nodes, 1)))
        q = self.create_input('q',val=np.zeros((num_nodes, 1)))
        r = self.create_input('r',val=np.zeros((num_nodes, 1)))
        phi = self.create_input('phi',val=np.zeros((num_nodes, 1)))
        theta = self.create_input('theta',val=theta)
        psi = self.create_input('psi',val=np.zeros((num_nodes, 1)))
        x = self.create_input('x',val=np.zeros((num_nodes, 1)))
        y = self.create_input('y',val=np.zeros((num_nodes, 1)))
        z = self.create_input('z',val=np.ones((num_nodes, 1))*1000)
        phiw = self.create_input('phiw',val=np.zeros((num_nodes, 1)))
        gamma = self.create_input('gamma',val=np.zeros((num_nodes, 1)))
        psiw = self.create_input('psiw',val=np.zeros((num_nodes, 1)))