from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
from caddee.caddee_core.system_model.design_scenario.design_condition.mechanics_group.mechanics_model.mechanics_model import MechanicsModel
from aframe.core.aframe import Aframe
import numpy as np

import csdl
import m3l
import array_mapper as am

from VAST.core.vlm_llt.NodalMapping import NodalMap,RadialBasisFunctions


class LinearBeam(m3l.Model):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('struct_solver', True)
        self.parameters.declare('compute_mass_properties', default=True, types=bool)

        self.parameters.declare('beams', default={})
        self.parameters.declare('bounds', default={})
        self.parameters.declare('joints', default={})
        self.num_nodes = None

    def construct_displacement_map(self, nodal_outputs_mesh):
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
        oml_mesh = nodal_outputs_mesh.value.reshape((-1, 3))
        displacement_map = self.umap(mesh.value.reshape((-1,3)), oml=oml_mesh)


    def construct_force_map(self, nodal_force):
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
        oml_mesh = nodal_force.mesh.value.reshape((-1, 3))
        force_map = self.fmap(mesh.value.reshape((-1,3)), oml=oml_mesh)
        return force_map
    
    # def construct_moment_map(self, nodal_moment):
    #     mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
    #     oml_mesh = nodal_moment.mesh.value.reshape((-1, 3))
    #     moment_map = self.fmap(mesh.value.reshape((-1,3)), oml=oml_mesh)
    #     return moment_map


        return displacement_map
    
    # def construct_rotation_map(self, nodal_outputs_mesh):
    #     mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
    #     oml_mesh = nodal_outputs_mesh.value.reshape((-1, 3))
    #     # rotation_map = self.mmap(mesh.value, oml=oml_mesh)

    #     rotation_map = np.zeros((oml_mesh.shape[0],mesh.shape[0]))

    #     return rotation_map
    
    def construct_invariant_matrix(self):
        pass

    def evaluate(self, nodal_outputs_mesh:am.MappedArray, nodal_force:m3l.FunctionValues=None, nodal_moment:m3l.FunctionValues=None):
        '''
        Evaluates the model.

        Parameters
        ----------
        nodal_outputs_mesh : am.MappedArray
            The mesh or point cloud representing the locations at which the nodal displacements and rotations will be returned.
        nodal_force : m3l.FunctionValues
            The nodal forces that will be mapped onto the beam.
        nodal_moment : m3l.FunctionValues
            The nodal moments that will be mapped onto the beam.
        
        Returns
        -------
        nodal_displacement : m3l.FunctionValues
            The displacements evaluated at the locations specified by nodal_outputs_mesh
        nodal_rotation : m3l.FunctionValues
            The rotations evluated at the locations specified by the nodal_outputs_mesh
        '''

        # NOTE: This is assuming one mesh. To handle multiple meshes, a method must be developed to figure out how mappings work.
        beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        component_name = self.parameters['component'].name

        input_modules = []
        if nodal_displacement is not None:
            displacement_map = self.construct_displacement_map(nodal_displacement=nodal_displacement)
            displacement_input_module = m3l.ModelInputModule(name='displacement_input_module', 
                                                  module_input=nodal_displacement, map=displacement_map, model_input_name=f'{beam_name}_displacement')
            input_modules.append(displacement_input_module)

        # if nodal_moment is not None:
        #     moment_map = self.construct_moment_map(nodal_moment=nodal_moment)
        #     moment_input_module = m3l.ModelInputModule(name='moment_input_module', 
        #                                            module_input=nodal_moment, map=moment_map, model_input_name=f'{beam_name}_moments')
        #     input_modules.append(moment_input_module)


        displacement_map = self.construct_displacement_map(nodal_outputs_mesh=nodal_outputs_mesh)
        # rotation_map = self.construct_rotation_map(nodal_outputs_mesh=nodal_outputs_mesh)

        force_output_module = m3l.ModelOutputModule(name='force_output_module',
                                                    model_output_name=f'{beam_name}_force',
                                                    map=force_map, module_output_name=f'beam_nodal_force_{component_name}',
                                                    module_output_mesh=nodal_outputs_mesh)

        # rotation_output_module = m3l.ModelOutputModule(name='rotation_output_module',
        #                                             model_output_name=f'{beam_name}_rotation',
        #                                             map=rotation_map, module_output_name=f'beam_nodal_rotation_{component_name}',
        #                                             module_output_mesh=nodal_outputs_mesh)

        nodal_displacement, nodal_rotation = self.construct_module_csdl(
                         model_map=self._assemble_csdl(), 
                         input_modules=input_modules,
                         output_modules=[force_output_module]
                         )
        
        return nodal_displacement, nodal_rotation


    def _assemble_csdl(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']

        csdl_model = LinearBeamCSDL(
            module=self,
            beams=beams,  
            bounds=bounds,
            joints=joints)

        return csdl_model

    def umap(self, mesh, oml):
        G_mat = NodalMap(self.solid_mesh.geometry.x, reshape_3D_array_to_2D(self.fluid_mesh), 
                              RBF_width_par=RBF_width_par, RBF_func=RBF_func, )

        return G_mat
    


    def fmap(self, mesh, oml):
        # Fs = W*Fpq

        x, y = mesh.copy(), oml.copy()
        n, m = len(mesh), len(oml)

        d = np.zeros((m,2))
        for i in range(m):
            dist = np.sum((x - y[i,:])**2, axis=1)
            d[i,:] = np.argsort(dist)[:2]

        # create the weighting matrix:
        weights = np.zeros((n, m))
        for i in range(m):
            ia, ib = int(d[i,0]), int(d[i,1])
            a, b = x[ia,:], x[ib,:]
            p = y[i,:]

            length = np.linalg.norm(b - a)
            norm = (b - a)/length
            t = np.dot(p - a, norm)
            # c is the closest point on the line segment (a,b) to point p:
            c =  a + t*norm

            ac, bc = np.linalg.norm(c - a), np.linalg.norm(c - b)
            l = max(length, bc)
            
            weights[ia, i] = (l - ac)/length
            weights[ib, i] = (l - bc)/length

        return weights




class LinearBeamMesh(Module):
    def initialize(self, kwargs):
        self.parameters.declare('meshes', types=dict)
        self.parameters.declare('mesh_units', default='m')



class LinearBeamCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('beams')
        self.parameters.declare('bounds')
        self.parameters.declare('joints')
    
    def define(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']

        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            cs = beams[beam_name]['cs']

            if cs == 'box':
                xweb = self.register_module_input(beam_name+'t_web_in',shape=(n-1), computed_upstream=False)
                xcap = self.register_module_input(beam_name+'t_cap_in',shape=(n-1), computed_upstream=False)
                self.register_output(beam_name+'_tweb',1*xweb)
                self.register_output(beam_name+'_tcap',1*xcap)
                
            elif cs == 'tube':
                thickness = self.register_module_input(beam_name+'thickness_in',shape=(n-1), computed_upstream=False)
                radius = self.register_module_input(beam_name+'radius_in',shape=(n-1), computed_upstream=False)
                self.register_output(beam_name+'_t', 1*thickness)
                self.register_output(beam_name+'_r', 1*radius)

        # solve the beam group:
        self.add_module(Aframe(beams=beams, bounds=bounds, joints=joints), name='Aframe')


if __name__ == "__main__":

    import csdl
    import numpy as np
    from VAST.core.fluid_problem import FluidProblem
    from VAST.utils.generate_mesh import *
    from VAST.core.submodels.input_submodels.create_input_module import CreateACSatesModule
    from VAST.core.vlm_llt.vlm_solver import VLMSolverModel
    from python_csdl_backend import Simulator

    num_nodes=1; nx=3; ny=11

    solver_option = 'VLM'
    problem_type = 'fixed_wake'
    fluid_problem = FluidProblem(solver_option=solver_option, problem_type=problem_type)
    ####################################################################
    # 1. Define VLM inputs that share the common names within CADDEE
    ####################################################################
    num_nodes = 1
    create_opt = 'create_inputs'
    model_1 = csdl.Module()
    alpha = np.deg2rad(np.ones((num_nodes,1))*5)

    vx = 248.136
    vz = 0

    u = model_1.create_input('u',val=np.ones((num_nodes,1))*vx)
    v = model_1.create_input('v',val=np.zeros((num_nodes, 1)))
    w = model_1.create_input('w',val=np.ones((num_nodes,1))*vz)
    p = model_1.create_input('p',val=np.zeros((num_nodes, 1)))
    q = model_1.create_input('q',val=np.zeros((num_nodes, 1)))
    r = model_1.create_input('r',val=np.zeros((num_nodes, 1)))
    phi = model_1.create_input('phi',val=np.zeros((num_nodes, 1)))
    theta = model_1.create_input('theta',val=alpha)
    psi = model_1.create_input('psi',val=np.zeros((num_nodes, 1)))
    x = model_1.create_input('x',val=np.zeros((num_nodes, 1)))
    y = model_1.create_input('y',val=np.zeros((num_nodes, 1)))
    z = model_1.create_input('z',val=np.ones((num_nodes, 1))*1000)
    phiw = model_1.create_input('phiw',val=np.zeros((num_nodes, 1)))
    gamma = model_1.create_input('gamma',val=np.zeros((num_nodes, 1)))
    psiw = model_1.create_input('psiw',val=np.zeros((num_nodes, 1)))

    ####################################################################
    # 2. add VLM meshes
    ####################################################################
    # single lifting surface
    nx = 3  # number of points in streamwise direction
    ny = 11  # number of points in spanwise direction

    surface_names = ['wing']
    surface_shapes = [(num_nodes, nx, ny, 3)]

    # chord = 1.49352
    # span = 16.2 / chord

    mesh_dict = {
        "num_y": ny,
        "num_x": nx,
        "wing_type": "rect",
        "symmetry": False,
        "span": 10.0,
        "chord": 1,
        "span_cos_sppacing": 1.0,
        "chord_cos_sacing": 1.0,
    }
    # Generate OML MESH for a rectangular wing with NACA0012 airfoil
    airfoil = np.loadtxt(fname='/home/lsdo/Documents/packages/VAST/VAST/core/vlm_llt/naca0012.txt')
    z_val = np.linspace(-5, 5, 13)
    oml_mesh = np.zeros((21, 13, 3))
    oml_mesh[:, :, 0] = np.outer(airfoil[:, 0],np.ones(13))-0.5
    oml_mesh[:, :, 1] = np.outer(np.ones(21),z_val)
    oml_mesh[:, :, 2] = np.outer(airfoil[:, 1],np.ones(13))
    
    

    # add oml mesh as a csdl variable
    oml = model_1.create_input('oml', val=oml_mesh.reshape(1, 21, 13, 3))

    # create a random displacement on the oml mesh
    disp_temp = np.linspace(0.6, 0,7)
    disp_z = np.outer(np.ones(21),np.concatenate((disp_temp, disp_temp[:-1][::-1])))
    disp = np.zeros((1, 21, 13, 3))
    disp[0, :, :, 2] = disp_z
    oml_disp = model_1.create_input('oml_displacement', val=disp)



    # Generate mesh of a rectangular wing
    mesh = generate_mesh(mesh_dict) #(nx,ny,3)
    
    ####################################################################
    # project the displacement on the oml mesh to the camber surface mesh
    G_mat = NodalMap(oml_mesh.reshape(-1,3), mesh.reshape(-1,3), RBF_width_par=2, RBF_func=RadialBasisFunctions.Gaussian)
    map_in = model_1.create_input('map_in', val=G_mat.map)
    ####################################################################

    mapped_disp = np.einsum('ij,jk->ik',G_mat.map,disp.reshape(-1,3)).reshape(mesh.shape)
    deformed_mesh = mesh + mapped_disp
    
    import pyvista as pv
    plotter = pv.Plotter()
    grid = pv.StructuredGrid(oml_mesh[:, :, 0]+disp[0,:, :, 0], oml_mesh[:, :, 1]+disp[0,:, :, 1], oml_mesh[:, :, 2]+disp[0,:, :, 2])
    grid_1 = pv.StructuredGrid(deformed_mesh[:, :, 0], deformed_mesh[:, :, 1], deformed_mesh[:, :, 2])

    plotter.add_mesh(grid, show_edges=False,opacity=0.5, color='red')
    plotter.add_mesh(grid_1, show_edges=True,color='grey')
    plotter.set_background('white')
    plotter.show()
    offset = 0

    mesh_val = np.zeros((num_nodes, nx, ny, 3))

    for i in range(num_nodes):
        mesh_val[i, :, :, :] = mesh
        mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0]
        mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] + offset

    wing = model_1.create_input('wing_undef', val=mesh_val)
    oml_disp_reshaped = csdl.reshape(oml_disp, (21* 13, 3))
    wing_def = wing + csdl.reshape(csdl.einsum(map_in, oml_disp_reshaped,subscripts='ij,jk->ik'), (num_nodes, nx, ny, 3))
    model_1.register_output('wing', wing_def)
    ####################################################################
    if fluid_problem.solver_option == 'VLM':
        eval_pts_shapes = [(num_nodes, x[1] - 1, x[2] - 1, 3) for x in surface_shapes]
        
        submodel = VLMSolverModel(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            num_nodes=num_nodes,
            eval_pts_shapes=eval_pts_shapes,
            AcStates='dummy',
        )
    wing_C_L_OAS = np.array([0.4426841725811703]).reshape((num_nodes, 1))
    wing_C_D_i_OAS = np.array([0.005878842561184834]).reshape((num_nodes, 1))

    model_1.add(submodel, 'VLMSolverModel')
    sim = Simulator(model_1)
    sim.run()


    # project the force on the oml mesh to the camber surface mesh
    G_mat_out = NodalMap(sim['wing_eval_pts_coords'].reshape(-1,3),oml_mesh.reshape(-1,3),  RBF_width_par=2, RBF_func=RadialBasisFunctions.Gaussian)
    map_out = model_1.create_input('map_in', val=G_mat_out.map)

    mapped_force = np.einsum('ij,jk->ik',G_mat_out.map,sim['panel_forces'].reshape(-1,3)).reshape(oml_mesh.shape)

    # formulate the invariant matrix to map the force on the quarter-chord of each panel (num_nodes, nx-1, ny-1, 3) to the vlm deformed mesh (num_nodes, nx, ny, 3)
    N_A = NodalMap(sim['wing_eval_pts_coords'].reshape(-1,3), 
                    mesh.reshape(-1,3), 
                    RBF_width_par=2,
                    RBF_func=RadialBasisFunctions.Gaussian).map