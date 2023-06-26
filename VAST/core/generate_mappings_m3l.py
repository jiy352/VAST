import csdl
from lsdo_modules.module.module import Module
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from VAST.core.submodels.input_submodels.create_input_model import CreateACSatesModel
from VAST.core.submodels.input_submodels.create_input_module import CreateACSatesModule
from VAST.core.vlm_llt.vlm_solver import VLMSolverModel

from VAST.core.fluid_problem import FluidProblem
import m3l
from VAST.core.vlm_llt.NodalMapping import NodalMap,RadialBasisFunctions


class VASTNodelDisplacements(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare(surface_names, types=list)
        self.parameters.declare(surface_shapes, types=list)
        self.parameters.declare(initial_mesh, default=list)

    def compute(self, nodal_displacements, nodal_displacements_mesh):
        surface_names = self.parameters['surface_names']  
        surface_shapes = self.parameters['surface_shapes']
        initial_mesh = self.parameters['initial_mesh']

        csdl_model = ModuleCSDL()

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            out_shape = int((surface_shape[1]-1)*(surface_shape[2]-1))
            initial_mesh = initial_mesh[i]
            displacements_map = self.disp_map(initial_mesh.reshape((-1,3)), oml=nodal_displacements_mesh.reshape((-1,3)))

            nodal_displacements = csdl_model.register_module_input(name=surface_name+'nodal_forces', shape=nodal_displacements.shape)
            displacements_map_csdl = csdl_model.create_input(f'nodal_to_{surface_name}_forces_map', val=displacements_map)
            flatenned_vlm_mesh_displacements = csdl.matmat(displacements_map_csdl, nodal_displacements)
            vlm_mesh_displacements = csdl.reshape(flatenned_vlm_mesh_forces, new_shape=(num_nodes,out_shape,3 ))
            csdl_model.register_module_output(f'{surface_name}_displacements', vlm_mesh_displacements)

        return csdl_model

    def evaluate(self, nodal_displacements, nodal_displacements_mesh):
        '''
        Maps nodal displacements_mesh from arbitrary locations to the mesh nodes.
        
        Parameters
        ----------
        nodal_displacements : a list of m3l.Variable
            The nodal_displacements to be mapped to the mesh nodes.
        nodal_displacements_mesh : a list of am.MappedArray
            The mesh that the nodal displacements_mesh are currently defined over.

        Returns
        -------
        mesh_forces : m3l.Variable
            The forces on the mesh.
        '''
        operation_csdl = self.compute(nodal_displacements=nodal_displacements, nodal_displacements_mesh=nodal_displacements_mesh)

        # beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        # mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
        arguments = {}
        for i in range(len(surface_names)):
            arguments[surface_names[i]+'_displacements'] = nodal_displacements[i]

        displacement_map_operation = m3l.CSDLOperation(name='ebbeam_force_map', arguments=arguments, operation_csdl=operation_csdl)
        output_shape = tuple(mesh.shape[:-1]) + (nodal_forces.shape[-1],)
        beam_forces = m3l.Variable(name=f'{beam_name}_forces', shape=output_shape, operation=force_map_operation)
        return beam_forces


    def disp_map(self, mesh, oml):

        # project the displacement on the oml mesh to the camber surface mesh
        weights = NodalMap(oml.reshape(-1,3), mesh.reshape(-1,3), RBF_width_par=2, RBF_func=RadialBasisFunctions.Gaussian)


        return weights

