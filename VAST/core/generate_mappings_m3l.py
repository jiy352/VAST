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

            nodal_displacements = csdl_model.register_module_input(name=surface_name+'_nodal_displacements', shape=nodal_displacements.shape)
            flatenned_vlm_mesh_displacements = csdl.sparsematmat(nodal_displacements, displacements_map)
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

if __name__ == "__main__":
    import numpy as np
    from VAST.utils.generate_mesh import *
    from python_csdl_backend import Simulator

    fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

    num_nodes=1; nx=3; ny=11

    v_inf = np.ones((num_nodes,1))*248.136
    theta = np.deg2rad(np.ones((num_nodes,1))*5)  # pitch angles


    surface_names = ['wing']
    surface_shapes = [(num_nodes, nx, ny, 3)]
    mesh_dict = {
        "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": 10.0,
        "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0,
    }
    # Generate mesh of a rectangular wing
    mesh = generate_mesh(mesh_dict)

    ###########################################
    # 1. Create a dummy m3l.Model()
    ###########################################
    dummy_model = m3l.Model()
 
    # fluid_model.set_module_input('wing_undef_mesh', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))

    input_dicts = {}
    input_dicts['v_inf'] = v_inf
    input_dicts['theta'] = theta
    input_dicts['undef_mesh'] = [np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh)]
    input_dicts['displacements'] = [np.zeros((num_nodes, nx, ny, 3))]


    airfoil = np.loadtxt(fname='/home/lsdo/Documents/packages/VAST/VAST/core/vlm_llt/naca0012.txt')
    z_val = np.linspace(-5, 5, 13)
    oml_mesh = np.zeros((21, 13, 3))
    oml_mesh[:, :, 0] = np.outer(airfoil[:, 0],np.ones(13))-0.5
    oml_mesh[:, :, 1] = np.outer(np.ones(21),z_val)
    oml_mesh[:, :, 2] = np.outer(airfoil[:, 1],np.ones(13))

    ###########################################
    # 2. Create fluid_model as VASTFluidSover 
    # (msl.explicit operation)
    ###########################################
    fluid_model = VASTNodelDisplacements(fluid_problem=fluid_problem,
                                 surface_names=surface_names,
                                 surface_shapes=surface_shapes,
                                 oml_mesh=[oml_mesh],)


    displacements = []
    for i in range(len(surface_names)):
        surface_name = surface_names[i]
        surface_shape = surface_shapes[i]
        displacement = m3l.Variable(f'{surface_name}_displacements',shape=surface_shape,value=np.zeros(surface_shape))
        fluid_model.set_module_input(f'{surface_name}_displacements', val=np.zeros(surface_shape))
        displacements.append(displacement)

    ###########################################
    # 4. call fluid_model.evaluate to get
    # surface panel forces
    ###########################################
    forces = fluid_model.evaluate(displacements)

    ###########################################
    # 5. register outputs to dummy_model
    ###########################################
    for i in range(len(surface_names)):
        surface_name = surface_names[i]
        dummy_model.register_output(forces[i])
        
    ###########################################
    # 6. call _assemble_csdl to get dummy_model_csdl
    ###########################################
    dummy_model_csdl = dummy_model._assemble_csdl()
    ###########################################
    # 7. use sim.run to run the csdl model
    ###########################################    

    sim = Simulator(dummy_model_csdl,analytics=False) # add simulator
    sim.run()



    # ####################################################################
    # # 2. add VLM meshes
    # ####################################################################
    # # single lifting surface
    # nx = 3  # number of points in streamwise direction
    # ny = 11  # number of points in spanwise direction
    # num_nodes = 1

    # surface_names = ['wing']
    # surface_shapes = [(num_nodes, nx, ny, 3)]

    # # chord = 1.49352
    # # span = 16.2 / chord

    # mesh_dict = {
    #     "num_y": ny,
    #     "num_x": nx,
    #     "wing_type": "rect",
    #     "symmetry": False,
    #     "span": 10.0,
    #     "chord": 1,
    #     "span_cos_sppacing": 1.0,
    #     "chord_cos_sacing": 1.0,
    # }
    # # Generate OML MESH for a rectangular wing with NACA0012 airfoil
    # airfoil = np.loadtxt(fname='/home/lsdo/Documents/packages/VAST/VAST/core/vlm_llt/naca0012.txt')
    # z_val = np.linspace(-5, 5, 13)
    # oml_mesh = np.zeros((21, 13, 3))
    # oml_mesh[:, :, 0] = np.outer(airfoil[:, 0],np.ones(13))-0.5
    # oml_mesh[:, :, 1] = np.outer(np.ones(21),z_val)
    # oml_mesh[:, :, 2] = np.outer(airfoil[:, 1],np.ones(13))
    
    
    # model_1 = csdl.Model()
    # # add oml mesh as a csdl variable
    # oml = model_1.create_input('oml', val=oml_mesh.reshape(1, 21, 13, 3))

    # # create a random displacement on the oml mesh
    # disp_temp = np.linspace(0.6, 0,7)
    # disp_z = np.outer(np.ones(21),np.concatenate((disp_temp, disp_temp[:-1][::-1])))
    # disp = np.zeros((1, 21, 13, 3))
    # disp[0, :, :, 2] = disp_z
    # oml_disp = model_1.create_input('oml_displacement', val=disp)



    # # Generate mesh of a rectangular wing
    # mesh = generate_mesh(mesh_dict) #(nx,ny,3)

    # ####################################################################
    # # project the displacement on the oml mesh to the camber surface mesh
    # G_mat = NodalMap(oml_mesh.reshape(-1,3), mesh.reshape(-1,3), RBF_width_par=2, RBF_func=RadialBasisFunctions.Gaussian)
    # map_in = model_1.create_input('map_in', val=G_mat.map)
    # ####################################################################

    # mapped_disp = np.einsum('ij,jk->ik',G_mat.map,disp.reshape(-1,3)).reshape(mesh.shape)
    # deformed_mesh = mesh + mapped_disp
    
    # import pyvista as pv
    # plotter = pv.Plotter()
    # grid = pv.StructuredGrid(oml_mesh[:, :, 0]+disp[0,:, :, 0], oml_mesh[:, :, 1]+disp[0,:, :, 1], oml_mesh[:, :, 2]+disp[0,:, :, 2])
    # grid_1 = pv.StructuredGrid(deformed_mesh[:, :, 0], deformed_mesh[:, :, 1], deformed_mesh[:, :, 2])

    # plotter.add_mesh(grid, show_edges=False,opacity=0.5, color='red')
    # plotter.add_mesh(grid_1, show_edges=True,color='grey')
    # plotter.set_background('white')
    # plotter.show()
    # offset = 0

    

    # mesh_val = np.zeros((num_nodes, nx, ny, 3))

    # for i in range(num_nodes):
    #     mesh_val[i, :, :, :] = mesh
    #     mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0]
    #     mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] + offset

    # wing = model_1.create_input('wing_undef', val=mesh_val)
    # oml_disp_reshaped = csdl.reshape(oml_disp, (21* 13, 3))
    # wing_def = wing + csdl.reshape(csdl.einsum(map_in, oml_disp_reshaped,subscripts='ij,jk->ik'), (num_nodes, nx, ny, 3))
    # model_1.register_output('wing', wing_def)
    # ####################################################################
    
    # submodel = VLMSolverModel(
    #     surface_names=surface_names,
    #     surface_shapes=surface_shapes,
    #     num_nodes=num_nodes,
    #     AcStates='dummy',
    # )
    # wing_C_L_OAS = np.array([0.4426841725811703]).reshape((num_nodes, 1))
    # wing_C_D_i_OAS = np.array([0.005878842561184834]).reshape((num_nodes, 1))

    # model_1.add(submodel, 'VLMSolverModel')
    # sim = Simulator(model_1)
    # sim.run()


    # # project the force on the oml mesh to the camber surface mesh
    # G_mat_out = NodalMap(sim['wing_eval_pts_coords'].reshape(-1,3),oml_mesh.reshape(-1,3),  RBF_width_par=2, RBF_func=RadialBasisFunctions.Gaussian)
    # map_out = model_1.create_input('map_in', val=G_mat_out.map)

    # mapped_force = np.einsum('ij,jk->ik',G_mat_out.map,sim['panel_forces'].reshape(-1,3)).reshape(oml_mesh.shape)

    # # formulate the invariant matrix to map the force on the quarter-chord of each panel (num_nodes, nx-1, ny-1, 3) to the vlm deformed mesh (num_nodes, nx, ny, 3)
    # N_A = NodalMap(sim['wing_eval_pts_coords'].reshape(-1,3), 
    #                 mesh.reshape(-1,3), 
    #                 RBF_width_par=2,
    #                 RBF_func=RadialBasisFunctions.Gaussian).map

