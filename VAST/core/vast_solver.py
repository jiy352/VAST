import csdl
from lsdo_modules.module.module import Module
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from VAST.core.submodels.input_submodels.create_input_model import CreateACSatesModel
from VAST.core.submodels.input_submodels.create_input_module import CreateACSatesModule
from VAST.core.vlm_llt.vlm_solver import VLMSolverModel

from VAST.core.fluid_problem import FluidProblem
import m3l


class VASTFluidSover(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('fluid_solver', True)
        self.parameters.declare('compute_mass_properties', default=True, types=bool)

        self.parameters.declare('fluid_problem', types=FluidProblem)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('solve_option', default='direct')
        self.parameters.declare('mesh_unit', default='m')
        self.parameters.declare('cl0', default=None)

    def compute(self):
        '''
        Creates a CSDL model to compute the solver outputs.

        Returns
        -------
        csdl_model : csdl.Model
            The csdl model which computes the outputs (the normal solver)
        '''
        fluid_problem = self.parameters['fluid_problem']


        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = surface_shapes[0][0]
        mesh_unit = self.parameters['mesh_unit']
        cl0 = self.parameters['cl0']
        csdl_model = ModuleCSDL()

        self.displacements = []

        # for i in range(len(surface_names)):
        #     surface_name = surface_names[i]
        #     surface_shape = self.parameters['surface_shapes'][i]
        #     displacement = csdl_model.register_module_input(f'{surface_name}_displacements', shape=(surface_shape),val=0.0)
        #     self.displacements.append(displacement)

        submodule = VASTCSDL(
            module=self,
            fluid_problem=fluid_problem,
            surface_names=surface_names,  
            surface_shapes=surface_shapes,
            mesh_unit=mesh_unit,
            cl0=cl0)

        csdl_model.add_module(submodule,'vast')
    
        return csdl_model      

    def compute_derivates(self,inputs,derivatives):
        pass

    def evaluate(self,displacements):
        '''
        Evaluates the vast model.
        
        Parameters
        ----------
        displacements : list of m3l.Variable = None
            The forces on the mesh nodes.

        Returns
        -------
        panel_forces : m3l.Variable
            The displacements of the mesh nodes.

        '''
        # Assembles the CSDL model
        operation_csdl = self.compute()

        # Gets information for naming/shapes
        # beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        # mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']

        arguments = {}
        print(displacements)

        # displacements = self.displacements 
        if displacements is not None:
            for i in range(len(surface_names)):
                surface_name = surface_names[i]

                arguments[f'{surface_name}_displacements'] = displacements[i]
        print(arguments)

        # Create the M3L graph operation
        vast_operation = m3l.CSDLOperation(name='vast_fluid_model', arguments=arguments, operation_csdl=operation_csdl)
        
        # Create the M3L variables that are being output
        forces = []
        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shapes = self.parameters['surface_shapes'][i]
            num_nodes = surface_shapes[0]
            nx = surface_shapes[1]
            ny = surface_shapes[2]
            force = m3l.Variable(name=f'{surface_name}_total_forces', shape=(num_nodes, int((nx-1)*(ny-1)),3), operation=vast_operation)
            forces.append(force)

        return forces


class VASTMesh(Module):
    def initialize(self, kwargs):
        self.parameters.declare('meshes', types=dict)
        self.parameters.declare('mesh_units', default='m')

class VASTCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('fluid_problem',default=None)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('solve_option', default='direct')
        self.parameters.declare('mesh_unit', default='m')
        self.parameters.declare('cl0', default=None)

    def define(self):
        fluid_problem = self.parameters['fluid_problem']
        solver_options = fluid_problem.solver_option
        problem_type = fluid_problem.problem_type

        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = surface_shapes[0][0]
        mesh_unit = self.parameters['mesh_unit']
        cl0 = self.parameters['cl0']

        # todo: connect the mesh to the solver
        # wing = model_1.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = self.parameters['surface_shapes'][i]
            displacements = self.register_module_input(f'{surface_name}_displacements', shape=(surface_shape),val=0.0)
            self.print_var(displacements)

            undef_mesh = self.declare_variable(f'{surface_name}_undef_mesh', shape=(surface_shape))
            mesh = undef_mesh  + displacements
            self.register_module_output(surface_name, mesh)


        if fluid_problem.solver_option == 'VLM' and fluid_problem.problem_type == 'fixed_wake':
            submodel = VLMSolverModel(
                surface_names=surface_names,
                surface_shapes=surface_shapes,
                AcStates='dummy',
                mesh_unit=mesh_unit,
                cl0=cl0,
            )
            self.add(submodel, 'VLMSolverModel')

        # TODO: make dynamic case works
        elif fluid_problem.solver_option == 'VLM' and fluid_problem.problem_type == 'prescribed_wake':
            sim = Simulator(UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                            surface_properties_dict=surface_properties_dict,mesh_val=mesh_val), mode='rev')

if __name__ == "__main__":

    import numpy as np
    from VAST.utils.generate_mesh import *
    from python_csdl_backend import Simulator
    import caddee.api as cd 

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

    ###########################################
    # 2. Create fluid_model as VASTFluidSover 
    # (msl.explicit operation)
    ###########################################
    fluid_model = VASTFluidSover(fluid_problem=fluid_problem,
                                 surface_names=surface_names,
                                 surface_shapes=surface_shapes,)

    # submodel = CreateACSatesModel(v_inf=v_inf, theta=theta, num_nodes=num_nodes)
    # model_1.add(submodel, 'InputsModule')
    
    ###########################################
    # 3. set fluid_model inputs 
    ###########################################
    fluid_model.set_module_input('u',val=v_inf)
    fluid_model.set_module_input('v',val=np.zeros((num_nodes, 1)))
    fluid_model.set_module_input('w',val=np.ones((num_nodes,1))*0)
    fluid_model.set_module_input('p',val=np.zeros((num_nodes, 1)))
    fluid_model.set_module_input('q',val=np.zeros((num_nodes, 1)))
    fluid_model.set_module_input('r',val=np.zeros((num_nodes, 1)))
    fluid_model.set_module_input('phi',val=np.zeros((num_nodes, 1)))
    fluid_model.set_module_input('theta',val=theta)
    fluid_model.set_module_input('psi',val=np.zeros((num_nodes, 1)))
    fluid_model.set_module_input('x',val=np.zeros((num_nodes, 1)))
    fluid_model.set_module_input('y',val=np.zeros((num_nodes, 1)))
    fluid_model.set_module_input('z',val=np.ones((num_nodes, 1))*1000)
    fluid_model.set_module_input('phiw',val=np.zeros((num_nodes, 1)))
    fluid_model.set_module_input('gamma',val=np.zeros((num_nodes, 1)))
    fluid_model.set_module_input('psiw',val=np.zeros((num_nodes, 1)))
 
    fluid_model.set_module_input('wing_undef_mesh', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))
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




    # import numpy as np
    # from VAST.utils.generate_mesh import *
    # from python_csdl_backend import Simulator

    # fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

    # num_nodes=1; nx=3; ny=11

    # v_inf = np.ones((num_nodes,1))*248.136
    # theta = np.deg2rad(np.ones((num_nodes,1))*5)  # pitch angles

    # model_1 = ModuleCSDL()

    # submodel = CreateACSatesModel(v_inf=v_inf, theta=theta, num_nodes=num_nodes)
    # model_1.add(submodel, 'InputsModule')
    
    # surface_names = ['wing']
    # surface_shapes = [(num_nodes, nx, ny, 3)]
    # mesh_dict = {
    #     "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": 10.0,
    #     "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0,
    # }
    # # Generate mesh of a rectangular wing
    # mesh = generate_mesh(mesh_dict) 
    # wing = model_1.create_input('wing_undef_mesh', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))

    # # add VAST fluid solver

    # submodel = VASTCSDL(
    #     fluid_problem=fluid_problem,
    #     surface_names=surface_names,
    #     surface_shapes=surface_shapes,
    # )
    # model_1.add_module(submodel, 'VASTSolverModule')
    # sim = Simulator(model_1) # add simulator
    # sim.run()
