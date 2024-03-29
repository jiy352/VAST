import csdl
from lsdo_modules.module.module import Module
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from VAST.core.submodels.input_submodels.create_input_model import CreateACSatesModel
from VAST.core.submodels.input_submodels.create_input_module import CreateACSatesModule
from VAST.core.vlm_llt.vlm_solver import VLMSolverModel

from VAST.core.fluid_problem import FluidProblem
import m3l
from typing import List
import numpy as np


class VASTFluidSover(m3l.ImplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('fluid_solver', True)
        self.parameters.declare('num_nodes', default=1, types=int)

        self.parameters.declare('fluid_problem', types=FluidProblem)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('solve_option', default='direct')
        self.parameters.declare('mesh_unit', default='m')
        self.parameters.declare('cl0', default=None)
        self.parameters.declare('input_dicts', default=None)

        self.parameters.declare('ML', default=False)


   

    def compute_derivates(self,inputs,derivatives):
        pass

    def evaluate(self, ac_states, surfaces, wake_circulations, displacements, bd_circulations : List[m3l.Variable]=None):
        '''
        Evaluates the vast model.
        
        Parameters
        ----------
        displacements : list of m3l.Variable = None
            The forces on the mesh nodes.

        Returns
        -------
        bd_circulations : m3l.Variable
            single m3l variable of the bound vortex circulations

        '''
        # Gets information for naming/shapes
        # beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        # mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = self.parameters['num_nodes']
        ML = self.parameters['ML']

        self.inputs = {}
        if displacements is not None:
            for i in range(len(surface_names)):
                surface_name = surface_names[i]
                self.inputs[f'{surface_name}_displacements'] = displacements[i]


        self.outputs ={}
        if circulations is not None:
            self.outputs[gamma_b] = bd_circulations

        self.arguments = {}
        self.name = f"{''.join(surface_names)}_vlm_implicit_model"

        # Declare residual partials - key is the residual csdl name, value is the m3l variable that's being partialed
        self.residual_partials = {}
        self.residual_partials['circulation_jacobian'] = displacements
        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            self.residual_partials[f'{surface_name}_displacement_jacobian'] = displacements[i]
        return

    def compute_residual(self):
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
        num_nodes = self.parameters['num_nodes'] #surface_shapes[0][0]
        mesh_unit = self.parameters['mesh_unit']
        cl0 = self.parameters['cl0']
        input_dicts = self.parameters['input_dicts']

        ML = self.parameters['ML']


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
            cl0=cl0,
            input_dicts=input_dicts,
            ML=ML,)

        csdl_model.add_module(submodule,'vast')
    
        return csdl_model   


    def solve_residual_equations(self):
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
        num_nodes = self.parameters['num_nodes'] #surface_shapes[0][0]
        mesh_unit = self.parameters['mesh_unit']
        cl0 = self.parameters['cl0']
        input_dicts = self.parameters['input_dicts']

        ML = self.parameters['ML']


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
            cl0=cl0,
            input_dicts=input_dicts,
            ML=ML,)

        csdl_model.add_module(submodule,'vast')
    
        return csdl_model  


    def compute_derivatives(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']
        mesh_units = self.parameters['mesh_units']

        # CSDL model computing residual jacobian - inputs and partials are named as defined in evaluate()
        csdl_model = LinearBeamResidualJacobiansCSDL(
            module=self,
            beams=beams,
            bounds=bounds,
            joints=joints,
            mesh_units=mesh_units)
        
        return csdl_model



#         self.arguments['u'] = ac_states['u']
#         self.arguments['v'] = ac_states['v']
#         self.arguments['w'] = ac_states['w']
#         self.arguments['p'] = ac_states['p']
#         self.arguments['q'] = ac_states['q']
#         self.arguments['r'] = ac_states['r']
#         self.arguments['theta'] = ac_states['theta']
#         self.arguments['psi'] = ac_states['psi']
#         self.arguments['gamma'] = ac_states['gamma']
#         # self.arguments['psiw'] = ac_states['psi_w']

        
#         # Create the M3L variables that are being output
#         forces = []
#         cl_spans = []
#         re_spans = []  
#         panel_areas = [] 
#         evaluation_pts = []
#         for i in range(len(surface_names)):
#             surface_name = surface_names[i]
#             surface_shapes = self.parameters['surface_shapes'][i]
#             num_nodes = surface_shapes[0]
#             nx = surface_shapes[1]
#             ny = surface_shapes[2]

#             force = m3l.Variable(name=f'{surface_name}_total_forces', shape=(num_nodes, int((nx-1)*(ny-1)), 3), operation=self)
#             cl_span = m3l.Variable(name=f'{surface_name}_cl_span_total', shape=(num_nodes, int(ny-1),1), operation=self)
#             re_span = m3l.Variable(name=f'{surface_name}_re_span', shape=(num_nodes, int(ny-1),1), operation=self)
#             panel_area = m3l.Variable(name=f'{surface_name}_s_panel', shape=(num_nodes,nx-1,ny-1), operation=self)
#             evaluation_pt = m3l.Variable(name=f'{surface_name}_eval_pts_coords', shape=(num_nodes,nx-1,ny-1,3), operation=self)

#             forces.append(force)
#             cl_spans.append(cl_span)
#             re_spans.append(re_span)
#             panel_areas.append(panel_area)
#             evaluation_pts.append(evaluation_pt)

#         total_force = m3l.Variable(name='F', shape=(num_nodes, 3), operation=self)
#         total_moment = m3l.Variable(name='M', shape=(num_nodes, 3), operation=self)
#         # return spanwise cl, forces on panels with vlm internal correction for cl0 and cdv, total force and total moment for trim
#         if ML:
#             return cl_spans, re_spans, forces, panel_areas, evaluation_pt, total_force, total_moment
#         else:
#             return forces, total_force, total_moment


# class VASTMesh(Module):
#     def initialize(self, kwargs):
#         self.parameters.declare('meshes', types=dict)
#         self.parameters.declare('mesh_units', default='m')

class VASTCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('fluid_problem',default=None)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('solve_option', default='direct')
        self.parameters.declare('mesh_unit', default='m')
        self.parameters.declare('cl0', default=None)
        self.parameters.declare('input_dicts', default=None)
        self.parameters.declare('ML', default=False)

    def define(self):
        fluid_problem = self.parameters['fluid_problem']
        solver_options = fluid_problem.solver_option
        problem_type = fluid_problem.problem_type

        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = surface_shapes[0][0]
        mesh_unit = self.parameters['mesh_unit']
        cl0 = self.parameters['cl0']

        ML = self.parameters['ML']

        # todo: connect the mesh to the solver
        # wing = model_1.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))
        # try:
        #     input_dicts = self.parameters['input_dicts']
        #     submodel = CreateACSatesModule(v_inf=input_dicts['v_inf'],theta=input_dicts['theta'],num_nodes=num_nodes)
        #     self.add_module(submodel, 'ACSates')

        # wing_incidence_angle = self.register_module_input('wing_incidence', shape=(1, ), computed_upstream=False)

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = self.parameters['surface_shapes'][i]
            # displacements = self.declare_variable(f'{surface_name}_displacements', shape=(surface_shape),val=0.)

            undef_mesh = self.declare_variable(f'{surface_name}_mesh', val=np.zeros(surface_shape))
            # mesh = undef_mesh  #+ displacements
            # self.register_module_output(surface_name, mesh)

        # except:
        #     pass

        if fluid_problem.solver_option == 'VLM' and fluid_problem.problem_type == 'fixed_wake':
            submodel = VLMSolverModel(
                surface_names=surface_names,
                surface_shapes=surface_shapes,
                AcStates='dummy',
                mesh_unit=mesh_unit,
                cl0=cl0,
                ML=ML
            )
            self.add_module(submodel, 'VLMSolverModel')

        # TODO: make dynamic case works
        elif fluid_problem.solver_option == 'VLM' and fluid_problem.problem_type == 'prescribed_wake':
            sim = Simulator(UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                            surface_properties_dict=surface_properties_dict,mesh_val=mesh_val), mode='rev')

if __name__ == "__main__":

    # import numpy as np
    # from VAST.utils.generate_mesh import *
    # from python_csdl_backend import Simulator
    # import caddee.api as cd 

    # fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

    # num_nodes=1; nx=3; ny=11

    # v_inf = np.ones((num_nodes,1))*248.136
    # theta = np.deg2rad(np.ones((num_nodes,1))*5)  # pitch angles


    # surface_names = ['wing']
    # surface_shapes = [(num_nodes, nx, ny, 3)]
    # mesh_dict = {
    #     "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": 10.0,
    #     "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0,
    # }
    # # Generate mesh of a rectangular wing
    # mesh = generate_mesh(mesh_dict)

    # ###########################################
    # # 1. Create a dummy m3l.Model()
    # ###########################################
    # dummy_model = m3l.Model()
    # # fluid_model = VASTFluidSover(fluid_problem=fluid_problem, surface_names=surface_names, surface_shapes=surface_shapes, mesh_unit='m', cl0=0.0)


    # # submodel = CreateACSatesModel(v_inf=v_inf, theta=theta, num_nodes=num_nodes)
    # # model_1.add(submodel, 'InputsModule')
    # fluid_model = VASTFluidSover(fluid_problem=fluid_problem,
    #                              surface_names=surface_names,
    #                              surface_shapes=surface_shapes,
    #                              input_dicts=None,)


    # ###########################################
    # # 3. set fluid_model inputs 
    # ###########################################
    # fluid_model.set_module_input('u',val=v_inf)
    # fluid_model.set_module_input('v',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('w',val=np.ones((num_nodes,1))*0)
    # fluid_model.set_module_input('p',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('q',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('r',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('phi',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('theta',val=theta)
    # fluid_model.set_module_input('psi',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('x',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('y',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('z',val=np.ones((num_nodes, 1))*1000)
    # fluid_model.set_module_input('phiw',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('gamma',val=np.zeros((num_nodes, 1)))
    # fluid_model.set_module_input('psiw',val=np.zeros((num_nodes, 1)))
 
    # fluid_model.set_module_input('wing_undef_mesh', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))

    # input_dicts = {}
    # # input_dicts['v_inf'] = v_inf
    # # input_dicts['theta'] = theta
    # # input_dicts['undef_mesh'] = [np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh)]
    # # input_dicts['displacements'] = [np.zeros((num_nodes, nx, ny, 3))]

    # ###########################################
    # # 2. Create fluid_model as VASTFluidSover 
    # # (msl.explicit operation)
    # ###########################################


    # displacements = []
    # for i in range(len(surface_names)):
    #     surface_name = surface_names[i]
    #     surface_shape = surface_shapes[i]
    #     displacement = m3l.Variable(f'{surface_name}_displacements',shape=surface_shape,value=np.ones(surface_shape)*10)
    #     fluid_model.set_module_input(f'{surface_name}_displacements', val=np.ones(surface_shape)*100)
    #     displacements.append(displacement)

    # ###########################################
    # # 4. call fluid_model.evaluate to get
    # # surface panel forces
    # ###########################################
    # forces = fluid_model.evaluate(displacements)

    # ###########################################
    # # 5. register outputs to dummy_model
    # ###########################################
    # for i in range(len(surface_names)):
    #     surface_name = surface_names[i]
    #     dummy_model.register_output(forces[i])
        
    # ###########################################
    # # 6. call _assemble_csdl to get dummy_model_csdl
    # ###########################################
    # dummy_model_csdl = dummy_model._assemble_csdl()
    # ###########################################
    # # 7. use sim.run to run the csdl model
    # ###########################################    

    # sim = Simulator(dummy_model_csdl,analytics=False) # add simulator
    # sim.run()




    import numpy as np
    from VAST.utils.generate_mesh import *
    from python_csdl_backend import Simulator

    fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

    num_nodes=1; nx=5; ny=11

    v_inf = np.ones((num_nodes,1))*57
    theta = np.deg2rad(np.ones((num_nodes,1))*5)  # pitch angles

    model_1 = ModuleCSDL()

    # submodel = CreateACSatesModel(v_inf=v_inf, theta=theta, num_nodes=num_nodes)
    # model_1.add(submodel, 'InputsModule')
    # model_1.add_design_variable('InputsModule.u')
    model_1.create_input('u', val=70, shape=(num_nodes, 1))
    model_1.add_design_variable('u', lower=50, upper=100, scaler=1e-2)

    surface_names = ['wing','tail']
    surface_shapes = [(num_nodes, nx, ny, 3),(num_nodes, nx-2, ny-2, 3)]

    mesh_dict = {
        "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": 10.0,
        "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0,
    }

    mesh_dict_1 = {
        "num_y": ny-2, "num_x": nx-2, "wing_type": "rect", "symmetry": False, "span": 10.0,
        "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0,
    }

    # Generate mesh of a rectangular wing
    mesh = generate_mesh(mesh_dict) 
    mesh_1 = generate_mesh(mesh_dict_1) 
    wing = model_1.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))
    wing = model_1.create_input('tail', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh_1))

    # add VAST fluid solver

    submodel = VASTCSDL(
        fluid_problem=fluid_problem,
        surface_names=surface_names,
        surface_shapes=surface_shapes,
    )
    model_1.add_module(submodel, 'VASTSolverModule')
    sim = Simulator(model_1, analytics=True) # add simulator

    
    model_1.add_objective('VASTSolverModule.VLMSolverModel.VLM_outputs.LiftDrag.total_drag')
    sim.run()
    sim.check_totals()

