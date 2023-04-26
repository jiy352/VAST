import numpy as np
import csdl

from VAST.core.submodels.kinematic_submodels.adapter_comp import AdapterComp
from VAST.core.submodels.geometric_submodels.wake_coords_comp import WakeCoords
from VAST.core.submodels.geometric_submodels.mesh_preprocessing_comp import MeshPreprocessingComp
from VAST.core.submodels.aerodynamic_submodels.seperate_gamma_b import SeperateGammab
# from VAST.core.submodels.implicit_submodels.solve_group import SolveMatrix
# from VAST.core.submodels.implicit_submodels.compute_residual import ComputeResidual
from VAST.core.submodels.aerodynamic_submodels.rhs_group import RHS


''' this is a UVLM solver for prescribed wake option '''
# Jiayao Yan init date: 2023-04-20

class UVLMSystemPrescibedPrecomp(csdl.Model):
    '''
    contains
    1. MeshPreprocessing_comp
    2. WakeCoords_comp
    3. solve_gamma_b_group
    3. seperate_gamma_b_comp
    4. extract_gamma_w_comp
    '''
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('delta_t', default=100)
        self.parameters.declare('mesh_unit', default='m')

        self.parameters.declare('AcStates', default=None)
        self.parameters.declare('n_wake_pts_chord', default=2)

        self.parameters.declare('solve_option',
                                default='direct',
                                values=['direct', 'optimization'])
        self.parameters.declare('TE_idx', default='last')

    def define(self):
        # rename parameters
        num_nodes = self.parameters['num_nodes']
        n_wake_pts_chord = self.parameters['n_wake_pts_chord']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        mesh_unit = self.parameters['mesh_unit']

        wake_coords_names = [x + '_wake_coords' for x in surface_names]

        bd_vortex_shapes = surface_shapes
        delta_t = self.parameters['delta_t']
        gamma_b_shape = sum((i[1] - 1) * (i[2] - 1) for i in bd_vortex_shapes)

        # frame_vel = self.declare_variable('frame_vel', shape=(3, ))
        v_total_wake_names = [x + '_wake_total_vel' for x in surface_names]
        wake_vortex_pts_shapes = [
            tuple((n_wake_pts_chord, item[1], 3)) for item in surface_shapes
        ]
        wake_vel_shapes = [(x[0] * x[1], 3) for x in wake_vortex_pts_shapes]

        self.add(MeshPreprocessingComp(surface_names=surface_names,
                                       surface_shapes=surface_shapes,
                                       mesh_unit=mesh_unit),
                 name='MeshPreprocessing_comp')
        AcStates = self.parameters["AcStates"]
        if AcStates != None:
            add_adapter = True
        else:
            add_adapter = False

        if add_adapter == True:
            m = AdapterComp(
                surface_names=surface_names,
                surface_shapes=surface_shapes,
            )
            # m.optimize_ir(False)
            self.add(m, name='adapter_comp')

        m = WakeCoords(surface_names=surface_names,
                       surface_shapes=surface_shapes,
                       n_wake_pts_chord=n_wake_pts_chord,
                       delta_t=delta_t,
                       TE_idx=self.parameters['TE_idx'])
        # m.optimize_ir(False)
        self.add(m, name='WakeCoords_comp')

        n_wake_pts_chord = self.parameters['n_wake_pts_chord']
        delta_t = self.parameters['delta_t']

        bd_coll_pts_shapes = [
            tuple(map(lambda i, j: i - j, item, (0, 1, 1, 0)))
            for item in bd_vortex_shapes
        ]

        # print('bd_coll_pts_shapes', bd_coll_pts_shapes)

        # aic_bd_proj_names = [x + '_aic_bd_proj' for x in surface_names]
        wake_vortex_pts_shapes = [
            tuple((n_wake_pts_chord, item[1], item[2]))
            for item in bd_vortex_shapes
        ]

        model = csdl.Model()
        '''1. add the rhs'''
        model.add(
            RHS(
                n_wake_pts_chord=n_wake_pts_chord,
                surface_names=surface_names,
                bd_vortex_shapes=bd_vortex_shapes,
                delta_t=delta_t,
            ), 'RHS_group')

        n_wake_pts_chord = self.parameters['n_wake_pts_chord']
        self.add(model, 'prepossing_before_Solve')


class UVLMSystemPrescibed(csdl.Model):
    pass

if __name__ == '__main__':

    from VAST.core.fluid_problem import FluidProblem
    from VAST.utils.generate_mesh import *
    from VAST.core.submodels.input_submodels.create_input_module import CreateACSatesModule
    from VAST.core.vlm_llt.vlm_solver import VLMSolverModel
    from python_csdl_backend import Simulator

    fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

    model_1 = csdl.Model()
    num_nodes=1
    nx=3
    ny=11
    ####################################################################
    # 1. add aircraft states
    ####################################################################
    v_inf = np.ones((num_nodes,1))*248.136
    theta = np.deg2rad(np.ones((num_nodes,1))*5)  # pitch angles

    submodel = CreateACSatesModule(v_inf=v_inf, theta=theta, num_nodes=num_nodes)
    model_1.add(submodel, 'InputsModule')
    ####################################################################
    # 2. add VLM meshes
    ####################################################################
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
    uvlm_model = UVLMSystemPrescibedPrecomp(
        num_nodes=num_nodes,
        surface_names=surface_names,
        surface_shapes=surface_shapes,
        AcStates='dummy')

    model_1.add(uvlm_model, 'uvlm_model_precomp')

    sim = Simulator(model_1)
    sim.run()
