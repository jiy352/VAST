from ozone.api import ODEProblem
import csdl

import numpy as np

from VAST.core.submodels.kinematic_submodels.adapter_comp import AdapterComp
from VAST.core.submodels.aerodynamic_submodels.combine_gamma_w import CombineGammaW
from VAST.core.submodels.implicit_submodels.solve_group import SolveMatrix
from VAST.core.submodels.aerodynamic_submodels.seperate_gamma_b import SeperateGammab
from VAST.core.submodels.geometric_submodels.mesh_preprocessing_comp import MeshPreprocessingComp
from VAST.core.submodels.output_submodels.vlm_post_processing.horseshoe_circulations import HorseshoeCirculations
from VAST.core.submodels.output_submodels.vlm_post_processing.eval_pts_velocities_mls import EvalPtsVel
from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_system import ODESystemModel
from VAST.core.submodels.output_submodels.vlm_post_processing.compute_thrust_drag_dynamic import ThrustDrag


class UVLMPostProc(csdl.Model):
    '''This class generates the solver for the prescribed VLM.'''

    def initialize(self):
        self.parameters.declare('num_times')
        self.parameters.declare('states_dict')
        self.parameters.declare('surface_properties_dict')
        # self.parameters.declare('mesh_val')
        self.parameters.declare('h_stepsize')
        self.parameters.declare('problem_type',default='prescirbed_wake')

    def define(self):
        num_times = self.parameters['num_times']
        h_stepsize = self.parameters['h_stepsize']


        # mesh_val = self.parameters['mesh_val']

        AcStates_val_dict = self.parameters['states_dict']
        surface_properties_dict = self.parameters['surface_properties_dict']
        surface_names = list(surface_properties_dict.keys())
        surface_shapes = list(surface_properties_dict.values())

        ####################################
        # Create parameters
        ####################################
        # for data in AcStates_val_dict:
        #     string_name = data
        #     val = AcStates_val_dict[data]            
        #     # print('{:15} = {},shape{}'.format(string_name, val, val.shape))
        #     variable = self.create_input(string_name,
        #                                  val=val)
        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            gamma_w_0_name = surface_name + '_gamma_w_0'
            wake_coords_0_name = surface_name + '_wake_coords_0'
            surface_shape = surface_shapes[i]
            nx = surface_shape[0]
            ny = surface_shape[1]

        eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
        ode_surface_shapes = [(num_times, ) + item for item in surface_shapes]
        op_surface_names = ['op_' + x for x in surface_names]
        eval_pts_shapes =        [
            tuple(map(lambda i, j: i - j, item, (0, 1, 1, 0)))
            for item in ode_surface_shapes
        ]
        self.add(MeshPreprocessingComp(surface_names=surface_names,
                                       surface_shapes=ode_surface_shapes,
                                       eval_pts_location=0.25,
                                       eval_pts_option='auto'),
                 name='MeshPreprocessing_comp')

        m = AdapterComp(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
        )
        self.add(m, name='adapter_comp')

        self.add(CombineGammaW(surface_names=op_surface_names, surface_shapes=ode_surface_shapes, n_wake_pts_chord=num_times-1),
            name='combine_gamma_w')

        self.add(SolveMatrix(n_wake_pts_chord=num_times-1,
                                surface_names=surface_names,
                                bd_vortex_shapes=ode_surface_shapes,
                                delta_t=h_stepsize,
                                problem_type='prescribed_wake'),
                    name='solve_gamma_b_group')
        self.add(SeperateGammab(surface_names=surface_names,
                                surface_shapes=ode_surface_shapes),
                 name='seperate_gamma_b')

        eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
        eval_pts_shapes =        [
            tuple(map(lambda i, j: i - j, item, (0, 1, 1, 0)))
            for item in ode_surface_shapes
        ]

        # compute lift and drag
        submodel = HorseshoeCirculations(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
        )
        self.add(submodel, name='compute_horseshoe_circulation')

        submodel = EvalPtsVel(
            eval_pts_names=eval_pts_names,
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_option='auto',
            eval_pts_location=0.25,
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
            n_wake_pts_chord=num_times-1,
            delta_t=h_stepsize,
            problem_type='prescribed_wake',
            eps=4e-5,
        )
        self.add(submodel, name='EvalPtsVel')

        submodel = ThrustDrag(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
            eval_pts_option='auto',
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_names=eval_pts_names,
            sprs=None,
            coeffs_aoa=None,
            coeffs_cd=None,
        )
        self.add(submodel, name='ThrustDrag')
