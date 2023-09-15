from VAST.core.vlm_llt.vlm_system import VLMSystem
from VAST.core.submodels.output_submodels.vlm_post_processing.compute_outputs_group import Outputs
import numpy as np
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL

# from VLM_package.VLM_preprocessing.utils.generate_simple_mesh import *

# Here n_wake_pts_chord is just a dummy variable that always equal to 2. since we are using a long wake panel,
# we can just make n_wake_pts_chord=2 and delta_t a large number.


class VLMSolverModel(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('num_nodes', default=None)

        self.parameters.declare('AcStates', default=None)

        self.parameters.declare('free_stream_velocities', default=None)

        self.parameters.declare('eval_pts_location', default=0.25)
        self.parameters.declare('eval_pts_names', default=None)

        self.parameters.declare('eval_pts_option', default='auto')
        self.parameters.declare('eval_pts_shapes', default=None)
        self.parameters.declare('sprs', default=None)
        self.parameters.declare('coeffs_aoa', default=None)
        self.parameters.declare('coeffs_cd', default=None)
        self.parameters.declare('solve_option', default='direct')
        self.parameters.declare('TE_idx', default='last')
        self.parameters.declare('mesh_unit', default='m')
        self.parameters.declare('cl0', default=[0])

        self.parameters.declare('ML', default=False)
        self.parameters.declare('ref_area', default=None)
        self.parameters.declare('compressible', default=False)
        self.parameters.declare('frame', default='wing_fixed')
        self.parameters.declare('symmetry',default=False)

    def define(self):
        # add the mesh info
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        cl0 = self.parameters['cl0']

        free_stream_velocities = self.parameters['free_stream_velocities']

        eval_pts_option = self.parameters['eval_pts_option']

        eval_pts_location = self.parameters['eval_pts_location']
        eval_pts_shapes = self.parameters['eval_pts_shapes']
        sprs = self.parameters['sprs']

        coeffs_aoa = self.parameters['coeffs_aoa']
        coeffs_cd = self.parameters['coeffs_cd']
        mesh_unit = self.parameters['mesh_unit']

        compressible = self.parameters['compressible']

        num_nodes = surface_shapes[0][0]
        if self.parameters['AcStates'] == None:
            frame_vel_val = -free_stream_velocities

            frame_vel = self.create_input('frame_vel', val=frame_vel_val)

        self.add_module(
            VLMSystem(
                surface_names=surface_names,
                surface_shapes=surface_shapes,
                num_nodes=num_nodes,
                AcStates=self.parameters['AcStates'],
                solve_option=self.parameters['solve_option'],
                TE_idx=self.parameters['TE_idx'],
                mesh_unit=mesh_unit,
                eval_pts_option=eval_pts_option,
                eval_pts_location=eval_pts_location,
                compressible=compressible,
                frame=self.parameters['frame'],
                symmetry=self.parameters['symmetry'],
            ), 'VLM_system')
        if eval_pts_option=='auto':
            eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
            eval_pts_shapes = [(num_nodes, x[1] - 1, x[2] - 1, 3) for x in surface_shapes]
        else:
            eval_pts_names=self.parameters['eval_pts_names']

        # compute lift and drag
        sub = Outputs(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            eval_pts_names=eval_pts_names,
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_option=eval_pts_option,
            eval_pts_location=eval_pts_location,
            sprs=sprs,
            coeffs_aoa=coeffs_aoa,
            coeffs_cd=coeffs_cd,
            mesh_unit=mesh_unit,
            cl0=cl0,
            ML=self.parameters['ML'],
            ref_area=self.parameters['ref_area'],
            compressible=compressible,
            symmetry=self.parameters['symmetry'],
        )
        self.add(sub, name='VLM_outputs')


if __name__ == "__main__":

    pass