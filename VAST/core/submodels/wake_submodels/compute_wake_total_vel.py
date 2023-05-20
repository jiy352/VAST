# 1. compute wake kinematic vel
# -frame_vel + rot_vel
# TODO: figure out this rot_vel for the plunging wing case


# 2. compute wake induced vel
# BS (surface_wake_coords, (induced by) all bound and wake coords) 
# gamma_b and gamma_W (depending on the ordering in the last line)


# from csdl_om import Simulator
from csdl import Model
import csdl
from matplotlib.pyplot import clabel
import numpy as np
from numpy.core.fromnumeric import size

from scipy.sparse import csc_matrix


from VAST.core.submodels.wake_submodels.compute_wake_kinematic_vel_temp import ComputeWakeKinematicVel
from VAST.core.submodels.wake_submodels.eval_pts_velocities_mls import EvalPtsVel

class ComputeWakeTotalVel(Model):
    """
    Compute various geometric properties for VLM analysis.
    These are used primarily to help compute postprocessing quantities,
    such as wave CD, viscous CD, etc.
    Some of the quantities, like `normals`, are used to compute the RHS
    of the AIC linear system.
    A \gamma_b = b - M \gamma_w
    parameters
    ----------

    collocation_pts[num_vortex_panel_x*num_vortex_panel_y] : csdl array
        all the bd vertices collocation_pts     
    wake_pts[num_vortex_panel_x*num_vortex_panel_y] : csdl array
        all the wake panel collcation pts 
    wake_circulations[num_wake_panel] : csdl array
        a concatenate vector of the wake circulation strength
    Returns
    -------
    vel_col_w[num_evel_pts_x*num_vortex_panel_x* num_evel_pts_y*num_vortex_panel_y,3]
    csdl array
        the velocities computed using the aic_col_w from biot svart's law
        on bound vertices collcation pts induces by the wakes
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)


        self.parameters.declare('n_wake_pts_chord') # we need this to combine bd and wake 
        # self.parameters.declare('delta_t', default=100)

        # self.parameters.declare('coeffs_aoa', default=None)
        # self.parameters.declare('coeffs_cd', default=None)

    def define(self):
        n_wake_pts_chord = self.parameters['n_wake_pts_chord']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        n_wake_pts_chord = self.parameters['n_wake_pts_chord']

        submodel = ComputeWakeKinematicVel(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            n_wake_pts_chord=n_wake_pts_chord
        )
        self.add(submodel, name='ComputeWakeKinematicVel')

        wake_coords_names = [x + '_wake_coords' for x in surface_names]
        wake_vortex_pts_shapes = [tuple((item[0],n_wake_pts_chord, item[2], 3)) for item in surface_shapes]

        submodel = EvalPtsVel(
            eval_pts_names=wake_coords_names,
            eval_pts_shapes=wake_vortex_pts_shapes,
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            n_wake_pts_chord=n_wake_pts_chord,
            problem_type='prescribed_wake',
            # eps=1e-2,
            
        )
        self.add(submodel, name='EvalPtsVel')

        wake_kinematic_vel_names = [x + '_wake_kinematic_vel' for x in surface_names]
        wake_total_vel_names = [x + '_wake_total_vel' for x in surface_names]
        eval_induced_velocities_names = [x + '_wake_induced_vel' for x in surface_names]

        for i in range(len(surface_names)):
            wake_kinematic_vel = self.declare_variable(wake_kinematic_vel_names[i],shape=wake_vortex_pts_shapes[i])
            wake_induced_vel = self.declare_variable(eval_induced_velocities_names[i],shape=wake_vortex_pts_shapes[i])
            ''''TODO: fix this hardcoding'''
            wake_total_vel = wake_kinematic_vel + wake_induced_vel*0
            
            self.register_output(wake_total_vel_names[i],wake_total_vel)

