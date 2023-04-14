# from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size


class KinematicVelocityComp(Model):
    """
    Compute various geometric properties for VLM analysis.
    These are used primarily to help compute postprocessing quantities,
    such as wave CD, viscous CD, etc.
    Some of the quantities, like `normals`, are used to compute the RHS
    of the AIC linear system.

    parameters
    ----------
    
    frame_vel
    p
    q
    r

    Returns
    -------
    ang_vel[num_nodes,3]: p, q, r
    kinematic_vel[n_wake_pts_chord, (num_evel_pts_x* num_evel_pts_y), 3] : csdl array
        undisturbed fluid velocity
    """
    def initialize(self):

        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        # get num_nodes from surface shape
        num_nodes = surface_shapes[0][0]

        # add_input name and shapes
        # compute rotatonal_vel_shapes from surface shape

        # print('line 49 bd_coll_pts_shapes', bd_coll_pts_shapes)

        # add_output name and shapes
        kinematic_vel_names = [
            x + '_kinematic_vel' for x in self.parameters['surface_names']
        ]

        frame_vel = self.declare_variable('frame_vel', shape=(num_nodes, 3))

        p = self.declare_variable('p', shape=(num_nodes, 1))
        q = self.declare_variable('q', shape=(num_nodes, 1))
        r = self.declare_variable('r', shape=(num_nodes, 1))

        ang_vel = self.create_output('ang_vel', shape=(num_nodes, 3))
        ang_vel[:, 0] = p
        ang_vel[:, 1] = q
        ang_vel[:, 2] = r

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            num_pts_chord = surface_shapes[i][1]
            num_pts_span = surface_shapes[i][2]
            kinematic_vel_name = kinematic_vel_names[i]
            out_shape = (num_nodes, (num_pts_chord - 1) * (num_pts_span - 1),
                         3)

            coll_pts_coords_name = surface_name + '_coll_pts_coords'

            coll_pts = self.declare_variable(coll_pts_coords_name,
                                             shape=(num_nodes,
                                                    num_pts_chord - 1,
                                                    num_pts_span - 1, 3))

            evaluation_pt = self.declare_variable('evaluation_pt',
                                                  val=np.zeros(3, ))
            evaluation_pt_exp = csdl.expand(
                evaluation_pt,
                (coll_pts.shape),
                'i->ljki',
            )

            r_vec = coll_pts - evaluation_pt_exp
            ang_vel_exp = csdl.expand(
                ang_vel, (num_nodes, num_pts_chord - 1, num_pts_span - 1, 3),
                indices='il->ijkl')
            rot_vel = csdl.reshape(csdl.cross(ang_vel_exp, r_vec, axis=3),
                                   out_shape)
            frame_vel_expand = csdl.expand(frame_vel,
                                           out_shape,
                                           indices='ij->ikj')
            # print('rot_vel shape', rot_vel.shape)
            # print('frame_vel_expand shape', frame_vel_expand.shape)

            kinematic_vel = -(rot_vel + frame_vel_expand)
            self.register_output(kinematic_vel_name, kinematic_vel)


