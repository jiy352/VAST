from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from VAST.utils.einsum_kij_kij_ki import EinsumKijKijKi
from VAST.utils.expand_ijk_ijlk import ExpandIjkIjlk
from VAST.utils.expand_ijk_iljk import ExpandIjkIljk

class BiotSavartComp(Model):
    """
    Compute AIC.

    parameters
    ----------
    eval_pts[num_nodes,nc, ns, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface that the 
        AIC matrix is computed on.
    vortex_coords[num_nodes,nc_v, ns_v, 3] : numpy array
        Array defining the nodal coordinates of background mesh that induces
        the AIC.

    Returns
    -------
    AIC[nc*ns*(nc_v-1)*(ns_v-1), nc*ns*(nc_v-1)*(ns_v-1), 3] : numpy array
        Aerodynamic influence coeffients (can be interprete as induced
        velocities given circulations=1)
    2023-06-13: 
        need to avoid repeated computation of induced velocities
        need to check the node order and the bound vector against OAS
    """
    def initialize(self):
        # evaluation points names and shapes
        self.parameters.declare('eval_pt_names', types=list)
        self.parameters.declare('eval_pt_shapes', types=list)
        # induced background mesh names and shapes
        self.parameters.declare('vortex_coords_names', types=list)
        self.parameters.declare('vortex_coords_shapes', types=list)
        # output aic names
        self.parameters.declare('output_names', types=list)
        # whether to enable the fixed vortex core model
        self.parameters.declare('vc', default=True)
        self.parameters.declare('eps', default=5e-6)

        self.parameters.declare('circulation_names', default=None)

    def define(self):
        eval_pt_names = self.parameters['eval_pt_names']
        eval_pt_shapes = self.parameters['eval_pt_shapes']
        vortex_coords_names = self.parameters['vortex_coords_names']
        vortex_coords_shapes = self.parameters['vortex_coords_shapes']
        output_names = self.parameters['output_names']
        vc = self.parameters['vc']
        eps = self.parameters['eps']
        circulation_names = self.parameters['circulation_names']

        for i in range(len(eval_pt_names)):
            # input_names
            eval_pt_name = eval_pt_names[i]
            vortex_coords_name = vortex_coords_names[i]
            # output_name
            output_name = output_names[i]
            # input_shapes
            eval_pt_shape = eval_pt_shapes[i]
            vortex_coords_shape = vortex_coords_shapes[i]
            # declare_inputs
            eval_pts = self.declare_variable(eval_pt_name, shape=eval_pt_shape)
            vortex_coords = self.declare_variable(vortex_coords_name, shape=vortex_coords_shape)

            # define panel points
            #                  C -----> D
            # ---v_inf-(x)-->  ^        |
            #                  |        v
            #                  B <----- A
            A = vortex_coords[:,1:, :vortex_coords_shape[2] - 1, :]
            B = vortex_coords[:,:vortex_coords_shape[1] -
                              1, :vortex_coords_shape[2] - 1, :]
            C = vortex_coords[:,:vortex_coords_shape[1] - 1, 1:, :]
            D = vortex_coords[:,1:, 1:, :]

            self.r_A = self.__compute_expand_vecs(eval_pts, A, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'A')
            self.r_B = self.__compute_expand_vecs(eval_pts, B, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'B')
            self.r_C = self.__compute_expand_vecs(eval_pts, C, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'C')
            self.r_D = self.__compute_expand_vecs(eval_pts, D, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'D')

            # openaerostruct
            # C = vortex_coords[:, 1:, :vortex_coords_shape[2] - 1, :]
            # B = vortex_coords[:, :vortex_coords_shape[1] -
            #                   1, :vortex_coords_shape[2] - 1, :]
            # A = vortex_coords[:, :vortex_coords_shape[1] - 1, 1:, :]
            # D = vortex_coords[:, 1:, 1:, :]

            v_ab = self._induced_vel_line(eval_pts, self.r_A, self.r_B, output_name,'AB')
            v_bc = self._induced_vel_line(eval_pts, self.r_B, self.r_C, output_name,'BC')
            v_cd = self._induced_vel_line(eval_pts, self.r_C, self.r_D, output_name,'CD')
            v_da = self._induced_vel_line(eval_pts, self.r_D, self.r_A, output_name,'DA')

            AIC = v_ab + v_bc + v_cd + v_da
            self.register_output(output_name, AIC)

    def _compute_expand_vecs(self, eval_pts, p_1, vortex_coords_shape, eval_pt_name, vortex_coords_name, output_name, point_name):

        vc = self.parameters['vc']
        num_nodes = eval_pts.shape[0]
        name = eval_pt_name + vortex_coords_name + output_name + point_name

        # 1 -> 2 eval_pts(num_pts_x,num_pts_y, 3)
        # v_induced_line shape=(num_panel_x*num_panel_y, num_panel_x, num_panel_y, 3)
        num_repeat_eval = p_1.shape[1] * p_1.shape[2]
        num_repeat_p = eval_pts.shape[1] * eval_pts.shape[2]

        eval_pts_expand = csdl.reshape(
            csdl.expand(
                csdl.reshape(eval_pts,
                             new_shape=(num_nodes, (eval_pts.shape[1] *
                                                    eval_pts.shape[2]), 3)),
                (num_nodes, eval_pts.shape[1] * eval_pts.shape[2],
                 num_repeat_eval, 3),
                'lij->likj',
            ),
            new_shape=(num_nodes,
                       eval_pts.shape[1] * eval_pts.shape[2] * num_repeat_eval,
                       3))

        p_1_expand = csdl.reshape(\
            csdl.expand(
                csdl.reshape(p_1,
                         new_shape=(num_nodes, (p_1.shape[1] * p_1.shape[2]),
                                    3)),
            (num_nodes, num_repeat_p, p_1.shape[1] * p_1.shape[2], 3),
            'lij->lkij'),
                        new_shape=(num_nodes,
                                    p_1.shape[1] *p_1.shape[2] * num_repeat_p,
                                    3))

        r1 = eval_pts_expand - p_1_expand
        return r1


    def _induced_vel_line(self, r_1, r_2):
        name = eval_pt_name + vortex_coords_name + output_name + point_name

        # book pg269
        # step 1 calculate r1_x_r2,r1_x_r2_norm_sq
        r1_x_r2 = csdl.cross(r1, r2, axis=2)

        r1_x_r2_norm_sq = csdl.expand(csdl.sum(r1_x_r2**2, axes=(2, )),
                                      shape=r1_x_r2.shape,
                                      indices=('ki->kij'))

        # step 2 r1_norm, r2_norm
        r1_norm = csdl.expand(csdl.sum(r1**2 + self.parameters['eps'],
                                       axes=(2, ))**0.5,
                              shape=r1_x_r2.shape,
                              indices=('ki->kij'))

        r2_norm = csdl.expand(csdl.sum(r2**2 + self.parameters['eps'],
                                       axes=(2, ))**0.5,
                              shape=r1_x_r2.shape,
                              indices=('ki->kij'))

        array1 = 1 / (np.pi * 4) * r1_x_r2

        if vc == False:
            in_3 = (r1 * r2_norm - r2 * r1_norm) / (r1_norm * r2_norm)
            in_4 = r1-r2

            array2 = csdl.sum((in_3 * in_4),axes=(2,))

            v_induced_line = array1 * csdl.expand(
                array2, array1.shape, 'ki->kij') / (r1_x_r2_norm_sq)
            del in_3
            del in_4
        return v_induced_line


if __name__ == "__main__":

    def generate_simple_mesh(nx, ny):
        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
        mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
        mesh[:, :, 2] = 0.
        return mesh

    n_wake_pts_chord = 2
    nc = 2
    ns = 3
    nc_v = 3
    ns_v = 4
    eval_pt_names = ['col']
    vortex_coords_names = ['vor']
    # eval_pt_shapes = [(nx, ny, 3)]
    # vortex_coords_shapes = [(nx, ny, 3)]

    eval_pt_shapes = [(nc, ns, 3), (nc, ns, 3)]
    vortex_coords_shapes = [(nc, ns, 3)]

    output_names = ['aic']

    model_1 = Model()

    circulations_val = np.ones((nx - 1, ny - 1)) * 0.5

    vor_val = generate_simple_mesh(nx, ny)
    col_val = 0.25 * (vor_val[:-1, :-1, :] + vor_val[:-1, 1:, :] +
                      vor_val[1:, :-1, :] + vor_val[1:, 1:, :])
    # col_val = generate_simple_mesh(nx, ny)

    vor = model_1.create_input('vor', val=vor_val)
    col = model_1.create_input('col', val=col_val)
    circulations = model_1.create_input('circulations',
                                        val=circulations_val.reshape(
                                            1, nx - 1, ny - 1))

    model_1.add(BiotSavartComp(eval_pt_names=eval_pt_names,
                               vortex_coords_names=vortex_coords_names,
                               eval_pt_shapes=eval_pt_shapes,
                               vortex_coords_shapes=vortex_coords_shapes,
                               output_names=output_names,
                               vc=False,
                               name='BiotSvart_group'))
    sim = Simulator(model_1)

    print(sim['vor'])
    print(sim[output_names[0]])
    # sim.visualize_implementation()
    sim.run()

