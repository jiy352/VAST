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
        self.parameters.declare('vc', default=False)
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

            self.r_A, self.r_A_norm = self.__compute_expand_vecs(eval_pts, A, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'A')
            self.r_B, self.r_B_norm = self.__compute_expand_vecs(eval_pts, B, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'B')
            self.r_C, self.r_C_norm = self.__compute_expand_vecs(eval_pts, C, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'C')
            self.r_D, self.r_D_norm = self.__compute_expand_vecs(eval_pts, D, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'D')

            # openaerostruct
            # C = vortex_coords[:, 1:, :vortex_coords_shape[2] - 1, :]
            # B = vortex_coords[:, :vortex_coords_shape[1] -
            #                   1, :vortex_coords_shape[2] - 1, :]
            # A = vortex_coords[:, :vortex_coords_shape[1] - 1, 1:, :]
            # D = vortex_coords[:, 1:, 1:, :]

            # v_ab = self._induced_vel_line(self.r_A, self.r_B, self.r_A_norm, self.r_B_norm, output_name,'AB')
            # v_bc = self._induced_vel_line(self.r_B, self.r_C, self.r_B_norm, self.r_C_norm, output_name,'BC')
            # v_cd = self._induced_vel_line(self.r_C, self.r_D, self.r_C_norm, self.r_D_norm, output_name,'CD')
            # v_da = self._induced_vel_line(self.r_D, self.r_A, self.r_D_norm, self.r_A_norm, output_name,'DA')

            v_ab = self._induced_vel_line(self.r_A, self.r_B, self.r_A_norm, self.r_B_norm)
            v_bc = self._induced_vel_line(self.r_B, self.r_C, self.r_B_norm, self.r_C_norm)
            v_cd = self._induced_vel_line(self.r_C, self.r_D, self.r_C_norm, self.r_D_norm)
            v_da = self._induced_vel_line(self.r_D, self.r_A, self.r_D_norm, self.r_A_norm)

            AIC = v_ab + v_bc + v_cd + v_da

            self.register_output(output_name, AIC)

    def __compute_expand_vecs(self, eval_pts, p_1, vortex_coords_shape, eval_pt_name, vortex_coords_name, output_name, point_name):

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
        r1_norm = csdl.sum(r1**2, axes=(2,))**0.5
        return r1, r1_norm


    def _induced_vel_line(self, r_1, r_2, r_1_norm, r_2_norm):

        vc = False

        num_nodes = r_1.shape[0]

        # 1 -> 2 eval_pts(num_pts_x,num_pts_y, 3)

        r0 = r_1 - r_2
        # the denominator of the induced velocity equation
        # shape = (num_nodes,num_panel_x*num_panel_y, num_panel_x* num_panel_y, 3)
        one_over_den = 1 / (np.pi * 4) * csdl.cross(r_1, r_2, axis=2)

        if vc == False:
            dor_r1_r2 = csdl.sum(r_1*r_2,axes=(2,))
            num = (1/(r_1_norm * r_2_norm + dor_r1_r2+1e-3)) * (1/r_1_norm + 1/r_2_norm+1e-3)
            # print('the shape of num is', num.shape)
            num_expand = csdl.expand(num, (num_nodes, num.shape[1], 3), 'ij->ijl')
            # num_expand = jnp.einsum('ij,l->ijl', num, jnp.ones(3))
            v_induced_line = num_expand * one_over_den
        else:
            raise NotImplementedError
        return v_induced_line


if __name__ == "__main__":
    import timeit
    from python_csdl_backend import Simulator
    import numpy as onp
    def generate_simple_mesh(nx, ny):
        mesh = onp.zeros((nx, ny, 3))
        mesh[:, :, 0] = onp.outer(onp.arange(nx), onp.ones(ny))
        mesh[:, :, 1] = onp.outer(onp.arange(ny), onp.ones(nx)).T
        mesh[:, :, 2] = 0.
        return mesh

    n_wake_pts_chord = 2
    nc = 10
    ns = 10

    eval_pt_names = ['coll_pts']
    vortex_coords_names = ['vtx_pts']
    # eval_pt_shapes = [(nx, ny, 3)]
    # vortex_coords_shapes = [(nx, ny, 3)]

    eval_pt_shapes = [(1, nc-1, ns-1, 3)]
    vortex_coords_shapes = [(1, nc, ns, 3)]

    output_names = ['aic']

    model_1 = csdl.Model()


    vor_val = generate_simple_mesh(nc, ns).reshape(1, nc, ns, 3)
    col_val = 0.25 * (vor_val[:,:-1, :-1, :] + vor_val[:,:-1, 1:, :] +
                      vor_val[:,1:, :-1, :] + vor_val[:,1:, 1:, :])
    # col_val = generate_simple_mesh(nx, ny)

    vor = model_1.create_input('vtx_pts', val=vor_val)
    col = model_1.create_input('coll_pts', val=col_val)

    # test if multiple ops work
    submodel=BiotSavartComp(eval_pt_names=eval_pt_names,
                               vortex_coords_names=vortex_coords_names,
                               eval_pt_shapes=eval_pt_shapes,
                               vortex_coords_shapes=vortex_coords_shapes,
                               output_names=output_names)

    model_1.add(submodel,'submodel')


    sim = Simulator(model_1)
    print(timeit.timeit(lambda: sim.run(), number=100), 'sec')
    print(timeit.timeit(lambda: sim.compute_totals(of='aic',wrt='vtx_pts'), number=10), 'sec')

    import cProfile
    profiler = cProfile.Profile()

    profiler.enable()
    rep = csdl.GraphRepresentation(model_1)
    profiler.disable()
    profiler.dump_stats('output_1')
    
    # print(sim['vor'])
    # print(sim[output_names[0]])
    # sim.visualize_implementation()
    import resource
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sim.run()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('memory usage', after-before)
    exit()
    
    a= sim.check_partials(compact_print=True)
    anal = a['aic', 'vtx_pts']['analytical_jac']
    fd = a['aic', 'vtx_pts']['fd_jac']
    import pandas as pd
    df_ana = pd.DataFrame(anal)
    df_fd = pd.DataFrame(fd)
    print(df_ana)
    print(df_fd)
    
    anal = a['aic', 'coll_pts']['analytical_jac']
    fd = a['aic', 'coll_pts']['fd_jac']
    df_ana = pd.DataFrame(anal)
    df_fd = pd.DataFrame(fd)
    print(df_ana)
    print(df_fd)


    print('aic is', sim['aic'])
    print('collocation points are ', sim['coll_pts'].shape, '\n',sim['coll_pts'])
    print('vortex points are ', sim['vtx_pts'].shape ,'\n',sim['vtx_pts'])
