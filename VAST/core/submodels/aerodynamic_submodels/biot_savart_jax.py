import csdl
import jax.numpy as jnp
from jax import grad, jit,  jacfwd, jacrev
import numpy as onp

class BiotSavartCompJax(csdl.CustomExplicitOperation):
    """
    Compute all AIC matrices.

    parameters
    ----------
    <eval_pts_names>[num_nodes,nc, ns, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface that the 
        AIC matrix is computed on.
    <vortex_coords_names>vortex_coords[num_nodes,nc_v, ns_v, 3] : numpy array
        Array defining the nodal coordinates of background mesh that induces
        the AIC.

    Returns
    -------
    <AIC_names>[nc*ns*(nc_v-1)*(ns_v-1), nc*ns*(nc_v-1)*(ns_v-1), 3] : numpy array
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
        num_nodes = eval_pt_shapes[0][0]
        
        vc = self.parameters['vc']
        eps = self.parameters['eps']
        circulation_names = self.parameters['circulation_names']

        for i in range(len(eval_pt_names)):
            # input names and shapes
            eval_pt_name = eval_pt_names[i]
            vortex_coords_name = vortex_coords_names[i]
            eval_pt_shape = eval_pt_shapes[i]
            vortex_coords_shape = vortex_coords_shapes[i]
            # output names and shapes
            out_name = output_names[i]
            out_shape = (num_nodes, eval_pt_shape[1]*eval_pt_shape[2]* (vortex_coords_shape[1]-1)*(vortex_coords_shape[2]-1), 3)
            # print('out_shape---------',out_shape)

            self.add_output(out_name, shape=out_shape)
            self.add_input(eval_pt_name, shape=eval_pt_shape)
            self.add_input(vortex_coords_name, shape=vortex_coords_shape)

            self.declare_derivatives(out_name, eval_pt_name)
            self.declare_derivatives(out_name, vortex_coords_name)

        self.AIC_jax = jit(self.__compute_AIC)
        self.grad_AIC_jax = jit(jacfwd(self.__compute_AIC,argnums=[0,1]))
        self.func = jit(self.__compute_expand_vecs)
        self.induced_vel_line = jit(self.__induced_vel_line)


        # self.AIC_jax = self.__compute_AIC
        # self.grad_AIC_jax = jacfwd(self.__compute_AIC,argnums=[0,1])
        # self.func = self.__compute_expand_vecs
        # self.induced_vel_line = self.__induced_vel_line

            # self.AIC_jax = self.__compute_AIC
            # self.func = self.__compute_expand_vecs
            # self.induced_vel_line = self.__induced_vel_line
    def compute(self, inputs, outputs):
        eval_pt_names = self.parameters['eval_pt_names']
        eval_pt_shapes = self.parameters['eval_pt_shapes']
        vortex_coords_names = self.parameters['vortex_coords_names']
        vortex_coords_shapes = self.parameters['vortex_coords_shapes']
        output_names = self.parameters['output_names']
        num_nodes = eval_pt_shapes[0][0]
        
        vc = self.parameters['vc']
        eps = self.parameters['eps']
        circulation_names = self.parameters['circulation_names']
        for i in range(len(eval_pt_names)):

            # input names and shapes
            eval_pt_name = eval_pt_names[i]
            vortex_coords_name = vortex_coords_names[i]
            eval_pt_shape = eval_pt_shapes[i]
            vortex_coords_shape = vortex_coords_shapes[i]
            # output names and shapes
            out_name = output_names[i]
            out_shape = (num_nodes, eval_pt_shape[1]*eval_pt_shape[2]* (vortex_coords_shape[1]-1)*(vortex_coords_shape[2]-1), 3)
            # convert to jax array
            eval_pts = jnp.array(inputs[eval_pt_name])
            vortex_coords =  jnp.array(inputs[vortex_coords_name])
            # compute AIC is a jax function
            # convert back to numpy array
            # print('------------------',self.AIC_jax(eval_pts, vortex_coords))
            outputs[out_name] = onp.array(self.AIC_jax(eval_pts, vortex_coords)) 
            # print('AIC_jax shape---------',AIC_jax.shape)

    def compute_derivatives(self, inputs, derivatives):
        eval_pt_names = self.parameters['eval_pt_names']
        eval_pt_shapes = self.parameters['eval_pt_shapes']
        vortex_coords_names = self.parameters['vortex_coords_names']
        vortex_coords_shapes = self.parameters['vortex_coords_shapes']
        output_names = self.parameters['output_names']
        num_nodes = eval_pt_shapes[0][0]
        
        vc = self.parameters['vc']
        eps = self.parameters['eps']
        circulation_names = self.parameters['circulation_names']
        for i in range(len(eval_pt_names)):

            # input names and shapes
            eval_pt_name = eval_pt_names[i]
            vortex_coords_name = vortex_coords_names[i]
            eval_pt_shape = eval_pt_shapes[i]
            vortex_coords_shape = vortex_coords_shapes[i]
            # output names and shapes
            out_name = output_names[i]
            out_shape = (num_nodes, eval_pt_shape[1]*eval_pt_shape[2]* (vortex_coords_shape[1]-1)*(vortex_coords_shape[2]-1), 3)
            # convert to jax array
            eval_pts = jnp.array(inputs[eval_pt_name])
            vortex_coords =  jnp.array(inputs[vortex_coords_name])
            # compute AIC is a jax function

            grad_fcn = self.grad_AIC_jax
            grad_fcn_val = grad_fcn(eval_pts, vortex_coords)
            # print(grad_fcn_val[0].shape)
            # print(grad_fcn_val[1].shape)
            # print(onp.array(grad_fcn_val[0]).shape)
            # print('derivative shape',derivatives[out_name, eval_pt_name].shape)


            derivatives[out_name, eval_pt_name] = onp.array(grad_fcn_val[0]).reshape(derivatives[out_name, eval_pt_name].shape)
            derivatives[out_name, vortex_coords_name] = onp.array(grad_fcn_val[1]).reshape(derivatives[out_name, vortex_coords_name].shape)
            del grad_fcn_val
    def __compute_AIC(self, eval_pts, vortex_coords):

        """
        Compute AIC matrix for a given lifting surface and background mesh.
        THIS IS A JAX FUNCTION

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
        """

        # define panel points
        #                  C -----> D
        # ---v_inf-(x)-->  ^        |
        #                  |        v
        #                  B <----- A
        vortex_coords_shape = vortex_coords.shape

        A = vortex_coords[:,1:, :vortex_coords_shape[2] - 1, :]
        B = vortex_coords[:,:vortex_coords_shape[1] -
                        1, :vortex_coords_shape[2] - 1, :]
        C = vortex_coords[:,:vortex_coords_shape[1] - 1, 1:, :]
        D = vortex_coords[:,1:, 1:, :]

        # openaerostruct
        # C = vortex_coords[:, 1:, :vortex_coords_shape[2] - 1, :]
        # B = vortex_coords[:, :vortex_coords_shape[1] -
        #                   1, :vortex_coords_shape[2] - 1, :]
        # A = vortex_coords[:, :vortex_coords_shape[1] - 1, 1:, :]
        # D = vortex_coords[:, 1:, 1:, :]



        r_A, r_A_norm = self.func(eval_pts, A)
        r_B, r_B_norm = self.func(eval_pts, B)
        r_C, r_C_norm = self.func(eval_pts, C)
        r_D, r_D_norm = self.func(eval_pts, D)

        v_ab = self.induced_vel_line(r_A, r_B, r_A_norm, r_B_norm)
        v_bc = self.induced_vel_line(r_B, r_C, r_B_norm, r_C_norm)
        v_cd = self.induced_vel_line(r_C, r_D, r_C_norm, r_D_norm)
        v_da = self.induced_vel_line(r_D, r_A, r_D_norm, r_A_norm)

        # AIC = jit(v_ab + v_bc + v_cd + v_da)
        AIC = (v_ab + v_bc + v_cd + v_da)
        # print('AIC shape---------',AIC.shape)

        return AIC

    def __compute_expand_vecs(self, eval_pts, p_1):
        # this is a jax function computing r_1, and r_1_norm

        num_nodes = eval_pts.shape[0]
        # 1 -> 2 eval_pts(num_pts_x,num_pts_y, 3)
        # v_induced_line shape=(num_panel_x*num_panel_y, num_panel_x, num_panel_y, 3)
        num_repeat_eval = p_1.shape[1] * p_1.shape[2]
        num_repeat_p = eval_pts.shape[1] * eval_pts.shape[2]

        eval_pts_reshaped = eval_pts.reshape((num_nodes, -1, 3))
        eval_pts_expand = jnp.einsum('ijk,l->ijlk',eval_pts_reshaped,jnp.ones(num_repeat_eval)).reshape((num_nodes, -1, 3))

        p_1_reshaped = p_1.reshape((num_nodes, -1, 3))
        p_1_expand = jnp.einsum('ijk,l->iljk',p_1_reshaped,jnp.ones(num_repeat_p)).reshape((num_nodes, -1, 3))

        r1 = eval_pts_expand - p_1_expand
        r1_norm = jnp.linalg.norm(r1, axis=-1)
        return r1, r1_norm

    def __induced_vel_line(self, r_1, r_2, r_1_norm, r_2_norm):
        # this is a jax function computing r_1, and r_1_norm
        vc = self.parameters['vc']

        num_nodes = r_1.shape[0]

        # 1 -> 2 eval_pts(num_pts_x,num_pts_y, 3)

        r0 = r_1 - r_2
        # the denominator of the induced velocity equation
        # shape = (num_nodes,num_panel_x*num_panel_y, num_panel_x* num_panel_y, 3)
        one_over_den = 1 / (jnp.pi * 4) * jnp.cross(r_1, r_2, axis=-1)

        if vc == False:
            # the numerator of the induced velocity equation
            # shape = (num_nodes, num_panel_x*num_panel_y, num_panel_x* num_panel_y)

            num = (1/(r_1_norm * r_2_norm + jnp.einsum('ijk,ijk->ij',r_1,r_2))) * (1/r_1_norm + 1/r_2_norm)
            num_expand = jnp.einsum('ij,l->ijl', num, jnp.ones(3))
            v_induced_line = num_expand * one_over_den
            # print('r_1 shape', r_1.shape)
            # print('r_2 shape', r_2.shape)
            # print('v_induced_line', v_induced_line.shape)
            # print('num_expand', num_expand.shape)
            # print('one_over_den', one_over_den.shape)
        return v_induced_line


class BiotSavartComp(csdl.Model):
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
        for i in range(len(self.parameters['eval_pt_names'])):
            eval_pt_name = self.parameters['eval_pt_names'][i]
            eval_pt_shape = self.parameters['eval_pt_shapes'][i]
            vortex_coords_name = self.parameters['vortex_coords_names'][i]
            vortex_coords_shape = self.parameters['vortex_coords_shapes'][i]
            output_name = self.parameters['output_names'][i]
            eval_pt = self.declare_variable(eval_pt_name, shape=eval_pt_shape)
            vortex_coords = self.declare_variable(vortex_coords_name, shape=vortex_coords_shape)
            
        aic = csdl.custom(eval_pt,vortex_coords,
                    op=BiotSavartCompJax(eval_pt_names=self.parameters['eval_pt_names'],
                            vortex_coords_names=self.parameters['vortex_coords_names'],
                            eval_pt_shapes=self.parameters['eval_pt_shapes'],
                            vortex_coords_shapes=self.parameters['vortex_coords_shapes'],
                            output_names=self.parameters['output_names']))
        self.register_output(self.parameters['output_names'][0], aic)

if __name__ == "__main__":
    import time
    import timeit
    ts = time.time()
    from python_csdl_backend import Simulator
    def generate_simple_mesh(nx, ny):
        mesh = onp.zeros((nx, ny, 3))
        mesh[:, :, 0] = onp.outer(onp.arange(nx), onp.ones(ny))
        mesh[:, :, 1] = onp.outer(onp.arange(ny), onp.ones(nx)).T
        mesh[:, :, 2] = 0.
        return mesh

    n_wake_pts_chord = 2
    nc = 10
    ns = 20

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

    #####################
    # finshed adding model
    ####################
    sim = Simulator(model_1)
    print('time', time.time() - ts)
    sim.run()
    print('time', time.time() - ts)
    sim.compute_totals(of='aic',wrt='vtx_pts')
    print('time', time.time() - ts)
    sim.compute_totals(of='aic',wrt='vtx_pts')
    print('time', time.time() - ts)
    sim.compute_totals(of='aic',wrt='vtx_pts')
    print('time', time.time() - ts)
    sim.compute_totals(of='aic',wrt='vtx_pts')
    print('time', time.time() - ts)
    sim.compute_totals(of='aic',wrt='vtx_pts')
    print('time', time.time() - ts)
    exit()
    # print(timeit.timeit(lambda: sim.run(), number=5), 'sec')
    
    
    # print(timeit.timeit(lambda: sim.compute_totals(of='aic',wrt='vtx_pts'), number=5), 'sec')

    # import cProfile
    # profiler = cProfile.Profile()

    # profiler.enable()
    # rep = csdl.GraphRepresentation(model_1)
    # profiler.disable()
    # profiler.dump_stats('output_1')

    # print(sim['vor'])
    # print(sim[output_names[0]])
    # sim.visualize_implementation()
    # import resource
    # before = (resource.RUSAGE_SELF).ru_maxrss
    # sim.run()
    
    # after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # print('memory usage', after-before)
    
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

sim['aic'].reshape(8,8,3)[:,:,2].reshape(2,4,2,4)
computed_aic = sim['aic'].reshape(8,8,3)[:,:,2].reshape(2,4,2,4)[:,:2,:,:]
generated_other_aic_temp = np.flip(computed_aic, axis=(1,3))
generated_other_aic = np.flip(generated_other_aic_temp, axis=1)

aic_whole = np.concatenate((computed_aic, generated_other_aic_temp), axis=1)