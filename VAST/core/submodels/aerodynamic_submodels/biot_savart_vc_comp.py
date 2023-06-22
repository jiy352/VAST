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

            symmetry = True

            if symmetry == False:
                self.r_A, self.r_A_norm = self.__compute_expand_vecs(eval_pts, A, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'A')
                self.r_B, self.r_B_norm = self.__compute_expand_vecs(eval_pts, B, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'B')
                self.r_C, self.r_C_norm = self.__compute_expand_vecs(eval_pts, C, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'C')
                self.r_D, self.r_D_norm = self.__compute_expand_vecs(eval_pts, D, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'D')
                v_ab = self._induced_vel_line(self.r_A, self.r_B, self.r_A_norm, self.r_B_norm)
                v_bc = self._induced_vel_line(self.r_B, self.r_C, self.r_B_norm, self.r_C_norm)
                v_cd = self._induced_vel_line(self.r_C, self.r_D, self.r_C_norm, self.r_D_norm)
                v_da = self._induced_vel_line(self.r_D, self.r_A, self.r_D_norm, self.r_A_norm)

                AIC = v_ab + v_bc + v_cd + v_da           
            
            else:
                nx = eval_pt_shape[1]
                ny = eval_pt_shape[2]
                self.r_A, self.r_A_norm = self.__compute_expand_vecs(eval_pts[:,:,:int(ny/2),:], A, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'A')
                self.r_B, self.r_B_norm = self.__compute_expand_vecs(eval_pts[:,:,:int(ny/2),:], B, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'B')
                self.r_C, self.r_C_norm = self.__compute_expand_vecs(eval_pts[:,:,:int(ny/2),:], C, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'C')
                self.r_D, self.r_D_norm = self.__compute_expand_vecs(eval_pts[:,:,:int(ny/2),:], D, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'D')
                
                v_ab = self._induced_vel_line(self.r_A, self.r_B, self.r_A_norm, self.r_B_norm)
                v_bc = self._induced_vel_line(self.r_B, self.r_C, self.r_B_norm, self.r_C_norm)
                v_cd = self._induced_vel_line(self.r_C, self.r_D, self.r_C_norm, self.r_D_norm)
                v_da = self._induced_vel_line(self.r_D, self.r_A, self.r_D_norm, self.r_A_norm)

                AIC_half = v_ab + v_bc + v_cd + v_da      
                # self.register_output('aic_half', AIC_half)             
                # model_2 = csdl.Model()
                AIC = csdl.custom(AIC_half, op = SymmetryJax(in_name=AIC_half.name, eval_pt_shape=eval_pt_shape, vortex_coords_shape=vortex_coords_shape, out_name=output_name))
                

            self.register_output(output_name, AIC)

                # self.print_var( AIC)
                # self.add(model_2, name='aic_symmetry')
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


class SymmetryJax(csdl.CustomExplicitOperation):
    """
    Compute the whole AIC matrix given half of it

    parameters
    ----------
    <aic_half_names>[nc*ns*(nc_v-1)*(ns_v-1)* nc*ns*(nc_v-1)*(ns_v-1)/2, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface that the 
        AIC matrix is computed on.

    Returns
    -------
    <aic_names>[nc*ns*(nc_v-1)*(ns_v-1), nc*ns*(nc_v-1)*(ns_v-1), 3] : numpy array
        Aerodynamic influence coeffients (can be interprete as induced
        velocities given circulations=1)
    """    
    def initialize(self):
        self.parameters.declare('in_name', types=str)
        self.parameters.declare('eval_pt_shape', types=tuple)
        self.parameters.declare('vortex_coords_shape', types=tuple)
        self.parameters.declare('out_name', types=str)
    def define(self):
        eval_pt_shape =self.eval_pt_shape= self.parameters['eval_pt_shape']
        vortex_coords_shape =self.vortex_coords_shape =  self.parameters['vortex_coords_shape']
        shape = eval_pt_shape[1]*eval_pt_shape[2]*(vortex_coords_shape[1]-1)*(vortex_coords_shape[2]-1)
        num_nodes = eval_pt_shape[0]
        self.add_input(self.parameters['in_name'],shape=(num_nodes,int(shape/2),3))
        self.add_output(self.parameters['out_name'],shape=(num_nodes,int(shape),3))

        self.full_aic_func = self.__get_full_aic_jax
        row_indices  = np.arange(int(num_nodes*shape*3))
        col_ind_temp = np.arange(int(num_nodes*shape/2*3)).reshape(num_nodes,eval_pt_shape[1],int(eval_pt_shape[2]/2),int(vortex_coords_shape[1]-1),int(vortex_coords_shape[2]-1),3)
        
        col_ind_flip = np.flip(col_ind_temp,axis=(2,4))
        col_indices = np.concatenate((col_ind_temp,col_ind_flip),axis=2).flatten()
        # col_indices = np.arange(int(num_nodes*shape/2*3))
        self.declare_derivatives(self.parameters['out_name'], self.parameters['in_name'],rows=row_indices,cols=col_indices,val=np.ones(row_indices.size))

    def compute(self, inputs, outputs):
        # print('in_name is', self.parameters['in_name'])
        # print('out_name is', self.parameters['out_name'])
        # print(onp.array(self.full_aic_func(inputs[self.parameters['in_name']])))
        outputs[self.parameters['out_name']] = self.full_aic_func(inputs[self.parameters['in_name']]).reshape(1,-1,3)
    # def compute_derivatives(self, inputs, derivatives):
    #     pass
    def __get_full_aic_jax(self,half_aic):
        nx_panel = self.eval_pt_shape[1]
        ny_panel = self.eval_pt_shape[2] 
        nx_panel_ind = self.vortex_coords_shape[1] - 1
        ny_panel_ind = self.vortex_coords_shape[2] - 1 
        num_nodes = self.eval_pt_shape[0]
        half_reshaped = half_aic.reshape(num_nodes,nx_panel,int(ny_panel/2),nx_panel_ind, ny_panel_ind, 3)
        # half_aic.reshape(nx_panel,int(ny_panel/2),nx_panel_ind, ny_panel_ind, 3)
        other_half_aic = np.flip(half_reshaped, (2,4))
        full_aic = np.concatenate((half_reshaped, other_half_aic), axis=2).reshape(num_nodes,-1,3)
        return full_aic


# class SymmetryJax(csdl.CustomExplicitOperation):
#     """
#     Compute the whole AIC matrix given half of it

#     parameters
#     ----------
#     <aic_half_names>[nc*ns*(nc_v-1)*(ns_v-1)* nc*ns*(nc_v-1)*(ns_v-1)/2, 3] : numpy array
#         Array defining the nodal coordinates of the lifting surface that the 
#         AIC matrix is computed on.

#     Returns
#     -------
#     <aic_names>[nc*ns*(nc_v-1)*(ns_v-1), nc*ns*(nc_v-1)*(ns_v-1), 3] : numpy array
#         Aerodynamic influence coeffients (can be interprete as induced
#         velocities given circulations=1)
#     """    
#     def initialize(self):
#         self.parameters.declare('in_name', types=str)
#         self.parameters.declare('eval_pt_shape', types=tuple)
#         self.parameters.declare('vortex_coords_shape', types=tuple)
#         self.parameters.declare('out_name', types=str)
#     def define(self):
#         shape = self.parameters['eval_pt_shape'][1]*self.parameters['eval_pt_shape'][2]*(self.parameters['vortex_coords_shape'][1]-1)*(self.parameters['vortex_coords_shape'][2]-1)
#         print('shape is', shape)
#         num_nodes = self.parameters['eval_pt_shape'][0]
#         self.add_input(self.parameters['in_name'],shape=(num_nodes,int(shape/2),3))
#         self.add_output(self.parameters['out_name'],shape=(num_nodes,int(shape/2),3))
#         self.eval_pt_shape = self.parameters['eval_pt_shape']
#         self.vortex_coords_shape = self.parameters['vortex_coords_shape']
#         # self.full_aic_func = jit((self.__get_full_aic_jax))
#         # self.full_aic_jac_func = jit((jacfwd(self.full_aic_func)))
#         # print('in_name is', self.parameters['in_name'])
#         # print('out_name is', self.parameters['out_name'])

#         self.full_aic_func = self.__get_full_aic_jax
#         # self.full_aic_jac_func = jacfwd(self.full_aic_func)
#         row_indices  = np.arange(int(num_nodes*shape/2*3))
#         col_indices = np.flip(row_indices.reshape(1,2,2,2,4,3),axis=(2,4)).flatten()
#         self.declare_derivatives(self.parameters['out_name'], self.parameters['in_name'],rows=row_indices,cols=col_indices,val=np.ones(row_indices.size))

#     def compute(self, inputs, outputs):
#         # print('in_name is', self.parameters['in_name'])
#         # print('out_name is', self.parameters['out_name'])
#         # print(onp.array(self.full_aic_func(inputs[self.parameters['in_name']])))
#         outputs[self.parameters['out_name']] = self.full_aic_func(inputs[self.parameters['in_name']]).reshape(1,-1,3)
#     # def compute_derivatives(self, inputs, derivatives):
#     #     pass
#     def __get_full_aic_jax(self,half_aic):
#         nx_panel = self.eval_pt_shape[1]
#         ny_panel = self.eval_pt_shape[2] 
#         nx_panel_ind = self.vortex_coords_shape[1] - 1
#         ny_panel_ind = self.vortex_coords_shape[2] - 1 
#         num_nodes = self.eval_pt_shape[0]
#         half_reshaped = half_aic.reshape(num_nodes,nx_panel,int(ny_panel/2),nx_panel_ind, ny_panel_ind, 3)
#         # half_aic.reshape(nx_panel,int(ny_panel/2),nx_panel_ind, ny_panel_ind, 3)
#         other_half_aic = np.flip(half_reshaped, (2,4))
#         full_aic = np.concatenate((half_reshaped, other_half_aic), axis=1).reshape(num_nodes,-1,3)
#         return other_half_aic

# import jax.numpy as jnp
# from jax import jacfwd, jacrev, jit
# from jax.experimental import sparse

# class SymmetryJax(csdl.CustomExplicitOperation):
#     """
#     Compute the whole AIC matrix given half of it

#     parameters
#     ----------
#     <aic_half_names>[nc*ns*(nc_v-1)*(ns_v-1)* nc*ns*(nc_v-1)*(ns_v-1)/2, 3] : numpy array
#         Array defining the nodal coordinates of the lifting surface that the 
#         AIC matrix is computed on.

#     Returns
#     -------
#     <aic_names>[nc*ns*(nc_v-1)*(ns_v-1), nc*ns*(nc_v-1)*(ns_v-1), 3] : numpy array
#         Aerodynamic influence coeffients (can be interprete as induced
#         velocities given circulations=1)
#     """    
#     def initialize(self):
#         self.parameters.declare('in_name', types=str)
#         self.parameters.declare('eval_pt_shape', types=tuple)
#         self.parameters.declare('vortex_coords_shape', types=tuple)
#         self.parameters.declare('out_name', types=str)
#     def define(self):
#         shape = self.parameters['eval_pt_shape'][1]*self.parameters['eval_pt_shape'][2]*(self.parameters['vortex_coords_shape'][1]-1)*(self.parameters['vortex_coords_shape'][2]-1)
#         self.add_input(self.parameters['in_name'],shape=(1,int(shape/2),3))
#         self.add_output(self.parameters['out_name'],shape=(1,shape,3))
#         self.eval_pt_shape = self.parameters['eval_pt_shape']
#         self.vortex_coords_shape = self.parameters['vortex_coords_shape']
#         # self.full_aic_func = jit((self.__get_full_aic_jax))
#         # self.full_aic_jac_func = jit((jacfwd(self.full_aic_func)))
#         # print('in_name is', self.parameters['in_name'])
#         # print('out_name is', self.parameters['out_name'])

#         self.full_aic_func = self.__get_full_aic_jax
#         self.full_aic_jac_func = jacfwd(self.full_aic_func)
#         self.declare_derivatives(self.parameters['out_name'], self.parameters['in_name'])

#     def compute(self, inputs, outputs):
#         # print('in_name is', self.parameters['in_name'])
#         # print('out_name is', self.parameters['out_name'])
#         # print(onp.array(self.full_aic_func(inputs[self.parameters['in_name']])))
#         outputs[self.parameters['out_name']] = onp.array(self.full_aic_func(inputs[self.parameters['in_name']]))
#     def compute_derivatives(self, inputs, derivatives):
#         # print(onp.array(
#         #     (self.full_aic_jac_func(inputs[self.parameters['in_name']]))
#         # ))
#         derivatives[self.parameters['out_name'], self.parameters['in_name']] = onp.array(
#             (self.full_aic_jac_func(inputs[self.parameters['in_name']]))
#         )
#         # print('the shape of the derivative is', derivatives[self.parameters['out_name'], self.parameters['in_name']].shape)
#         # print(derivatives[self.parameters['out_name'], self.parameters['in_name']] )
#     def __get_full_aic_jax(self,half_aic):
#         nx_panel = self.eval_pt_shape[1]
#         ny_panel = self.eval_pt_shape[2] 
#         nx_panel_ind = self.vortex_coords_shape[1] - 1
#         ny_panel_ind = self.vortex_coords_shape[2] - 1 
#         num_nodes = self.eval_pt_shape[0]
#         half_reshaped = half_aic.reshape(nx_panel,int(ny_panel/2),nx_panel_ind, ny_panel_ind, 3)
#         # half_aic.reshape(nx_panel,int(ny_panel/2),nx_panel_ind, ny_panel_ind, 3)
#         other_half_aic = jnp.flip(half_reshaped, (1,3))
#         full_aic = jnp.concatenate((half_reshaped, other_half_aic), axis=1).reshape(num_nodes,-1,3)
#         return full_aic



if __name__ == "__main__":
    # import timeit
    # from python_csdl_backend import Simulator
    # import numpy as onp
    # AIC_half_val = onp.random.rand(1, 32, 3)
    # eval_pt_shape = (1,2,4)
    # vortex_coords_shape = (1,3,5)
    # m = csdl.Model()
    # AIC_half = m.declare_variable('AIC_half', val = AIC_half_val)
    # output_name = 'AIC'
    # AIC = csdl.custom(AIC_half, op = SymmetryJax(in_name=AIC_half.name, eval_pt_shape=eval_pt_shape, vortex_coords_shape=vortex_coords_shape, out_name=output_name))
    # m.register_output(output_name, AIC)
    # sim = Simulator(m)
    # sim.run()

    import time
    import timeit
    ts = time.time()
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
    ns = 11

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