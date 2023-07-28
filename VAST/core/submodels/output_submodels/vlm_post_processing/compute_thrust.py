from turtle import shape
# from csdl_om import Simulator
from csdl import Model
import csdl
from matplotlib.pyplot import axis
import numpy as np

from VAST.core.submodels.output_submodels.vlm_post_processing.compute_effective_aoa_cd_v import AOA_CD


class ThrustDrag(Model):
    """
    L,D,cl,cd
    parameters
    ----------

    bd_vec : csdl array
        tangential vec    
    velocities: csdl array
        force_pts vel 
    gamma_b[num_bd_panel] : csdl array
        a concatenate vector of the bd circulation strength
    frame_vel[3,] : csdl array
        frame velocities
    Returns
    -------
    L,D,cl,cd
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

        self.parameters.declare('eval_pts_option')
        self.parameters.declare('eval_pts_shapes')
        self.parameters.declare('sprs')

        # self.parameters.declare('rho', default=0.9652)
        self.parameters.declare('eval_pts_names', types=None)

        self.parameters.declare('coeffs_aoa', default=None)
        self.parameters.declare('coeffs_cd', default=None)
        self.parameters.declare('delta_t',default=0.5)

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        delta_t = self.parameters['delta_t']

        num_nodes = surface_shapes[0][0]
        frame_vel = self.declare_variable('frame_vel', shape=(num_nodes, 3))

        cl_span_names = [x + '_cl_span' for x in surface_names]

        system_size = 0
        for i in range(len(surface_names)):
            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]
            system_size += (nx - 1) * (ny - 1)

        rho = self.declare_variable('density', shape=(num_nodes, 1))
        rho_expand = csdl.expand(csdl.reshape(rho, (num_nodes, )), (num_nodes, system_size, 3), 'k->kij')
        alpha = self.declare_variable('alpha', shape=(num_nodes, 1))
        beta = self.declare_variable('beta', shape=(num_nodes, 1))

        eval_pts_option = self.parameters['eval_pts_option']
        eval_pts_shapes = self.parameters['eval_pts_shapes']

        v_total_wake_names = [x + '_eval_total_vel' for x in surface_names]

        bd_vec = self.declare_variable('bd_vec',
                                       shape=((num_nodes, system_size, 3)))

        circulations = self.declare_variable('horseshoe_circulation',
                                             shape=(num_nodes, system_size))
        circulation_repeat = csdl.expand(circulations,
                                         (num_nodes, system_size, 3),
                                         'ki->kij')

        eval_pts_names = self.parameters['eval_pts_names']

        if eval_pts_option == 'auto':
            velocities = self.create_output('eval_total_vel',
                                            shape=(num_nodes, system_size, 3))
            s_panels_all = self.create_output('s_panels_all',
                                              shape=(num_nodes, system_size))
            eval_pts_all = self.create_output('eval_pts_all',
                                              shape=(num_nodes, system_size,
                                                     3))
            start = 0
            for i in range(len(v_total_wake_names)):

                nx = surface_shapes[i][1]
                ny = surface_shapes[i][2]
                delta = (nx - 1) * (ny - 1)

                vel_surface = self.declare_variable(v_total_wake_names[i], shape=(num_nodes, delta, 3))
                s_panels = self.declare_variable(surface_names[i] + '_s_panel',  shape=(num_nodes, nx - 1, ny - 1))

                chords = self.declare_variable(
                    surface_names[i] + '_chord_length',
                    shape=(num_nodes, nx - 1, ny - 1))
                eval_pts = self.declare_variable(eval_pts_names[i],
                                                 shape=(num_nodes, nx - 1,
                                                        ny - 1, 3))

                velocities[:, start:start + delta, :] = vel_surface
                s_panels_all[:, start:start + delta] = csdl.reshape( s_panels, (num_nodes, delta))
                eval_pts_all[:, start:start + delta, :] = csdl.reshape( eval_pts, (num_nodes, delta, 3))
                start = start + delta

            panel_forces = rho_expand * circulation_repeat * csdl.cross(
                velocities, bd_vec, axis=2)

            self.register_output('panel_forces', panel_forces)
            wing_inital = self.declare_variable(surface_names[0],shape=(num_nodes,nx,ny,3))
            # TODO: delete this hardcoding


            gamma_b = self.declare_variable('gamma_b',shape=(num_nodes, system_size))
            gamma_b_repeat = csdl.expand(gamma_b* s_panels_all,(num_nodes, system_size, 3),'ki->kij')   
            # gamma_b_repeat = csdl.expand(gamma_b,(num_nodes, system_size, 3),'ki->kij')   
          
            c_bar_exp = csdl.expand(csdl.reshape(chords,(num_nodes,system_size)),(num_nodes,system_size,3),'ki->kij')




            dcirculation_repeat_dt = self.create_output('dcirculation_repeat_dt',shape=(num_nodes,system_size,3))

            dcirculation_repeat_dt[0,:,:] = (gamma_b_repeat[0,:,:])/delta_t
            dcirculation_repeat_dt[1:num_nodes-1,:,:] = (gamma_b_repeat[2:num_nodes,:,:]-gamma_b_repeat[0:num_nodes-2,:,:])/(delta_t*2)
            dcirculation_repeat_dt[num_nodes-1,:,:] = (gamma_b_repeat[num_nodes-1,:,:]-gamma_b_repeat[num_nodes-2,:,:])/delta_t

            # panel_forces_dynamic = rho_expand * dcirculation_repeat_dt * c_bar_exp * csdl.cross(
            #     velocities, bd_vec, axis=2)

            normals = self.declare_variable(surface_names[0] + '_bd_vtx_normals',shape=(num_nodes,system_size,3))
            panel_forces_dynamic = -rho_expand * dcirculation_repeat_dt * normals


            # print('compute lift drag panel_forces', panel_forces.shape)
            b = frame_vel[:, 0]**2 + frame_vel[:, 1]**2 + frame_vel[:, 2]**2


            traction_panel = panel_forces / csdl.expand(
                s_panels_all, panel_forces.shape, 'ij->ijk')

            s_panels_sum = csdl.reshape(csdl.sum(s_panels_all, axes=(1, )),
                                        (num_nodes, 1))


            total_forces_temp = csdl.sum(panel_forces, axes=(1, ))




            '''this is hardcodes with one surface'''
            initial_mesh_names = [
                x + '_initial_mesh' for x in self.parameters['surface_names']
            ]


            # panel_forces_dynamic = rho_expand * dcirculation_repeat_dt* c_bar_exp * csdl.cross(
            #     velocities, bd_vec, axis=2)
            total_forces_temp_dynamic = csdl.sum(panel_forces_dynamic, axes=(1, ))
            self.register_output('rho_expand',rho_expand)
            # self.register_output('dcirculation_repeat_dt',dcirculation_repeat_dt)
            self.register_output('c_bar_exp',c_bar_exp)
            self.register_output('total_forces_temp_dynamic',total_forces_temp_dynamic)
            self.register_output('crosspd',csdl.cross(
                velocities, bd_vec, axis=2))

            panel_forces_all = panel_forces + panel_forces_dynamic
            self.register_output('panel_forces_all', panel_forces_all)
            self.register_output('panel_forces_dynamic', panel_forces_dynamic)
            panel_forces_all_mag = csdl.sum(panel_forces_all**2,axes=(2,))**0.5
            velocities_mag = csdl.sum(velocities**2,axes=(2,))**0.5
            panel_power = csdl.sum(panel_forces_all_mag*velocities_mag,axes=(1,))

            # panel_power = csdl.sum(csdl.dot(panel_forces_all,velocities,axis=2),axes=(1,))
            # panel_power = csdl.dot(panel_forces_all,velocities,axis=2)
            self.register_output('panel_power',panel_power)

            F = self.create_output('F', shape=(num_nodes, 3))
            F[:, 0] = total_forces_temp[:, 0] + total_forces_temp_dynamic[:, 0]
            F[:, 1] = total_forces_temp[:, 1] + total_forces_temp_dynamic[:, 1]
            F[:, 2] = -total_forces_temp[:, 2] - total_forces_temp_dynamic[:, 2]

            F_s = self.create_output('F_s', shape=(num_nodes, 3))
            F_s[:, 0] = total_forces_temp[:, 0] 
            F_s[:, 1] = total_forces_temp[:, 1] 
            F_s[:, 2] = -total_forces_temp[:, 2] 
            self.register_output('thrust',F[:,0]) # thurst is negative x force

            CD_0 = 0.1936
            CD_1 = 0.1412
            alpha_deg = 0
            alpha = alpha_deg / 180 * np.pi
            CD_v = CD_0 + CD_1 * alpha**2
            s_panels_sum = csdl.reshape(csdl.sum(s_panels_all, axes=(1, )),
                                        (num_nodes, 1))



            evaluation_pt = self.declare_variable('evaluation_pt',
                                                  val=np.zeros(3, ))
            evaluation_pt_exp = csdl.expand(
                evaluation_pt,
                (eval_pts_all.shape),
                'i->jki',
            )
            r_M = eval_pts_all - evaluation_pt_exp
            total_moment = csdl.sum(csdl.cross(r_M, panel_forces, axis=2),
                                    axes=(1, ))
            M = self.create_output('M', shape=total_moment.shape)

            M[:, 0] = total_moment[:, 0] 
            M[:, 1] = -total_moment[:, 1] 
            M[:, 2] = total_moment[:, 2]



if __name__ == "__main__":

    nx = 3
    ny = 4
    model_1 = Model()
    surface_names = ['wing']
    surface_shapes = [(nx, ny, 3)]

    frame_vel_val = np.array([-1, 0, -1])
    f_val = np.einsum(
        'i,j->ij',
        np.ones(6),
        np.array([-1, 0, -1]) + 1e-3,
    )

    # coll_val = np.random.random((4, 5, 3))

    frame_vel = model_1.create_input('frame_vel', val=frame_vel_val)
    gamma_b = model_1.create_input('gamma_b',
                                   val=np.random.random(((nx - 1) * (ny - 1))))
    force_pt_vel = model_1.create_input('force_pt_vel', val=f_val)

    model_1.add(
        LiftDrag(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
        ))

    frame_vel = model_1.declare_variable('L', shape=(1, ))
    frame_vel = model_1.declare_variable('D', shape=(1, ))

    sim = Simulator(model_1)
    sim.run()
