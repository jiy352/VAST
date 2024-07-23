import numpy as np
import csdl
from csdl import Model
from VAST.core.submodels.output_submodels.vlm_post_processing.compute_effective_aoa_cd_v import AOA_CD

class ThrustDrag(Model):
    """
    Compute lift, drag, CL, and CD.

    Parameters
    ----------
    surface_names : list
        List of surface names.
    surface_shapes : list
        List of shapes corresponding to the surfaces.
    eval_pts_option : str
        Option for evaluation points.
    eval_pts_shapes : list
        List of shapes for evaluation points.
    sprs : list
        List of sparse matrices.
    eval_pts_names : list
        List of names for evaluation points.
    coeffs_aoa : list, optional
        Coefficients for angle of attack.
    coeffs_cd : list, optional
        Coefficients for drag.
    delta_t : float, default=0.5
        Time step for calculations.

    Returns
    -------
    L : csdl variable
        Lift.
    D : csdl variable
        Drag.
    cl : csdl variable
        Lift coefficient.
    cd : csdl variable
        Drag coefficient.
    """

    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

        self.parameters.declare('eval_pts_names', types=list)
        self.parameters.declare('coeffs_aoa', default=None)
        self.parameters.declare('coeffs_cd', default=None)
        self.parameters.declare('delta_t', default=0.5)

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']

        delta_t = self.parameters['delta_t']
        num_nodes = surface_shapes[0][0]

        frame_vel = self.declare_variable('frame_vel', shape=(num_nodes, 3))
        rho = self.declare_variable('density', shape=(num_nodes, 1))
        alpha = self.declare_variable('alpha', shape=(num_nodes, 1))
        beta = self.declare_variable('beta', shape=(num_nodes, 1))

        system_size, rho_expand, bd_vec, circulations, circulation_repeat = self.initialize_variables(surface_shapes, num_nodes, rho)

        sina, cosa, sinb, cosb = self.compute_trig_functions(alpha, beta, num_nodes, system_size)
        panel_forces = self.compute_panel_forces(rho_expand, circulation_repeat, frame_vel, bd_vec)
        self.register_output('panel_forces', panel_forces)

        eval_pts_option = self.parameters['eval_pts_option']
        eval_pts_names = self.parameters['eval_pts_names']

        if eval_pts_option == 'auto':
            self.process_auto_evaluation(
                surface_names, surface_shapes, num_nodes, system_size, delta_t, panel_forces, rho, frame_vel, sina, cosa, sinb, cosb, eval_pts_names
                )

    def initialize_variables(self, surface_shapes, num_nodes, rho):
        system_size = sum((nx - 1) * (ny - 1) for _, nx, ny, _ in surface_shapes)
        # system size is the total number of panels
        rho_expand = csdl.expand(csdl.reshape(rho, (num_nodes,)), (num_nodes, system_size, 3), 'k->kij')
        bd_vec = self.declare_variable('bd_vec', shape=(num_nodes, system_size, 3))
        circulations = self.declare_variable('horseshoe_circulation', shape=(num_nodes, system_size))
        circulation_repeat = csdl.expand(circulations, (num_nodes, system_size, 3), 'ki->kij')
        return system_size, rho_expand, bd_vec, circulations, circulation_repeat

    def compute_trig_functions(self, alpha, beta, num_nodes, system_size):
        sina = csdl.expand(csdl.sin(alpha), (num_nodes, system_size, 1), 'ki->kji')
        cosa = csdl.expand(csdl.cos(alpha), (num_nodes, system_size, 1), 'ki->kji')
        sinb = csdl.expand(csdl.sin(beta), (num_nodes, system_size, 1), 'ki->kji')
        cosb = csdl.expand(csdl.cos(beta), (num_nodes, system_size, 1), 'ki->kji')
        return sina, cosa, sinb, cosb

    def compute_panel_forces(self, rho_expand, circulation_repeat, frame_vel, bd_vec):
        return rho_expand * circulation_repeat * csdl.cross(frame_vel, bd_vec, axis=2)

    def process_auto_evaluation(self, surface_names, surface_shapes, num_nodes, system_size, delta_t, panel_forces, rho, frame_vel, sina, cosa, sinb, cosb, eval_pts_names):
        velocities, s_panels_all, eval_pts_all, start = self.initialize_auto_evaluation_variables(num_nodes, system_size, surface_names, surface_shapes, eval_pts_names)
        gamma_b_repeat, dcirculation_repeat_dt, normals, panel_forces_dynamic = self.compute_dynamic_forces(rho, panel_forces, s_panels_all, delta_t, num_nodes, system_size, velocities, bd_vec, frame_vel)

        panel_forces_all = panel_forces + panel_forces_dynamic
        panel_forces_all_mag = csdl.sum(panel_forces_all**2, axes=(2,))**0.5
        velocities_mag = csdl.sum(frame_vel**2, axes=(2,))**0.5
        panel_power = csdl.sum(panel_forces_all_mag * velocities_mag, axes=(1,))

        self.register_output('panel_forces_all', panel_forces_all)
        self.register_output('panel_power', panel_power)

        self.compute_and_register_lift_drag(surface_names, surface_shapes, num_nodes, system_size, panel_forces_all, sina, cosa, sinb, cosb, s_panels_all, rho, frame_vel)

    def initialize_auto_evaluation_variables(self, num_nodes, system_size, surface_names, surface_shapes, eval_pts_names):
        velocities = self.create_output('eval_total_vel', shape=(num_nodes, system_size, 3))
        s_panels_all = self.create_output('s_panels_all', shape=(num_nodes, system_size))
        eval_pts_all = self.create_output('eval_pts_all', shape=(num_nodes, system_size, 3))

        start = 0
        for i in range(len(surface_names)):
            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]
            delta = (nx - 1) * (ny - 1)

            vel_surface = self.declare_variable(surface_names[i] + '_eval_total_vel', shape=(num_nodes, delta, 3))
            s_panels = self.declare_variable(surface_names[i] + '_s_panel', shape=(num_nodes, nx - 1, ny - 1))
            eval_pts = self.declare_variable(eval_pts_names[i], shape=(num_nodes, nx - 1, ny - 1, 3))

            velocities[:, start:start + delta, :] = vel_surface
            s_panels_all[:, start:start + delta] = csdl.reshape(s_panels, (num_nodes, delta))
            eval_pts_all[:, start:start + delta, :] = csdl.reshape(eval_pts, (num_nodes, delta, 3))
            start += delta
        return velocities, s_panels_all, eval_pts_all, start

    def compute_dynamic_forces(self, rho, panel_forces, s_panels_all, delta_t, num_nodes, system_size, velocities, bd_vec, frame_vel):
        gamma_b = self.declare_variable('gamma_b', shape=(num_nodes, system_size))
        gamma_b_repeat = csdl.expand(gamma_b * s_panels_all, (num_nodes, system_size, 3), 'ki->kij')

        c_bar = eval_pts_all[0, surface_shapes[0][1] - 1, 0, 0] - eval_pts_all[0, surface_shapes[0][1] - 2, 0, 0]
        c_bar_exp = csdl.reshape(csdl.expand(csdl.reshape(c_bar, (1,)), (num_nodes * system_size * 3, 1), 'i->ji'), (num_nodes, system_size, 3))

        dcirculation_repeat_dt = self.create_output('dcirculation_repeat_dt', shape=(num_nodes, system_size, 3))
        dcirculation_repeat_dt[0, :, :] = gamma_b_repeat[0, :, :] / delta_t
        if num_nodes > 2:
            dcirculation_repeat_dt[1:num_nodes-1, :, :] = (gamma_b_repeat[2:num_nodes, :, :] - gamma_b_repeat[0:num_nodes-2, :, :]) / (2 * delta_t)
        dcirculation_repeat_dt[num_nodes-1, :, :] = (gamma_b_repeat[num_nodes-1, :, :] - gamma_b_repeat[num_nodes-2, :, :]) / delta_t

        normals = self.declare_variable(surface_names[0] + '_bd_vtx_normals', shape=(num_nodes, system_size, 3))
        panel_forces_dynamic = rho_expand * dcirculation_repeat_dt * normals
        return gamma_b_repeat, dcirculation_repeat_dt, normals, panel_forces_dynamic

    def compute_and_register_lift_drag(self, surface_names, surface_shapes, num_nodes, system_size, panel_forces_all, sina, cosa, sinb, cosb, s_panels_all, rho, frame_vel):
        L_panel = -panel_forces_all[:, :, 0] * sina + panel_forces_all[:, :, 2] * cosa
        D_panel = panel_forces_all[:, :, 0] * cosa * cosb + panel_forces_all[:, :, 2] * sina * cosb - panel_forces_all[:, :, 1] * sinb

        s_panels_sum = csdl.reshape(csdl.sum(s_panels_all, axes=(1,)), (num_nodes, 1))
        start = 0
        for i in range(len(surface_names)):
            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]
            delta = (nx - 1) * (ny - 1)

            L_panel_surface = L_panel[:, start:start + delta, :]
            D_panel_surface = D_panel[:, start:start + delta, :]
            traction_surfaces = panel_forces_all[:, start:start + delta, :] / csdl.expand(s_panels_all[:, start:start + delta], panel_forces_all[:, start:start + delta, :].shape, 'ij->ijk')

            self.register_output(surface_names[i] + '_L_panel', L_panel_surface)
            self.register_output(surface_names[i] + '_D_panel', D_panel_surface)
            self.register_output(surface_names[i] + '_traction_surfaces', traction_surfaces)

            L = csdl.sum(L_panel_surface, axes=(1,))
            D = csdl.sum(D_panel_surface, axes=(1,))
            self.register_output(surface_names[i] + '_L', csdl.reshape(L, (num_nodes, 1)))
            self.register_output(surface_names[i] + '_D', csdl.reshape(D, (num_nodes, 1)))

            c_l = L / (0.5 * rho * s_panels_sum * csdl.sum(frame_vel**2, axes=(1,)))
            c_d = D / (0.5 * rho * s_panels_sum * csdl.sum(frame_vel**2, axes=(1,)))

            self.register_output(surface_names[i] + '_C_L', csdl.reshape(c_l, (num_nodes, 1)))
            self.register_output(surface_names[i] + '_C_D_i', csdl.reshape(c_d, (num_nodes, 1)))

            start += delta

        if self.parameters['coeffs_aoa'] is not None:
            sub = AOA_CD(surface_names=surface_names, surface_shapes=surface_shapes, coeffs_aoa=self.parameters['coeffs_aoa'], coeffs_cd=self.parameters['coeffs_cd'])
            self.add(sub, name='AOA_CD')

        total_forces_temp = csdl.sum(panel_forces, axes=(1,))
        total_forces_temp_dynamic = csdl.sum(panel_forces_dynamic, axes=(1,))

        F = self.create_output('F', shape=(num_nodes, 3))
        F[:, 0] = total_forces_temp[:, 0] + total_forces_temp_dynamic[:, 0]
        F[:, 1] = total_forces_temp[:, 1] + total_forces_temp_dynamic[:, 1]
        F[:, 2] = -total_forces_temp[:, 2] - total_forces_temp_dynamic[:, 2]

        self.register_output('thrust', F[:, 0])

        # Drag computation using CD_0 and CD_1
        CD_0 = 0.1936
        CD_1 = 0.1412
        alpha_deg = 0
        alpha = alpha_deg / 180 * np.pi
        CD_v = CD_0 + CD_1 * alpha**2
        Drag = 0.5 * rho * csdl.sum(frame_vel**2, axes=(1,)) * s_panels_sum * CD_0
        self.register_output('Drag', Drag)

        # Residual computation
        res = (csdl.sum(F[:, 0]) / num_nodes - csdl.sum(Drag) / num_nodes)**2
        self.register_output('res', res)

        # Moment computation
        evaluation_pt = self.declare_variable('evaluation_pt', val=np.zeros(3,))
        evaluation_pt_exp = csdl.expand(evaluation_pt, (eval_pts_all.shape), 'i->jki')
        r_M = eval_pts_all - evaluation_pt_exp
        total_moment = csdl.sum(csdl.cross(r_M, panel_forces, axis=2), axes=(1,))
        M = self.create_output('M', shape=total_moment.shape)
        M[:, 0] = total_moment[:, 0]
        M[:, 1] = -total_moment[:, 1]
        M[:, 2] = total_moment[:, 2]
        self.register_output('M', M)


if __name__ == "__main__":
    nx = 3
    ny = 4
    model_1 = Model()
    surface_names = ['wing']
    surface_shapes = [(nx, ny, 3)]

    frame_vel_val = np.array([-1, 0, -1])
    f_val = np.einsum('i,j->ij', np.ones(6), np.array([-1, 0, -1]) + 1e-3)

    frame_vel = model_1.create_input('frame_vel', val=frame_vel_val)
    gamma_b = model_1.create_input('gamma_b', val=np.random.random(((nx - 1) * (ny - 1))))
    force_pt_vel = model_1.create_input('force_pt_vel', val=f_val)

    model_1.add(ThrustDrag(
        surface_names=surface_names,
        surface_shapes=surface_shapes,
        eval_pts_option='auto',
        eval_pts_shapes=[(nx, ny, 3)],
        sprs=[],
        eval_pts_names=['wing_eval_pts']
    ))

    sim = csdl.Simulator(model_1)
    sim.run()
