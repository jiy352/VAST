import csdl
import numpy as np
from VAST.utils.generate_mesh import *
import jax
import jax.numpy as jnp
from jax import jacfwd


class ActuatorActuation(csdl.CustomExplicitOperation):
    '''
    This operation is used to index a matrix with a vector of indices.

    parameters
    ----------
    matrix : csdl 2d array [num_nodes, nn]
        Matrix to be indexed
    ind_array : numpy 0d array [n_ind,]
        array of indices

    Returns
    -------
    indexed_matrix : csdl 2d array [num_nodes, n_ind]
        Indexed matrix    
    '''
    def initialize(self):
        self.parameters.declare('surface_name', types=str)
        self.parameters.declare('surface_shape', types=tuple)
        # self.parameters.declare('ind_array', types=np.ndarray)
        # self.parameters.declare('out_name', types=str)
    def define(self):
        surface_name = self.parameters['surface_name']
        surface_shape = self.parameters['surface_shape']
        # ind_array = self.parameters['ind_array']
        # out_name = self.parameters['out_name']

        num_nodes = surface_shape[0]
        nx = surface_shape[1]
        ny = surface_shape[2]


        self.add_input('frequency',shape=(1, ))
        self.add_output(surface_name,shape=surface_shape)
        self.add_output(surface_name+'_coll_vel',shape=(num_nodes, nx-1, ny-1, 3))

        # row_indices  = np.arange(num_nodes*n_ind)
        # col_indices = np.arange(in_shape[0]*in_shape[1]).reshape(in_shape)[:,ind_array].flatten()
        
        self.declare_derivatives(surface_name, 'frequency')
        self.declare_derivatives(surface_name+'_coll_vel', 'frequency')

    def compute(self, inputs, outputs):
        surface_name = self.parameters['surface_name']
        surface_shape = self.parameters['surface_shape']
        nx = surface_shape[1]
        ny = surface_shape[2]


        outputs[surface_name] = self.compute_mesh(inputs['frequency'])
        outputs[surface_name+'_coll_vel'] = self.compute_mesh_vel(inputs['frequency'])
        self.mesh = outputs[surface_name]
        self.mesh_vel = outputs[surface_name+'_coll_vel']
        '''
        derivative_fcn = jacfwd(self.compute_mesh)
        self.derivative_val = derivative_fcn(inputs['frequency'])

        derivative_fcn_vel = jacfwd(self.compute_mesh_vel)
        self.derivative_val_vel = derivative_fcn_vel(inputs['frequency'])
        '''
        # outputs[surface_name+'_coll_vel'][:] = inputs['frequency']

    def fit_actuator_angles(self,frequency):
        surface_shape = self.parameters['surface_shape']        
        num_nodes = surface_shape[0]
        N_period=3
        t = jnp.linspace(1e-3, N_period/frequency, num_nodes)
        coeff_a_1 = jnp.array([-10.52532715,  77.66921897])
        coeff_b_1 = 0
        a_0 = 0
        a_1 = coeff_a_1[0] * frequency + coeff_a_1[1]
        b_1 = coeff_b_1

        angles = a_0 + a_1 * jnp.cos(t * 2 * jnp.pi * frequency + b_1)
        return angles

    def compute_mesh(self, frequency):
        surface_shape = self.parameters['surface_shape']        
        angles_deg = self.fit_actuator_angles(frequency)
        angles = jnp.deg2rad(angles_deg)
        L = 0.065
        nx = surface_shape[1]
        ny = surface_shape[2]
        num_nodes = surface_shape[0]


        x_0 = np.linspace(0, L, nx) # size = nx
        R = L/angles # size = num_nodes
        alpha = jnp.outer(angles,x_0)/L # size = num_nodes, nx
        R_expanded = jnp.outer(R, np.ones(nx))

        x = R_expanded*jnp.sin(alpha)
        sign_angles_expanded = jnp.outer(jnp.tanh(50*angles), np.ones(nx))
        y = -sign_angles_expanded*(R_expanded**2 - x**2)**0.5 + R_expanded 

        x_expand = jnp.einsum('ij,l->ijl',x,np.ones(ny))
        y_expand = jnp.einsum('ij,l->ijl',y,np.ones(ny))
        mesh_dict = {
            "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": 0.042,
            "root_chord": 0.065, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0,
        }

        unactuated_mesh = np.outer(np.ones((num_nodes)),generate_mesh(mesh_dict)).reshape((num_nodes,nx,ny,3))

        print('shape of unactuated_mesh',unactuated_mesh.shape)
        print('shape of x_expand',x_expand.shape)
        print('shape of y_expand',y_expand.shape)
        # mesh = jnp.zeros((num_nodes, nx, ny, 3))

        # concatenate x,y,z
        mesh = jnp.concatenate((x_expand.reshape(num_nodes,nx,ny,1),y_expand.reshape(num_nodes,nx,ny,1),unactuated_mesh[:,:,:,1].reshape(num_nodes,nx,ny,1)),axis=3)

        
        # mesh[:,:,:,0] = x_expand
        # mesh[:,:,:,1] = y_expand
        # mesh[:,:,:,2] = unactuated_mesh[:,:,2]
        return mesh


    def compute_mesh_vel(self, frequency):
        num_nodes = self.parameters['surface_shape'][0]
        nx = self.parameters['surface_shape'][1]
        ny = self.parameters['surface_shape'][2]
        N_period = 3
        L = 0.065
        t = jnp.linspace(0, N_period/frequency, num_nodes)
        x_0 = jnp.linspace(0, L, nx)[1:]

        vx = self.jax_expression_vx(L, frequency, t, x_0)
        vy = self.jax_expression_vy(L, frequency, t, x_0)
        vz = jnp.zeros(vx.shape)

        v_line = jnp.concatenate((vx.reshape(num_nodes,nx-1, 1),vy.reshape(num_nodes,nx-1, 1),vz.reshape(num_nodes,nx-1, 1)),axis=2)
        v_mesh = jnp.einsum('ijk,l->ijlk',v_line,jnp.ones(ny-1))
        return v_mesh
    

    def jax_expression_vx(self, L, f, t, x_0):
        term1 = 4.63499420625724 * L * f * jnp.sin(6.28318530717959 * f * t)
        term2 = jnp.sin(x_0 * (-0.183608637309803 * f + 1.355597230024 * jnp.cos(6.28318530717959 * f * t)) / L)
        term3 = (-0.135444830693962 * f + jnp.cos(6.28318530717959 * f * t))**2

        term4 = -8.51746859814012 * f * x_0 * jnp.sin(6.28318530717959 * f * t)
        term5 = jnp.cos(x_0 * (-0.183608637309803 * f + 1.355597230024 * jnp.cos(6.28318530717959 * f * t)) / L)
        term6 = (-0.183608637309803 * f + 1.355597230024 * jnp.cos(6.28318530717959 * f * t))

        return (term1 * term2 / term3) + (term4 * term5 / term6)

    def jax_expression_vy(self, L, f, t_actual, x_0_actual):
        t= jnp.outer(t_actual, jnp.ones(x_0_actual.shape))
        x_0 = jnp.outer(jnp.ones(t_actual.shape), x_0_actual)
        return (4.63499420625724*L*f*jnp.sin(6.28318530717959*f*t)/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2
                + 314.159265358979*f*(1 - jnp.tanh(9.18043186549017*f - 67.7798615011998*jnp.cos(6.28318530717959*f*t))**2)*(-L**2*jnp.sin(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2 + L**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2)**0.5*jnp.sin(6.28318530717959*f*t)
                + 0.737682239128136*(-6.28318530717959*L**2*f*jnp.sin(6.28318530717959*f*t)*jnp.sin(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**3 + 6.28318530717959*L**2*f*jnp.sin(6.28318530717959*f*t)/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**3
                                    + 8.51746859814012*L*f*x_0*jnp.sin(6.28318530717959*f*t)*jnp.sin(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)*jnp.cos(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2)*jnp.tanh(9.18043186549017*f - 67.7798615011998*jnp.cos(6.28318530717959*f*t))/(-L**2*jnp.sin(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2 + L**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2)**0.5)


    def fd_derivative(self, inputs):
        mesh = self.mesh
        frequency = inputs['frequency']
        mesh_vel = self.mesh_vel
        num_nodes = self.parameters['surface_shape'][0]
        # implement finite difference derivative for mesh with respect to frequency
        delta_f = 1e-6
        mesh_delta_f = self.compute_mesh(frequency+delta_f)
        mesh_delta_f_vel = self.compute_mesh_vel(frequency+delta_f)
        # mesh_delta_f_minus = self.compute_mesh(frequency-delta_f)
        # mesh_delta_f_minus_vel = self.compute_mesh_vel(frequency-delta_f)
        derivative_val = (mesh_delta_f - mesh)/(delta_f)
        derivative_val_vel = (mesh_delta_f_vel - mesh_vel)/(delta_f)

        return derivative_val, derivative_val_vel
    
    def compute_derivatives(self, inputs, derivatives):
    #     in_name_1 = self.parameters["in_name_1"]
    #     in_name_2 = self.parameters["in_name_2"]
    #     out_name = self.parameters["out_name"]
    #     in_shape = self.parameters["ijk"]
        surface_name = self.parameters['surface_name']

        fd_derivative_mesh, fd_derivative_mesh_vel = self.fd_derivative(inputs)
        derivatives[surface_name, 'frequency'] = fd_derivative_mesh
        derivatives[surface_name+'_coll_vel', 'frequency'] = fd_derivative_mesh_vel

        # derivatives[surface_name, 'frequency'] = np.array(self.derivative_val)
        # derivatives[surface_name+'_coll_vel', 'frequency'] = np.array(self.derivative_val_vel)
        

if __name__ == "__main__":
    import python_csdl_backend

    num_nodes = 10
    nx = 11
    ny = 5

    surface_name = 'eel'
    surface_shape = (num_nodes, nx, ny, 3)

    model = csdl.Model()
    frequency = model.create_input('frequency', val=np.ones((1,))*5.)
    out_mat = csdl.custom(frequency,
                          op=ActuatorActuation(surface_name=surface_name,
                                            surface_shape=surface_shape,
                                            ))

    model.register_output(surface_name, out_mat[0])
    model.register_output(surface_name+'_coll_vel', out_mat[1])
    sim = python_csdl_backend.Simulator(model)

    sim.run()
    sim.check_partials(compact_print=True)
    # print('frequency:',sim['frequency'].shape,'\n',sim['frequency'])
    # print('surface_name:',sim[surface_name].shape,'\n',sim[surface_name])
    # print('surface_name_coll_vel:',sim[surface_name+'_coll_vel'].shape,'\n',sim[surface_name+'_coll_vel'])


    import pyvista as pv
    mesh_array = np.array(sim[surface_name])
    for i in range(num_nodes):
        
        x_mesh_i = mesh_array[i,:,:,0]
        y_mesh_i = mesh_array[i,:,:,1]
        z_mesh_i = mesh_array[i,:,:,2]
        mesh_i = pv.StructuredGrid(x_mesh_i,y_mesh_i,z_mesh_i)
        mesh_i.save(f'mesh_{i}.vtk')