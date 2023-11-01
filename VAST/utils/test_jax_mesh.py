import csdl
import numpy as np
from VAST.utils.generate_mesh import *
import jax
import jax.numpy as jnp
from jax import jacfwd

def time_freq(frequency):
    '''
    Compute the frequency of a time series
    '''
    t = jnp.linspace(1e-3, 1/frequency, num_nodes)
    return t

def fit_actuator_angles(frequency,surface_shape):
    num_nodes = surface_shape[0]
    t = jnp.linspace(1e-3, 1/frequency, num_nodes)
    coeff_a_1 = jnp.array([-10.52532715,  77.66921897])
    coeff_b_1 = 0
    a_0 = 0
    a_1 = coeff_a_1[0] * frequency + coeff_a_1[1]
    b_1 = coeff_b_1

    angles = a_0 + a_1 * jnp.cos(t * 2 * jnp.pi * frequency + b_1)
    return angles

def compute_mesh(frequency,surface_shape):
    angles_deg = fit_actuator_angles(frequency,surface_shape)
    angles = jnp.deg2rad(angles_deg)
    L = 0.065
    nx = surface_shape[1]
    ny = surface_shape[2]


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

# def compute_mesh_velocity(frequency,surface_shape):
    
def compute_mesh_wrt_t(t,surface_shape):
    frequency = 1/t[-1]
    angles_deg = fit_actuator_angles(frequency,surface_shape)
    angles = jnp.deg2rad(angles_deg)
    L = 0.065
    nx = surface_shape[1]
    ny = surface_shape[2]


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


frequency_val = 5.
num_nodes = 10
nx = 11
ny = 5
surface_shape = (num_nodes, nx, ny)

mesh = compute_mesh(frequency_val,surface_shape)

mesh_array = np.array(mesh)

# Compute the Jacobian function
jacobian_fn = jacfwd(compute_mesh,argnums=0)
frequency_val = frequency_val  # example value
jacobian_value = jacobian_fn(frequency_val, surface_shape)


# # compute dmeshdt jacobian as dmeshdf * dfdt
# dmeshdt_jacobian_value = np.einsum('ijkl,i->ijkl',jacobian_value,dfdt_jacobian_value)

# import pyvista as pv
# for i in range(num_nodes):
#     x_mesh_i = mesh_array[i,:,:,0]
#     y_mesh_i = mesh_array[i,:,:,1]
#     z_mesh_i = mesh_array[i,:,:,2]
#     mesh_i = pv.StructuredGrid(x_mesh_i,y_mesh_i,z_mesh_i)
#     mesh_i.save(f'mesh_{i}.vtk')
    