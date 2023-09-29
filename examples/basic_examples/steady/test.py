
'''Example 1 : simulation of a rectangular wing'''
import csdl
import numpy as np
from VAST.core.fluid_problem import FluidProblem
from VAST.utils.generate_mesh import *
from VAST.core.submodels.input_submodels.create_input_model import CreateACSatesModel
from VAST.core.vlm_llt.vlm_solver import VLMSolverModel
from python_csdl_backend import Simulator
from scipy.spatial.transform import Rotation as R
def generate_image_mesh(wing_mesh, h, theta, test_plot=False):
    # SHIFT ORIGIN TO THE TE
    wing_origin_shift = np.min(wing_mesh[:,:,0])
    wing_mesh_shifted = wing_mesh.copy()
    wing_mesh_shifted[:,:,0] -= wing_origin_shift
    c = np.max(wing_mesh[:,:,0]) - np.min(wing_mesh[:,:,0])
    print('chord length:', c)
    axis_shift = h - c*np.sin(theta)
    print(axis_shift)

    # CONVERT TO REFERENCE FRAME ON GROUND
    print(wing_mesh_shifted)
    t_mesh_1 = np.zeros_like(wing_mesh_shifted)
    t_mesh_1[:,:,1] = wing_mesh_shifted[:,:,1]
    t_mesh_1[:,:,0] = np.cos(theta)*(wing_mesh_shifted[:,:,0]) - np.sin(theta)*wing_mesh_shifted[:,:,2]
    t_mesh_1[:,:,2] = axis_shift + np.sin(theta)*(wing_mesh_shifted[:,:,0]) + np.cos(theta)*wing_mesh_shifted[:,:,2]
    # print(t_mesh_1)
    # exit()

    # APPLY INVERSION IN Z-AXIS
    t_mesh_2 = t_mesh_1[:,:,:].copy()
    t_mesh_2[:,:,2] *= -1.

    # SYMMETRY PLANE (GROUND)
    symmetry_plane_gf = np.zeros((2,3))
    symmetry_plane_gf[0,:] = np.linspace(0., c, 3)

    # CONVERT BACK TO FRAME OF ORIGINAL LIFTING SURFACE
    wing_image_mesh = t_mesh_2[:,:,:].copy()
    wing_image_mesh[:,:,0] = np.cos(theta)*t_mesh_2[:,:,0] + np.sin(theta)*(t_mesh_2[:,:,2] - axis_shift) + wing_origin_shift
    wing_image_mesh[:,:,2] = -np.sin(theta)*t_mesh_2[:,:,0] + np.cos(theta)*(t_mesh_2[:,:,2] - axis_shift)

    symmetry_plane = np.zeros_like(symmetry_plane_gf)
    symmetry_plane[0,:] = np.cos(theta)*symmetry_plane_gf[0,:] + np.sin(theta)*(symmetry_plane_gf[1,:] - axis_shift) + wing_origin_shift
    symmetry_plane[1,:] = -np.sin(theta)*symmetry_plane_gf[0,:] + np.cos(theta)*(symmetry_plane_gf[1,:] - axis_shift)

    print(wing_mesh)
    print(t_mesh_1)
    print(t_mesh_2)
    print(wing_image_mesh)
    # exit()

    wing_line = wing_mesh[0:3,0,0:3]
    wing_image_line = wing_image_mesh[0:3,0,0:3]

    print(wing_line)

    # WING INFLOW LINE
    wing_inflow_x = np.array([wing_line[0,0],wing_line[0,0],wing_line[0,0]])
    wing_inflow_z = np.array([wing_line[0,0],wing_line[0,0],wing_line[0,0]])

    wing_inflow_x[0] = wing_line[0,0]
    wing_inflow_z[0] = wing_line[0,2]


    wing_inflow_x = symmetry_plane[0,:] + wing_line[-1,0]
    wing_inflow_z = symmetry_plane[1,:] + wing_line[-1,2]

    # WING IMAGE INFLOW LINE


    import matplotlib.pyplot as plt
    plt.plot(wing_line[:,0], wing_line[:,2], 'k-*', label='wing')
    plt.plot(wing_image_line[:,0], wing_image_line[:,2], 'b-*', label='image')
    plt.plot(symmetry_plane[0,:], symmetry_plane[1,:], 'r--', label='ground plane')
    # plt.plot(wing_inflow_x, wing_inflow_z, 'g', label='inflow')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.axis('equal')
    plt.grid()
    if test_plot:
        plt.show()
        # exit()
    return wing_image_mesh

# region inputs
nx, ny = 3, 11
num_nodes = 1

h = 1.

mach = 0.02
sos = 340.3
v_inf_scalar = mach*sos

pitch_scalar = 10. # degrees
# endregion

fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

model = csdl.Model()

# region aircraft states
v_inf = np.ones((num_nodes,1)) * v_inf_scalar
theta = np.deg2rad(np.ones((num_nodes,1))*pitch_scalar)

acstates_model = CreateACSatesModel(v_inf=v_inf, theta=theta, num_nodes=num_nodes)
model.add(acstates_model, 'ac_states_model')
# endregion

# region VLM meshes
surface_names, surface_shapes = [], []

# WING
mesh_dict = {
    "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": 10.0,
    "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0, # 'offset': np.array([0., 0., h])
}

wing_mesh_temp = generate_mesh(mesh_dict)
# wing_mesh = generate_mesh(mesh_dict)

theta = pitch_scalar



r = R.from_euler('y', -theta, degrees=True).as_matrix() # num_nodes,3,3
wing_mesh = np.einsum('ijk,kl->ijl', wing_mesh_temp, r)


surface_names.append('wing')
surface_shapes.append((num_nodes, nx, ny, 3))
wing = model.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), wing_mesh))

# IMAGE OF WING
wing_image_mesh = generate_image_mesh(wing_mesh, h, theta, test_plot=False)

# mesh_dict = {
#     "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": 10.0,
#     "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0, 'offset': np.array([0., 0., -h])
# }

wing_image_mesh_temp = generate_mesh(mesh_dict)
height = 20

r = R.from_euler('y', theta, degrees=True).as_matrix() # num_nodes,3,3
wing_image_mesh = np.einsum('ijk,kl->ijl', wing_image_mesh_temp, r)
wing_image_mesh[:,:,2] += height

surface_names.append('wing_image')
surface_shapes.append((num_nodes, nx, ny, 3))
wing_image = model.create_input('wing_image', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), wing_image_mesh))

# endregion

# VAST SOLVER
eval_pts_shapes = [(num_nodes, x[1] - 1, x[2] - 1, 3) for x in surface_shapes]
solver_model = VLMSolverModel(
    surface_names=surface_names,
    surface_shapes=surface_shapes,
    num_nodes=num_nodes,
    eval_pts_shapes=eval_pts_shapes,
    AcStates='dummy',
    frame='inertial',
    cl0 = [0.0,0.0],
)

model.add(solver_model, 'VLMSolverModel')

sim = Simulator(model, analytics=True)
sim.run()

print('==== RESULTS ====')
print('\n')
for surface in surface_names:
    print(f'==== Results for surface: {surface} ====')
    print(f'{surface} Lift (N)', sim[f'{surface}_L'])
    print(f'{surface} Drag (N)', sim[f'{surface}_D'])
    print(f'{surface} CL', sim[f'{surface}_C_L'])
    print(f'{surface} CD_i', sim[f'{surface}_C_D_i'])

    print('\n')

print('==== Results for total values ====')
print('Total Lift (N)', sim['total_lift'])
print('Total Drag (N)', sim['total_drag'])
print('L/D', sim['L_over_D'])
print('Total CL', sim['total_CL'])
print('Total CD', sim['total_CD'])
print('Total Moments', sim['M'])