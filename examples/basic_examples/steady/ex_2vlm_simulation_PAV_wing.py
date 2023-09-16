'''Example 1 : simulation of a rectangular wing'''
import csdl
import numpy as np
from VAST.core.fluid_problem import FluidProblem
from VAST.utils.generate_mesh import *
from VAST.core.submodels.input_submodels.create_input_model import CreateACSatesModel
from VAST.core.vlm_llt.vlm_solver import VLMSolverModel
from python_csdl_backend import Simulator
from scipy.spatial.transform import Rotation as R

fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')
nx=16; ny=40
model_1 = csdl.Model()
####################################################################
# 1. add aircraft states
####################################################################
theta = np.deg2rad(np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]))  # pitch angles
num_nodes=theta.shape[0]
v_inf = np.ones((num_nodes,1))*248.136
submodel = CreateACSatesModel(v_inf=v_inf, theta=theta, num_nodes=num_nodes)
model_1.add(submodel, 'InputsModule')
####################################################################
# 2. add VLM meshes
####################################################################
# single lifting surface 
# (nx: number of points in streamwise direction; ny:number of points in spanwise direction)
surface_names = ['wing']
surface_shapes = [(num_nodes, nx, ny, 3)]

# load PAV wing
mesh_half = np.swapaxes(np.loadtxt('v&v_meshes/mesh.txt').reshape(20,16,3), 0, 1)
mesh_other_half = mesh_half[:,::-1,:].copy()
mesh_other_half[:,:,1] =-mesh_other_half.copy()[:,:,1]
mesh = np.concatenate((mesh_other_half,mesh_half),axis=1)

wing = model_1.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))

model_1.create_input('density', val=1*np.ones((num_nodes,1)))
####################################################################
# 3. add VAST solver
####################################################################
if fluid_problem.solver_option == 'VLM':
    eval_pts_shapes = [(num_nodes, x[1] - 1, x[2] - 1, 3) for x in surface_shapes]
    submodel = VLMSolverModel(
        surface_names=surface_names,
        surface_shapes=surface_shapes,
        num_nodes=num_nodes,
        eval_pts_shapes=eval_pts_shapes,
        AcStates='dummy',
        cl0 = [0.0,],
        frame = 'wing_fixed'
    )
model_1.add(submodel, 'VLMSolverModel')
####################################################################

sim = Simulator(model_1) # add simulator

sim.run()

# print('wing_C_L\n',sim['wing_C_L'])
# print('wing_C_D_i\n',sim['wing_C_D_i'])

wing_C_L_AVL = np.array([-0.77020, -0.61878, -0.465579217135667, -0.311163208143354,-0.155784316259393,0.00000,
                          0.155784316259393, 0.311163208143354, 0.465579217135667, 0.61878, 0.77020]).reshape((num_nodes, 1))

if np.linalg.norm(wing_C_L_AVL - sim["wing_C_L"])/(np.linalg.norm((wing_C_L_AVL)))<3e-2:
    import sys

    # if the relative error is less than 1%, we consider it as a pass
    print('-'*90)
    print(sys.argv[0],f'Test passed! Relative error is {np.linalg.norm(wing_C_L_AVL - sim["wing_C_L"])/(np.linalg.norm((wing_C_L_AVL)))*100}% less than the tolerance.')
    print('-'*90)

    print('\n'*3)



try:
    import pyvista as pv

    plotter = pv.Plotter()
    grid = pv.StructuredGrid(mesh[:, :, 0], mesh[:, :, 1], mesh[:, :, 2])

    plotter.add_mesh(grid, show_edges=True,opacity=0.5, color='red')
    plotter.set_background('white')
    plotter.show()
except:
    pass
