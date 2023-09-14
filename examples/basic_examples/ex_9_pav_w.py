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
num_nodes=3; nx=16; ny=40
num_twist_cp=5
compressible=False
model_1 = csdl.Model()
####################################################################
# 1. add aircraft states
####################################################################
v_inf = np.ones((num_nodes,1))*248.136
# theta = np.deg2rad(np.ones((num_nodes,1))*5)  # pitch angles
# theta = np.deg2rad(np.array([-10,-1,0,5,10]))  # pitch angles
theta = np.deg2rad(np.array([12,14,16]))  # pitch angles

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
mesh_half = np.swapaxes(np.loadtxt('mesh.txt').reshape(20,16,3), 0, 1)
mesh_other_half = mesh_half[:,::-1,:].copy()
mesh_other_half[:,:,1] =-mesh_other_half.copy()[:,:,1]
mesh = np.concatenate((mesh_other_half,mesh_half),axis=1)

# mesh = np.swapaxes(np.loadtxt('mesh.txt').reshape(20,16,3), 0, 1)


if compressible:
    # r = R.from_euler('y', np.array([-10,-1,0,5,10]), degrees=True).as_matrix() # num_nodes,3,3
    r = R.from_euler('y', np.array([-10,-5,0]), degrees=True).as_matrix() # num_nodes,3,3
    rotated_mesh = np.einsum('ijk,lmk->ilmj', r, mesh)
    rotated_mesh_csdl = model_1.create_input('wing', val=rotated_mesh)
else:
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
        cl0 = [0.0,0.0],
        # ref_area=205.2824,
        compressible=compressible,
        frame = 'wing_fixed'
    )
# wing_C_L_OAS = np.array([0.4426841725811703]).reshape((num_nodes, 1))
# wing_C_D_i_OAS = np.array([0.005878842561184834]).reshape((num_nodes, 1))
model_1.add(submodel, 'VLMSolverModel')
####################################################################

sim = Simulator(model_1) # add simulator

sim.run()
Ma = 0.84
b = (1-Ma**2)**0.5
print('wing_C_L',sim['wing_C_L'])
# print('wing_C_L*b',sim['wing_C_L']*b)
print('wing_C_D_i',sim['wing_C_D_i'])
# print('wing_C_L*b',sim['wing_C_D_i']*b)
# print('The number of nan in num_00e9 is: ', np.count_nonzero(np.isinf(sim['num_00e9'])))
# print('The number of nan in num_00eB is: ', np.count_nonzero(np.isinf(sim['num_00eB'])))
# print('The number of nan in num_00f2 is: ', np.count_nonzero(np.isinf(sim['num_00f2'])))
# print('The number of nan in num_00fu is: ', np.count_nonzero(np.isinf(sim['num_00fu'])))


Ma = 0.84
b = (1-Ma**2)**0.5

import pyvista as pv

plotter = pv.Plotter()
grid = pv.StructuredGrid(mesh[:, :, 0], mesh[:, :, 1], mesh[:, :, 2])

plotter.add_mesh(grid, show_edges=True,opacity=0.5, color='red')
plotter.set_background('white')
plotter.show()
