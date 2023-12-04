'''Example 1 : simulation of a rectangular wing'''
import csdl
import numpy as np
from VAST.core.fluid_problem import FluidProblem
from VAST.utils.generate_mesh import *
from VAST.core.submodels.input_submodels.create_input_model import CreateACSatesModel
from VAST.core.vlm_llt.vlm_solver import VLMSolverModel
from python_csdl_backend import Simulator
from scipy.spatial.transform import Rotation as R

'''
TODO: confirm with the V&V team
'''
fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

compressible=False
model_1 = csdl.Model()
####################################################################
# 1. add aircraft states
####################################################################

theta = np.deg2rad(np.array([10]))  # pitch angles
num_nodes=theta.shape[0]
v_inf = np.ones((num_nodes,1))*248.136
submodel = CreateACSatesModel(v_inf=v_inf, theta=theta*0, num_nodes=num_nodes)
model_1.add(submodel, 'InputsModule')
####################################################################
# 2. add VLM meshes
####################################################################
# single lifting surface 
# (nx: number of points in streamwise direction; ny:number of points in spanwise direction)
surface_names = ['wing','h_fuselage']


# load LPC wing
# load npy file
mesh_1 = np.load('v&v_meshes/mesh_1.npy')
nx_1 = mesh_1.shape[1]
ny_1 = mesh_1.shape[2]
mesh_2 = np.swapaxes(np.load('v&v_meshes/mesh_2.npy'), 1, 2)
nx_2 = mesh_2.shape[1]
ny_2 = mesh_2.shape[2]
surface_shapes = [(num_nodes, nx_1, ny_1, 3),(num_nodes, nx_2, ny_2, 3)]


import pyvista as pv

plotter = pv.Plotter()
grid = pv.StructuredGrid(mesh_1.reshape(nx_1, ny_1, 3)[:, :, 0], mesh_1.reshape(nx_1, ny_1, 3)[:, :, 1], mesh_1.reshape(nx_1, ny_1, 3)[:, :, 2])
grid_fuselage = pv.StructuredGrid(mesh_2.reshape(nx_2, ny_2, 3)[:, :, 0], mesh_2.reshape(nx_2, ny_2, 3)[:, :, 1], mesh_2.reshape(nx_2, ny_2, 3)[:, :, 2])

plotter.add_mesh(grid, show_edges=True,opacity=0.5, color='red')
plotter.add_mesh(grid_fuselage, show_edges=True,opacity=0.5, color='blue')
plotter.set_background('white')
plotter.show()


# exit()

wing = model_1.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh_1.reshape(nx_1,ny_1,3)))
h_fuselage = model_1.create_input('h_fuselage', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh_2.reshape(nx_2,ny_2,3)))
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
        cl0 = [0.,0.0],
        # ref_area=11.3926998,
        compressible=compressible,
        frame = 'inertia'
    )
model_1.add(submodel, 'VLMSolverModel')
####################################################################

sim = Simulator(model_1) # add simulator

sim.run()

print('wing_C_L',sim['wing_C_L'])
print('h_fuselage_C_L',sim['h_fuselage_C_L'])
print('h_fuselage_C_D_i',sim['wing_C_D_i'])
print('h_fuselage_C_D_i',sim['h_fuselage_C_D_i'])
# print('area_wing',sim['panel_area_wing'][0]) 
# print('area_tail',sim['panel_area_tail'][0])
print('total_CL',sim['total_CL'])
print('total_CD_i',sim['total_CD'])
import sys
print('-'*90)
print(sys.argv[0],'Auto-testing has not been implemented yet.')
print('-'*90)
print('\n'*3)

plotter = pv.Plotter()
grid = pv.StructuredGrid(mesh_1.reshape(nx_1, ny_1, 3)[:, :, 0], mesh_1.reshape(nx_1, ny_1, 3)[:, :, 1], mesh_1.reshape(nx_1, ny_1, 3)[:, :, 2])
grid_fuselage = pv.StructuredGrid(mesh_2.reshape(nx_2, ny_2, 3)[:, :, 0], mesh_2.reshape(nx_2, ny_2, 3)[:, :, 1], mesh_2.reshape(nx_2, ny_2, 3)[:, :, 2])

plotter.add_mesh(grid, show_edges=True,opacity=0.5, color='red')
plotter.add_mesh(grid_fuselage, show_edges=True,opacity=0.5, color='blue')
plotter.set_background('white')
plotter.show()

