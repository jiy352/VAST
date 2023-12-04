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
nx=30; ny=40
num_twist_cp=5
compressible=False
model_1 = csdl.Model()
####################################################################
# 1. add aircraft states
####################################################################

theta = np.deg2rad(np.array([-6]))  # pitch angles
# theta = np.deg2rad(np.array([ -10]))  # pitch angles
num_nodes=theta.shape[0]
v_inf = np.ones((num_nodes,1))*248.136
submodel = CreateACSatesModel(v_inf=v_inf, theta=theta, num_nodes=num_nodes)
model_1.add(submodel, 'InputsModule')

#################################################
# 2. add VLM meshes
####################################################################
# single lifting surface 
# (nx: number of points in streamwise direction; ny:number of points in spanwise direction)
surface_names = ['tail']
# nx_tail = 16; ny_tail = 6
nx_tail = 30; ny_tail = 40
surface_shapes = [(num_nodes, nx_tail, ny_tail*2, 3)]

# load PAV wing
# mesh_half = np.swapaxes(np.loadtxt('v&v_meshes/mesh.txt').reshape(20,16,3), 0, 1)
# mesh_other_half = mesh_half[:,::-1,:].copy()
# mesh_other_half[:,:,1] =-mesh_other_half.copy()[:,:,1]
# mesh = np.concatenate((mesh_other_half,mesh_half),axis=1)

# load PAV htail 16*6
tail_half = np.swapaxes(np.loadtxt('tail_mesh_fine.txt').reshape(ny_tail,nx_tail,3), 0, 1)
# tail_half = np.swapaxes(np.loadtxt('v&v_meshes/tail.txt').reshape(ny_tail,nx_tail,3), 0, 1)
tail_other_half = tail_half[:,::-1,:].copy()
tail_other_half[:,:,1] =-tail_other_half.copy()[:,:,1]
tail = np.concatenate((tail_other_half,tail_half),axis=1) * 0.3048

# wing = model_1.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))
wing = model_1.create_input('tail', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), tail))



# exit()
###################

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
        cl0 = [0,0.0],
        ref_area=4.587,
        compressible=compressible,
        frame = 'wing_fixed',
        symmetry=False,
    )
model_1.add(submodel, 'VLMSolverModel')
####################################################################

sim = Simulator(model_1) # add simulator

sim.run()

print('tail_C_L',sim['tail_C_L'])
print('total_CL',sim['total_CL'])
print('tail_C_D_i',sim['tail_C_D_i'])
print('total_CD_i',sim['total_CD'])
# print('area_wing',sim['panel_area_wing'][0]) 
# print('area_tail',sim['panel_area_tail'][0])

import sys
print('-'*90)
print(sys.argv[0],'Auto-testing has not been implemented yet.')
print('-'*90)
print('\n'*3)

try:
    import pyvista as pv

    plotter = pv.Plotter()
    # grid = pv.StructuredGrid(mesh[:, :, 0], mesh[:, :, 1], mesh[:, :, 2])
    grid_t = pv.StructuredGrid(tail[:, :, 0], tail[:, :, 1], tail[:, :, 2])

    # plotter.add_mesh(grid, show_edges=True,opacity=0.5, color='red')
    plotter.add_mesh(grid_t, show_edges=True,opacity=0.5, color='blue')
    plotter.set_background('white')
    plotter.show()
except:
    pass

# try:
#     import pyvista as pv

#     plotter = pv.Plotter()
#     grid = pv.StructuredGrid(mesh[:, :, 0], mesh[:, :, 1], mesh[:, :, 2])

#     plotter.add_mesh(grid, show_edges=True,opacity=0.5, color='red')
#     plotter.set_background('white')
#     plotter.show()

import matplotlib.pyplot as plt
data = np.loadtxt('flat_tail.txt')
plt.plot(data[:,0], data[:,1], '.-', label='wing')
plt.xlim([-15,20])
plt.ylim([-1,2.5])
# save the figure with a transparent background
plt.savefig('tail_CL.png', transparent=True)
plt.show()

np.savetxt('flat_tail.txt', data, delimiter=',')
