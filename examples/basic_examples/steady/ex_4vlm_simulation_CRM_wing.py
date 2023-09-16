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
num_nodes=5; nx=2; ny=7
num_twist_cp=5
compressible=True
Ma = 0.84
model_1 = csdl.Model()
####################################################################
# 1. add aircraft states
####################################################################
v_inf = np.ones((num_nodes,1))*248.136
theta = np.deg2rad(np.array([-10,-5,0,5,10]))  # pitch angles

submodel = CreateACSatesModel(v_inf=v_inf, theta=theta, num_nodes=num_nodes)
model_1.add(submodel, 'InputsModule')
####################################################################
# 2. add VLM meshes
####################################################################
# single lifting surface 
# (nx: number of points in streamwise direction; ny:number of points in spanwise direction)
surface_names = ['wing']
surface_shapes = [(num_nodes, nx, ny, 3)]

# Generate mesh of a rectangular wing
mesh_dict = {"num_y": ny, "num_x": nx, "wing_type": "CRM", "symmetry": False,'num_twist_cp':num_twist_cp, "span": 10.0,}

# Generate the aerodynamic mesh based on the previous dictionary
mesh, twist_cp = generate_mesh(mesh_dict)

if compressible:
    r = R.from_euler('y', np.rad2deg(theta), degrees=True).as_matrix() # num_nodes,3,3
    rotated_mesh = np.einsum('ijk,lmk->ilmj', r, mesh)
    rotated_mesh_csdl = model_1.create_input('wing', val=rotated_mesh)
else:
    wing = model_1.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))


model_1.create_input('density', val=.38*np.ones((num_nodes,1)))
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
        Ma=Ma,
        frame = 'inertia',
        symmetry=True,
    )

model_1.add(submodel, 'VLMSolverModel')
####################################################################

sim = Simulator(model_1) # add simulator

sim.run()
Ma = 0.84
b = (1-Ma**2)**0.5
# print('wing_C_L',sim['wing_C_L'])
# print('wing_C_D_i',sim['wing_C_D_i'])

wing_C_L_OAS = np.array([-1.0434, -0.5215, 0.0, 0.5215, 1.0434]).reshape((num_nodes, 1))
wing_C_D_i_OAS = np.array([0.0292, 0.0073, 0.0, 0.0073, 0.0292]).reshape((num_nodes, 1))



if np.linalg.norm((wing_C_L_OAS - sim["wing_C_L"])/np.linalg.norm(wing_C_L_OAS+1e-10))<1e-2 and \
 np.linalg.norm((wing_C_D_i_OAS - sim["wing_C_D_i"])/np.linalg.norm(wing_C_D_i_OAS+1e-10))<1e-2:
    import sys
    
    rel_error_cl = np.linalg.norm((wing_C_L_OAS - sim["wing_C_L"])/ np.linalg.norm(wing_C_L_OAS))
    rel_error_cd = np.linalg.norm((wing_C_D_i_OAS - sim["wing_C_D_i"])/ np.linalg.norm(wing_C_D_i_OAS))  
    max_rel_error = max(rel_error_cl,rel_error_cd)

    # if the relative error is less than 1%, we consider it as a pass
    print('-'*90)
    print(sys.argv[0],f'Test passed! Max relative error is {max_rel_error*100}% less than the tolerance.')
    print('-'*90)

    print('\n'*3)

# AOA	OAS CL	   OAS CDi
# -10	-1.0434	   0.0292
# -5	-0.5215	   0.0073
# 0	     0.0000	   0.0000
# 5	     0.5215	   0.0073
# 10	 1.0434	   0.0292

Ma = 0.84
b = (1-Ma**2)**0.5

try:
    import pyvista as pv

    plotter = pv.Plotter()
    grid = pv.StructuredGrid(mesh[:, :, 0], mesh[:, :, 1], mesh[:, :, 2])

    plotter.add_mesh(grid, show_edges=True,opacity=0.5, color='red')
    plotter.set_background('white')
    plotter.show()
except:
    pass