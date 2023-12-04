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
nx=5; ny=11
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

mesh = np.array([[[[ 1.23431866e+01,  2.52445156e+01,  7.61899803e+00],
         [ 1.02636671e+01,  2.01968038e+01,  7.84026851e+00],
         [ 9.85856812e+00,  1.51500005e+01,  8.05305898e+00],
         [ 9.54411219e+00,  1.01000002e+01,  8.25003528e+00],
         [ 9.23152998e+00,  5.04999985e+00,  8.44626689e+00],
         [ 8.91777899e+00,  0.00000000e+00,  8.63965299e+00],
         [ 9.23109567e+00, -5.04999983e+00,  8.44627909e+00],
         [ 9.52079852e+00, -1.00966690e+01,  8.24109855e+00],
         [ 9.85807481e+00, -1.51500005e+01,  8.05319336e+00],
         [ 1.02636218e+01, -2.01968059e+01,  7.84025587e+00],
         [ 1.23431833e+01, -2.52445097e+01,  7.62035439e+00]],
 
        [[ 1.26074255e+01,  2.52496092e+01,  7.63634716e+00],
         [ 1.11314606e+01,  2.01967165e+01,  7.88454826e+00],
         [ 1.08748464e+01,  1.51491142e+01,  8.08919497e+00],
         [ 1.06737815e+01,  1.00991184e+01,  8.28657663e+00],
         [ 1.04730231e+01,  5.04914472e+00,  8.48320551e+00],
         [ 1.02704958e+01,  1.15338158e-08,  8.68014120e+00],
         [ 1.04726972e+01, -5.04914472e+00,  8.48382958e+00],
         [ 1.06562905e+01, -1.00966204e+01,  8.28744874e+00],
         [ 1.08744763e+01, -1.51491142e+01,  8.08967195e+00],
         [ 1.11314436e+01, -2.01967163e+01,  7.88514837e+00],
         [ 1.26074210e+01, -2.52402361e+01,  7.64125422e+00]],
 
        [[ 1.28702360e+01,  2.52495714e+01,  7.64012685e+00],
         [ 1.19992897e+01,  2.01966255e+01,  7.88056201e+00],
         [ 1.18911309e+01,  1.51482274e+01,  8.07768242e+00],
         [ 1.18034571e+01,  1.00982362e+01,  8.26962410e+00],
         [ 1.17145226e+01,  5.04828917e+00,  8.46108873e+00],
         [ 1.16243793e+01, -1.88206491e-08,  8.64616896e+00],
         [ 1.17143053e+01, -5.04828917e+00,  8.46082343e+00],
         [ 1.17917964e+01, -1.00965709e+01,  8.26968325e+00],
         [ 1.18908842e+01, -1.51482274e+01,  8.07747316e+00],
         [ 1.19992784e+01, -2.01966254e+01,  7.88033843e+00],
         [ 1.28699629e+01, -2.52495714e+01,  7.63996310e+00]],
 
        [[ 1.31352796e+01,  2.52495714e+01,  7.63953430e+00],
         [ 1.28671185e+01,  2.01965345e+01,  7.86226331e+00],
         [ 1.29074151e+01,  1.51473407e+01,  8.04937062e+00],
         [ 1.29331324e+01,  1.00973540e+01,  8.23384285e+00],
         [ 1.29560217e+01,  5.04743361e+00,  8.41815255e+00],
         [ 1.29746097e+01,  0.00000000e+00,  8.60336981e+00],
         [ 1.29559131e+01, -5.04743360e+00,  8.41863501e+00],
         [ 1.29273021e+01, -1.00965214e+01,  8.23462539e+00],
         [ 1.29072918e+01, -1.51473406e+01,  8.04977137e+00],
         [ 1.28671129e+01, -2.01965345e+01,  7.86258651e+00],
         [ 1.31352779e+01, -2.52495714e+01,  7.63961839e+00]],
 
        [[ 1.34001688e+01,  2.52495588e+01,  7.61381501e+00],
         [ 1.37342952e+01,  2.01964192e+01,  7.76010941e+00],
         [ 1.39228347e+01,  1.51464215e+01,  7.92224549e+00],
         [ 1.40617839e+01,  1.00964445e+01,  8.08795828e+00],
         [ 1.41963437e+01,  5.04654663e+00,  8.25398512e+00],
         [ 1.43306176e+01, -3.70259180e-17,  8.41987809e+00],
         [ 1.41963398e+01, -5.04654653e+00,  8.25391354e+00],
         [ 1.40617806e+01, -1.00964444e+01,  8.08789313e+00],
         [ 1.39228319e+01, -1.51464215e+01,  7.92218686e+00],
         [ 1.37342932e+01, -2.01964191e+01,  7.76005960e+00],
         [ 1.34002051e+01, -2.52495644e+01,  7.61380566e+00]]]]).reshape(nx,ny,3)


mesh_half = mesh[:,:int(ny/2),:]
mesh_symetric = mesh.copy()

mesh_symetric[:,int(ny/2)+1:,0] = mesh[:,:int(ny/2),0][:,::-1]
mesh_symetric[:,int(ny/2)+1:,1] = -mesh[:,:int(ny/2),1][:,::-1]
mesh_symetric[:,int(ny/2)+1:,2] = mesh[:,:int(ny/2),2][:,::-1]
# mesh_symetric[:,:,0] = np.einsum('i,j->ij',np.ones(nx),mesh[2,:,0])
try:
    import pyvista as pv

    plotter = pv.Plotter()
    grid = pv.StructuredGrid(mesh_symetric[:, :, 0], mesh_symetric[:, :, 1], mesh_symetric[:, :, 2])

    plotter.add_mesh(grid, show_edges=True,opacity=0.5, color='red')
    plotter.set_background('white')
    plotter.show()
except:
    pass         
# exit()
# wing = model_1.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))
wing = model_1.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh_symetric))

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
