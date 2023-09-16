'''Example 3 : verification of prescibed vlm with Theodorsen solution'''
from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver import UVLMSolver
from VAST.core.submodels.actuation_submodels.pitching_wing_actuation import PitchingModel
import time
import numpy as np
from VAST.core.submodels.output_submodels.vlm_post_processing.efficiency import EfficiencyModel
from visualization import run_visualization
import os

########################################
# This is a test case to check the prescribed wake solver
# against the Theodorsen solution
# with a wing pitching sinusoidally from 1 to -1 deg
# and the reduced frequency k = [0.2, 0.6, 1, 3]
########################################


def run_pitching_theodorsen_verification(k,num_nodes,N_period,A=1,save_results=True, save_vtk=False, path = "theodorsen"):
    '''
    This is a test case to check the prescribed wake solver against the Theodorsen solution
    with a wing pitching sinusoidally from 1 to -1 deg and the reduced frequency k = [0.2, 0.6, 1, 3]
    '''
    ########################################
    # 1. define geometry
    ########################################

    nx = 31; ny = 3         # num_pts in chordwise and spanwise direction
    chord = 1; span = 100   # chord and span of the wing   
    b = chord/2

    omega = 1
    v_inf = omega*b/k
    T = 2*np.pi/(omega)

    t_vec = np.linspace(0, N_period*T, num_nodes) 

    u_val = np.ones(num_nodes) * v_inf
    w_vel = np.zeros((num_nodes,1)) *np.tan(np.deg2rad(0))

    states_dict = {
        'u': u_val, 'v': np.zeros((num_nodes, 1)), 'w': w_vel,
        'p': np.zeros((num_nodes, 1)), 'q': np.zeros((num_nodes, 1)), 'r': np.zeros((num_nodes, 1)),
        'theta': np.zeros((num_nodes,1)), 'psi': np.zeros((num_nodes, 1)),
        'x': np.zeros((num_nodes, 1)), 'y': np.zeros((num_nodes, 1)), 'z': np.zeros((num_nodes, 1)),
        'phiw': np.zeros((num_nodes, 1)), 'gamma': np.zeros((num_nodes, 1)),'psiw': np.zeros((num_nodes, 1)),
    }

    surface_properties_dict = {'surface_names':['wing'],
                                'surface_shapes':[(nx, ny, 3)],
                            'frame':'inertia',}

    h_stepsize = delta_t = t_vec[1] 

    model = csdl.Model()
    ode_surface_shapes = [(num_nodes, ) + item for item in surface_properties_dict['surface_shapes']]

    Pitching = PitchingModel(surface_names=['wing'], surface_shapes=[(nx,ny)], num_nodes=num_nodes,A=A, k=k,
                            v_inf=v_inf, c_0=chord, N_period=N_period, AR=span/chord)

    model.add(Pitching, 'pitching')
    model.add(UVLMSolver(num_times=num_nodes, h_stepsize=h_stepsize,states_dict=states_dict,
                                        surface_properties_dict=surface_properties_dict,mesh_val=None), 'uvlm_solver')

    model.add(EfficiencyModel(surface_names=['wing'],surface_shapes=ode_surface_shapes),name='EfficiencyModel')

    sim = python_csdl_backend.Simulator(model)
        
    t_start = time.time()
    sim.run()

    print('simulation time is', time.time() - t_start)

    alpha = A * np.cos(omega*t_vec)
    plt.plot(alpha[int(num_nodes/N_period):], sim['wing_C_L'][int(num_nodes/N_period):])
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$C_L$')
    # Check whether the specified path exists or not
    
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
    if save_results:
        np.savetxt('theodorsen/C_L_theodorsen'+str(k)+'.txt',sim['wing_C_L'][int(num_nodes/N_period):])
        np.savetxt('theodorsen/alpha_theodorsen'+str(k)+'.txt',alpha[int(num_nodes/N_period):])
        plt.savefig('theodorsen/C_L_theodorsen'+str(k)+'.png',dpi=300,transparent=True)

    if save_vtk:
        run_visualization(sim,h_stepsize,folder_name='theodorsen_verfi')
    del sim


if __name__ == '__main__':
    import python_csdl_backend
    import csdl
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    import matplotlib.pyplot as plt

    run_pitching_theodorsen_verification(k=0.2,num_nodes=200,N_period=4,A=1,save_vtk=False)
    run_pitching_theodorsen_verification(k=0.6,num_nodes=200,N_period=4,A=1,save_vtk=False)
    run_pitching_theodorsen_verification(k=1,num_nodes=200,N_period=4,A=1,save_vtk=False)
    run_pitching_theodorsen_verification(k=3,num_nodes=200,N_period=4,A=1,save_vtk=False)
    plt.savefig('theodorsen/C_L_theodorsen_nx_31.png',dpi=400,transparent=True)
    plt.show()
