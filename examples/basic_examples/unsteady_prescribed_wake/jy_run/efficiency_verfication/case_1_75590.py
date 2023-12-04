'''Example 3 : verification of prescibed vlm with Theodorsen solution'''
from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver import UVLMSolver
from VAST.core.submodels.actuation_submodels.anderson_actuation import PitchingModel
import time
import numpy as np
from VAST.core.submodels.output_submodels.vlm_post_processing.efficiency import EfficiencyModel
from VAST.utils.visualization import run_visualization
import os

########################################
# This is a test case to check the prescribed wake solver
# pitching and plunging efficiency with Anderson paper
# "Oscillating foils of high propulsive efficiency"
# kinematic parameters are 
# h_0/c = 0.75, alpha_0 = 5 deg, Psi = 90 deg (pitch leading heave)
########################################


def run_anderson_verfication(St,h_0_star, num_nodes,N_period,alpha_0_deg,save_results=False, save_vtk=False, path = "verfication_data/theodorsen"):
    '''
    This is a test case to check the prescribed wake solver against the Theodorsen solution
    with a wing pitching sinusoidally from 1 to -1 deg and the reduced frequency k = [0.2, 0.6, 1, 3]
    '''
    ########################################
    # 1. define geometry
    ########################################

    nx = 11; ny = 7         # num_pts in chordwise and spanwise direction
    chord = 1; span = 6   # chord and span of the wing
    b = chord/2

    omega = 1
    h_0 = h_0_star * chord
    v_inf = omega*h_0 / (np.pi * St)
    k = omega * b/v_inf
    T = 2*np.pi/(omega)

    theta_0 = np.rad2deg(np.arctan2(omega*h_0, v_inf) - np.deg2rad(alpha_0_deg))

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
                            'frame':'wing_fixed',}

    h_stepsize = delta_t = t_vec[1] 

    model = csdl.Model()
    ode_surface_shapes = [(num_nodes, ) + item for item in surface_properties_dict['surface_shapes']]

    Pitching = PitchingModel(surface_names=['wing'], surface_shapes=[(nx,ny)], num_nodes=num_nodes,A=theta_0, k=k,
                            v_inf=v_inf, c_0=chord, N_period=N_period, AR=span/chord,h_0=h_0)

    model.add(Pitching, 'pitching')
    model.add(UVLMSolver(num_times=num_nodes, h_stepsize=h_stepsize,states_dict=states_dict,
                                        surface_properties_dict=surface_properties_dict,mesh_val=None), 'uvlm_solver')

    model.add(EfficiencyModel(surface_names=['wing'],surface_shapes=ode_surface_shapes,n_ignore=int(num_nodes/N_period)),name='EfficiencyModel')

    sim = python_csdl_backend.Simulator(model)
        
    t_start = time.time()
    sim.run()

    print('simulation time is', time.time() - t_start)

    alpha = theta_0 * np.cos(omega*t_vec)
    plt.plot(alpha[int(num_nodes/N_period):], sim['wing_C_L'][int(num_nodes/N_period):])
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$C_L$')
    # Check whether the specified path exists or not
    
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
    if save_results:
        np.savetxt('verfication_data/theodorsen/C_L_theodorsen'+str(k)+'.txt',sim['wing_C_L'][int(num_nodes/N_period):])
        np.savetxt('verfication_data/theodorsen/alpha_theodorsen'+str(k)+'.txt',alpha[int(num_nodes/N_period):])
        plt.savefig('verfication_data/theodorsen/C_L_theodorsen'+str(k)+'.png',dpi=300,transparent=True)

    if save_vtk:
        run_visualization(sim,h_stepsize,folder_name='theodorsen_verfi')
    return sim


if __name__ == '__main__':
    import python_csdl_backend
    import csdl
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    import matplotlib.pyplot as plt

    num_nodes = 150
    N_period=3
    alpha_0_deg = 5

    

    sim = run_anderson_verfication(St=0.4,h_0_star=0.75, num_nodes=num_nodes,N_period=N_period,alpha_0_deg=alpha_0_deg,save_vtk=False)

    alpha_file_name = '../../verfication_data/theodorsen/analytical_solution/alpha_1deg0.2.txt'
    Cl_file_name = [
                    # '../../verfication_data/theodorsen/analytical_solution/Cl_1deg0.2.txt',
    #                 '../../verfication_data/theodorsen/analytical_solution/Cl_1deg0.6.txt',
                    '../../verfication_data/theodorsen/analytical_solution/Cl_1deg1.txt',
                    # '../../verfication_data/theodorsen/analytical_solution/Cl_1deg3.txt'
                    ]

    def plot_cl(alpha_file_name, Cl_file_name, L_file_name=None):
        for i in range(len(Cl_file_name)):
            alpha = np.loadtxt(alpha_file_name)
            if alpha.max()<0.2:
                alpha = np.rad2deg(alpha)
            Cl = np.loadtxt(Cl_file_name[i])
            print('i', Cl_file_name[i])

            if L_file_name is not None:
                L = np.loadtxt(L_file_name[i])
            else:
                L = None
            if np.loadtxt(alpha_file_name).max()>0.2:
                plt.plot(alpha, Cl,'--')
            else:
                plt.plot(alpha, Cl,'--')

    plot_cl(alpha_file_name, Cl_file_name)
    plt.legend(['VAST k = 0.2', 'VAST k = 0.6', 'VAST k = 1', 'VAST k = 3',
                'Theodorsen k = 0.2', 'Theodorsen k = 0.6', 'Theodorsen k = 1', 'Theodorsen k = 3'])
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$C_l$')
    plt.savefig('verfication_data/theodorsen/C_L_theodorsen_vs_analytical.png',dpi=400,transparent=True)
    plt.show()

