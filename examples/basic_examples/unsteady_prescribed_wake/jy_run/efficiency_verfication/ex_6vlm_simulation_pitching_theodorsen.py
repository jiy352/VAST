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

# TODO: there's always a memory leak when running this script
# need to contact backend developer to fix this issue, but for now it is not a big problem

# @profile
def run_pitching_theodorsen_verification(k,num_nodes,N_period,A=1,save_results=False, save_vtk=False, path = "verfication_data/theodorsen"):
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
        np.savetxt('verfication_data/theodorsen/C_L_theodorsen'+str(k)+'.txt',sim['wing_C_L'][int(num_nodes/N_period):])
        np.savetxt('verfication_data/theodorsen/alpha_theodorsen'+str(k)+'.txt',alpha[int(num_nodes/N_period):])
        plt.savefig('verfication_data/theodorsen/C_L_theodorsen'+str(k)+'.png',dpi=300,transparent=True)

    if save_vtk:
        run_visualization(sim,h_stepsize,folder_name='theodorsen_verfi')
    del sim


if __name__ == '__main__':
    import python_csdl_backend
    import csdl
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    import matplotlib.pyplot as plt

    num_nodes = 220
    N_period=3
    A=1

    run_pitching_theodorsen_verification(k=0.2,num_nodes=num_nodes,N_period=N_period,A=A,save_vtk=False)
    run_pitching_theodorsen_verification(k=0.6,num_nodes=num_nodes,N_period=N_period,A=A,save_vtk=False)
    run_pitching_theodorsen_verification(k=1,num_nodes=num_nodes,N_period=N_period,A=A,save_vtk=False)
    run_pitching_theodorsen_verification(k=3,num_nodes=num_nodes,N_period=N_period,A=A,save_vtk=False)
    alpha_file_name = 'verfication_data/theodorsen/analytical_solution/alpha_1deg0.2.txt'
    Cl_file_name = ['verfication_data/theodorsen/analytical_solution/Cl_1deg0.2.txt',
                    'verfication_data/theodorsen/analytical_solution/Cl_1deg0.6.txt',
                    'verfication_data/theodorsen/analytical_solution/Cl_1deg1.txt',
                    'verfication_data/theodorsen/analytical_solution/Cl_1deg3.txt']

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


# for name in all_variables:
#     # Print the item if it doesn't start with '__'
#     if not name.startswith('__'):
#         myvalue = eval(name)
#         print(name, "is", type(myvalue), "and is equal to ", myvalue)

# import sys
# def sizeof_fmt(num, suffix='B'):
#     ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
#     for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
#         if abs(num) < 1024.0:
#             return "%3.1f %s%s" % (num, unit, suffix)
#         num /= 1024.0
#     return "%.1f %s%s" % (num, 'Yi', suffix)

# for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
#                           locals().items())), key= lambda x: -x[1])[:10]:
#     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))