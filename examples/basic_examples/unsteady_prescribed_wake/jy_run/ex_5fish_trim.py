'''Example 5 : fish kinematic optimization'''

from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver import UVLMSolver
from VAST.utils.make_video_vedo import make_video as make_video_vedo
import time
import numpy as np
import resource
import csdl

# from visualization import run_visualization
run_optimizaton=0

from VAST.core.submodels.actuation_submodels.eel_actuation_model import EelActuationModel

from VAST.core.submodels.friction_submodels.eel_viscous_force import EelViscousModel
from VAST.core.submodels.output_submodels.vlm_post_processing.efficiency import EfficiencyModel
from VAST.utils.visualization import run_visualization
########################################
# define mesh resolution and num_nodes
########################################
before_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def run_fish(v_inf):
    # nx = 12; ny = 3
    nx = 41; ny = 5
    num_nodes = 100;  
    nt = num_nodes

    # kinematics variables
    # v_inf = 0.38467351

    lambda_ = 1
    N_period= 2         
    st = 0.15
    A = 0.125
    f = 0.48
    surface_properties_dict = {'surface_names':['eel'],
                                'surface_shapes':[(nx, ny, 3)],
                            'frame':'wing_fixed',}

    u_val = (np.ones(num_nodes)).reshape((num_nodes,1)) * v_inf
    w_vel = np.zeros((num_nodes, 1))

    alpha_equ = np.arctan2(w_vel, u_val)

    states_dict = {
        'v': np.zeros((num_nodes, 1)), 'w': w_vel,
        'p': np.zeros((num_nodes, 1)), 'q': np.zeros((num_nodes, 1)), 'r': np.zeros((num_nodes, 1)),
        'theta': alpha_equ, 'psi': np.zeros((num_nodes, 1)),
        'x': np.zeros((num_nodes, 1)), 'y': np.zeros((num_nodes, 1)), 'z': np.zeros((num_nodes, 1)),
        'phiw': np.zeros((num_nodes, 1)), 'gamma': np.zeros((num_nodes, 1)),'psiw': np.zeros((num_nodes, 1)),
    }
    t_vec = np.linspace(0,N_period/f,num_nodes)
    h_stepsize = t_vec[1]

    #####################
    # define the problem
    #####################
    import python_csdl_backend
    model = csdl.Model()
    v_x = model.create_input('v_x', val=v_inf)
    #v_x = model.create_input('v_x', val=0.35)
    tail_amplitude = model.create_input('tail_amplitude', val=A)
    tail_frequency = model.create_input('tail_frequency', val=f)
    wave_number = model.create_input('wave_number', val=lambda_)
    linear_relation = model.create_input('linear_relation', val=0.03125)
    # v_x = model.create_input('v_x', val=0.8467351)
    u = model.register_output('u', csdl.expand(v_x,shape=(num_nodes,1)))
    density = model.create_input('density',val=np.ones((num_nodes,1))*997)



    surface_names = surface_properties_dict['surface_names']
    surface_shapes = surface_properties_dict['surface_shapes']
    ode_surface_shapes = [(num_nodes, ) + item for item in surface_shapes]

    s_1_ind = 3
    s_2_ind = None
    if s_2_ind==None:
        s_2_ind = int(ode_surface_shapes[0][1]-2)

    model.add(EelViscousModel(),name='EelViscousModel')

    model.add(EelActuationModel(surface_names=surface_names,
                                surface_shapes=ode_surface_shapes,
                                n_period=N_period,
                                s_1_ind=s_1_ind,
                                s_2_ind=s_2_ind,
                                ),name='EelActuationModel')

    model.add(UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                        surface_properties_dict=surface_properties_dict), 'fish_model')
    model.add(EfficiencyModel(surface_names=surface_names, surface_shapes=ode_surface_shapes),name='EfficiencyModel')
    model.add_design_variable('v_x',upper=0.8,lower=0.05)
    # model.add_design_variable('tail_amplitude',upper=0.2,lower=0.05)
    # model.add_design_variable('tail_frequency',upper=0.5,lower=0.2)
    # model.add_design_variable('wave_number',upper=2,lower=1)
    # model.add_design_variable('linear_relation',upper=0.03125*3,lower=0.03125*0.5)
    thrust = model.declare_variable('thrust',shape=(num_nodes,1))
    C_F = model.declare_variable('C_F')
    area = model.declare_variable('eel_s_panel',shape=(num_nodes,int((nx-1)*(ny-1))))
    avg_area = csdl.sum(area)/num_nodes
    avg_C_T = -csdl.sum(thrust)/(0.5*csdl.reshape(density[0,0],(1,))*v_x**2*avg_area)/num_nodes
    model.register_output('avg_C_T', avg_C_T)
    thrust_coeff_avr = (avg_C_T - C_F)**2

    model.register_output('thrust_coeff_avr', thrust_coeff_avr)
    # model.add_constraint('thrust_coeff_avr',equals=0.)
    model.add_objective('thrust_coeff_avr',scaler=1e3)

    sim = python_csdl_backend.Simulator(model)
        
    t_start = time.time()
    sim.run()

    return sim

v_inf = np.array([0.5])

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt


sim_list = [None]*len(v_inf)
efficiency = np.zeros(len(v_inf))   
thrust_power = np.zeros(len(v_inf))   
panel_thrust_power = np.zeros(len(v_inf))   
for i in range(len(v_inf)):
    sim_list[i] = run_fish(v_inf[i])  
    efficiency[i] = sim_list[i]['efficiency']
    thrust_power[i] = sim_list[i]['thrust_power']
    panel_thrust_power[i] = sim_list[i]['panel_thrust_power']

plt.plot(v_inf,efficiency,'.')
h_stepsize = 0.04208754
run_visualization(['eel'], sim_list[0], h_stepsize,folder_name='fish_new_vc',filename='fish')