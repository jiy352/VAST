'''Example 5 : fish kinematic optimization'''

from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver import UVLMSolver
from VAST.utils.make_video_vedo import make_video as make_video_vedo
import time
import numpy as np
import resource
import csdl

# from visualization import run_visualization
run_optimizaton=0

from VAST.core.submodels.actuation_submodels.eel_actuation_model_simple import EelActuationModel

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
    num_nodes = 70;  
    nt = num_nodes

    # kinematics variables
    # v_inf = 0.38467351

    lambda_ = 2
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

    s_1_ind = 7
    # s_1_ind = 3
    s_2_ind = None
    if s_2_ind==None:
        s_2_ind = int(ode_surface_shapes[0][1]-5)
        # s_2_ind = int(ode_surface_shapes[0][1]-3)

    model.add(EelViscousModel(),name='EelViscousModel')

    model.add(EelActuationModel(surface_names=surface_names,
                                surface_shapes=ode_surface_shapes,
                                n_period=N_period,
                                s_1_ind=s_1_ind,
                                s_2_ind=s_2_ind,
                                ),name='EelActuationModel')

    model.add(UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                        surface_properties_dict=surface_properties_dict), 'fish_model')
    model.add(EfficiencyModel(surface_names=surface_names, surface_shapes=ode_surface_shapes,n_ignore=int(num_nodes/N_period)),name='EfficiencyModel')
    # model.add_design_variable('v_x',upper=0.8,lower=0.5)
    model.add_design_variable('v_x',upper=0.2,lower=0.2)
    # '''
    if True:
        # model.add_design_variable('tail_amplitude',upper=0.2,lower=0.05)
        # model.add_design_variable('tail_frequency',upper=0.6,lower=0.2)
        # model.add_design_variable('wave_number',upper=2,lower=0.5)

        model.add_design_variable('tail_amplitude',upper=0.4,lower=0.05)
        model.add_design_variable('tail_frequency',upper=0.4,lower=0.05)
        # model.add_design_variable('wave_number',upper=2,lower=1.9)
        # model.add_design_variable('linear_relation',upper=0.03125*3,lower=0.03125*0.5)

        model.print_var(tail_amplitude)
        model.print_var(tail_frequency)
        model.print_var(wave_number)
        model.print_var(linear_relation)

    # '''
    thrust = model.declare_variable('thrust',shape=(num_nodes,1))
    C_F = model.declare_variable('C_F')
    area = model.declare_variable('eel_s_panel',shape=(num_nodes,int((nx-1)*(ny-1))))
    avg_area = csdl.sum(area)/num_nodes
    avg_C_T = -csdl.sum(thrust)/(0.5*csdl.reshape(density[0,0],(1,))*v_x**2*avg_area)/num_nodes
    model.register_output('avg_C_T', avg_C_T)
    thrust_coeff_avr = (avg_C_T - C_F)**2
    ############################################################################################
    # eel_kinematic_vel = model.declare_variable('eel_kinematic_vel',shape=(num_nodes,int((nx-1)*(ny-1)),3))
    # model.print_var(eel_kinematic_vel+0)
    ############################################################################################

    model.register_output('thrust_coeff_avr', thrust_coeff_avr)
    if True:
        model.add_constraint('thrust_coeff_avr',equals=0.)
        model.add_objective('efficiency',scaler=-1)
    else:
        model.add_objective('thrust_coeff_avr',scaler=1e3)

    sim = python_csdl_backend.Simulator(model)
        
    t_start = time.time()
    sim.run()

    return sim

v_inf = np.array([4.164472e-01])
# v_inf = np.array([0.1])

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt

import time
t_start = time.time()
sim_list = [None]*len(v_inf)
efficiency = np.zeros(len(v_inf))   
thrust_power = np.zeros(len(v_inf))   
panel_thrust_power = np.zeros(len(v_inf))   
for i in range(len(v_inf)):
    sim_list[i] = run_fish(v_inf[i])  
    efficiency[i] = sim_list[i]['efficiency']
    thrust_power[i] = sim_list[i]['thrust_power']
    panel_thrust_power[i] = sim_list[i]['panel_thrust_power']
print('simulation time is', time.time() - t_start)
plt.plot(v_inf,efficiency,'.')
h_stepsize = 0.04208754
run_visualization(['eel'], sim_list[0], h_stepsize,folder_name='fish_new_vc',filename='fish')


sim = sim_list[0]

total_fx = np.sum(sim['panel_forces_x'], axis=1)
thrust = sim['thrust']


print('percentage of thrust C_F\n',(-np.average(sim['eel_C_D_i'])-sim['C_F'])* 100/sim['C_F'],'%')

# exit()

#####################
# optimizaton
#####################
from modopt.csdl_library import CSDLProblem

from modopt.scipy_library import SLSQP
from modopt.snopt_library import SNOPT
# Define problem for the optimization
prob = CSDLProblem(
    problem_name='eel_kinematic_opt_lv',
    simulator=sim,
)
# optimizer = SLSQP(prob, maxiter=1)
optimizer = SNOPT(
    prob, 
    Major_iterations=30,
    # Major_optimality=1e-6,
    Major_optimality=1e-7,
    # Major_feasibility=1e-5,
    Major_feasibility=1e-3,
    append2file=True,
    Major_step_limit=.1,
)

optimizer.solve()
optimizer.print_results(summary_table=True)
# print('total time is', time.time() - t_start)

print('v_x is',sim['v_x'])
print('efficiency is',sim['efficiency'])
print('tail amplitude is',sim['tail_amplitude'])
print('tail frequency is',sim['tail_frequency'])
print('wave number is',sim['wave_number'])
print('strouhal number is',sim['tail_amplitude']*sim['tail_frequency']*2/sim['v_x'])
print('percentage of thrust C_F\n',(-np.average(sim['eel_C_D_i'])-sim['C_F'])* 100/sim['C_F'],'%')
