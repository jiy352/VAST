'''Example 5 : fish kinematic optimization'''

from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver import UVLMSolver
from VAST.utils.make_video_vedo import make_video as make_video_vedo
import time
import numpy as np
import csdl

from VAST.core.submodels.actuation_submodels.eel_actuation_model_simple import EelActuationModel

from VAST.core.submodels.friction_submodels.eel_viscous_force import EelViscousModel
from VAST.core.submodels.output_submodels.vlm_post_processing.efficiency import EfficiencyModel
from VAST.utils.visualization import run_visualization


###################################################
# 1. Inputs: define mesh resolution, num_time_steps
# and kinematics variables
###################################################

# number of nodes in fish longitudinal and lateral direction (number of panels = (nx-1)*(ny-1))
nx = 41; ny = 5 
num_time_steps = 70;  
nt = num_time_steps

# kinematics variables
v_inf = 0.4 # forward velocity of the fish
lambda_ = 1 # wave number 
N_period= 2 # total number of periods in the simulation (t_total = N_period * T)        
A = 0.125   # amplitude of the tail oscillation 
f = 0.48    # frequency of the tail oscillation

# this dictionary is used to define the geometry that goes into the UVLMSolver
surface_properties_dict = {'surface_names':['eel'], 'surface_shapes':[(nx, ny, 3)], 'frame':'wing_fixed',}

u_val = (np.ones(num_time_steps)).reshape((num_time_steps,1)) * v_inf
w_vel = np.zeros((num_time_steps, 1))

alpha_equ = np.arctan2(w_vel, u_val)

# this dictionary is used to define the states into the UVLMSolver
states_dict = {
    'v': np.zeros((num_time_steps, 1)), 'w': w_vel,
    'p': np.zeros((num_time_steps, 1)), 'q': np.zeros((num_time_steps, 1)), 'r': np.zeros((num_time_steps, 1)),
    'theta': alpha_equ, 'psi': np.zeros((num_time_steps, 1)),
    'x': np.zeros((num_time_steps, 1)), 'y': np.zeros((num_time_steps, 1)), 'z': np.zeros((num_time_steps, 1)),
    'phiw': np.zeros((num_time_steps, 1)), 'gamma': np.zeros((num_time_steps, 1)),'psiw': np.zeros((num_time_steps, 1)),
}
t_vec = np.linspace(0,N_period/f,num_time_steps)
h_stepsize = t_vec[1]

##########################################
# 2. define the simulation model
##########################################
import python_csdl_backend
fish_fluid_model = csdl.Model()
v_x = fish_fluid_model.create_input('v_x', val=v_inf)
tail_amplitude = fish_fluid_model.create_input('tail_amplitude', val=A)
tail_frequency = fish_fluid_model.create_input('tail_frequency', val=f)
wave_number = fish_fluid_model.create_input('wave_number', val=lambda_)
# amplitue growth rate
linear_relation = fish_fluid_model.create_input('linear_relation', val=0.03125)
# x-direction velocity
u = fish_fluid_model.register_output('u', csdl.expand(v_x,shape=(num_time_steps,1)))
# density of the fluid (water)
density = fish_fluid_model.create_input('density',val=np.ones((num_time_steps,1))*997)

surface_names = surface_properties_dict['surface_names']
surface_shapes = surface_properties_dict['surface_shapes']
ode_surface_shapes = [(num_time_steps, ) + item for item in surface_shapes]

# s_1_ind is the number of nodes in the head region (s_1_ind = 7 means 7 nodes in the head region)
# s_2_ind is the number of nodes in the tail region (s_2_ind = 3 means 3 nodes in the tail region)
# rest of the nodes are in the body region (nx-1-s_1_ind-s_2_ind)
s_1_ind = 7
s_2_ind = None
if s_2_ind==None:
    s_2_ind = int(ode_surface_shapes[0][1]-5)

fish_fluid_model.add(EelViscousModel(),name='EelViscousModel')

fish_fluid_model.add(EelActuationModel(surface_names=surface_names,
                            surface_shapes=ode_surface_shapes,
                            n_period=N_period,
                            s_1_ind=s_1_ind,
                            s_2_ind=s_2_ind,
                            ),name='EelActuationModel')

fish_fluid_model.add(UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                    surface_properties_dict=surface_properties_dict), 'fish_model')
fish_fluid_model.add(EfficiencyModel(surface_names=surface_names, surface_shapes=ode_surface_shapes,n_ignore=int(num_time_steps/N_period)),name='EfficiencyModel')

thrust = fish_fluid_model.declare_variable('thrust',shape=(num_time_steps,1))
C_F = fish_fluid_model.declare_variable('C_F')
area = fish_fluid_model.declare_variable('eel_s_panel',shape=(num_time_steps,int((nx-1)*(ny-1))))
avg_area = csdl.sum(area)/num_time_steps
avg_C_T = -csdl.sum(thrust)/(0.5*csdl.reshape(density[0,0],(1,))*v_x**2*avg_area)/num_time_steps
fish_fluid_model.register_output('avg_C_T', avg_C_T)
thrust_coeff_avr = (avg_C_T - C_F)**2
############################################################################################
# eel_kinematic_vel = fish_fluid_model.declare_variable('eel_kinematic_vel',shape=(num_time_steps,int((nx-1)*(ny-1)),3))
############################################################################################

# run the simulator
sim.run()
