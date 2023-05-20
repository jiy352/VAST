'''Example 5 : fish kinematic optimization'''

from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver_eel_1 import UVLMSolver
from VAST.utils.make_video_vedo import make_video as make_video_vedo
import time
import numpy as np
import resource
import csdl

########################################
# define mesh resolution and num_nodes
########################################
before_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

nx = 15; ny = 5
num_nodes = 20;  
nt = num_nodes

# kinematics variables
v_inf = 0.38467351
lambda_ = 1
N_period= 1         
st = 0.15
A = 0.125
f = 0.48
surface_properties_dict = {'eel':(nx,ny,3)}

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
v_x = model.create_input('v_x', val=0.35)
# v_x = model.create_input('v_x', val=0.8467351)
u = model.register_output('u', csdl.expand(v_x,shape=(num_nodes,1)))

model.add(UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,n_period=N_period,
                                    surface_properties_dict=surface_properties_dict), 'fish_model')

model.add_design_variable('v_x',upper=0.6,lower=0.1)
thrust = model.declare_variable('thrust',shape=(num_nodes,1))
C_F = model.declare_variable('C_F')
density = model.declare_variable('density',shape=(num_nodes,1))

thrust_coeff_avr = (csdl.sum(thrust)/(0.5*csdl.reshape(density[0,0],(1,))*v_x**2*0.13826040386294708)/num_nodes - C_F)**2


model.register_output('thrust_coeff_avr', thrust_coeff_avr)
model.add_objective('thrust_coeff_avr')

sim = python_csdl_backend.Simulator(model)
    
t_start = time.time()
sim.run()
# make_video_vedo(surface_properties_dict, num_nodes, sim,xrange=(-0.5,3),yrange=(-0.3,0.3),zrange=(-0.5,0.5))

exit()

# panel_forces = sim['panel_forces_all']




# exit()

#####################
# optimizaton
#####################
from modopt.csdl_library import CSDLProblem

from modopt.scipy_library import SLSQP
from modopt.snopt_library import SNOPT
# Define problem for the optimization
prob = CSDLProblem(
    problem_name='eel_kinematic_opt',
    simulator=sim,
)
optimizer = SLSQP(prob, maxiter=1)
# optimizer = SNOPT(
#     prob, 
#     Major_iterations=100,
#     # Major_optimality=1e-6,
#     Major_optimality=1e-9,
#     Major_feasibility=1e-9,
#     append2file=True,
#     # Major_step_limit=.25,
# )

optimizer.solve()
optimizer.print_results(summary_table=True)
print('total time is', time.time() - t_start)

#####################
# memory usage
#####################
after_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

ALLOCATED_MEMORY = (after_mem - before_mem)/(1024**3) # for mac
# ALLOCATED_MEMORY = (after_mem - before_mem)*1e-6 # for linux

print('Allocated memory: ', ALLOCATED_MEMORY, 'Gib')


thrust = np.sum(sim['thrust'])
v_x = sim['v_x']

# Force_sum = thrust

effi = thrust*v_x/np.sum(sim['panel_power'])