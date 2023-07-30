'''Example 5 : fish kinematic optimization'''

from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver_eel_1 import UVLMSolver
from VAST.utils.make_video_vedo import make_video as make_video_vedo
import time
import numpy as np
import resource
import csdl

from visualization import run_visualization
run_optimizaton=0


from VAST.core.submodels.actuation_submodels.eel_actuation_model import EelActuationModel



from VAST.core.submodels.friction_submodels.eel_viscous_force import EelViscousModel

########################################
# define mesh resolution and num_nodes
########################################
before_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# nx = 12; ny = 3
nx = 15; ny = 3
num_nodes = 40;  
nt = num_nodes

# kinematics variables
v_inf = 0.38467351
lambda_ = 1
N_period= 2         
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
v_x = model.create_input('v_x', val=v_inf)
#v_x = model.create_input('v_x', val=0.35)
tail_amplitude = model.create_input('tail_amplitude', val=A)
tail_frequency = model.create_input('tail_frequency', val=f)
wave_number = model.create_input('wave_number', val=lambda_)
linear_relation = model.create_input('linear_relation', val=0.03125)
# v_x = model.create_input('v_x', val=0.8467351)
u = model.register_output('u', csdl.expand(v_x,shape=(num_nodes,1)))
density = model.create_input('density',val=np.ones((num_nodes,1))*997)



surface_names = list(surface_properties_dict.keys())
surface_shapes = list(surface_properties_dict.values())
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

model.add(UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,n_period=N_period,
                                    surface_properties_dict=surface_properties_dict), 'fish_model')

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
# exit()
# np.sum(thrust)/(0.5*np.reshape(997,(1,))*v_x**2*0.13826040386294708)/num_nodes 

# make_video_vedo(ssurface_properties_dict,num_nodes, sim)

# sim.compute_total_derivatives()
# exit()

# panel_forces = sim['panel_forces_all']

if run_optimizaton:

    #####################
    # optimizaton
    #####################
    from modopt.csdl_library import CSDLProblem

    from modopt.scipy_library import SLSQP
    from modopt.snopt_library import SNOPT
    # Define problem for the optimization
    prob = CSDLProblem(
        problem_name='ee',
        simulator=sim,
    )
    # optimizer = SLSQP(prob, maxiter=1)
    optimizer = SNOPT(
        prob, 
        Major_iterations=10,
        # Major_optimality=1e-6,
        Major_optimality=1e-9,
        Major_feasibility=1e-9,
        append2file=True,
        Major_step_limit=.25,
    )

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

import pyvista as pv

# # def save_fish_mesh():
# surface = sim['eel']
# vel = sim['eel_kinematic_vel'].copy()
# # forces = sim['eel_forces'].copy()
# vel[:,:,0] = 0
# # vel[:,:,2] = 0
# # vel[:,:,1] = np.arange(100)
# for i in range(num_nodes):
#     x = surface[i,:,:,0]
#     y = surface[i,:,:,1]
#     z = surface[i,:,:,2]
#     grid = pv.StructuredGrid(x,y,z)
#     grid.cell_data.set_vectors(np.swapaxes(vel[i].reshape(nx-1,ny-1,3), 0,1).reshape(-1,3),'normal_kinematic_vel')
#     # grid.cell_data.set_vectors(vel[i].reshape(-1,3),'test')
#     # print(vel[i].reshape(-1,3)[:50,1])
#     # print(vel[i].reshape(-1,3)[:50,1] - vel[i].reshape(-1,3)[50:,1])
#     # grid.cell_data.set_vectors(vel[i].reshape(-1,3),'test')
#     grid.save(filename='fish_vtk/fish_'+str(i)+'.vtk')
# #     return grid

# # grid = save_fish_mesh()
