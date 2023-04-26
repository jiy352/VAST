import matplotlib.pyplot as plt
import numpy as np
import time

import pyvista as pv

from panel_method.utils.generate_eel_vlm_mesh import (
    generate_eel_carling_vlm,)

from panel_method.mesh_preprocessing.generate_kinematics import eel_kinematics


from lsdo_uvlm.examples.simple_wing_constant_aoa_sls_outputs.plunging_system_free import ODESystemModel
from lsdo_uvlm.examples.simple_wing_constant_aoa_sls_outputs.plunging_profile_outputs import ProfileSystemModel
from ozone.api import ODEProblem
import csdl

from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show
num_nodes = 10
u_val = np.ones((num_nodes, 1))*0.4
AcStates_val_dict = {
    'u': u_val,
    'v': np.zeros((num_nodes, 1)),
    'w': np.zeros((num_nodes, 1)),
    'p': np.zeros((num_nodes, 1)),
    'q': np.zeros((num_nodes, 1)),
    'r': np.zeros((num_nodes, 1)),
    'theta': np.ones((num_nodes, 1))*0,
    'psi': np.zeros((num_nodes, 1)),
    'x': np.zeros((num_nodes, 1)),
    'y': np.zeros((num_nodes, 1)),
    'z': np.zeros((num_nodes, 1)),
    'phiw': np.zeros((num_nodes, 1)),
    'gamma': np.zeros((num_nodes, 1)),
    'psiw': np.zeros((num_nodes, 1)),
}

class ODEProblemTest(ODEProblem):

    
    def setup(self):
        # surface_name = self.parameter['surface_name']
        # surface_shapes = self.parameter['surface_shapes']
        ####################################
        # ode parameter names
        ####################################        
        self.add_parameter('u', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('v', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('w', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('p', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('q', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('r', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('theta',dynamic=True, shape=(self.num_times, 1))
        # self.add_parameter('x',dynamic=True, shape=(self.num_times, 1))
        # self.add_parameter('y',dynamic=True, shape=(self.num_times, 1))
        # self.add_parameter('z',dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('psi',dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('gamma',dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('psiw',dynamic=True, shape=(self.num_times, 1))

        gamma_w_name_list = []
        wing_wake_coords_name_list = []

        for i in range(len(surface_names)):
            # print('surface_names in run.py',surface_names)
            # print('surface_shapes in run.py',surface_shapes)
            ####################################
            # ode parameter names
            ####################################
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            nx = surface_shape[0]
            ny = surface_shape[1]
            self.add_parameter(surface_name,
                               dynamic=True,
                               shape=(self.num_times, nx, ny, 3))
            ####################################
            # ode states names
            ####################################
            gamma_w_name = surface_name + '_gamma_w'
            wing_wake_coords_name = surface_name + '_wake_coords'
            # gamma_w_name_list.append(gamma_w_name)
            # wing_wake_coords_name_list.append(wing_wake_coords_name)
            # Inputs names correspond to respective upstream CSDL variables
            ####################################
            # ode outputs names
            ####################################
            dgammaw_dt_name = surface_name + '_dgammaw_dt'
            dwake_coords_dt_name = surface_name + '_dwake_coords_dt'
            ####################################
            # IC names
            ####################################
            gamma_w_0_name = surface_name + '_gamma_w_0'
            wake_coords_0_name = surface_name + '_wake_coords_0'
            ####################################
            # states and outputs names
            ####################################
            gamma_w_int_name = surface_name + '_gamma_w_int'
            wake_coords_int_name = surface_name + '_wake_coords_int'
            self.add_state(gamma_w_name,
                           dgammaw_dt_name,
                           shape=(nt - 1, ny - 1),
                           initial_condition_name=gamma_w_0_name,
                           output=gamma_w_int_name)
            self.add_state(wing_wake_coords_name,
                           dwake_coords_dt_name,
                           shape=(nt - 1, ny, 3),
                           initial_condition_name=wake_coords_0_name,
                           output=wake_coords_int_name)

            ####################################
            # profile outputs
            ####################################
            # F_name = surface_name + '_F'
            # self.add_profile_output(F_name)
        

        self.add_times(step_vector='h')

        # Define ODE and Profile Output systems (Either CSDL Model or Native System)
        self.set_ode_system(ODESystemModel)



class RunModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_times')
        self.parameters.declare('h_stepsize')
        self.parameters.declare('surface_names')
        self.parameters.declare('surface_shapes')
        self.parameters.declare('mesh_val')

    def define(self):
        num_times = self.parameters['num_times']

        h_stepsize = self.parameters['h_stepsize']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        mesh_val = self.parameters['mesh_val']

        nt = num_times+1


        ####################################
        # Create parameters
        ####################################
        for data in AcStates_val_dict:
            string_name = data
            val = AcStates_val_dict[data]            
            # print('{:15} = {},shape{}'.format(string_name, val, val.shape))

            variable = self.create_input(string_name,
                                         val=val)

        initial_mesh_names = [
            x + '_initial_mesh' for x in surface_names
        ]

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            gamma_w_0_name = surface_name + '_gamma_w_0'
            wake_coords_0_name = surface_name + '_wake_coords_0'
            surface_shape = surface_shapes[i]
            nx = surface_shape[0]
            ny = surface_shape[1]
            ####################################
            # Create parameters
            ####################################
            '''1. wing'''
            wing_val = mesh_val
            wing = self.create_input(surface_name, wing_val)

            ########################################
            # Initial condition for states
            ########################################
            '''1. wing_gamma_w_0'''
            wing_gamma_w_0 = self.create_input(gamma_w_0_name, np.zeros((nt - 1, ny - 1)))

            '''2. wing_wake_coords_0'''
            wing_wake_coords_0_val = np.zeros((nt - 1, ny, 3))
            wing_wake_coords_0 = self.create_input(wake_coords_0_name, wing_wake_coords_0_val)

        ########################################
        # Timestep vector
        ########################################
        h_vec = np.ones(num_times - 1) * h_stepsize
        h = self.create_input('h', h_vec)
        ########################################
        # params_dict to the init of ODESystem
        ########################################
        params_dict = {
            'surface_names': surface_names,
            'surface_shapes': surface_shapes,
            'delta_t': h_stepsize,
            'nt': nt,
        }

        profile_params_dict = {
            'num_nodes': nt-1,
            'surface_names': surface_names,
            'surface_shapes': surface_shapes,
            'delta_t': h_stepsize,
            'nt': nt,
        }

        # add an actuation model on the upstream
        # self.add(ActuationModel(surface_names=surface_names, surface_shapes=surface_shapes, num_nodes=nt-1),'actuation_temp')

        # Create Model containing integrator
        # ODEProblem = ODEProblemTest('ForwardEuler', 'time-marching', num_times, display='default', visualization='None')
        ODEProblem = ODEProblemTest('ForwardEuler', 'time-marching checkpointing', num_times, display='default', visualization='None')

        self.add(ODEProblem.create_solver_model(ODE_parameters=params_dict), 'subgroup')
        self.add(ProfileSystemModel(**profile_params_dict),'profile_outputs')
        self.add_design_variable('u',lower=1e-3, upper=10)
        # self.add_objective('res')



