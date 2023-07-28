from VAST.core.submodels.actuation_submodels.eel_actuation_model import EelActuationModel
import csdl

v_inf = 0.4
lambda_ = 1
st = 0.15
A = 0.125
f = st*v_inf/A 
# omg = 2*np.pi*f


import python_csdl_backend
import numpy as np
import pyvista as pv
# simulator_name = 'csdl_om'
num_nodes = 20
num_pts_chord = 13 # nx
num_pts_span = 3
N_period=1
# tail_amplitude = 0.125
# tail_frequency = 0.48
# vx = 0.38643524

tail_amplitude = 0.125
tail_frequency = 0.48
vx = 0.38108754

model = csdl.Model()

tail_amplitude = model.create_input('tail_amplitude', val=A)
tail_frequency = model.create_input('tail_frequency', val=f)
wave_number = model.create_input('wave_number', val=lambda_)
linear_relation = model.create_input('linear_relation', val=0.03125)



t_temp_val = np.linspace(0,N_period/0.48,num_nodes)

model.add(EelActuationModel(surface_names=['surface'],surface_shapes=[(num_nodes,num_pts_chord,num_pts_span)],n_period=N_period),'actuation')


sim = python_csdl_backend.Simulator(model)
sim.run()

surface = sim["surface"]
print(surface.shape)

for i in range(num_nodes):
    # x = surface[i,:,:,0] - t_temp_val[i]*0.38108754
    x = surface[i,:,:,0]# - t_temp_val[i]*vx
    y = surface[i,:,:,1]
    z = surface[i,:,:,2]

    grid = pv.StructuredGrid(x,y,z)
    # grid.plot()
    grid.save(f'eel_actuation_{i}.vtk')
    # filename=f'eel_actuation_{i}.vtk'