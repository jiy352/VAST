import numpy as np

from VAST.utils.generate_mesh import *

A = 5
v_inf = 1
c_0 = 1
k = [0.1,0.3,0.5]
N_period = 4
num_nodes = 80





import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt
for k_i in k:
    T = 2*np.pi/(2*v_inf*k_i/c_0)
    t = np.linspace(0,N_period*T,num_nodes)
    ###########################################
    f = A * np.cos(2*v_inf*k_i/c_0*t)
    ###########################################
    plt.plot(t,f)
    plt.legend(['k=0.1','k=0.3','k=0.5'])
plt.ylim([-5,5])
plt.yticks(np.arange(-5,5.1,1))
plt.xlabel('time')
plt.ylabel('piching angle')
# plt.show()

# generate initial mesh
nx = 5; ny = 15
chord = 1; span = 6

mesh_dict = {"num_y": ny, "num_x": nx, "wing_type": "rect",  "symmetry": False,
                 "span": span, "root_chord": chord,"span_cos_spacing": False, "chord_cos_spacing": False}
mesh = generate_mesh(mesh_dict)

# visualize mesh
import pyvista as pv
for i in range(1):
    x = mesh[:,:,0] + 0.5
    y = mesh[:,:,1]
    z = mesh[:,:,2]

    grid = pv.StructuredGrid(x,y,z)
    grid.plot(show_edges=True, show_grid=True)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html

