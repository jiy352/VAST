import numpy as np
import matplotlib.pyplot as plt

# def eel_kinematics_pure_py(tail_amplitude, tail_frequency, v_x, num_nodes, nx, ny, L):


nx = 20
ny = 5
# define the x (length) discretization of the mesh
L = 1.0
v_inf = 0.4
num_nodes = 30
lambda_ = 1
N_period= 2
st = 0.15 # Strouhal number= fA/v_inf
A = 0.125 # amplitude of the tail lateral movement
# A = 0.15 # amplitude of the tail lateral movement
f = st*v_inf/A


s_1_ind = 5
s_2_ind = int(nx-3)

s1 = 0.04 * L
s2 = 0.95 * L
x_1 = (1-np.cos(np.linspace(0, np.pi/2,s_1_ind,endpoint=False)))/1*s1
x_2 = np.linspace(s1, s2, int(s_2_ind-s_1_ind),endpoint=False)
x_3 = np.linspace(s2, L, int(nx-s_2_ind))
x = np.concatenate((x_1,x_2,x_3))

t = np.linspace(0,N_period/f,num_nodes)

omg = 2*np.pi*f
y = np.zeros((num_nodes,nx,ny,1))
y_dot = np.zeros((num_nodes,nx,ny,1))

t_exp = np.outer(t, np.ones(nx))
x_exp = np.outer(np.ones(num_nodes), x)

y_nx = 0.125*((x_exp+0.03125)/(1.03125))*np.sin(np.pi*2*x_exp/lambda_ - omg*t_exp)
y_dot_nx =  0.125*((x_exp+0.03125)/(1.03125))*np.cos(np.pi*2*x_exp/lambda_ - omg*t_exp)*(-omg)
for i in range(num_nodes):
    plt.plot(x,y_nx[i,:])
    # y_dot[i,:,0,0] = y_dot_nx[i,:]

plt.show()
# y[i,:] = 0.125*((x+0.03125)/(1.03125))*np.sin(np.pi*2*x/lambda_ - omg*t)
# y_dot[i,:] =  0.125*((x+0.03125)/(1.03125))*np.cos(np.pi*2*x/lambda_ - omg*t)*(-omg)