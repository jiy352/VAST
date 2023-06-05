import matplotlib.pyplot as plt
import numpy as np
N_period= 1
num_nodes = 20
f = 0.48
t_vec = np.linspace(0,N_period/f,num_nodes)

# folder_name = 'sweeps_ny'
# variable_name = 'ny'
# variable_name_values = [3, 5, 7, 9, 11, 13]


# nx_list = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
# variable_name = 'nx'
# variable_name_values = nx_list
# folder_name = 'sweeps_nx'

# nn = [15, 30, 45, 60, 75,]
# variable_name = 'num_nodes'
# variable_name_values = nn
# folder_name = 'sweeps_nn_10'

nn = [15, 30, 45,60]
variable_name = 'num_nodes'
variable_name_values = nn
folder_name = 'sweeps_nn_opt'

legend_list = []

for i in range(len(variable_name_values)):
    print('i = ---------------------------------------',i)
    thrust_coeff = np.loadtxt(folder_name+'/thrust_coeff_'+str(i)+'.txt')
    thrust = np.loadtxt(folder_name+'/thrust_'+str(i)+'.txt')
    # np.savetxt('sweeps_mesh/C_F'+str(i)+'.txt',sim['C_F'])
    # np.savetxt('sweeps_mesh/v_x_'+str(i)+'.txt',sim['v_x'])
    num_nodes = thrust.shape[0]
    t_vec = np.linspace(0,N_period/f,num_nodes)
    plt.plot(t_vec, thrust_coeff,'.-')
    legend_list.append(variable_name+'='+str(variable_name_values[i]))
    plt.xlabel('time',fontsize=20)
    plt.ylabel('thrust_coeff',fontsize=20)

plt.legend(legend_list,fontsize=20,loc='upper right')
plt.tick_params(axis='both', which='major', labelsize=20)
# plt.show()

# folder_name = 'sweeps_nn'

# legend_list = []
# plt.figure()
# for i in range(len(variable_name_values)):
#     print('i = ---------------------------------------',i)
#     thrust_coeff = np.loadtxt(folder_name+'/thrust_coeff_'+str(i)+'.txt')
#     thrust = np.loadtxt(folder_name+'/thrust_'+str(i)+'.txt')
#     # np.savetxt('sweeps_mesh/C_F'+str(i)+'.txt',sim['C_F'])
#     # np.savetxt('sweeps_mesh/v_x_'+str(i)+'.txt',sim['v_x'])
#     num_nodes = thrust.shape[0]
#     t_vec = np.linspace(0,N_period/f,num_nodes)
#     plt.plot(t_vec, thrust_coeff,'.-')
#     legend_list.append(variable_name+'='+str(variable_name_values[i]))
#     plt.xlabel('time',fontsize=20)
#     plt.ylabel('thrust_coeff',fontsize=20)

# plt.legend(legend_list,fontsize=20,loc='upper right')
# plt.tick_params(axis='both', which='major', labelsize=20)
# plt.show()