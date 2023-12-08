import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from io import BytesIO
from PIL import Image
import imageio

# Define the function to fit
def fourier_1st(tomg, a0, a1, b1,):
    return a0 + a1*np.cos(tomg+b1)

# Load the data
data_list = []
# T_list = [0.1, 0.15, 0.2, 0.25, 0.5]
# T_list = [0.1,0.125, 1/3/2, 0.2, 0.25, 0.5]
T_list = [0.1,0.125, 1/3/2,  0.25]
data_01 = np.loadtxt('01_times_2_data.txt')
data_0125 = np.loadtxt('4hz_data.txt')
data_016667 = np.loadtxt('3hz_data.txt')
data_02 = np.loadtxt('020_times_2_data.txt')

data_025 = np.loadtxt('025_times_2_data.txt')
# data_05 = np.loadtxt('05_times_2_data.txt')

data_list.append(data_01)
data_list.append(data_0125)
data_list.append(data_016667)
# data_list.append(data_02)
data_list.append(data_025)
# data_list.append(data_05)

# Fit the data
# subplots
fig, axs = plt.subplots(4, 1)
popt_array = np.zeros((len(data_list), 3))
for i in range(len(data_list)):
    T = T_list[i]
    omega = np.pi*2/(T*2)
    popt, pcov = curve_fit(fourier_1st, data_list[i][:,0]*omega, data_list[i][:,1],bounds=([0,0,-np.pi],[np.inf,np.inf,np.pi]))
    popt_array[i,:] = popt
    print('the fourier coefficient for', 1/(T*2), 'Hz is', popt)
    # plt.figure()
    # Plot the data and the fit
    axs[i].plot(data_list[i][:,0]/T/2, data_list[i][:,1], 'bo', label='Data')
    t = np.linspace(0, 3*T*2, 100)
    axs[i].plot(t/T/2, fourier_1st(t*omega, *popt), 'r-', label='Fit')
    # set axis limit
    axs[i].set_xlim([0, 3])

    axs[i].set_xlabel('t/T (s)')
    axs[i].set_ylabel('angle $(\degree)$')
    axs[i].set_title('frequency = {:.2f} Hz'.format(1/(T*2)))

    # plt.plot(data_list[i][:,0]/T, data_list[i][:,1], 'bo', label='Data')

    # t = np.linspace(0, 3*T*2, 100)
    # plt.plot(t/T, fourier_1st(t*omega, *popt), 'r-', label='Fit')
    # plt.xlabel('t/T (s)')
    # plt.ylabel('Bending angle (degree)')
    plt.tight_layout()
    plt.legend()

plt.figure()
plt.plot((1/np.array(T_list)/2), popt_array[:,1], 'bo', label='a1')
plt.plot((1/np.array(T_list)/2), popt_array[:,2], 'ro', label='b1')
plt.show()

exit()

import warnings
frequency = (1/np.array(T_list)/2)
a_1_all = popt_array[:,1]
coeff_a_1 = np.polyfit(frequency, a_1_all, 1)

# plot the fitted curve
plt.figure()
plt.plot(frequency, a_1_all, 'o', label='data')
plt.plot(frequency, coeff_a_1[0]*frequency+coeff_a_1[1], 'r-', label='fit')
# print R^2
print(np.corrcoef(frequency, a_1_all)[0, 1]**2)
plt.legend()
# plt.show()

# fit b1
b_1_all = popt_array[:,2]
coeff_b_1 = np.polyfit(frequency, b_1_all, 0)

# # plot the fitted curve
# plt.figure()
plt.plot(frequency, b_1_all, 'o', label='data')
plt.plot(frequency, coeff_b_1*np.ones(frequency.size), 'r-', label='fit')
# # plt.plot(x, z[0]*x**2+z[1]*x+z[2], 'r-', label='fit')
# # plt.plot(x, z[0]*x**3+z[1]*x**2+z[2]*x+z[3], 'r-', label='fit')
# plt.show()


# exit()
# def fit_actuator_angles(frequency, a_0, coeff_a_1, coeff_b_1,  t):
def fit_actuator_angles(frequency, t):
    coeff_a_1 = np.array([-10.52532715,  77.66921897])
    # coeff_b_1 = -16.46683619
    coeff_b_1 = 0
    # a_0 = 4.477525252134109
    a_0 = 0
    a_1 = coeff_a_1[0]*frequency+coeff_a_1[1]
    b_1 = coeff_b_1#*np.ones(frequency.size)  

    print('a_0, a_1, b_1-------------', a_0, a_1, b_1)  
    angle = a_0 + a_1*np.cos(t*2*np.pi*frequency+b_1) 
    return angle


def get_actuator_kinematics(alpha):
    actuator_length = 0.07  # 7 cm actuator
    frames = []
    for idx, alpha_i in enumerate(alpha):
        plt.figure(figsize=(6, 3))
        alpha_i_abs = np.abs(alpha_i)
        R_i = actuator_length / alpha_i_abs
        x = np.linspace(0, R_i * np.sin(alpha_i_abs), 100)  # discretization along the arc
        # print('R_i, x, y-------------', R_i, x, -(R_i**2 - x**2)**0.5 + R_i)
        if alpha_i < 0:
            y = (R_i**2 - x**2)**0.5 - R_i
        else:

            y = -(R_i**2 - x**2)**0.5 + R_i
        plt.plot(x, y, label='alpha = {:.1f} deg, t = {:.2f}s'.format(np.rad2deg(alpha_i), t[idx]))

        plt.legend(loc='best', fontsize='small')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.xlim(0., 0.1)
        plt.ylim(-0.05, 0.05)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()

        # Convert the Matplotlib plot to an image using PIL
        buf = BytesIO()
        
        plt.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        image = Image.open(buf)

        frames.append(np.array(image))

        buf.close()
        plt.close()
        plt.clf()
    return frames


a_0 = np.average(popt_array[:,0])
frequency = 5

t = np.linspace(0, 1/frequency, 10)
# angle = fit_actuator_angles(frequency, a_0, coeff_a_1, coeff_b_1, t)
angle_deg = fit_actuator_angles(frequency, t)
angle = np.deg2rad(angle_deg)

plt.figure()
plt.plot(data_list[2][:,0], data_list[2][:,1], 'ro', label='Data')
plt.plot(t, angle, '-', label='data')
# plt.show()


# frames = get_actuator_kinematics(np.deg2rad(angle))
# imageio.mimwrite('actuator_kinematics_5hz.mp4', frames, fps=90)
L = 0.065
x_0 = np.linspace(0, L, 11)

plt.figure()
for i in range(len(angle)):
    print('angle', angle[i])
    R = L/angle[i]#/np.sign(angle[i])
    alpha = x_0/L*angle[i]#/np.sign(angle[i])
    x = R*np.sin(alpha)
    y = -np.tanh(angle[i])*(R**2 - x**2)**0.5 + R 
    plt.plot(x, y, 'r-')

plt.gca().set_aspect('equal', adjustable='box')


R = L/angle[i]#/np.sign(angle[i])
alpha = x_0/L*angle[i]#/np.sign(angle[i])
x = R*np.sin(alpha)
y = -np.tanh(angle[i])*(R**2 - x**2)**0.5 + R 