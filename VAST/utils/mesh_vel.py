import csdl
import numpy as np
from VAST.utils.generate_mesh import *
import jax
import jax.numpy as jnp
from jax import jacfwd
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt

from sympy import Symbol, cos, diff, sin, tanh, simplify, sign

t = Symbol('t')
f = Symbol('f')
L = Symbol('L')
x_0 = Symbol('x_0')

a_1 = -10.52*f+77.67
theta = (a_1 * cos(2 * np.pi * f * t)) / 180 * np.pi

diff(theta, t)

R = L/theta

alpha = x_0/L * theta

x = R * sin(alpha)

# y = -tanh(50*theta)*(R**2 - x**2)**0.5 + R
y = -sign(theta)*(R**2 - x**2)**0.5 + R
diff(x, t)
diff(y, t)

# exit()

import jax.numpy as jnp

L = 0.065
f = 5.
t_actual = np.linspace(0, 1/f, 30)
x_0_actual = np.linspace(1e-2, L, 5)

# def jax_expression_y(L, f, t_actual, x_0_actual):
t = jnp.outer(t_actual, jnp.ones(x_0_actual.shape))
x_0 = jnp.outer(jnp.ones(t_actual.shape), x_0_actual)
a_1 = -10.52*f+77.67
theta = (a_1 * jnp.cos(2 * jnp.pi * f * t)) / 180 * jnp.pi
R = L / theta
alpha = x_0 / L * theta

x = R * jnp.sin(alpha)

# y = -jnp.tanh(100 * theta) * (R ** 2 - x ** 2) ** 0.5 + R
y = -np.sign(theta) * (R ** 2 - x ** 2) ** 0.5 + R
# y = -jnp.tanh(1000*theta)   * (R**2 - x**2)**0.5 + R 
plt.figure()
plt.plot(t[:,0],y[:,-1])
plt.show()
# for i in range(10):
#     # plt.plot(x[i,:],y[i,:])
#     plt.plot(t[:,0],y[:,i])
# plt.show()

    # return y


# y = jax_expression_y(L, f, t, x_0)
# exit()
# def jax_expression_vy(L, f, t_actual, x_0_actual):
#     t= jnp.outer(t_actual, jnp.ones(x_0_actual.shape))
#     x_0 = jnp.outer(jnp.ones(t_actual.shape), x_0_actual)
#     return (4.63499420625724*L*f*jnp.sin(6.28318530717959*f*t)/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2
#             + 314.159265358979*f*(1 - jnp.tanh(9.18043186549017*f - 67.7798615011998*jnp.cos(6.28318530717959*f*t))**2)*(-L**2*jnp.sin(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2 + L**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2)**0.5*jnp.sin(6.28318530717959*f*t)
#             + 0.737682239128136*(-6.28318530717959*L**2*f*jnp.sin(6.28318530717959*f*t)*jnp.sin(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**3 + 6.28318530717959*L**2*f*jnp.sin(6.28318530717959*f*t)/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**3
#                                 + 8.51746859814012*L*f*x_0*jnp.sin(6.28318530717959*f*t)*jnp.sin(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)*jnp.cos(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2)*jnp.tanh(9.18043186549017*f - 67.7798615011998*jnp.cos(6.28318530717959*f*t))/(-L**2*jnp.sin(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2 + L**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2)**0.5)

L = 0.065
f = 5.
t = np.linspace(0, 1/f, 30)
x_0 = np.linspace(1e-2, L, 20)
# vy = jax_expression_vy(L, f, t, x_0)

def jax_expression_vx(L, f, t_actual, x_0_actual):
    t= jnp.outer(t_actual, jnp.ones(x_0_actual.shape))
    x_0 = jnp.outer(jnp.ones(t_actual.shape), x_0_actual)
    term1 = 360.0 * L * f * jnp.sin(6.28318530717959 * f * t) 
    term1 *= jnp.sin(0.0174532925199433 * x_0 * (77.67 - 10.52 * f) * jnp.cos(6.28318530717959 * f * t) / L)
    term1 /= ((77.67 - 10.52 * f) * jnp.cos(6.28318530717959 * f * t)**2)
    
    term2 = 6.28318530717959 * f * x_0 * jnp.sin(6.28318530717959 * f * t)
    term2 *= jnp.cos(0.0174532925199433 * x_0 * (77.67 - 10.52 * f) * jnp.cos(6.28318530717959 * f * t) / L)
    term2 /= jnp.cos(6.28318530717959 * f * t)
    
    return term1 - term2

def jax_expression_vy(L, f, t_actual, x_0_actual):
    t= jnp.outer(t_actual, jnp.ones(x_0_actual.shape))
    x_0 = jnp.outer(jnp.ones(t_actual.shape), x_0_actual)
    term1 = 360.0 * L * f * jnp.sin(6.28318530717959 * f * t) / ((77.67 - 10.52 * f) * jnp.cos(6.28318530717959 * f * t)**2)
    
    term2a = 1 - jnp.tanh(0.872664625997165 * (77.67 - 10.52 * f) * jnp.cos(6.28318530717959 * f * t))**2
    term2b = -L**2 * jnp.sin(0.0174532925199433 * x_0 * (77.67 - 10.52 * f) * jnp.cos(6.28318530717959 * f * t) / L)**2
    term2b /= ((1 - 0.135444830693962 * f)**2 * jnp.cos(6.28318530717959 * f * t)**2)
    term2b += L**2 / ((1 - 0.135444830693962 * f)**2 * jnp.cos(6.28318530717959 * f * t)**2)
    term2 = 4.0447954855025 * f * term2a * (77.67 - 10.52 * f) * jnp.sqrt(term2b) * jnp.sin(6.28318530717959 * f * t)
    
    term3a = -6.28318530717959 * L**2 * f * jnp.sin(6.28318530717959 * f * t) * jnp.sin(0.0174532925199433 * x_0 * (77.67 - 10.52 * f) * jnp.cos(6.28318530717959 * f * t) / L)**2
    term3a /= ((1 - 0.135444830693962 * f)**2 * jnp.cos(6.28318530717959 * f * t)**3)
    term3a += 6.28318530717959 * L**2 * f * jnp.sin(6.28318530717959 * f * t) / ((1 - 0.135444830693962 * f)**2 * jnp.cos(6.28318530717959 * f * t)**3)
    
    term3b = 0.109662271123215 * L * f * x_0 * (77.67 - 10.52 * f) * jnp.sin(6.28318530717959 * f * t)
    term3b *= jnp.sin(0.0174532925199433 * x_0 * (77.67 - 10.52 * f) * jnp.cos(6.28318530717959 * f * t) / L) * jnp.cos(0.0174532925199433 * x_0 * (77.67 - 10.52 * f) * jnp.cos(6.28318530717959 * f * t) / L)
    term3b /= ((1 - 0.135444830693962 * f)**2 * jnp.cos(6.28318530717959 * f * t)**2)
    
    term3c = -L**2 * jnp.sin(0.0174532925199433 * x_0 * (77.67 - 10.52 * f) * jnp.cos(6.28318530717959 * f * t) / L)**2
    term3c /= ((1 - 0.135444830693962 * f)**2 * jnp.cos(6.28318530717959 * f * t)**2)
    term3c += L**2 / ((1 - 0.135444830693962 * f)**2 * jnp.cos(6.28318530717959 * f * t)**2)
    
    term3 = -0.737682239128136 * (term3a + term3b) * jnp.tanh(0.872664625997165 * (77.67 - 10.52 * f) * jnp.cos(6.28318530717959 * f * t)) / jnp.sqrt(term3c)
    
    return term1 + term2 + term3
    # return (4.63499420625724*L*f*jnp.sin(6.28318530717959*f*t)/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2
    #         + 314.159265358979*f*(1 - jnp.tanh(9.18043186549017*f - 67.7798615011998*jnp.cos(6.28318530717959*f*t))**2)*(-L**2*jnp.sin(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2 + L**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2)**0.5*jnp.sin(6.28318530717959*f*t)
    #         + 0.737682239128136*(-6.28318530717959*L**2*f*jnp.sin(6.28318530717959*f*t)*jnp.sin(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**3 + 6.28318530717959*L**2*f*jnp.sin(6.28318530717959*f*t)/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**3
    #                             + 8.51746859814012*L*f*x_0*jnp.sin(6.28318530717959*f*t)*jnp.sin(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)*jnp.cos(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2)*jnp.tanh(9.18043186549017*f - 67.7798615011998*jnp.cos(6.28318530717959*f*t))/(-L**2*jnp.sin(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2 + L**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2)**0.5)

vy = jax_expression_vy(L,f,t,x_0)
vx = jax_expression_vx(L,f,t,x_0)



plt.plot(t,vx[:,0])
plt.plot(t,vx[:,-1])
plt.figure()

plt.plot(t,vy[:,0])
plt.plot(t,vy[:,-1],'.-')
plt.ylim([-3,3])

plt.show()