import csdl
import numpy as np
from VAST.utils.generate_mesh import *
import jax
import jax.numpy as jnp
from jax import jacfwd


from sympy import Symbol, cos, diff, sin, tanh, simplify

t = Symbol('t')
f = Symbol('f')
L = Symbol('L')
x_0 = Symbol('x_0')

theta =( -10.52 * f + 77.67* cos(2 * np.pi * f * t))/180*np.pi

diff(theta, t)

R = L/theta

alpha = x_0/L * theta

x = R * sin(alpha)

y = -tanh(50*theta)*(R**2 - x**2)**0.5 + R

diff(x, t)
diff(y, t)

import jax.numpy as jnp

def jax_expression_vy(L, f, t_actual, x_0_actual):
    t= jnp.outer(t_actual, jnp.ones(x_0_actual.shape))
    x_0 = jnp.outer(jnp.ones(t_actual.shape), x_0_actual)
    return (4.63499420625724*L*f*jnp.sin(6.28318530717959*f*t)/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2
            + 314.159265358979*f*(1 - jnp.tanh(9.18043186549017*f - 67.7798615011998*jnp.cos(6.28318530717959*f*t))**2)*(-L**2*jnp.sin(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2 + L**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2)**0.5*jnp.sin(6.28318530717959*f*t)
            + 0.737682239128136*(-6.28318530717959*L**2*f*jnp.sin(6.28318530717959*f*t)*jnp.sin(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**3 + 6.28318530717959*L**2*f*jnp.sin(6.28318530717959*f*t)/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**3
                                + 8.51746859814012*L*f*x_0*jnp.sin(6.28318530717959*f*t)*jnp.sin(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)*jnp.cos(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2)*jnp.tanh(9.18043186549017*f - 67.7798615011998*jnp.cos(6.28318530717959*f*t))/(-L**2*jnp.sin(x_0*(-0.183608637309803*f + 1.355597230024*jnp.cos(6.28318530717959*f*t))/L)**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2 + L**2/(-0.135444830693962*f + jnp.cos(6.28318530717959*f*t))**2)**0.5)

L = 0.065
f = 5.
t = np.linspace(0, 1/f, 100)
x_0 = np.linspace(0, L, 20)
vy = jax_expression_vy(L, f, t, x_0)

def jax_expression_vx(L, f, t, x_0):
    term1 = 4.63499420625724 * L * f * jnp.sin(6.28318530717959 * f * t)
    term2 = jnp.sin(x_0 * (-0.183608637309803 * f + 1.355597230024 * jnp.cos(6.28318530717959 * f * t)) / L)
    term3 = (-0.135444830693962 * f + jnp.cos(6.28318530717959 * f * t))**2

    term4 = -8.51746859814012 * f * x_0 * jnp.sin(6.28318530717959 * f * t)
    term5 = jnp.cos(x_0 * (-0.183608637309803 * f + 1.355597230024 * jnp.cos(6.28318530717959 * f * t)) / L)
    term6 = (-0.183608637309803 * f + 1.355597230024 * jnp.cos(6.28318530717959 * f * t))

    return (term1 * term2 / term3) + (term4 * term5 / term6)