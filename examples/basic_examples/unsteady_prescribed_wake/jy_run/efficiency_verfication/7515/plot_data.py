import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt

import numpy as np

analytical_solution = np.loadtxt('Andersen_7515_analytical.txt')
exp = np.loadtxt('Andersen_7515_exp.txt')
nu = np.loadtxt('Andersen_7515_nu.txt')
plt.figure(figsize=(5,5))
plt.plot(analytical_solution[:,0], analytical_solution[:,1], '-',label='Andersen linear', color='black')
plt.plot(nu[:,0], nu[:,1], 'o', label='Andersen nonlinear', color='black')
plt.plot(exp[:,0], exp[:,1],'x',  label='Andersen experimental', color='black')

plt.legend()
plt.xlabel('St')    
plt.ylabel('$C_T$')
plt.savefig('Andersen_7515.png',dpi=400,transparent=True)
plt.show()