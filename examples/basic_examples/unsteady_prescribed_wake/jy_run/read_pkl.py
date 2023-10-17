import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

with open('x_height.pkl','rb') as f:
    x, height = pkl.load(f)

plt.figure(figsize=(4,3))
plt.plot(x,height)
plt.xlabel('x/L')
plt.ylabel('height/L')
plt.tight_layout()
plt.savefig('geo_height_L.pdf',dpi=600,transparent=False)
plt.show()
