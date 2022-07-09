import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
from pathlib import Path

cfile = Path(sys.argv[-2])
mine = Path(sys.argv[-1])
run_name = mine.parts[1]

Lx = 25
Ly = 25
Lz = 1

print(f"plotting KE for {run_name}")
curtis = np.loadtxt(cfile)
ts = h5py.File(mine,"r")
t  = ts['scales/sim_time'][:]
KE = ts['tasks/KE'][:,0,0,0]/(Lx*Ly*Lz)
Re = ts['tasks/Re_rms'][:,0,0,0]
Nu = ts['tasks/Nusselt'][:,0,0,0]
ts.close()

plt.figure()
plt.plot(t,KE, 'kx', label='mine')
plt.plot(curtis[:,0], curtis[:,1], label='Curtis')
plt.xlabel(r'$t/t_{\kappa}$')
plt.ylabel('KE')
plt.xlim(0,200)
plt.legend()
plt.savefig(f"KE_vs_t_{run_name}.png",dpi=300)
