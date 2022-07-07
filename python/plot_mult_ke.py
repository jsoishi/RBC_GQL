import matplotlib.pyplot as plt
import numpy as np
import h5py

curtis = np.loadtxt("eps_0.4_run/fig_kinetic_t.dat")

Lx = 25
Ly = 25
Lz = 1
t = {}
KE = {}
Re = {}
Nu = {}

runs = ("128x128x72", "128x128x32", "64x64x16")
dirs = ("eps_0.4_run_128_128_72","eps_0.4_run_128_128_32","eps_0.4_run")

for name, rdir in zip(runs, dirs):
    
    ts = h5py.File(rdir + "/scalar/scalar.h5","r")
    t[name]  = ts['scales/sim_time'][:]
    KE[name] = ts['tasks/KE'][:,0,0,0]/(Lx*Ly*Lz)
    Re[name] = ts['tasks/Re_rms'][:,0,0,0]
    Nu[name] = ts['tasks/Nusselt'][:,0,0,0]
    ts.close()

plt.figure()
for k,v in t.items():
    plt.plot(t[k],KE[k], label=k)
plt.plot(curtis[:,0], curtis[:,1], label='Curtis')
plt.xlabel(r'$t/t_{\kappa}$')
plt.ylabel('KE')
#plt.xlim(0,400)
plt.legend()
plt.savefig("KE_vs_t.png",dpi=300)
