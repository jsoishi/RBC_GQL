import matplotlib.pyplot as plt
import numpy as np
import h5py

runs = {"DNS":"runs/eps_0.4_run_GQL_test", "L0":"runs/eps_0.4_run_GQL_L0", "L1":"runs/eps_0.4_run_GQL_L1", "L5":"runs/eps_0.4_run_GQL_L5"}
curtis = {"DNS":"fig_kinetic_t_eps0.4.dat", "L0":"fig_kinetic_t_eps0.4_L0.dat", "L1":"fig_kinetic_t_eps0.4_L1.dat", "L5":"fig_kinetic_t_eps0.4_L5.dat"}

Lx = 25
Ly = 25
Lz = 1
t = {}
KE = {}
Re = {}
Nu = {}
tc = {}
KEc = {}
for k, v in runs.items():
    ts = h5py.File(v + "/scalar/scalar_s1/scalar_s1_p0.h5","r")
    t [k]  = ts['scales/sim_time'][:]
    KE[k] = ts['tasks/KE'][:,0,0,0]/(Lx*Ly*Lz)
    Re[k] = ts['tasks/Re_rms'][:,0,0,0]
    Nu[k] = ts['tasks/Nusselt'][:,0,0,0]
    ts.close()
    c = np.loadtxt(curtis[k])
    tc[k] = c[:,0]
    KEc[k] = c[:,1]

plt.figure()
for k,v in t.items():
    plt.plot(t[k],KE[k], 'x', label=k)

plt.gca().set_prop_cycle(None)
for k, v in tc.items():
    plt.plot(tc[k], KEc[k])
plt.xlabel(r'$t/t_{\kappa}$')
plt.ylabel('KE')
plt.xlim(0,150)
plt.ylim(13,13.6)
plt.legend()
plt.savefig("../figs/mult_KE_vs_t.png",dpi=300)



