"""
Dedalus script for 3D Rayleigh-Benard convection.

"""

import numpy as np
from mpi4py import MPI
import time
import os, fnmatch
import argparse

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Runtime value assignment?
parser = argparse.ArgumentParser(description='fingerprint_NL')
parser.add_argument('--resume', default=0, help='resume? 0 = no, 1 = yes')
parser.add_argument('--hiss', default=1.e-3, help='initial noise amplitude')
parser.add_argument('--dtin', default=1.e-6, help='initial timestep (1.e-6)')
parser.add_argument('--twall', default=47., help='wall time limit (47 hours)')
parser.add_argument('--tend', default=5.0, help='simulation end time')
parser.add_argument('--tout', default=0.01, help='2D output time')
parser.add_argument('--tout3d', default=0.01, help='3D output time')
parser.add_argument('--res_x', default=128, help='X resolution (default 128)')
parser.add_argument('--res_y', default=128, help='Y resolution (default 128)')
parser.add_argument('--res_z', default=72, help='Z resolution (default 72)')
parser.add_argument('--epsilon', default=0.8, help='parameter (default 0.8)')
#parser.add_argument('--Lambda', default=10, help='GQL cutoff (default 10)')
parser.add_argument('--ibcv', default=0, help='ibc inner boundary condition')
parser.add_argument('--obcv', default=0, help='obc inner boundary condition')
parser.add_argument('--ibct', default=0, help='ibc inner boundary condition')
parser.add_argument('--obct', default=0, help='obc inner boundary condition')
run_args = parser.parse_args()

#  Classify the boundary conditions on velocity and temperature.
ibcv=int(run_args.ibcv)
obcv=int(run_args.obcv)
ibct=int(run_args.ibct)
obct=int(run_args.obct)
Ra_crit = 1707.762  # No-slip top & bottom, by default.
if((ibct==0) and (obct==0)) :
   #  fixed-temperature conditions at inner and outer boundaries
   if((ibcv==0) and (obcv==0)) : Ra_crit = 1.70776177710371e+03
   if((ibcv==1) and (obcv==1)) : Ra_crit = 6.57511364481250e+02
   if((ibcv==0) and (obcv==1)) : Ra_crit = 1.10064960688828e+03
   if((ibcv==1) and (obcv==0)) : Ra_crit = 1.10064960688828e+03
if((ibct==1) and (obct==1)) :
   #  fixed-flux conditions at inner and outer boundaries
   if((ibcv==0) and (obcv==0)) : Ra_crit = 720.0
   if((ibcv==1) and (obcv==1)) : Ra_crit = 120.0
   if((ibcv==0) and (obcv==1)) : Ra_crit = 320.0
   if((ibcv==1) and (obcv==0)) : Ra_crit = 320.0 # ?
if((ibct==1) and (obct==0)) :
   #  fixed-flux IBC and fixed-temperature OBC
   if((ibcv==0) and (obcv==0)) : Ra_crit = 1.29577786545293e+03
   if((ibcv==1) and (obcv==1)) : Ra_crit = 3.84692841442969e+02
   if((ibcv==0) and (obcv==1)) : Ra_crit = 8.16744431749805e+02
   if((ibcv==1) and (obcv==0)) : Ra_crit = 6.68998252019900e+02
if((ibct==0) and (obct==1)) :
   #  fixed-temperature IBC and fixed-flux OBC
   if((ibcv==0) and (obcv==0)) : Ra_crit = 1.29577786545100e+03
   if((ibcv==1) and (obcv==1)) : Ra_crit = 3.84692841442969e+02
   if((ibcv==0) and (obcv==1)) : Ra_crit = 8.16744431749805e+02 # ?
   if((ibcv==1) and (obcv==0)) : Ra_crit = 6.68998252019900e+02 # ?

# Should an extant simulation be resumed?
resume=int(run_args.resume)

#  Grid resolutions
res_x=int(run_args.res_x)
res_y=int(run_args.res_y)
res_z=int(run_args.res_z)

#  Output time intervals and finishing time.
dtin=float(run_args.dtin)
tout=float(run_args.tout)
tout3d=float(run_args.tout3d)
twall=float(run_args.twall)
tend=float(run_args.tend)
hiss=float(run_args.hiss)

# Parameters
Lx, Ly, Lz = (25., 25., 1.)
Pr = 1.0

#  Rayleigh-Benard parameter: how far above critical Ra?
epsilon=float(run_args.epsilon)
Ra = Ra_crit * (1 + epsilon)

#  GQL low/high break index
#Lambda=int(run_args.Lambda)
#Lambda=('%i' %(Lambda))

#  Save the assumptions and basic parameters to a file.
file=open("ASSUME.txt","w")
file.write('## epsilon=%f\n' %(epsilon))
#file.write('## Lambda=%i\n' %(int(Lambda)))
file.write('## Lambda=∞\n')
file.write('## res_x=%i\n' %(res_x))
file.write('## res_y=%i\n' %(res_y))
file.write('## res_z=%i\n' %(res_z))
file.write('## Lx=%f\n' %(Lx))
file.write('## Ly=%f\n' %(Ly))
file.write('## Lz=%f\n' %(Lz))
file.write('## tout=%f\n' %(tout))
file.write('## tout3d=%f\n' %(tout3d))
file.write('## tend=%f\n' %(tend))
file.write('## ibct=%i\n' %(ibct))
file.write('## obct=%i\n' %(obct))
file.write('## ibcv=%i\n' %(ibcv))
file.write('## obcv=%i\n' %(obcv))
file.write('## Ra_crit=%f\n' %(Ra_crit))
file.write('## Ra=%f\n' %(Ra))
file.close()


# Create bases and domain
start_init_time = time.time()
x_basis = de.Fourier('x', res_x, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', res_y, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', res_z, interval=(-Lz/2, Lz/2), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=[8,8])

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','b','u','v','w','bz','uz','vz','wz'], time='t')
problem.parameters['P'] = 1
problem.parameters['R'] = Pr
problem.parameters['F'] = F = Ra*Pr

#  restate the box size
problem.parameters['Lx'] = Lx
problem.parameters['Ly'] = Ly
problem.parameters['Lz'] = Lz
problem.parameters['Axy'] = Lx * Ly
problem.parameters['Vol'] = Lx * Ly * Lz
problem.parameters['zobc'] = Lz/2

#  abbreviations for the low/high variable conditions:
case_zero="(nx == 0) and (ny == 0)"
case_nonz="(abs(nx)+abs(ny) > 0)"
#case_nonz="((nx != 0) or (ny != 0))"

#  incompressibility condition:
problem.add_equation("dx(u) + dy(v) + wz = 0")

#  evolution of the buoyancy variable:
problem.add_equation("dt(b) - P*(dx(dx(b)) + dy(dy(b)) + dz(bz))             = - u*dx(b) - v*dy(b) - w*bz")

#  x-momentum equation:
problem.add_equation("dt(u) - R*(dx(dx(u)) + dy(dy(u)) + dz(uz)) + dx(p)     = - u*dx(u) - v*dy(u) - w*uz")

#  y-momentum equation:
problem.add_equation("dt(v) - R*(dx(dx(v)) + dy(dy(v)) + dz(vz)) + dy(p)     = - u*dx(v) - v*dy(v) - w*vz")

#  z-momentum equation:
problem.add_equation("dt(w) - R*(dx(dx(w)) + dy(dy(w)) + dz(wz)) + dz(p) - b = - u*dx(w) - v*dy(w) - w*wz")

#  identification of gradient variables:
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")

#  inner boundary conditions
if(ibct == 0)	: problem.add_bc("left(b) = -left(F*z)")
else		: problem.add_bc("left(bz) = -F")
if(ibcv == 0)	:
   problem.add_bc("left(u) = 0")
   problem.add_bc("left(v) = 0")
else :
   problem.add_bc("left(uz) = 0")
   problem.add_bc("left(vz) = 0")
problem.add_bc("left(w) = 0")

#  outer boundary conditions
if(obct == 0)	: problem.add_bc("right(b) = -right(F*z)")
else		: problem.add_bc("right(bz) = -F")
if(obcv == 0)	:
   problem.add_bc("right(u) = 0")
   problem.add_bc("right(v) = 0")
else :
   problem.add_bc("right(uz) = 0")
   problem.add_bc("right(vz) = 0")
problem.add_bc("right(w) = 0", condition=case_nonz)

#  condition on integrated pressure
problem.add_bc("integ_z(p) = 0", condition=case_zero)

# Build solver
solver = problem.build_solver(de.timesteppers.MCNAB2)
#solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# Analysis:
snap = solver.evaluator.add_file_handler('snapshots', sim_dt=tout, max_writes=10)

# Are any checkpoint files available?
if resume :
   logger.info('RESUME?')
   check_fn=fnmatch.filter(os.listdir('RESUME/'), '*_s*.h5')
   n_check=len(check_fn)
   logger.info('found %i files' %(n_check))
   logger.info(check_fn)
   #print(check_fn)
   if(n_check < 1) :
      resume=0
   else :
      check_n=[]
      for i in check_fn :
         cut=i.replace('.h5','')
         j=cut.find('_s')
         if(j > 1) :
            val=int(cut[j+2:len(cut)])
         else :
            val=0
         check_n.append(val)
      best_fn=check_fn[check_n.index(max(check_n))]
      logger.info('RESTART: '+best_fn)

if resume:
   # Load restart
   write, dt = solver.load_state('RESUME/'+best_fn, -1)
else:
   # Initial conditions
   z = domain.grid(2)
   b = solver.state['b']
   bz = solver.state['bz']

   # Random perturbations, initialized globally for same results in parallel
   gshape = domain.dist.grid_layout.global_shape(scales=1)
   slices = domain.dist.grid_layout.slices(scales=1)
   rand = np.random.RandomState(seed=23)
   noise = rand.standard_normal(gshape)[slices]

   # Linear background + perturbations damped at walls
   zb, zt = z_basis.interval
   pert =  hiss * noise * (zt - z) * (z - zb)
   b['g'] = -F*(z - pert)
   b.differentiate('z', out=bz)

# Integration parameters
solver.stop_sim_time = tend
solver.stop_wall_time = 60.*60.*twall
solver.stop_iteration = np.inf


# horizontal cross sections
snap.add_task("interp(p, z=0)", scales=1, name='p midplane')
snap.add_task("interp(b, z=0)", scales=1, name='b midplane')
snap.add_task("interp(u, z=0)", scales=1, name='u midplane')
snap.add_task("interp(v, z=0)", scales=1, name='v midplane')
snap.add_task("interp(w, z=0)", scales=1, name='w midplane')
snap.add_task("integ(b, 'z')", scales=1, name='b integral')
snap.add_task("interp(p, z=0.25)", scales=1, name='p xy_plane')
snap.add_task("interp(b, z=0.25)", scales=1, name='b xy_plane')
snap.add_task("interp(u, z=0.25)", scales=1, name='u xy_plane')
snap.add_task("interp(v, z=0.25)", scales=1, name='v xy_plane')
snap.add_task("interp(w, z=0.25)", scales=1, name='w xy_plane')
#
snap.add_task("interp(b, z=0.25)", layout='c', scales=1, name='b test')

# vertical cross sections
snap.add_task("interp(p, x=0)", scales=1, name='p yz_plane')
snap.add_task("interp(b, x=0)", scales=1, name='b yz_plane')
snap.add_task("interp(u, x=0)", scales=1, name='u yz_plane')
snap.add_task("interp(v, x=0)", scales=1, name='v yz_plane')
snap.add_task("interp(w, x=0)", scales=1, name='w yz_plane')

# functions of z at given time
snap.add_task("integ(b,'x','y')/Axy", scales=1, name='b profile')
snap.add_task("integ(b, 'x','y')/(Axy*F)", scales=1,name='T profile')
snap.add_task("integ(bz,'x','y')/(Axy*F)", scales=1, name='dTdz profile')
snap.add_task("integ(u*dx(b) + v*dy(b) + w*bz, 'x', 'y')/(Axy*F)", scales=1, name='heat convection')
snap.add_task("integ((dx(dx(b)) + dy(dy(b)) + dz(bz)), 'x', 'y')/(Axy*F)", scales=1, name='heat conduction')
snap.add_task("integ(w*b,'x','y')/(Axy*F) -integ(w,'x','y')*integ(b,'x','y')/(Axy*Axy*F*F)", scales=1, name='Tw profile')
nuss="(((w)*(b)-bz)/F)"
z_nuss="integ("+nuss+",'x','y')/Axy"
snap.add_task("integ("+nuss+",'x','y')/Axy", scales=1, name='z_Nusselt')

# integrated values at given time
snap.add_task("integ("+nuss+",'x','y','z')/Vol", scales=1, name='V_Nusselt')
snap.add_task("integ(b*b,'x','y','z')/Vol-(integ(b,'x','y','z')/Vol)**2", scales=1, name='b variance')
snap.add_task("integ(b, 'x', 'y', 'z')/Vol", scales=1, name='b mean')
snap.add_task("0.5*integ(u*u+v*v+w*w,'x','y','z')/Vol", scales=1, name='kinetic')
snap.add_task("integ(b, 'x', 'y', 'z')/(Vol*F)", name='mean temperature')

# raw coefficient slabs of the principal variables
slab = solver.evaluator.add_file_handler('slabs', sim_dt=tout3d, max_writes=10)
#
slab.add_task("b", layout='g', scales=1, name='b grid')
slab.add_task("u", layout='g', scales=1, name='u grid')
slab.add_task("v", layout='g', scales=1, name='v grid')
slab.add_task("w", layout='g', scales=1, name='w grid')
slab.add_task("p", layout='g', scales=1, name='p grid')


# CFL
CFL = flow_tools.CFL(solver, initial_dt=dtin, cadence=5, safety=1.2,
                     max_change=1.5, min_change=0.5, max_dt=tout*0.5)
CFL.add_velocities(('u', 'v', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + v*v + w*w) / R", name='Re')
flow.add_property("0.5 *(u*u + v*v + w*w)", name='KE')
flow.add_property("sqrt(u*u + v*v)", name='amax_uv')
flow.add_property("sqrt(u*u)", name='amax_u')
flow.add_property("sqrt(v*v)", name='amax_v')
flow.add_property("sqrt(w*w)", name='amax_w')

# Save checkpoints.
checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=tout, max_writes=1)
checkpoints.add_system(solver.state)

# Main loop
end_init_time = time.time()
logger.info('epsilon = %f' %(epsilon))
logger.info('Lambda  = ∞')
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('max Re = %e, avg KE = %e' %(flow.max('Re'), flow.volume_average('KE')))
        # if((np.isnan(flow.max('Re'))) or (flow.max('Re') > 1.e9)) :
        #     logger.inro("Error: infinite Re.  Try higher resolution?")
        #     logger.info('max u  = %e' %flow.max('amax_u'))
        #     logger.info('max v  = %e' %flow.max('amax_v'))
        #     logger.info('max w  = %e' %flow.max('amax_w'))
        #     break
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))
    from dedalus.tools import post
    logger.info('beginning join operation')
    logger.info('checkpoints/')
    post.merge_analysis('checkpoints/')

