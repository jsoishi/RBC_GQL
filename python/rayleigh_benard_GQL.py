"""
Dedalus script for 3D Rayleigh-Benard convection.

This script uses parity-bases in the x and y directions to mimick stress-free,
insulating sidewalls.  The equations are scaled in units of the thermal
diffusion time (Pe = 1).

This script should be ran in parallel, and would be most efficient using a
2D process mesh.  It uses the built-in analysis framework to save 2D data slices
in HDF5 files.  The `merge_procs` command can be used to merge distributed analysis
sets from parallel runs, and the `plot_slices.py` script can be used to plot
the slices.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ mpiexec -n 4 python3 plot_slices.py snapshots/*.h5

The simulation should take roughly 400 process-minutes to run, but will
automatically stop after an hour.

"""

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post
from GQLProjection import GQLProjection

import logging
logger = logging.getLogger(__name__)

restart = None
# Parameters
Lx, Ly, Lz = (25., 25., 1.)
Lambda_x = 0
Lambda_y = 0
epsilon = 0.4
Pr = 1.0

Ra_crit = 1707.762  # No-slip top & bottom
Ra = Ra_crit * (1 + epsilon)

# Create bases and domain
start_init_time = time.time()
x_basis = de.Fourier('x', 64, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', 64, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', 16, interval=(-Lz/2, Lz/2), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64, mesh=[4,8])

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','b','u','v','w','bz','uz','vz','wz'], time='t')
problem.parameters['Lx'] = Lx
problem.parameters['Ly'] = Ly
problem.parameters['Lz'] = Lz
problem.parameters['P'] = 1
problem.parameters['R'] = Pr
problem.parameters['F'] = F = Ra*Pr
problem.substitutions['KE'] = "0.5*(u*u + v*v + w*w)"
problem.substitutions['vol_avg(A)'] = 'integ(A)/Lx/Ly/Lz'
# substitutions for projecting onto the low and high modes
problem.substitutions['Project_high(A)'] = "Project(A,[{:d},{:d}],'h')".format(Lambda_x, Lambda_y)
problem.substitutions['Project_low(A)'] = "Project(A,[{:d},{:d}],'l')".format(Lambda_x, Lambda_y)
# projected variables
problem.substitutions['b_l'] = "Project_low(b)"
problem.substitutions['b_h'] = "Project_high(b)"
problem.substitutions['u_l'] = "Project_low(u)"
problem.substitutions['u_h'] = "Project_high(u)"
problem.substitutions['v_l'] = "Project_low(v)"
problem.substitutions['v_h'] = "Project_high(v)"
problem.substitutions['w_l'] = "Project_low(w)"
problem.substitutions['w_h'] = "Project_high(w)"

problem.substitutions['U_l_dotGrad(f)'] = "u_l*dx(f) + v_l*dy(f) + w_l*dz(f)"
problem.substitutions['U_h_dotGrad(f)'] = "u_h*dx(f) + v_h*dy(f) + w_h*dz(f)"

problem.add_equation("dx(u) + dy(v) + wz = 0")
problem.add_equation("dt(b) - P*(dx(dx(b)) + dy(dy(b)) + dz(bz))             = - Project_low(U_l_dotGrad(b_l) + U_h_dotGrad(b_h)) - Project_high(U_l_dotGrad(b_h) + U_h_dtGrad(h_l))")
problem.add_equation("dt(u) - R*(dx(dx(u)) + dy(dy(u)) + dz(uz)) + dx(p)     = - Project_low(U_l_dotGrad(u_l) + U_h_dotGrad(u_h)) - Project_high(U_l_dotGrad(u_h) + U_h_dtGrad(u_l))")
problem.add_equation("dt(v) - R*(dx(dx(v)) + dy(dy(v)) + dz(vz)) + dy(p)     = - Project_low(U_l_dotGrad(v_l) + U_h_dotGrad(v_h)) - Project_high(U_l_dotGrad(v_h) + U_h_dtGrad(v_l))")
problem.add_equation("dt(w) - R*(dx(dx(w)) + dy(dy(w)) + dz(wz)) + dz(p) - b = - Project_low(U_l_dotGrad(w_l) + U_h_dotGrad(w_h)) - Project_high(U_l_dotGrad(w_h) + U_h_dtGrad(w_l))")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")

problem.add_bc("left(b) = -left(F*z)")
problem.add_bc("left(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(b) = -right(F*z)")
problem.add_bc("right(u) = 0")
problem.add_bc("right(v) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_bc("integ_z(p) = 0", condition="(nx == 0) and (ny == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.MCNAB2)
logger.info('Solver built')
write_mode = "overwrite"
if restart:
    write_mode = "append"
    logger.info("Restarting from file {}".format(restart))
    write, last_dt = solver.load_state(restart, -1)
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
    pert =  1e-3 * noise * (zt - z) * (z - zb)
    b['g'] = -F*(z - pert)
    b.differentiate('z', out=bz)

# Integration parameters
solver.stop_sim_time = 120
solver.stop_wall_time = 24*60 * 60.
solver.stop_iteration = np.inf

# Analysis
analysis_tasks = []
snap = solver.evaluator.add_file_handler('snapshots', sim_dt=0.2, max_writes=10, mode=write_mode)
snap.add_task("interp(p, z=0)", scales=1, name='p midplane')
snap.add_task("interp(b, z=0)", scales=1, name='b midplane')
snap.add_task("interp(u, z=0)", scales=1, name='u midplane')
snap.add_task("interp(v, z=0)", scales=1, name='v midplane')
snap.add_task("interp(w, z=0)", scales=1, name='w midplane')
snap.add_task("integ(b, 'z')", name='b integral x4', scales=4)
analysis_tasks.append(snap)
# checkpoints
check = solver.evaluator.add_file_handler('checkpoints', sim_dt=10, max_writes=1, mode=write_mode)
check.add_system(solver.state)
analysis_tasks.append(check)
# scalars
analysis_scalar = solver.evaluator.add_file_handler('scalar', parallel=False,sim_dt=0.01, mode=write_mode)
analysis_scalar.add_task("integ(KE)", name="KE")
analysis_scalar.add_task("vol_avg(sqrt(u*u + v*v + w*w))/R", name="Re_rms")
analysis_scalar.add_task("vol_avg(w*b-bz)/F",name='Nusselt')
analysis_tasks.append(analysis_scalar)


# CFL
CFL = flow_tools.CFL(solver, initial_dt=1e-6, cadence=5, safety=1.2,
                     max_change=1.5, min_change=0.5, max_dt=1.1)
CFL.add_velocities(('u', 'v', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + v*v + w*w) / R", name='Re')
flow.add_property("0.5*(u*u + v*v + w*w)", name='KE')
# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    logger.info('Starting loop')
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('KE = %f; Max Re = %f' %(flow.volume_average('KE'), flow.max('Re')))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

solver.evaluate_handlers_now(dt)
for task in analysis_tasks:
    post.merge_analysis(task.base_path)
