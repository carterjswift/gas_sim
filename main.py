"""
Usage: python main.py [--runmode] 
                        [--animate] [--event_limit] 
                        [--num_atoms] [--volume] [--energy] 
                        [--mass] [--radius] [--num_points]

Flags:

--runmode: str = 'vsweep' or 'single' (default 'single')

    Determines how the simulation is run, once or swept across a range of volumes.

--animate: bool (default False)

    Only used if runmode == 'single'. Determines whether animation frames of the speed
    distribution are produced.

--event_limit: int (default 10000)

    Determines how many events the simulation allows per run before ending.

--num_atoms: int (default 200)

    The number of atoms to simulate.

--volume: float (default 200)

    Only valid if runmode == 'single'. Sets volume of cube in which particles are simulated. 

--energy: float (default 100)

    The total energy of the particles.

--mass: float (default 0.1)

    The mass of each particle.

--radius: float (default 0.1)

    The radius of each particle.

--num_points: int (default 100)

    The number times to run the simulation if runmode == 'vsweep'.

Usage Notes:

    -If running the simulation produces excessive output of "changing position of particle _",
    the volume has been set too small for all of the atoms to be placed inside the box in a uniform random distribution.

    -If animate == True, frames will be produced and placed in ../frames, which must be created before running. Be
    aware that a very large number of frames could be produced, and you will need to stitch them together with ffmpeg or gifski
    to create a video or gif.

    -Data from vsweep are saved in ../vsweep.csv. This can then be processed however you want, though I would personally
    recommend using a Jupyter Notebook.

"""

import sim_runners as sr
import sim
import argparse

p = argparse.ArgumentParser(description='Run a gas simulation')
p.add_argument('--runmode',dest='mode',type=str, default='single')
p.add_argument('--animate',dest='animate',type=bool, default=False)
p.add_argument('--event_limit',dest='lim', type=int, default=10000)
p.add_argument('--num_atoms',dest='n_atoms', type=int, default=200)
p.add_argument('--volume', dest='vol', type=float, default=200)
p.add_argument('--energy', dest='U', type=float, default=100)
p.add_argument('--mass', dest='mass', type=float, default=0.1)
p.add_argument('--radius', dest = 'radius', type=float, default=0.1)
p.add_argument('--num_points', dest='num_points', type=int, default=100)


a = p.parse_args()

if a.mode == 'single':
    sim.run(a.lim, a.n_atoms, a.vol, a.U, a.mass, a.radius, animate=a.animate, interact=True)
elif a.mode == 'vsweep':
    sr.volume_sweep(a.lim, a.n_atoms, a.U, a.mass, a.radius, a.num_points)