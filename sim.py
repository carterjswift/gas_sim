import heapq
import math
import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns

from collideable import Atom, Wall

Collideable = Union[Wall, Atom]


class Event:
    """Represents a collision Event occuring at time t."""
    def __init__(self, t: float, collision: Tuple[Atom, Collideable, bool, int]) -> None:
        self.t = t
        self.collision = collision

    def __lt__(self, other: 'Event') -> bool:
        return self.t < other.t


def run(event_limit: int, 
        num_atoms: int, 
        volume: float, 
        energy: float, 
        mass: float, 
        radius: float,
        animate: bool = False,
        interact: bool = False,
        fpath: str = '../frames') -> Tuple[float, float, float]:
    """Run the simulation and return pressure, volume, and NkT as a tuple."""

    walls: List[Wall] = build_walls(volume)
    atoms: List[Atom] = build_atoms(num_atoms, energy, volume, mass, radius)

    print("finding events")
    event_queue: List[Event] = find_events(atoms, walls)

    #store the number of collisions for each Atom to invalidate old collisions
    col_dict: Dict[Collideable, int] = {}
    for atom in atoms:
        col_dict[atom] = 0

    sim_time: float = 0
    num_events: int = 0
    delta_p_tot: float = 0

    sns.set_style('darkgrid')
    if animate:
        s: List[float] = get_all_speed(atoms)
        a: float = math.sqrt(2 * energy / (3 * num_atoms * mass))
        xs: np.ndarray = np.linspace(0,8,100)
        f: Callable = lambda x: scipy.stats.maxwell.pdf(x, scale=a)
        ys: List[float] = list(map(f,xs))

        g = sns.lineplot(xs,ys) #ideal final velocity distribution
        sns.distplot(s,norm_hist=True,hist=False) #actual initial velocity distribution
        axes = g.axes
        axes.set_xlim(0,8)
        axes.set_ylim(0,0.4)

        path: str = fpath
        fileid: int = 0
        ext: str = "png"
        plt.savefig(f"{path}/{fileid:07}.{ext}")
    

    while num_events < event_limit:
        print(f"Simulating. t = {sim_time:.3f}, events = {num_events}",end='\r')
        # get next event
        event: Event = heapq.heappop(event_queue)

        # extract event details
        p1: Atom
        p2: Collideable
        with_wall: bool
        pred_cols: int
        p1, p2, with_wall, pred_cols = event.collision
        col_time: float = event.t

        #check if event still valid
        p_cols: int = col_dict[p1]
        if not with_wall:
            p_cols += col_dict[p2]
        if p_cols > pred_cols:
            continue

        num_events += 1

        # advance state of simulation to the time that the event should occur
        move_all(atoms, sim_time, col_time)
        sim_time = col_time

        # perform collision
        delta_p = p1.collide(p2)

        if with_wall:
            delta_p_tot += delta_p

        # update number of collisions for each atom involved in the event
        col_dict[p1] += 1
        if not with_wall:
            col_dict[p2] += 1

        # predict new events for involved atoms
        refind_events(event_queue, p1, atoms, walls,
                      sim_time, col_dict, ignore=p2)

        if not isinstance(p2, Wall):
            refind_events(event_queue, p2, atoms, walls,
                          sim_time, col_dict, ignore=p1)

        if not with_wall and animate:
            plt.cla()
            sns.lineplot(xs,ys) #MB distribution
            sns.distplot(get_all_speed(atoms),norm_hist=True,hist=False) #actual current distribution
            axes.set_xlim(0,8)
            axes.set_ylim(0,0.4)
            fileid += 1
            plt.savefig(f"{path}/{fileid:07}.{ext}")

    print(f"\n{num_events} events occurred",end='\n\n')

    s = get_all_speed(atoms)

    # Show distribution of particle speeds after sim has run
    if interact:
        plt.figure()
        sns.distplot(s)
        plt.show()

    #calculate pressure
    p = (delta_p_tot / sim_time) / (6 * volume**(2/3))

    print(f"PV: {p * volume}")

    #calculate energy
    U: float = 0
    for speed in s:
        U += speed**2 * mass / 2

    print(f"NkT: {U * 2 / 3}")

    return (p, volume, energy * 2 / 3)




# builds distribution of atoms in center-ish of box with same velocity in random direction
def build_atoms(num_atoms: int, energy: float, volume: float, mass: float, radius: float) -> List[Atom]:
    """Produce atoms uniformly distributed in posiition space with the same velocity in random directions."""

    vels = [(energy * 2 / mass / num_atoms)**(1/2) for i in range(num_atoms)]

    atoms: List[Atom] = []
    # all particles will start in the middle of the box
    max_coord: float = (volume)**(1/3) / 2 - radius
    positions: List[List[float]] = []

    for i in range(num_atoms):
        # get random direction and position
        direction: np.ndarray = np.array([random.random() - 0.5 for j in range(3)])
        dir_unit: np.ndarray = direction / np.linalg.norm(direction)
        pos: List[float] = [(random.random()-0.5) * 2 * max_coord for j in range(3)]

        # don't allow atoms to start inside of eachother
        while intersects(pos, positions, radius):
            print(f"changing position of particle {i}")
            pos = [(random.random()-0.5) * 2 * max_coord for j in range(3)]

        positions.append(pos)

        atoms.append(Atom(mass, radius, pos, dir_unit * vels[i]))

    return atoms


def build_walls(volume: float) -> List[Wall]:
    """Produce the walls that bound the Atoms."""

    # create walls that bound a cube with given volume
    pos: float = volume**(1/3) / 2
    walls: List[Wall] = []
    walls.append(Wall([1, 0, 0], pos))
    walls.append(Wall([-1, 0, 0], pos))
    walls.append(Wall([0, 1, 0], pos))
    walls.append(Wall([0, -1, 0], pos))
    walls.append(Wall([0, 0, 1], pos))
    walls.append(Wall([0, 0, -1], pos))

    return walls

# Builds priority queue of events
def find_events(atoms: List[Atom], walls: List[Wall]) -> List[Event]:
    """Create and populate the priority queue of Events."""

    event_queue: List[Event] = []
    for idx, atom1 in enumerate(atoms):
        for atom2 in atoms[idx:]:
            t = atom1.forecast(atom2)
            if t is not None:
                heapq.heappush(event_queue, Event(t, (atom1, atom2, False, 0)))

        for wall in walls:
            t = atom1.forecast(wall)
            if t is not None:
                heapq.heappush(event_queue, Event(t, (atom1, wall, True, 0)))

    return event_queue


def refind_events(event_queue: List[Event], 
                    atom: Atom, 
                    atoms: List[Atom], 
                    walls: List[Wall], 
                    cur_time: float, 
                    col_dict: Dict[Collideable, int], 
                    ignore: Optional[Collideable] = None) -> None:
    """Recalculate events involving atom and add to the event_queue."""

    for atom2 in atoms:
        if atom2 == ignore:
            continue

        t = atom.forecast(atom2)
        if t is not None:
            heapq.heappush(event_queue, Event(
                cur_time+t, (atom, atom2, False, col_dict[atom]+col_dict[atom2])))

    for wall in walls:
        t = atom.forecast(wall)

        if t is not None:
            heapq.heappush(event_queue, Event(
                cur_time+t, (atom, wall, True, col_dict[atom])))


def move_all(atoms: List[Atom], cur_time: float, end_time: float) -> None:
    """Update positions of all atoms."""
    time: float = end_time - cur_time
    for atom in atoms:
        atom.move(time)


def get_all_speed(atoms: List[Atom]) -> List[float]:
    """Get the speed of every atom."""
    s: List[float] = []
    for atom in atoms:
        s.append(np.linalg.norm(atom.v))

    return s


def get_all_pos(atoms: List[Atom]) -> List[np.ndarray]:
    """Get the position of every atom."""
    p: List[np.ndarray] = []
    for atom in atoms:
        p.append(atom.pos)
    return p


def intersects(pos: List[float], positions: List[List[float]], radius: float) -> bool:
    """Determine whether a new Atom at position pos will overlap with any previous Atoms at positions in positions."""
    for position in positions:
        if np.linalg.norm(np.subtract(position, pos)) < 2 * radius:
            return True
    return False
