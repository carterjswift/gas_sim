import heapq
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import scipy.stats
from collideable import Atom, Wall


def run_sim(event_limit, num_atoms, volume, energy, mass, radius, mb_dis=False, animate=False,interact=False):
    walls = build_walls(volume)
    atoms = build_atoms(num_atoms, energy, volume, mass, radius, mb_dis)
    print("finding events")
    event_queue = find_events(atoms, walls)
    col_dict = {}  # dictionary that stores number of collisions for each particle for the
    # purpose of invalidating events that will not occur
    for atom in atoms:
        col_dict[atom] = 0
    sim_time = 0
    num_events = 0

    delta_p_tot = 0

    sns.set_style('darkgrid')
    if animate:
        s = get_all_speed(atoms)
        a = math.sqrt(2 * energy / (3 * num_atoms * mass))
        xs = np.linspace(0,8,100)
        f = lambda x: scipy.stats.maxwell.pdf(x, scale=a)
        ys = list(map(f,xs))
        g = sns.lineplot(xs,ys) #ideal final velocity distribution
        sns.distplot(s,norm_hist=True,hist=False)
        axes = g.axes
        axes.set_xlim(0,8)
        axes.set_ylim(0,0.4)

        path = "../frames"
        fileid = 0
        ext = "png"
        plt.savefig(f"{path}/{fileid:07}.{ext}")
    

    while num_events < event_limit:
        print(f"Simulating. t = {sim_time:.3f}, events = {num_events}",end='\r')
        # get next event
        event = heapq.heappop(event_queue)

        # extract event details
        p1, p2, with_wall, pred_cols = event.collision
        col_time = event.t

        p_cols = col_dict[p1]
        if not with_wall:
            p_cols += col_dict[p2]

        # if either particle has collided since this event was predicted, collision is no longer valid
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
        if not with_wall:
            refind_events(event_queue, p2, atoms, walls,
                          sim_time, col_dict, ignore=p1)

        # Plotting code for purpose of making animation
        #position = np.array(get_all_pos(atoms))
        # plt.cla()
        # sns.scatterplot(x=position[:,0],y=position[:,1])
        # plt.xlim([-volume**(1/3),volume**(1/3)])
        # plt.ylim([-volume**(1/3),volume**(1/3)])
        # plt.title(str(sim_time))
        #name = "pos/" + str(num_events)
        #name = name + ".png"
        # plt.savefig(name)

        if not with_wall and animate:
            plt.cla()
            sns.lineplot(xs,ys)
            sns.distplot(get_all_speed(atoms),norm_hist=True,hist=False)
            axes.set_xlim(0,8)
            axes.set_ylim(0,0.4)
            fileid += 1
            plt.savefig(f"{path}/{fileid:07}.{ext}")

    print(f"\n{num_events} events occurred",end='\n\n')

    # Show distribution of particle speeds after sim has run

    s = get_all_speed(atoms)
    if interact:
        plt.figure()
        sns.distplot(s)
        plt.show()

    #calculate pressure
    p = (delta_p_tot / sim_time) / (6 * volume**(2/3))

    print(f"PV: {p * volume}")

    #calculate energy
    U = 0
    for speed in s:
        U += speed**2 * mass / 2

    print(f"NkT: {U * 2 / 3}")

    return (p, volume, energy * 2 / 3)




# builds distribution of atoms in center-ish of box with same velocity in random direction
def build_atoms(num_atoms, energy, volume, mass, radius, mb_dis):
    if mb_dis:
        a = math.sqrt(2 * energy / (3 * num_atoms * mass))
        vels = scipy.stats.maxwell.rvs(scale=a,size=num_atoms)
    else:
        vels = [(energy * 2 / mass / num_atoms)**(1/2) for i in range(num_atoms)]
    atoms = []
    # all particles will start in the middle of the box
    max_coord = (volume)**(1/3) / 2 - radius
    positions = []

    for i in range(num_atoms):
        # get random direction and position
        dir = [random.random() - 0.5 for j in range(3)]
        dir_unit = np.divide(dir, np.linalg.norm(dir))
        pos = [(random.random()-0.5) * 2 * max_coord for j in range(3)]

        # don't allow atoms to start inside of eachother
        while intersects(pos, positions, radius):
            print(f"changing position of particle {i}")
            pos = [(random.random()-0.5) * 2 * max_coord for j in range(3)]

        positions.append(pos)

        atoms.append(Atom(mass, radius, pos, np.multiply(dir_unit, vels[i])))

    return atoms


def build_walls(volume):
    # create walls that bound a cube with given volume
    pos = volume**(1/3) / 2
    walls = []
    walls.append(Wall([1, 0, 0], pos))
    walls.append(Wall([-1, 0, 0], pos))
    walls.append(Wall([0, 1, 0], pos))
    walls.append(Wall([0, -1, 0], pos))
    walls.append(Wall([0, 0, 1], pos))
    walls.append(Wall([0, 0, -1], pos))

    return walls

# Builds priority queue of events
def find_events(atoms, walls):
    event_queue = []
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


def refind_events(event_queue, atom, atoms, walls, cur_time, col_dict, ignore=None):
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

def move_all(atoms, cur_time, end_time):
    # update positions of all atoms
    time = end_time - cur_time
    for atom in atoms:
        atom.move(time)


def get_all_speed(atoms):
    s = []
    for atom in atoms:
        s.append(np.linalg.norm(atom.v))

    return s


def get_all_pos(atoms):
    p = []
    for atom in atoms:
        p.append(atom.pos)
    return p

def intersects(pos, positions, radius):
    for position in positions:
        if np.linalg.norm(np.subtract(position, pos)) < 2 * radius:
            return True
    return False

class Event:
    def __init__(self, t, collision):
        self.t = t
        self.collision = collision

    def __lt__(self, other):
        return self.t < other.t


# Run the simulation
if __name__=='__main__':
    run_sim(10000, 200, 200, 100, 0.1, 0.1, mb_dis=False, animate=False, interact=True)
