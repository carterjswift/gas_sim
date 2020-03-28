import heapq
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from collideable import Atom, Wall

def run_sim(duration, num_atoms, volume, energy, mass, radius):
    walls = build_walls(volume)
    atoms = build_atoms(num_atoms, energy, volume, mass, radius)
    print("finding events")
    event_queue = find_events(atoms, walls)
    col_dict = {} #dictionary that stores number of collisions for each particle for the
                        #purpose of invalidating events that will not occur
    for atom in atoms:
        col_dict[atom] = 0
    sim_time = 0
    num_events = 0
    prev_col_time = 0

    position = np.array(get_all_pos(atoms))
    sns.set_style('darkgrid')
    sns.scatterplot(position[:,0],position[:,1])
    plt.xlim([-volume**(1/3),volume**(1/3)])
    plt.ylim([-volume**(1/3),volume**(1/3)])
    plt.title(str(sim_time))

    plt.savefig("pos/0.png")
    num_events = 1

    while sim_time < duration:
        event = heapq.heappop(event_queue)

        p1, p2, with_wall, pred_cols = event.collision
        col_time = event.t

        #disallow simultaneous collisions to avoid particles getting stuck in eachother
        #if col_time == prev_col_time and prev_col_time != 0:
        #    continue

        p_cols = col_dict[p1]
        if not with_wall:
            p_cols += col_dict[p2]

        if p_cols > pred_cols:
            continue
        num_events += 1

        if not with_wall:
            print("#########",col_time, p1, p2)
        else:
            #print(col_time,p1,p2)
            pass
        prev_col_time = col_time

        move_all(atoms,sim_time,col_time)
        sim_time = col_time

        p1.collide(p2)

        col_dict[p1] += 1
        if not with_wall:
            col_dict[p2] += 1
        refind_events(event_queue,p1,atoms,walls,sim_time,col_dict, ignore=p2)

        if not with_wall:
            refind_events(event_queue,p2,atoms,walls,sim_time,col_dict,ignore=p1)

        position = np.array(get_all_pos(atoms))
        plt.cla()
        sns.scatterplot(x=position[:,0],y=position[:,1])
        plt.xlim([-volume**(1/3),volume**(1/3)])
        plt.ylim([-volume**(1/3),volume**(1/3)])
        plt.title(str(sim_time))
        name = "pos/" + str(num_events)
        name = name + ".png"
        plt.savefig(name)
        
    s = get_all_speed(atoms)
    print(s)
    print(num_events)
    #sns.set_style('darkgrid')
    plt.figure()
    sns.distplot(s)
    plt.show()

                


    
#builds distribution of atoms in center of box with same velocity in random direction
def build_atoms(num_atoms, energy, volume,mass, radius):
    vel = (energy * 2 / mass / num_atoms)**(1/2)
    atoms = []
    max_coord = (volume)**(1/3) / 1.2 #all particles will start in the middle of the box
    positions = []

    for _ in range(num_atoms):
        dir = [random.random() - 0.5 for j in range(3)]
        dir_unit = np.divide(dir,np.linalg.norm(dir))
        pos = [(random.random()-0.5)* 2 * max_coord for j in range(3)]
        
        while intersects(pos,positions,radius):
           print("changing position")
           pos = [(random.random()-0.5)* 2 * max_coord for j in range(3)]
           
        positions.append(pos)

        atoms.append(Atom(mass,radius,pos,np.multiply(dir_unit,vel)))


    return atoms


def build_walls(volume):
    #create walls that bound a cube with given volume
    pos = volume**(1/3)
    walls = []
    walls.append(Wall([1,0,0],pos))
    walls.append(Wall([-1,0,0],pos))
    walls.append(Wall([0,1,0],pos))
    walls.append(Wall([0,-1,0],pos))
    walls.append(Wall([0,0,1],pos))
    walls.append(Wall([0,0,-1],pos))

    return walls

#Builds priority queue of events
def find_events(atoms, walls):
    event_queue = []
    for idx, atom1 in enumerate(atoms):
        print("finding events for", atom1)
        for atom2 in atoms[idx:]:
            t = atom1.forecast(atom2)
            if t is not None:
                heapq.heappush(event_queue,Event(t,(atom1,atom2,False,0)))

        for wall in walls:
            t = atom1.forecast(wall)
            if t is not None:
                heapq.heappush(event_queue,Event(t,(atom1,wall,True,0)))

    for event in event_queue:
        print(event.t, event.collision[0].pos,event.collision[1].pos,event.collision[2])
    return event_queue

def refind_events(event_queue,atom, atoms, walls,cur_time,col_dict, ignore=None):
    for atom2 in atoms:
        if atom2 == ignore:
            continue
        t = atom.forecast(atom2) 
        if t is not None:
            heapq.heappush(event_queue, Event(cur_time+t,(atom,atom2,False,col_dict[atom]+col_dict[atom2])))
        
    for wall in walls:
        t = atom.forecast(wall)

        if t is not None:
            heapq.heappush(event_queue,Event(cur_time+t,(atom,wall,True,col_dict[atom])))

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

class Event:
    def __init__(self, t, collision):
        self.t = t
        self.collision = collision

    def __lt__(self, other):
        return self.t < other.t
        
def intersects(pos, positions,radius):
    for position in positions:
        if np.linalg.norm(np.subtract(position,pos)) < 2 * radius:
            return True
    return False

run_sim(10000,800,600,100,2,0.5)