import sys
from abc import ABC, abstractmethod
import numpy as np
import math


class Collideable(ABC):

    @abstractmethod
    def collide(self, other):
        pass


class Wall(Collideable):

    def __init__(self, norm, pos):
        # make sure normal vector of wall is normalized
        self.norm = np.divide(norm, np.linalg.norm(norm))

        self.pos = pos

    def collide(self, other):
        if isinstance(other, Wall):
            return

        # perform collision
        initialv = other.v
        v_par = np.multiply(self.norm, np.dot(initialv, self.norm))
        deltaV = np.multiply(-2,v_par)
        other.v = np.add(initialv, deltaV)

        # Return the change in momentum of the particle
        return np.linalg.norm(np.multiply(other.mass,deltaV))


class Atom(Collideable):

    def __init__(self, mass, radius, pos, v):
        self.mass = mass
        self.radius = radius
        self.pos = pos
        self.v = v

    def move(self, time):
        self.pos = np.add(self.pos, np.multiply(time, self.v))

    def collide(self, other):
        if isinstance(other, Wall):
            return other.collide(self)

        vis = self.v
        vio = other.v

        # calculate collision normal vector
        col_norm = np.divide(np.subtract(self.pos, other.pos),
                             np.linalg.norm(np.subtract(self.pos, other.pos)))

        # project velocities into collision coordinate system
        sv_par = np.multiply(col_norm, np.dot(self.v, col_norm))
        ov_par = np.multiply(col_norm, np.dot(other.v, col_norm))

        sv_perp = self.v - sv_par
        ov_perp = other.v - ov_par
        # perform collision with transformed vectors
        sv_parf = ov_par
        ov_parf = sv_par

        # reconstruct xy velocities from collision components
        other.v = np.add(ov_perp, ov_parf)
        self.v = np.add(sv_perp, sv_parf)

        # return the total momentum change of the collision (should be zero)
        p_init = np.add(np.multiply(other.mass, vio),
                        np.multiply(self.mass, vis))
        p_fin = np.add(np.multiply(other.mass, other.v),
                       np.multiply(self.mass, self.v))

        return np.subtract(p_fin, p_init)

    # predict when a collision will occur between self and other
    def forecast(self, other):

        if other == self:
            return None

        if isinstance(other, Wall):
            rel_v = np.dot(self.v, other.norm)

            if rel_v <= 0:
                return None
            
            dist_to_wall = other.pos - np.dot(self.pos,other.norm)

            if dist_to_wall < self.radius and dist_to_wall >= 0:
                return 0

            t = (dist_to_wall - self.radius) / rel_v

            if t > 0:
                return t

            return None

        else:

            delta_x = np.subtract(self.pos, other.pos)
            delta_v = np.subtract(self.v, other.v)
            d = self.radius + other.radius

            a = np.dot(delta_v, delta_v)
            b = np.dot(delta_x, delta_v)
            c = np.dot(delta_x, delta_x) - d**2
            det = b**2 - a * c

            if b > 0:
                return None
            elif det < 0:
                return None
            elif c < 0:
                return 0
            else:
                return (-b + math.sqrt(det)) / a
