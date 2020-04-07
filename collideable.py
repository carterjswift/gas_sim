import sys
from typing import List, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
import math


class Collideable(ABC):

    @abstractmethod
    def collide(self, other) -> Optional[float]:
        pass


class Wall(Collideable):

    def __init__(self, norm: List[float], pos: float) -> None:
        # make sure normal vector of wall is normalized
        self.norm: np.ndarray = np.divide(norm, np.linalg.norm(norm))

        self.pos = pos

    def collide(self, other: Union['Atom', 'Wall']) -> Optional[float]:
        if isinstance(other, Wall):
            return None

        # perform collision
        initialv = other.v
        v_par: np.ndarray = self.norm * (initialv @ self.norm)
        deltaV: np.ndarray = -2 * v_par
        other.v = initialv + deltaV

        #return the momentum change of the collision
        return np.linalg.norm(other.mass * other.v)


class Atom(Collideable):

    def __init__(self, mass: float, radius: float, pos: List[float], v: List[float]) -> None:
        self.mass = mass
        self.radius = radius
        self.pos: np.ndarray = np.array(pos)
        self.v: np.ndarray = np.array(v)

    def move(self, time: float):
        self.pos = self.pos + time * self.v

    def collide(self, other: Union[Wall, 'Atom']) -> Optional[float]:
        if isinstance(other, Wall):
            return other.collide(self)

        vis: np.ndarray = self.v
        vio: np.ndarray = other.v

        # calculate collision normal vector
        col_norm: np.ndarray = (self.pos - other.pos) / np.linalg.norm(self.pos - other.pos)

        # project velocities into collision coordinate system
        sv_par = col_norm * (self.v @ col_norm)
        ov_par = col_norm * (other.v @ col_norm)

        sv_perp = self.v - sv_par
        ov_perp = other.v - ov_par
        # perform collision with transformed vectors
        sv_parf = ov_par
        ov_parf = sv_par

        # reconstruct xy velocities from collision components
        other.v = ov_perp + ov_parf
        self.v = sv_perp + sv_parf

        # return the total momentum change of the collision (should be zero)
        p_init: np.ndarray = other.mass * vio + self.mass * vis
        p_fin: np.ndarray = other.mass * other.v + self.mass * self.v

        return np.linalg.norm(p_fin - p_init)

    # predict when a collision will occur between self and other
    def forecast(self, other: Union[Wall, 'Atom']) -> Optional[float]:

        if other == self:
            return None

        if isinstance(other, Wall):
            rel_v: float = self.v @ other.norm

            if rel_v <= 0:
                return None
            
            dist_to_wall: float = other.pos - (self.pos @ other.norm)

            if dist_to_wall < self.radius and dist_to_wall >= 0:
                return 0

            t = (dist_to_wall - self.radius) / rel_v

            if t > 0:
                return t

            return None

        else:

            delta_x: np.ndarray = self.pos - other.pos
            delta_v: np.ndarray = self.v - other.v
            d = self.radius + other.radius

            a: float = delta_v @ delta_v
            b: float = delta_x @ delta_v
            c: float = delta_x @ delta_x - d**2
            det: float = b**2 - a * c

            if b > 0:
                return None
            elif det < 0:
                return None
            elif c < 0:
                return 0
            else:
                return (-b + math.sqrt(det)) / a
