from abc import ABC, abstractmethod
import numpy as np

class Collideable(ABC):

    @abstractmethod
    def collide(self, other):
        pass

    @abstractmethod
    def getv(self):
        pass

    @abstractmethod
    def setv(self,newv):
        pass

class Wall(Collideable):

    def __init__(self, norm, start, length):
        self.norm = norm
        self.start = start
        self.length = length

    def collide(self, other):
        if isinstance(other,Wall):
            return
        initialv = other.getv()
        v_par = self.norm * np.dot(initialv, self.norm) / np.dot(self.norm, self.norm)
        v_perp = initialv - v_par

        other.setv(np.subtract(v_perp, v_par))

        # Return the change in momentum of the particle
        return other.mass * np.subtract(initialv,other.getv())

    def getv(self):
        return [0,0]

    def setv(self,newv):
        pass
