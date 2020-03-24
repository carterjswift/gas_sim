from abc import ABC, abstractmethod

class Collideable(ABC):

    @abstractmethod
    def collide(self, other):
        pass

    @abstractmethod
    def get_vy(self):
        pass

    @abstractmethod
    def get_vx(self):
        pass

    @abstractmethod
    def set_vy(self):
        pass

    @abstractmethod
    def set_vx(self):
        pass

class Wall(Collideable):

    def collide(self, other):
       pass 

