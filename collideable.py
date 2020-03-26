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

    def __init__(self, norm, pos):
        self.norm = np.divide(norm,np.linalg.norm(norm))
        self.pos = pos

    def collide(self, other):
        if isinstance(other,Wall):
            return
        initialv = other.getv()
        v_par = np.multiply(self.norm,np.dot(initialv, self.norm) / np.dot(self.norm, self.norm))
        v_perp = initialv - v_par

        other.setv(np.subtract(v_perp, v_par))

        # Return the change in momentum of the particle
        return np.multiply(other.mass, np.subtract(other.getv(),initialv))

    def getv(self):
        return [0,0]

    def setv(self,newv):
        pass

class Atom(Collideable):

    def __init__(self, mass, radius, pos, v):
        self.mass = mass
        self.radius = radius
        self.pos = pos
        self.v = v
    
    def getv(self):
        return self.v

    def setv(self, v):
        self.v = v

    def move(self, time):
        self.pos = np.add(np.multiply(time,self.v))

    def collide(self, other):
        if isinstance(other,Wall):
            return other.collide(self)

        vis = self.v
        vio = other.getv()
        
        #calculate collision normal vector
        col_norm = np.divide(np.subtract(self.pos,other.pos),np.linalg.norm(np.subtract(self.pos,other.pos)))

        #project velocities into collision coordinate system
        sv_par = np.multiply(col_norm, np.dot(self.v,col_norm))
        ov_par = np.multiply(col_norm,np.dot(other.getv(),col_norm))

        sv_perp = self.v - sv_par
        ov_perp = other.getv() - sv_perp


        #perform collision with transformed vectors
        sv_parf = (self.mass - other.mass)/(self.mass + other.mass)*sv_par + 2*other.mass/(self.mass+other.mass)*ov_par
        ov_parf = (other.mass - self.mass)/(other.mass + self.mass)*ov_par + 2*self.mass/(other.mass+self.mass)*sv_par

        #reconstruct xy velocities from collision components
        other.setv(np.add(ov_perp,ov_parf))
        self.v = np.add(sv_perp,sv_parf)

        #return the total momentum change of the collision (should be zero)
        p_init = np.add(np.multiply(other.mass, vio),np.multiply(self.mass,vis))
        p_fin = np.add(np.multiply(other.mass,other.getv()),np.multiply(self.mass,self.v))

        return np.subtract(p_fin,p_init)

    def forecast(self,other):
        if isinstance(other, Wall):
            if np.dot(self.v,other.norm) == 0:
                return None
            t = (other.pos - self.radius - np.dot(self.pos,other.norm)) / np.dot(self.v,other.norm)
        else:
            t = 0
            fut_poss = np.add(self.pos,np.multiply(t,self.v))
            fut_poso = np.add(other.pos,np.multiply(t,other.v))
            dist = np.linalg.norm(np.subtract(fut_poss,fut_poso))
            min_dist = dist

            while dist > self.radius + other.radius:
                t+=0.000001 
                fut_poss = np.add(self.pos,np.multiply(t,self.v))
                fut_poso = np.add(other.pos,np.multiply(t,other.v))
                dist = np.linalg.norm(np.subtract(fut_poss,fut_poso))

                if dist > min_dist:
                    return None
                
                min_dist = dist
                

        if t >= 0:
            return t

        return None