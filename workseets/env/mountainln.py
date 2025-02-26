'''
    This simple class allows us to deal with an environment with continuous state unlike the Grid class
    which a discrete state environment. To be able to deal with it you will need to discretise it. 
    So we provide you with this ability in a simple way here.

'''
import numpy as np
from numpy.random import rand
from math import floor
import matplotlib.pyplot as plt
from IPython.display import clear_output
from gym.envs.classic_control import MountainCarEnv
import time

# ============================ Mountain Car Class ==================================================
''''
    Mountain Car using Gym environment or our own bespoke made mountain car environment
'''
def Mountain(Base=object): # MountainCarEnv
    class Mountain(Base):
        def __init__(self, ntiles=8, **kw):   #  ω: position window, rd: velocity window
            super().__init__(**kw)
            self.base = int(Base==object)

            # constants                          
            self.X0,  self.Xn  = -1.2, .5       # position range
            self.Xv0, self.Xvn = -.07, .07      # velocity range
            self.η = 3                          # we rescale by 3 to get the wavy valley/hill

            # for render()
            self.X  = np.linspace(self.X0,  self.Xn, 100)     # car's position
            self.Xv = np.linspace(self.Xv0, self.Xvn, 100)    # car's speed
            self.Y  = np.sin(self.X*self.η)
            
            # for state encoding (indexes)
            self.ntiles  = ntiles
            # number of states is nS*nSd but number of features is nS+nSd with an econdoing power of 2^(nS+nSd)>>nS*nSd!
            self.nF = self.nS = 2*(self.ntiles+1)
            self.nA = 3
            # for compatability
            self.Vstar = None
            
            # reset
            self.x = -.6 + rand()*(-.4+.6) if self.base else super().reset()[0]
            self.xv = 0

            # figure setup
            self.figsize0 = (12, 2) # for compatibility

            self.render_mode="rgb_array"
            self.render = self.render_ if self.base else self.render_gym
            self.step   = self.step_   if self.base else self.step_gym

        def s(self, tilings=1):
            s = (tilings*self.ntiles*(self.x  - self.X0 )/(self.Xn  - self.X0 ))
            return int(s) if self.base else s.astype(int)
        
        def sv(self, tilings=1):
            return int(tilings*self.ntiles*(self.xv - self.Xv0)/(self.Xvn - self.Xv0))

        def reset(self):
            self.x = -.6 + rand()*(-.4+.6) if self.base else super().reset(seed=0)[0]
            self.xv = 0
            return self.s_()
            
        def s_(self):
            φ = np.zeros(self.nF)
            φ[self.s()] = 1
            φ[self.sv() + self.ntiles + self.base] = 1
            return φ

        # for compatibility
        def S_(self):
            return np.eye(self.nF)
        
        def isatgoal(self):
            return self.x>=self.Xn
        
        
        def step_gym(self,a):
            obs, r, done, _,_ = super().step(a)
            self.x, self.xv = obs[0], obs[1]
            return self.s_(), r, done, {}

        def step_(self,a):
            a-=1       # to map from 0,1,2 to -1,0,+1
            self.xv += .001*a - .0025*np.cos(self.η*self.x); self.xv = max(min(self.xv, self.Xvn), self.Xv0)
            self.x  += self.xv;                              self.x  = max(min(self.x,  self.Xn ), self.X0 )
            
            # reset speed to 0 when reaching far left
            if self.x<=self.X0:  self.xv = 0
            
            return self.s_(), -1.0, self.isatgoal(), {}
    
        def render_gym(self, visible=True, pause=0, subplot=131, animate=True, **kw):

            if not visible: return
            self.ax0 = plt.subplot(subplot)

            plt.gcf().set_size_inches(self.figsize0[0], self.figsize0[1])
            plt.imshow(super().render())
            plt.axis('off')
            
            if animate: 
                clear_output(wait=True)
                plt.show(); time.sleep(pause)

        def render_(self,  visible=True, pause=0, subplot=131, animate=True, **kw):
            if not visible: return
            
            self.ax0 = plt.subplot(subplot)
            plt.gcf().set_size_inches(self.figsize0[0], self.figsize0[1])
            
            car = '\ō͡≡o˞̶' # fastemoji
            bbox = {'fc': '1','pad': -5}
            
            X = self.X
            Y = self.Y
            η = self.η
            
            plt.plot(X+.1,Y, 'k')
            plt.plot(X[-1]+.1,Y[-1]-.05,'sg')
            plt.text(X[-1],Y[-1]+.2,'Goal', color='g', size=14)
            plt.title('Mountain Car', size=10)
            plt.axis("off")
            
            # plot the mountain car 
            # take the derivative of the terrain to know the rotation of the car to make it more realistic
            rotation = np.arctan(np.cos(self.x*η))*90  
            plt.text(self.x, np.sin(self.x*η)+.05, car, va='center', rotation=rotation,  size=13, fontweight='bold', bbox=bbox)

            if animate: clear_output(wait=True); plt.show(); time.sleep(pause)
    
    return Mountain
        
# we can use any of the classes either with the gym base or ours
MountainCar = Mountain()
# MountainCar = Mountain(MountainCarEnv)
# ===========================tile coding================================================================

class tiledMountainCar(MountainCar):
    def __init__(self, ntilings=1, **kw): # ntilings: is number of tiles
        super().__init__(**kw)
        
        self.ntilings = ntilings
        self.dim = (self.ntilings, self.ntiles+2-self.base, self.ntiles+3) # the redundancy to mach the displacements of position(x) and velocity(xv)
        self.nF = self.dim[0]*self.dim[1]*self.dim[2]


    def inds(self):
        s_tiling = self.s(self.ntilings)
        sv_tiling = self.sv(self.ntilings)
        
        inds = []
        for tiling in range(self.ntilings):
            s  = (s_tiling  + 1*tiling )//self.ntilings
            sv = (sv_tiling + 3*tiling )//self.ntilings
            inds.append((tiling,s,sv))
        
        return inds
        
    def s_(self):
        φ = np.zeros(self.dim)
        for ind in self.inds(): φ[ind]=1
        return φ.flatten()


# ===========================hashed tile coding============================================================
class hashedtiledMountainCar(tiledMountainCar):
    def __init__(self, hash_size=1024,**kw): 
        super().__init__(**kw)
        self.nF = hash_size # fixed size that does not vary with the ntilings* ntiles
        
    def s_(self):
        φ = np.zeros(self.nF)
        for ind in self.inds():
            φ[hash(ind)%self.nF]=1
        return φ

# ===========================Index Hash Table tile coding==================================================
class IHTtiledMountainCar(tiledMountainCar):
    def __init__(self, iht_size=1024, **kw): # by default we have 8*8*8 (position tiles * velocity tiles * tilings)
        super().__init__(**kw)
        self.nF = iht_size

    def s_(self):
        φ = np.zeros(self.nF)
        inds = np.where(super().s_()!=0)[0]
        φ[inds%self.nF]=1
        return φ

