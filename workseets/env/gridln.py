'''
    This is a library to handle Grid world but with vectroised state representation.
    In this case each state s instead of being an index (as in the Grid class), it can be 
    encoded as a one hot encoding vector or a set of vectors as in tile coding.
    So, we define vectorised Grid class for using with linear function approximation algorithms
'''

from env.grid import *

# ============================== vectorised Grid class ===============================================
class vGrid(Grid):
    def __init__(self, nF=None, **kw):
        super().__init__( **kw)
        # num of features to encode a state
        self.nF = nF if nF is not None else self.nS 
        self.S = None
        
    # vectorised state representation: one-hot encoding (1 component represents a state)
    def s_(self):
        φ = np.zeros(self.nF)
        φ[self.s] = 1 
        return φ

    def S_(self):
        if self.S is not None: return self.S
        # S is a *matrix* that represents the full state space, this is only needed for Grid visualisation
        sc = self.s  # store current state to be retrieved later
        for self.s in range(self.nS): 
            self.S = np.c_[self.S, self.s_()] if self.s else self.s_()
        self.s = sc 
        return self.S

# ============================== aggregated Grid ====================================================
class aggGrid(vGrid):
    def __init__(self, tilesize=1, **kw):
        super().__init__(**kw)
        self.tilesize = self.jump = tilesize
        self.nF = -(-self.nS//self.tilesize)
        
    def s_(self):
        φ = np.zeros(self.nF) 
        φ[self.s//self.tilesize] = 1 
        return φ

# ============================== Tile-encoded Grid ===================================================
class tiledGrid(vGrid):
    def __init__(self, ntilings, offset=4, tilesize=50, **kw):
        super().__init__(**kw)
        self.tilesize = self.jump = tilesize
        self.ntilings = ntilings
        self.offset = offset
        self.ntiles = -(-self.nS//self.tilesize) 
        self.nF = self.ntiles*self.ntilings
    
    def s_(self):
        φ = np.zeros((self.ntilings, self.ntiles))
        
        for tiling in range(self.ntilings):
            ind = min((self.s + tiling*self.offset)//self.tilesize, self.ntiles-1)
            φ[tiling, ind] = 1
            
        return φ.flatten()

# =========================== useful vec grid env for prediciotn and control =========================
def vrandwalk(**kw):  return randwalk  (vGrid, **kw)
def vrandwalk_(**kw): return randwalk_ (vGrid, **kw)
def vgrid(**kw):      return grid      (vGrid, **kw)
def vmaze(**kw):      return maze      (vGrid, **kw)
def vcliffwalk(**kw): return cliffwalk (vGrid, **kw)
def vwindy(**kw):     return windy     (vGrid, **kw)

# ============================ useful vec grid env for prediction ====================================

# assuming that vstar is a function that returns Vstar values
def aggrandwalk_(nS=1000, tilesize=50, vstar=None, **kw): 
    env = randwalk_(aggGrid, nS=nS, tilesize=tilesize, **kw)
    if vstar is not None: env.Vstar = vstar(env) # vstar is a function
    return env

def tiledrandwalk_(nS=1000, ntilings=1, tilesize=200, vstar=None, **kw):
    env = randwalk_(tiledGrid, nS=nS, ntilings=ntilings, tilesize=tilesize,  **kw)
    if vstar is not None: env.Vstar = vstar(env) 
    return env

