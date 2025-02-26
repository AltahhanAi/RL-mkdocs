'''
    this library implement a *linear function approximation* for well-known 
    RL algorithms. It works by inheriting from the classes in the 
    rl.tabular library. We added v prefix to the MRP and MDP base classes to 
    differentiate them from their ancestor but we could have kept the same names.
    As usual we start by defining an MRP class for prediction, then MDP for control
    then make other rl algorithms inherit forn them as needed.
'''

from rl.rl import *
from env.gridln import *
from env.mountainln import *

from math import floor
# ======================================= prediction master class==========================================
class vMRP(MRP):
        
    # set up the weights, must be done whenever we train
    def init(self):
        self.w = np.ones(self.env.nF)*self.v0
        self.V = self.V_ # this allows us to use a very similar syntax for our updates
        self.S_= None
        
    #-------------------------------------------buffer related-------------------------------------
    # allocate a suitable buffer
    def allocate(self): 
        super().allocate()
        self.s = np.ones ((self.max_t, self.env.nF), dtype=np.uint32) *(self.env.nS+10)    
    
    #---------------------------------------- retrieve Vs ------------------------------------------
    def V_(self, s=None):
        return self.w.dot(s) if s is not None else self.w.dot(self.env.S_()) 
        
    def ΔV(self,s): # gradient: we should have used ∇ but jupyter does not like it
        return s


# ======================================= prediction algorithms==========================================
class MC(vMRP):
    def __init__(self,  **kw):
        super().__init__(**kw)
        self.store = True 
        
    def init(self):
        super().init() # this is needed to bring w to the scope of the child class
        self.store = True 
        
    # ----------------------------- 🌘 offline, MC learning: end-of-episode learning ----------------------    
    def offline(self):
        # obtain the return for the latest episode
        Gt = 0
        for t in range(self.t, -1, -1):
            s = self.s[t]
            rn = self.r[t+1]
            
            Gt = self.γ*Gt + rn
            self.w += self.α*(Gt - self.V(s))*self.ΔV(s)

class TDf(vMRP):
    def init(self):
        super().init()
        self.store = True        
    # ----------------------------- 🌘 offline TD learning ----------------------------   
    def offline(self):
        for t in range(self.t, -1, -1):
            s = self.s[t]
            sn = self.s[t+1]
            rn = self.r[t+1]
            done = self.done[t+1]
            
            self.w += self.α*(rn + (1-done)*self.γ*self.V(sn) - self.V(s))*self.ΔV(s)

class TD(vMRP):
    # ----------------------------- 🌖 online learning ----------------------    
    def online(self, s, rn,sn, done, *args): 
        self.w += self.α*(rn + (1-done)*self.γ*self.V(sn) - self.V(s))*self.ΔV(s)

class TDn(vMRP):

    def init(self):
        super().init()
        self.store = True # there is a way to save storage by using t%(self.n+1) but we left it for clarity

    # ----------------------------- 🌖 online learning ----------------------    
    def online(self,*args):
        τ = self.t - (self.n-1);  n=self.n
        if τ<0: return
        
        # we take the min so that we do not exceed the episode limit (last step+1)
        τn = τ+n ; τn=min(τn, self.t+1 - self.skipstep)
        τ1 = τ+1
        
        sτ = self.s[τ ]
        sn = self.s[τn]
        done = self.done[τn]
        
        # n steps τ+1,..., τ+n inclusive of both ends
        self.w += self.α*(self.G(τ1,τn)+ (1-done)*self.γ**n *self.V(sn) - self.V(sτ))*self.ΔV(sτ)
        
class TDnf(vMRP):

    def init(self):
        super().init()
        self.store = True # offline method we need to store anyway

    # ----------------------------- 🌘 offline TD learning ----------------------------   
    def offline(self):
        n=self.n        
        for t in range(self.t+n): # T+n to reach T+n-1
            τ  = t - (n-1)
            if τ<0: continue
        
            # we take the min so that we do not exceed the episode limit (last step+1)
            τ1 = τ+1
            τn = τ+n ; τn=min(τn, self.t+1)
            
            sτ = self.s[τ ]
            sn = self.s[τn]
            done = self.done[τn]
            
            # n steps τ+1,..., τ+n inclusive of both ends
            self.w += self.α*(self.G(τ1,τn)+ (1-done)*self.γ**n *self.V(sn) - self.V(sτ))*self.ΔV(sτ)


# ======================================= control master class==========================================

class vMDP(MDP(vMRP)):

    def init(self):
        super().init()
        self.W = np.ones((self.env.nA, self.env.nF))*self.q0
        self.Q = self.Q_

    def Q_(self, s=None, a=None):
        #print(s.shape)
        W = self.W if a is None else self.W[a]
        return W.dot(s) if s is not None else np.matmul(W, self.env.S_()).T 

    # we should have used ∇ but python does not like it
    def ΔQ(self,s): 
        return s

# ======================================= control algorithms===================================
class MCC(vMDP):
    def init(self):
        super().init()
        self.store = True
    # ---------------------------- 🌘 offline, MC learning: end-of-episode learning-------------    
    def offline(self):  
        # obtain the return for the latest episode
        Gt = 0
        for t in range(self.t, -1, -1):
            s = self.s[t]
            a = self.a[t]
            rn = self.r[t+1]
            
            Gt = self.γ*Gt + rn
            self.W[a] += self.α*(Gt - self.Q(s,a))*self.ΔQ(s)

# -------------------------------------🌖 Sarsa online learning ----------------------------------
class Sarsa(vMDP):
    def init(self): #α=.8
        super().init()
        self.step = self.step_an # for Sarsa we want to decide the next action in time step t

    def online(self, s, rn,sn, done, a,an):
        self.W[a] += self.α*(rn + (1-done)*self.γ*self.Q(sn,an) - self.Q(s,a))*self.ΔQ(s)
 
#--------------------------------------🌖 Q-learning online --------------------------------------
class Qlearn(vMDP):
    def online(self, s, rn,sn, done, a,_):
        self.W[a] += self.α*(rn + (1-done)*self.γ*self.Q(sn).max() - self.Q(s,a))*self.ΔQ(s)
    
# --------------------- 🌖 XSarsa (value function) online learning ------------------------------------
class XSarsa(vMDP):
    def online(self, s, rn,sn, done, a,_):      
        # obtain the ε-greedy policy probabilities, then obtain the expecation via a dot product for efficiency
        π = self.π(sn)
        v = self.Q(sn).dot(π)
        self.W[a] += self.α*(rn + (1-done)*self.γ*v - self.Q(s,a))*self.ΔQ(s)

# ---------------------- 🌖 Actor-Critic (policy Gradient) online learning ----------------------------
class Actor_Critic(PG(vMDP)):
    def step0(self):
        self.γt = 1 # powers of γ

    def online(self, s, rn,sn, done, a,_): 
        π, γ, γt, α, τ, t, ΔV, ΔQ = self.π, self.γ, self.γt, self.α, self.τ, self.t, self.ΔV, self.ΔQ
        
        δ = (1- done)*γ*self.V(sn) + rn - self.V(s)    # TD error is based on the critic estimate
        
        self.w    += α*δ*ΔV(s)                         # critic
        self.W[a] += α*δ*ΔQ(s)*(1 - π(s,a))*γt/τ       # actor
        self.γt *= γ  

# ------------------------ 🌖 multi-step (value function) online learning -----------------------------      
class Sarsan(vMDP):
    def init(self):
        super().init()
        self.store = True        # although online but we need to access *some* of earlier steps,
        self.step = self.step_an # for Sarsa we want to decide the next action in time step t
      
    def online(self, *args):
        τ = self.t - (self.n-1);  n=self.n
        if τ<0: return
        
        # we take the min so that we do not exceed the episode limit (last step+1)
        τ1 = τ+1
        τn = τ+n ; τn=min(τn, self.t+1 - self.skipstep)
        
        sτ = self.s[τ];  aτ = self.a[τ]
        sn = self.s[τn]; an = self.a[τn]
        done = self.done[τn]
        
        # n steps τ+1,..., τ+n inclusive of both ends
        self.W[aτ] += self.α*(self.G(τ1,τn) + (1-done)*self.γ**n *self.Q(sn,an) - self.Q(sτ,aτ))*self.ΔQ(sτ)

# ------------------------ 🌖 multi-step (value function prediction) online learning -----------------------   
class TDλ(vMRP):
    def __init__(self, λ=.5, **kw):
        super().__init__(**kw)
        self.λ = λ
    def step0(self):
        self.z = self.w*0
    
    def online(self, s, rn,sn, done, *args): 
        α, γ, λ = self.α, self.γ, self.λ
        self.z = λ*γ*self.z + self.ΔV(s)
        self.w += α*(rn + (1-done)*γ*self.V(sn) - self.V(s))*self.z
    
# ------------------------ 🌖 multi-step (value function prediction) online learning -----------------------
class trueTDλ(vMRP):
    def __init__(self, λ=.5, **kw):
        super().__init__(**kw)
        self.λ = λ

    def step0(self):
        self.z = self.w*0
        self.vo = 0
  
    def online(self, s, rn,sn, done, *args): 
        α, γ, λ = self.α, self.γ, self.λ
        
        self.v = self.V(s)
        self.vn= self.V(sn)*(1-done)
        δ = rn + γ*self.vn - self.v
        self.z = λ*γ*self.z + (1-α*λ*γ*self.z.dot(s))*s
        
        self.w += α*(δ + self.v - self.vo )*self.z - α*(self.v - self.vo)*s
        self.vo = self.vn
    
# ------------------------ 🌖 multi-step (value function control) online learning -----------------------
class Sarsaλ(vMDP):
    def __init__(self, λ=.5, **kw):
        super().__init__(**kw)
        self.λ = λ
        self.step = self.step_an # for Sarsa we want to decide the next action in time step t
    
    def step0(self):
        self.Z = self.W*0

    def online(self, s, rn,sn, done, a,an):
        self.Z[a] = self.λ*self.γ*self.Z[a] + self.ΔQ(s)
        self.W[a] += self.α*(rn + (1-done)*self.γ*self.Q(sn,an)- self.Q(s,a))*self.Z[a]

# ------------------------ 🌖 multi-step, value function control, online learning -----------------------
class trueSarsaλ(vMDP):
    def __init__(self, λ=.5, **kw):
        super().__init__(**kw)
        self.λ = λ
        self.step = self.step_an # for Sarsa we want to decide the next action in time step t
    def step0(self):
        self.Z = self.W*0
        self.qo = 0
    # --------🌖 online learning ----------
    def online(self, s, rn,sn, done, a,an):
        α, γ, λ = self.α, self.γ, self.λ
        
        self.q = self.Q(s,a)
        self.qn= self.Q(sn,an)*(1-done)
        δ = rn + γ*self.qn - self.q
        self.Z[a] = λ*γ*self.Z[a] + (1-α*λ*γ*self.Z[a].dot(s))*s
        
        self.W[a] += α*(δ + self.q - self.qo )*self.Z[a] - α*(self.q - self.qo)*s
        self.qo = self.qn

# =========================a set of useful prediction comparisons =========================================

def TDtiledwalk(ntilings):
    env=tiledrandwalk_(nS=20, tilesize=4, offset=1, ntilings=ntilings)
    TD(env=env, α=.02, episodes=200, **demoV()).interact(label='TD learning, %d tilings'%ntilings)


# =========================a set of useful control comparisons =========================================
def MountainCarRuns(runs=20, algo=Sarsa, env=MountainCar(), label='', ε=0):
    for α in [.1, .2, .5]:
        sarsaRuns = Runs(algorithm=algo(env=env, α=α/8, episodes=500, ε=ε),
                         runs=runs, seed=1, plotT=True).interact(label='α=%.2f/8'%α)
    plt.ylim((10**2,10**3))
    plt.yscale('log')
    plt.title('Semi Gradient ' + algo.__name__  +' on Mountain Car '+label)


def MountainCarTiledCompare_n(runs=5, ntilings=8,  env=IHTtiledMountainCar): # 10
    xsticks = np.array([0, .5 , 1, 1.5, 2, 2.3])/ntilings
    plt.xticks(ticks=xsticks, labels=xsticks*ntilings)
    plt.yticks([220, 240, 260, 280, 300])
    plt.ylim(210, 300)
    plt.title('Steps per episode averaged over first 50 episodes')

    for n in range(5):
        if n==0: αs = np.arange(.4,  1.8,  .1)
        if n==1: αs = np.arange(.2,  1.8,  .1)
        if n==2: αs = np.arange(.1,  1.8,  .1)
        if n==3: αs = np.arange(.1,  1.2,  .07)
        if n==4: αs = np.arange(.1,  1.0,  .07)
    
        Compare(algorithm=Sarsan(env=env(ntiles=8, ntilings=ntilings), n=2**n, episodes=50, ε=0), runs=runs, 
                                  hyper={'α':αs/ntilings}, 
                                  plotT=True).compare(label='%d-step Sarsa'%2**n)
    plt.xlabel(r'$\alpha \times 8$ since we used 8 tiles for each tilings')
    plt.show()

figure_10_4_n = MountainCarTiledCompare_n


def SarsaOnMountainCar(ntilings, env=tiledMountainCar):
    sarsa = Sarsa(env=env(ntilings=ntilings), α=.5/ntilings, episodes=500, seed=1, ε=0, plotT=True).interact(label='ntilings=%d'%ntilings)
    plt.gcf().set_size_inches(20,4)
    plt.ylim(100,1000)
    return sarsa


def MountainCarTilings(runs=20, α=.3, algo=Sarsa, env=tiledMountainCar):
    plt.title('Sarsa on mountain car: comparison of different tilings with α=%.2f/8'%α)
    for ntilings in [2, 4, 8, 16, 32]:
        sarsaRuns = Runs(algorithm=algo(env=env(ntiles=8, ntilings=ntilings),α=α/ntilings,episodes=500, ε=0), 
                         runs=runs, seed=1, plotT=True).interact(label='%d tilings'%ntilings)
    plt.ylim((10**2,10**3))
    plt.yscale('log')
    plt.show()