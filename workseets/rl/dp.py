
'''
    This is a Dynamic Programming Library, mainly 
    1. policy evaluation for prediction
    2. policy iteration and value iteration for control
'''
from env.grid import *

# =====================================================================================================

def dynamics(env=randwalk(), stoch=False, show=False, repeat=1000): # , maxjump=1

    rewards = env.rewards_set()
    nS, nA, nR = env.nS, env.nA, rewards.shape[0]
    p  = np.zeros((nS,nR,  nS,nA))
    randjump = env.randjump
    env.randjump = False # so that probability of all intermed. jumps is correctly calculated
    for i in trange(repeat if stoch else 1): # in case the env is stochastic (non-deterministic)
        for s in range(nS):
            if s in env.goals: continue # uncomment to explicitly make pr of terminal states=0
            for a in range(nA):
                for jump in (range(1,env.jump+1) if randjump else [env.jump]):
                    if not i and show: env.render() # render the first repetition only
                    env.s = s
                    env.jump = jump
                    rn = env.step(a)[1]
                    sn = env.s
                    rn_ = np.where(rewards==rn)[0][0] # get reward index we need to update
                    p[sn,rn_, s,a] +=1
                    
    env.randjump = randjump
    # making sure that it is a conditional probability that satisfies Bayes rule
    for s in range(nS):
        for a in range(nA):
            sm=p[:,:, s,a].sum()
            if sm: p[:,:, s,a] /= sm
            
    return p

# =====================================================================================================
def Policy_evaluation(env=randwalk(), p=None, V0=None, π=None, γ=.99, θ=1e-3, show=False): 
    
    # env parameters
    nS, nA, nR, rewards = env.nS, env.nA, env.nR, env.rewards_set()
    p = dynamics(env) if p is None else np.array(p)

    # policy parameters
    V = np.zeros(nS)     if V0 is None else np.array(V0); V[env.goals] = 0 # initial state values
    π = np.zeros(nS,int) if π  is None else np.array(π) # policy to be evaluated **stochastic/deterministic**

    i=0
    # policy evaluation --------------------------------------------------------------
    while True:
        Δ = 0
        i+= 1
        # show indication to keep us informed
        if show: clear_output(wait=True); rng = trange(nS) # 
        else: rng = range(nS)
        for s in rng:
            if s in env.goals: continue # only S not S+
            v, V[s] = V[s], 0            
            for sn in range(nS): # S+
                for rn_, rn in enumerate(rewards): # get the next reward rn and its index rn_
                    if π.ndim == 1: # deterministic policy
                        V[s] += p[sn,rn_, s,π[s]]*(rn + γ*V[sn])
                    else:           # stochastic policy 
                        V[s] += sum(π[s,a]*p[sn,rn_, s,a]*(rn + γ*V[sn]) for a in range(nA))
            Δ = max(Δ, abs(v-V[s]))
        if Δ<θ: break
    if show: 
        env.render(underhood='V', V=V)
        print('policy evaluation stopped @ iteration %d:'%i)

    return V

# =====================================================================================================
def Policy_iteration(env=randwalk(), p=None, V0=None, π0=None, γ=.99, θ=1e-3, show=False, epochs=None): 
    
    # env parameters
    nS, nA, nR, rewards = env.nS, env.nA, env.nR, env.rewards_set()
    p = dynamics(env) if p is None else np.array(p)

    # policy parameters 
    V = np.zeros(nS)     if V0 is None else np.array(V0); V[env.goals] = 0 # initial state values
    π = np.zeros(nS,int) if π0 is None else np.array(π0); # initial **deterministic** policy 
    Q = np.zeros((nS,nA))  # state action values storage
    # π = randint(0,nA,nS)
    
    j=0
    while True if epochs is None else j<epochs:
        j+=1
        # 1. Policy evaluation---------------------------------------------------
        i=0
        while True if epochs is None else i<epochs:
            Δ = 0
            i+= 1
            for s in range(nS): 
                if s in env.goals: continue # S not S+
                v, V[s] = V[s], 0
                for sn in range(nS): # S+
                    for rn_, rn in enumerate(rewards): # get the reward rn and its index rn_
                        V[s] += p[sn,rn_,  s, π[s]]*(rn + γ*V[sn])

                Δ = max(Δ, abs(v-V[s]))
            if Δ<θ: print('policy evaluation stopped @ iteration %d:'%i); break
        
        # 2. Policy improvement----------------------------------------------------
        policy_stable=True
        for s in range(nS):
            if s in env.goals: continue # S not S+
            πs = π[s]
            for a in range(nA):
                Q[s,a]=0
                for sn in range(nS): # S+
                    for rn_, rn in enumerate(rewards): # get the reward rn and its index rn_
                        Q[s,a] += p[sn,rn_,  s,a]*(rn + γ*V[sn]) 
            
            π[s] = Q[s].argmax() # simple greedy step
            if π[s]!=πs: policy_stable=False
           
        if policy_stable: print('policy improvement stopped @ iteration %d:'%j); break
        if show: env.render(underhood='π', π=π)
        
    return π, Q

# =====================================================================================================
# stochast policy iteration
def Policy_iteration_stoch(env=randwalk(), p=None, V0=None, π0=None, γ=.99, θ=1e-4, ε=.1, show=False): 
    
    # env parameters
    nS, nA, nR, rewards = env.nS, env.nA, env.nR, env.rewards_set()
    p = dynamics(env) if p is None else np.array(p)

    # policy parameters
    V = np.zeros(nS)      if V0 is None else np.array(V0); V[env.goals] = 0 # initial state values
    π = np.ones ((nS,nA)) if π0 is None else np.array(π0); π=π/π.sum(1)[:,None] # initial **stochastic** policy 
    Q = np.zeros((nS, nA)) # state action values storage
    # π = randint(1,nA,(nS,nA))
    
    j=0
    while True:
        j+=1
        # 1. Policy evaluation---------------------------------------------------
        i=0
        while True:
            Δ = 0
            i+= 1
            for s in range(nS):
                if s in env.goals: continue # S not S+
                v, V[s] = V[s], 0
                for sn in range(nS): # S+
                    for rn_, rn in enumerate(rewards): # get the reward rn and its index rn_ 
                        # stochastic policy
                        # V[s] += sum(π[s,a]*p[sn,rn_, s,a ]*(rn + γ*V[sn]) for a in range(nA)) 
                        V[s] += p[sn,rn_,  s, π[s].argmax()]*(rn + γ*V[sn])
                        
                Δ = max(Δ, abs(v-V[s]))
            if Δ<θ: print('policy evaluation stopped @ iteration %d:'%i); break

        # 2. Policy improvement----------------------------------------------------
        policy_stable=True
        for s in range(nS):
            if s in env.goals: continue # S not S+
            Qs = Q[s]
            for a in range(nA):
                Q[s,a]=0
                for sn in range(nS): # S+
                    for rn_, rn in enumerate(rewards): # get the reward rn and its index rn_
                        Q[s,a] += p[sn,rn_, s,a]*(rn + γ*V[sn])
                        
                if abs(Q[s,a]-Qs[a]) > 0: policy_stable=False
            π[s] = Qs*0 + ε/nA
            π[s,Q[s].argmax()] += 1-ε     # greedy step  
    
        if policy_stable: print('policy improvement stopped @ iteration %d:'%j); break
        
    if show: env.render(underhood='maxQ', Q=Q)
    
    return π, Q

# =====================================================================================================

def value_iteration(env=randwalk(), p=None, V0=None, γ=.99, θ=1e-4, epochs=None, show=False): 
    # np.random.seed(1)
    # env parameters
    nS, nA, nR, rewards, i = env.nS, env.nA, env.nR, env.rewards_set(), 0
    p = dynamics(env) if p is None else np.array(p)

    # policy parameters
    V = np.zeros(nS) if V0 is None else np.array(V0); V[env.goals] = 0 # initial state values
    Q = np.zeros((nS,nA)) # state action values storage

    while True if epochs is None else i<epochs:
        Δ = 0
        i+= 1
        for s in range(nS):
            if s in env.goals: continue
            v, Q[s] = V[s], 0
            for a in range(nA):
                for sn in range(nS):
                    for rn_, rn in enumerate(rewards):            # get the reward rn and its index rn_
                        Q[s,a] += p[sn,rn_,  s,a]*(rn + γ*V[sn])  # max operation is embedded now in the evaluation
                        
            V[s] = Q[s].max()                                     # step which made the algorithm more concise 
            Δ = max(Δ, abs(v-V[s]))
            
        if Δ<θ: print('loop stopped @ iteration: %d , Δ = %2.f'% (i, Δ)); break
        if show: env.render(underhood='π', π=Q.argmax(1))
        
    return Q

# =====================================================================================================
