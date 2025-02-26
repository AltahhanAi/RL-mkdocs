
'''
    This is a model/algorithm selection library for RL algorithms
    Runs class handles several runs of the same algorithm
    Compare class handles comparing different algorithms each with several runs
'''
# ===========================================================================================
import time
import numpy as np
import random
import matplotlib.pyplot as plt

from env.grid import *

# ======================== runs the same algorithm several times ============================
'''
    this is a useful class to extensively run an algorithm several times with different seeds
    and get the average performance. This is a common requirment in RL to obtain a more
    reliable perfromance estimate, especially given the stocastic nature of the algorithms and 
    sometimes the environment itself.
'''
class Runs: # experimental trials

    def __init__(self, 
                 algorithm=None,
                 runs=10, 
                 plotR=False, 
                 plotT=False, 
                 plotE=False,
                 **kw): 
        
        self.algorithm = algorithm if algorithm is not None else MRP(**kw)
        self.runs = runs
        self.plotR = plotR
        self.plotT = plotT
        self.plotE = plotE
        
       
    def header(self):
        return 'runs|'          + self.algorithm.header()
    def values(self, results):
        return '%4d|'%self.runs + self.algorithm.values(results)
   
    def init(self):
        np.random.seed(1)# for binomial, choice and randint
        random.seed(1)   # for using choices
        
        self.Rs = np.zeros((self.runs, self.algorithm.episodes))
        self.Ts = np.zeros((self.runs, self.algorithm.episodes))
        self.Es = np.zeros((self.runs, self.algorithm.episodes))
    
    def isplot(self):
        return self.plotR or self.plotT or self.plotE

    def interact(self, label='', frmt='-', **kw):
        self.init()
        runs,  algorithm = self.runs, self.algorithm
        if not label: 
            label = 'no label passed'
        
        start_time = time.time()
        for self.run in range(runs):
            run = self.run

            # visual env in the last run, usually no need to visualise other runs
            visual = algorithm.visual and run==runs-1  
            label_ = ', run=%d'%(run+1)
            
            algorithm = algorithm.interact(visual=visual, label=label_, seed=run, **kw)
            self.Rs[run] = algorithm.Rs 
            self.Ts[run] = algorithm.Ts
            self.Es[run] = algorithm.Es            
            self.runstime = progress(self.run, runs, start_time, self.isplot())
        
        if self.isplot(): self.plot(label, frmt)
        
        return self
            
    def plot(self, label='', frmt='-') :
        Rs, Ts, Es, algorithm = self.Rs, self.Ts, self.Es, self.algorithm
        label_ =' averaged over %d runs'%(self.runs)
        if self.plotT: plt.plot(algorithm.eplist, Ts.mean(0), frmt, label=label); plt.xlabel('episodes,'+label_); plt.legend()
        if self.plotR: plt.plot(algorithm.eplist, Rs.mean(0), frmt, label=label); plt.xlabel('episodes,'+label_); plt.legend()
        if self.plotE: plt.plot(algorithm.eplist, Es.mean(0), frmt, label=label); plt.xlabel('episodes,'+label_); plt.legend()
        
        return self

# ============================ useful progress bar ======================================
'''
    useful progress bar function we can use tqdm but it does not play well sometime
'''
def progress(i, I, start, show=True, color=0):
    if show:
        percent = int(100 * (i+1)//I)
        print(f'\r{percent}%|\033[9{color}m{"█" * int(percent*.9)}\033[0m|{i+1}/{I}', \
              end='\r' if i+1<I else '\n')

    return int((time.time()- start)*1000)  

# ============================= compare algorithms =======================================
'''
    this is a useful class to extensively compare different algorithms by running them
    several times with different hyper parameters.
'''
class Compare:
    
    def __init__(self, 
                 algoruns=None,
                 hyper={'α':np.round(np.arange(.1,1,.2),1)},
                 plotR=False, 
                 plotT=False,
                 plotE=False,
                 print=False, 
                 **kw):
        
        self.algoruns = algoruns if algoruns is not None else Runs(**kw)
        self.hyper = hyper
        self.plotR = plotR
        self.plotT = plotT
        self.plotE = plotE
        self.print = print
    
    def isFunction(self, hyperval): 
        return isinstance(hyperval, str) # not in ['α','γ','ε','λ']

    def compare(self, label='',frmt='-', **kw):
            
        algoruns = self.algoruns
        algorithm  = self.algoruns.algorithm
        runs, episodes = algoruns.runs, algorithm.episodes
        
        hypername = list(self.hyper.keys())[0]
        hypervals = list(self.hyper.values())[0]
        nhypervals = len(hypervals)

        self.Rs = np.zeros((nhypervals, runs, episodes))
        self.Ts = np.zeros((nhypervals, runs, episodes))
        self.Es = np.zeros((nhypervals, runs, episodes))
        
        # now call the algorithm for each parameters set
        if self.print: print(algoruns.header())
        start = time.time()
        for h, hyperval in enumerate(hypervals):
            
            if self.isFunction(hyperval):  
                  label_ =   '%s'% hyperval; hyperval = getattr(algorithm, hyperval)
            else: label_ = '%.4f'% hyperval
            setattr(algorithm, hypername, hyperval)

            algoruns.interact(label= '%s %s=%s'%(label, hypername, label_), **kw)
            
            self.Rs[h] = algoruns.Rs
            self.Ts[h] = algoruns.Ts
            self.Es[h] = algoruns.Es
            
            # take the mean over the trials
            results = (algoruns.Rs.mean(), algoruns.Ts.mean(), algoruns.Es.mean())
            if self.print: print(algoruns.values(results))
            # for the progress bar we use a differernt color for compare
            self.comparetime = progress(h, len(hypervals), start, color=2)

        if self.print: print('comparison time = %.2f'% self.comparetime,'\n')
        if self.plotR or self.plotT or self.plotE: self.plot(label=label, frmt=frmt)
            
        return self
            
    def plot(self, label, frmt):
        hypername = list(self.hyper.keys())[0]#'α'
        hypervals = list(self.hyper.values())[0]
        ishyperNum= not self.isFunction(hypervals[0])
        hyperrng  = hypervals if ishyperNum else list(range(len(hypervals)))
        # [.1, .2, .3, .4...,1], ['reward0', 'reward1', 'reward10']

        compareGT = [self.plotR, self.plotT, self.plotE ]
        labels    = ['Rewards', 'Steps', 'Errors']
        cs        = ['r', 'b', 'g']
        
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        
        HyperMeansGT = [self.Rs.mean(axis=(1,2)), self.Ts.mean(axis=(1,2)), self.Es.mean(axis=(1,2))]
        for h, HyperMeans in enumerate(HyperMeansGT):
            
            # plot if it is required
            if not compareGT[h]: continue
            
            if label: plt.plot(hyperrng, HyperMeans, frmt, label=label)
            else:     plt.plot(hyperrng, HyperMeans, cs[h]+'--', label=labels[h])
            plt.xlabel(hypername, fontsize=14)
            plt.legend()
            
            # need to annotate if the hyper parameters are policy or rewards etc
            if ishyperNum: continue  
            bottom, top = plt.ylim()
            for i, hval in enumerate(HyperMeans):
                anot = str(hypervals[i]) +', %s'%hval
                plt.annotate(anot, xy=(i,hval+.1))
                
        
        return self

