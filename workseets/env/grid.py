

'''
    This library is to handle a set of Grid world with thier visualisation.
    Below we provide a series of Grid classes that build on top of each other, 
    each time we add a bit more funcitonality. you do not need to study or understand 
    the code, you need to know how to utilise the Grid() class which is explained in Lesson3.

    Enjoy!
'''

#=================================================================================================
'''
    imports
'''
import numpy as np
import time
import io
import os
import random

from IPython.display import clear_output, display, HTML
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import colors
from numpy.random import rand, seed, randint, choice
from random import choices, sample
from tqdm import trange, tqdm

#===============================simple Grid world (no visualisation)==============================

'''
Actions and States
    If the action is right, then we add 1, whiel if it is left, we subtract 1. 
    When we want the agent to move up, we add one row _number of cols in the grid_, 
    and when we want the agent to move down, we subtract cols. Moving diagonally is similar, 
    but we need to combine moving up and left, down and left, up and right, and down and right. 
    Checking if the move is allowed (new state is allowed) as per the agent's current state 
    consists of checking for 4 cases. 
    1. the agent is not bypassing outside the grid to boundaries (top or bottom)
    2. the agent is not bypassing the right edge of the 2-d grid
    3. the agent is not bypassing the left edge of the 2-d grid
    4. the agent is not stepping into an obstacle

Rewards
    The set of possible rewards is 4; one for 2 possible goals (terminal states), 
    one for intermediate (non-terminal) states and one special for cliff falling. 
    The class caller can assign one of the special rewards by passing its name, 
    and the setattr() function will handle the rest.


'''
class Grid:
    def __init__(self, gridsize=[6,9], nA=4, s0=3*9, goals=None, Vstar=None):
        self.rows = gridsize[0]
        self.cols = gridsize[1]
        self.nS = self.cols*self.rows # we assume cells IDs go left to right and top down
        self.goals = [self.nS-1, self.nS-1] if goals is None else ([goals[0], goals[0]] if len(goals)==1 else goals)
        self.Vstar = Vstar # optimal state value, needed for some of the environments
        self.s0 = s0
        self.s = s0
        self.trace = [self.s0]
        
        # actions ---------------------------------------------------------------------       
        cols = self.cols
        self.actions_2 = [-1, +1]
        self.actions_4 = [-1, +1, -cols, +cols]     # left, right, down and up
        self.actions_8 = [-1, +1, -cols, +cols, -1-cols, -1+cols, 1-cols, 1+cols] # left-down, left-up, right-down, right-up

        self.nA = nA
        if nA==2: self.actions = self.actions_2
        if nA==4: self.actions = self.actions_4
        if nA==8: self.actions = self.actions_8
        
        # rewards types-----------------------------------------------------------------
        self.nR = 4
        self.rewards = [0, 1, 0, -100] # intermediate, goal1, goal2, cliff
        self.obstacles, self.cliffs = [], [] # lists that will be checked when doing actions
        
        
    def reset(self, withtrace=True):
        self.s = self.s0
        if withtrace: self.trace = [self.s0]
        return self.s_()
    #-----------------------------------------rewards related-------------------------------------------
    def rewards_set(self):
        return np.array(list(set(self.rewards)))
        
    def reward(self):
        stype = self.stype()
        reward = self.rewards[stype]
        if stype==3: self.reset(False)    # s in cliffs
        return reward, 2>=stype>=1        # either at goal1 or goal2
    
    #-----------------------------------------actions related-------------------------------------------
    def invalid(self, s,a):
        cols = self.cols
        # invalid moves are 
        # 1. off grid boundaries
        # 2. off the right edge (last and is for right up and down diagonal actions)
        # 3. off the left edge  (last and is for left  up and down diagonal actions)
        # 4. into an obstacle
        return      not(0<=(s+a)<self.nS) \
                    or (s%cols!=0 and (s+a)%cols==0 and (a==1 or a==cols+1 or a==-cols+1))  \
                    or (s%cols==0 and (s+a)%cols!=0 and (a==-1 or a==cols-1 or a==-cols-1)) \
                    or (s+a) in self.obstacles

    def step(self, a, *args):
        a = self.actions[a]
        if not self.invalid(self.s,a): self.s += a
        
        self.trace.append(self.s)
        reward, done = self.reward()       # must be done in this order for the cliff reset to work properly
        return self.s_(), reward, done, {} # empty dict for compatibility
    
    #-----------------------------------------state related-------------------------------------------
    # useful for inheritance, observation can be a state (index) or a state representation (vector or image)
    def s_(self):
        return self.s
    
    # returns the number of states that are available for the agent to occupy
    def nS_available(self):
        return self.nS - len(self.obstacles)
    
    #-----------------------------------------goals related-------------------------------------------
    # returns the type of the current state (0: intermediate, 1 or 2 at goal1 or goal2, 3:off cliff)
    def stype(self):
        s, goals, cliffs = self.s, self.goals, self.cliffs
        # the order is significant and must not be changed
        return [s not in goals+cliffs, s==goals[0], s==goals[1], s in cliffs].index(True)
    
    def isatgoal(self):
        return self.stype() in [1,2] # either at goal1 or goal2


#=================================================================================================
#===========================differen Grid world styles no visualisation===========================

'''
    In all of our treatments we will follow a trend of using the same class name 
    if possible for the child and parent class so that we do not need to deal 
    with different class names when we import theses classes. So Grid(Grid),
    means the new Gris class inherits from the previous Grid class. This allows us 
    to gradually build the capabilities of our classes in a concise and manageable 
    manner. 
    
    The getattr(self, reward) function allows us to pass a string to the class setter 
    ex. Grid(reward='cliffwalk') and then python will search and return a corresponding 
    attribute or function with the same name ex. self.cliffwalk.

    Finally, due to the wind adding to the displacement of the robot, we had to override 
    the step(a) function. The function checks the validity of an action and then attempts 
    to add as much of the wind displacement as the grid boundaries allow.
'''
class Grid(Grid):
    def __init__(self, reward='',  style='', **kw):
        super().__init__(**kw)
    
        # explicit rewards for[intermediate,goal0,goal1, cliff] states
        self.reward_    = [0,    1,   0, -100] # this is the default value for the rewards
        self.cliffwalk  = [-1,  -1,  -1, -100]
        self.randwalk   = [ 0,   0,   1,    0]
        self.randwalk_  = [ 0,  -1,   1,    0]
        self.reward0    = [-1,   0,   0,   -1]
        self.reward_1   = [-1,  -1,  -1,   -1]
        self.reward1    = [-1,   1,   1,   -1]
        self.reward10   = [-1,  10,  10,   -1]
        self.reward100  = [-1, 100, 100,   -1]
        

        if reward: self.rewards  = getattr(self, reward)
        self.style = style
        
        # accommodating grids styles -------------------------------------------------------------
        self.X, self.Y = None, None
        self.Obstacles = self.Cliffs = 0 # np arrays for display only, related to self.obstacles, self.cliffs
        self.wind = [0]*10               # [0,0,0,0,0,0,0,0,0,0]
        
        if self.style=='cliff':
            self.Cliffs = None           # for displaying only, to be filled when render() is called
            self.cliffs = list(range(1,self.cols-1))
            
        elif self.style=='maze':
            self.Obstacles = None        # for displaying only, to be filled when render() is called
            rows = self.rows
            cols = self.cols
            # midc = int(cols/2)
            obstacles1 = list(range(2+2*cols, 2+(rows-1)*cols, cols))    # set of vertical obstacles near the start
            obstacles2 = list(range(5+cols, 2*cols-3))                   # set of horizontal obstacles
            obstacles3 = list(range(-2+4*cols,-2+(rows+1)*cols, cols))   # set of vertical obstacles near the end
            self.obstacles = obstacles1 + obstacles2 + obstacles3        # concatenate them all 

        # upward winds intensity for each column
        elif self.style=='windy':
            self.wind = [0,0,0,1,1,1,2,2,1,0] # as in example 6.5 of the book
    
    # override the step() function so that it can deal with wind
    def step(self, a, *args):
        a = self.actions[a]
        if not self.invalid(self.s,a): self.s += a
        
        if self.style=='windy':
            maxwind = self.wind[self.s%self.cols]
            for wind in range(maxwind, 0, -1): # we need to try apply all the wind or at least part of it
                if not self.invalid(self.s, wind*self.cols): self.s += wind*self.cols; break
        
        self.trace.append(self.s)
        reward, done = self.reward()       # must be done in this order for the cliff reset to work properly
        return self.s_(), reward, done, {} # empty dict for compatibility



#=================================================================================================
#===========================differen Grid world styles with visualisation=========================

'''
    Ok, dealing with the bare minimum grid without visualisation makes it difficult to observe 
    the behaviour of an agent. Now we add useful set of visualisation routines to make this 
    possible. 

    This is will add an overhead so we try to minimise it by only initialising and calling when 
    a render() function is called. 

    We are moving from 1-d list of states to a 2-d set of coordinates. We use the modulus % 
    and the floor division operators to achieve this. Both are built-in operators and very efficient. 
    The function to_pos() convert a state into its correspondent position coordinates. 

    The render function does all the heavy lifting of visualising the environment and the agent's 
    current state s. It basically renders a 2-d grid as per the dimension of the grid along the side 
    with the agent and any obstacles (which block the agent pathway) or cliff cells (which cause the 
    agent to reinitialise its position to state s0). We also call a placeholder function render_(), 
    which will be called to render further info. such as the states' representation in the grid, 
    as we will see next.
'''

class Grid(Grid):
    def __init__(self, pause=0, figsize=None, **kw):
        super().__init__(**kw)
        
        self.figsize = figsize # desired figure size  
        self.figsize0 = (12, 2) # default figure size
        self.fig = None        # figure handle, may have several subplots        
        self.ax0 = None        # Grid subplot handle
        
        self.pause = pause     # pause to slow animaiton
        self.arrows = None     # policy arrows (direction of action with max value)
        
        # assuming env is not dynamic, otherwise should be moved to render() near self.to_pos(self.s)
        self.start = self.to_pos(self.s0)         
        self.goal1 = self.to_pos(self.goals[0])
        self.goal2 = self.to_pos(self.goals[1])
        self.cmap = colors.ListedColormap(['w', 'darkgray'])

    # state representation function that converts 1-d list of state representation into a 2-d coordinates
    def to_pos(self, s):
        return [s%self.cols + 1, s//self.cols + 1]

    #------------------------------------------initialise------------------------------------------------- 
    def init_cells(self, cells): 
        Cells = np.zeros((self.rows+1, self.cols+1),  dtype=bool)
        Cells[0,0] = True # to populate for drawing 
        poses = self.to_pos(np.array(cells))
        Cells[poses[1], poses[0]] = True
        return Cells[1:,1:]
    
    #------------------------------------------render ✍️-------------------------------------------------
    # this function is to protect render() called twice for Gridi
    def render(self, **kw):
        self.render__(**kw)

    # we have placed most of the render overhead in the render() function to keep the rest efficient.
    # this funciton must not be called directly instead render() is to be called
    def render__(self, underhood='', pause=None, label='', subplot=131, large=False, 
               animate=True, image=False, saveimg=False,  **kw):
        
        if self.figsize is None:
            self.figsize = self.figsize0  # (13, 2)
            if   self.rows==1:             self.figsize = (15,.5) 
            elif underhood=='Q':           self.figsize = (12, 3)#(20, 10)
            elif underhood=='V' and large: self.figsize = (12, 3)#(35, 25)                        
        if image: self.figsize = (17, 3) # changing the default figure size is dissallowed for games

        if self.fig is None: self.fig = plt.figure(1)
        #if self.ax0 is None: self.ax0 = plt.subplot(subplot)
        plt.gcf().set_size_inches(self.figsize[0], self.figsize[1])
            
        #if   animate: self.ax0 = plt.subplot(subplot)
        #elif image:   plt.cla() 
        self.ax0 = plt.subplot(subplot)
        if image and not animate: plt.cla()
        
        
        # get hooks for self properties
        rows, cols = self.rows, self.cols
        pos, start, goal1, goal2 = self.to_pos(self.s), self.start, self.goal1, self.goal2
        
        pause = self.pause if pause is None else pause
        
        # a set of properties for the grid subplot
        
        prop = {'xticks': np.linspace(0, cols, cols+1),     'xticklabels':[],
                'yticks': np.linspace(0, rows, rows+1)+.01, 'yticklabels':[],
                'xlim':(0, cols), 'ylim':(0, rows), 'xlabel': label} # useful info
        self.ax0.update(prop)
        self.ax0.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        if self.style not in ['maze', 'cliff']: self.ax0.grid(True)

        # robot visuals :-)
        mrgn = .6
        eyes = ['˚-˚','ˇ-ˇ','ˆ-ˆ'][self.s%2 if not self.s in self.goals else 2]
        eyes, body = (eyes, 'ro') if underhood!='Q' else ('' , 'co')
        
        # plot goals and start state
        for (x,y), s in zip([goal1, goal2, start], ['G', 'G', 'S']):
            self.ax0.text(x-mrgn, y-mrgn, s, fontsize=11)
        
        # plot robot
        self.ax0.text(pos[0]-mrgn-.17, pos[1]-mrgn-.25, eyes, fontsize=9)
        self.ax0.plot(pos[0]-mrgn,     pos[1]-mrgn,     body, markersize=12) 
        #self.ax0.plot(pos, body, markersize=15) # this causes the body not be up to date in later lessons

        # to reduce overhead, pre-store coordinates in the grid only when render is needed
        if self.X is None: 
            self.X, self.Y = np.array(self.to_pos(np.arange(self.nS))) 
            self.Ox, self.Oy = np.arange(cols+1), np.arange(rows+1)

        # underhood obstacles and a cliffs
        if self.style=='maze':  
            if self.Obstacles is None: self.Obstacles = self.init_cells(self.obstacles)
            self.ax0.pcolormesh(self.Ox, self.Oy, self.Obstacles, edgecolors='lightgray', cmap=self.cmap)
        
        if self.style=='cliff': 
            if self.Cliffs is None: self.Cliffs = self.init_cells(self.cliffs)
            self.ax0.pcolormesh(self.Ox, self.Oy, self.Cliffs, edgecolors='lightgray', cmap=self.cmap)

        # this means that the user wants to draw the policy arrows (actions)
        if 'Q' in kw and underhood=='': underhood='maxQ'
        
        # a placeholder function for extra rendering jobs
        render_ = getattr(self, 'render_'+ underhood)(**kw)
        # windy style needs a bespoke rendering
        if self.style =='windy': self.render_windy()

        if image: self.render_image(saveimg=saveimg)
            
        # to animate clear and plot the Grid
        if animate: clear_output(wait=True); plt.show(); time.sleep(pause)
        #else: plt.subplot(subplot)
    
    #-------------------------helper functions for rendering policies and value functions-----------
    def render_(self, **kw):
        pass # a placeholder for a another drawing if needed
    
    def render_image(self, **kw):
        pass # a placeholder for capturing and saving Grid as images
    
    # renders all states numbers' reprsentation on the grid
    def render_states(self, **kw):
        X,Y  = self.X, self.Y
        for s in range(self.nS): 
            self.ax0.text(X[s]-.5,Y[s]-.5, s, fontsize=13, color='g')




# =============================================================================================
# ================================visualising the policy within the grid=======================

'''
    Next we further add more rendering routines to enhance and enrich the Grid class. 
    Mainly these rendering routines will be used to visualise the policy of an agent either 
    via  π or via Q function. The arrows are used with the quiver function which yields much 
    faster results than using the plt.text(x,y, '→') function.
'''
class Grid(Grid):
    def __init__(self, **kw):
        super().__init__(**kw)

    def init_arrows(self):       
        self._left,      self._right,   self._down,       self._up       = tuple(range(0,4))
        self._left_down, self._left_up, self._right_down, self._right_up = tuple(range(4,8))
        
        # works for quiver and pos, max action can potentially go upto 8! if we are dealing with a grid world
        self.arrows = np.zeros((self.nA,2), dtype=int)
        
        self.arrows[self._left ] =[-1, 0]  # '←'
        self.arrows[self._right] =[ 1, 0]  # '→'
        
        if self.nA>2:
            self.arrows[self._down ] =[ 0,-1]  # '↓'
            self.arrows[self._up   ] =[ 0, 1]  # '↑'

        if self.nA>4:
            self.arrows[self._left_down ]=[-1,-1]  # '↓←'
            self.arrows[self._left_up   ]=[-1, 1]  # '↑←'
            self.arrows[self._right_down]=[ 1,-1]  # '→↓'
            self.arrows[self._right_up  ]=[ 1, 1]  # '→↑'
    

    # renders a policy
    def render_π(self, π=None, **kw): 
        if π is None: π=np.ones(self.nS, dtype=int)
        if self.arrows is None: self.init_arrows()
        X, Y = self.X, self.Y
        U, Z = self.arrows[π].T
        ind = [s for s in range(self.nS) if s not in self.goals and s not in self.obstacles + self.cliffs]
        ind = np.array(ind)
        if ind.any()==False: return
        plt.quiver(X[ind]-.5,Y[ind]-.5,  U[ind],Z[ind],color='b')
  
    # renders a policy deduced from a Q function
    def render_maxQ(self, Q=None, **kw): 
        if Q is None: Q=np.ones((self.nS, self.nA ))
        X, Y = self.X, self.Y
        if self.arrows is None: self.init_arrows()
        U, Z = self.arrows[np.argmax(Q,1)].T
        ind  = np.sum(Q,1)!=0
        if ind.any()==False: return
        plt.quiver(X[ind]-.5,Y[ind]-.5,  U[ind],Z[ind],color='b')
    
    # renders state value function
    def render_V(self, V=None, **kw):
        if V is None: V=np.ones(self.nS)
        X,Y  = self.X, self.Y
        fntsz, clr = 14 - int(self.cols/5), 'b'
        for s in range(self.nS):
            if s in self.obstacles or s in self.goals: continue
            plt.text(X[s]-.7,Y[s]-.7, '%.1f  '% V[s], fontsize=fntsz, color=clr) 
    
    # renders action-state value function
    def render_Q(self, Q=None, **kw):
        if Q is None: Q=np.ones((self.nS, self.nA ))
        X,Y  = self.X, self.Y
        fntsz, mrgn, clr = 12 - (5-self.nA) - int(self.cols/5), 0.4, 'b'
        for s in range(self.nS):
            if s in self.obstacles: continue        
            #  '→', '←', '↑', '↓'
            plt.text(X[s]-mrgn,Y[s]-mrgn, '←%.2f, '% Q[s,0], ha='right', va='bottom', fontsize=fntsz, color=clr) 
            plt.text(X[s]-mrgn,Y[s]-mrgn, '%.2f→  '% Q[s,1], ha='left' , va='bottom', fontsize=fntsz, color=clr)
            if self.nA==2: continue
            plt.text(X[s]-mrgn,Y[s]-mrgn, '↓%.2f, '% Q[s,2], ha='right', va='top'   , fontsize=fntsz, color=clr) 
            plt.text(X[s]-mrgn,Y[s]-mrgn, '%.2f↑  '% Q[s,3], ha='left' , va='top'   , fontsize=fntsz, color=clr) 



#============================================================================================
# ====================== Visualisation for a specialist Grids================================

class Grid(Grid):
    def __init__(self, **kw):
        super().__init__(**kw)
        
        # randwalk related
        self.letters = None                    # letter rep. for states
        
    #--------------------------helper functions specific for some env and exercises---------
    # renders winds values on a grid
    def render_windy(self, **kw):
        for col in range(self.cols): # skipping the first and final states
            plt.text(col+.2,-.5, self.wind[col], fontsize=13, color='k')
        plt.text(6.15,1, '⬆',fontsize=60, color='lightgray')
        plt.text(6.15,4, '⬆',fontsize=60, color='lightgray')
    
    # renders a trace path on a grid
    def render_trace(self, **kw):
        poses = self.to_pos(np.array(self.trace))
        plt.plot(poses[0]-.5, poses[1]-.5, '->c')

    def render_V(self, **kw):
        super().render_V(**kw)
        if self.rows==1: self.render_letters()

    # renders all states letters' reprsentation on the grid
    def render_letters(self, **kw): # for drawing states numbers on the grid
        if self.nS>26: return
        X,Y  = self.X, self.Y
        # to reduce overhead, create the list only when render_letters is needed
        if self.letters is None: self.letters = self.letters_list() 
        for s in range(1,self.nS-1): # skipping the first and final states
            plt.text(X[s]-.5,Y[s]+.02, self.letters[s], fontsize=13, color='g')
    
    def letters_list(self, **kw):
        letters = [chr(letter) for letter in range(ord('A'),ord('A')+(self.nS-2))]
        letters.insert(0, 'G1')
        letters.append('G2')
        return letters



#============================================================================================
#==================================jumping grid !============================================
'''
    We can define a class that allows the agent to jump randomly or to a specific location 
    in the grid without going through intermediate states. This will be used later in other 
    lessons that deal with state representations. Here we pass jGrid to the maze function to 
    obtain an instance of a jumping Grid class without redefining the maze.
'''


class Grid(Grid):
    def __init__(self, jump=1, randjump=True, **kw):
        super().__init__(**kw)
        self.jump = jump
        self.randjump = randjump
        
    #-----------------------------------actions related---------------------------------------
    def step(self, a):
        jump = randint(1, min(self.jump, self.nS - self.s) +1) if self.randjump else self.jump
        if self.jump==1: return super().step(a)
            
        a = self.actions[a]*jump
        if not self.invalid(self.s, a):  
            #print('valid jump')
            self.s += a
        else: 
            #print('invalid jump')
            self.s = max(min(self.s+a, self.nS-1),0)
        
        self.trace.append(self.s)
        reward, done = self.reward() 
        return self.s_(), reward, done, {}
    

# ======================A set of handy functions that will be used a lot===========================
#-------------------------------suitable for control------------------------------------------------
def grid(Grid=Grid, **kw):
    return Grid(gridsize=[8, 10], s0=31, goals=[36], **kw)

def grid8(Grid=Grid, **kw): 
    return grid(Grid=Grid, nA=8, **kw)

def windy(Grid=Grid,  **kw):
    return Grid(gridsize=[7, 10], s0=30, goals=[37], style='windy', **kw)

def cliffwalk(Grid=Grid, **kw):
    return Grid(gridsize=[4, 12], figsize=[12,2], s0=0,  goals=[11], style='cliff', reward='cliffwalk', **kw)

def maze(Grid=Grid, r=6, c=9, **kw):
    return Grid(gridsize=[r,c], s0=r//2*c, goals=[r*c-1], style='maze', **kw)

def maze_large(Grid=Grid, **kw):
    return maze(Grid=Grid, r=16, c=26, figsize=[25,4],**kw)

def maze8(Grid=Grid, **kw): 
    return maze(Grid=Grid, nA=8, **kw)

#-------------------------------suitable for prediction------------------------------------------------
def randwalk(Grid=Grid, nS=5+2, Vstar=None, **kw):
    if Vstar is None: Vstar = np.arange(0,nS)/(nS-1)
    return Grid(gridsize=(1,nS), reward='randwalk', nA=2, goals=[0,nS-1], s0=nS//2, Vstar=Vstar, **kw)

def randwalk_(Grid=Grid, nS=19+2, Vstar=None, **kw):
    if Vstar is None: Vstar = np.arange(-(nS-1),nS,2)/(nS-1)
    return Grid(gridsize=(1,nS), reward='randwalk_', nA=2, goals=[0,nS-1], s0=nS//2, Vstar=Vstar,**kw)

# ------------------------------change default size of cells in jupyter: an indicaiton of successful import----
def resize_cells(size=90):
    display(HTML('<style>.container {width:' +str(size) +'% !important}</style>'))

resize_cells()