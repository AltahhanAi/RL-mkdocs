import rclpy as ros
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg  import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel

import numpy as np
from numpy import Inf
from random import randint
from math import atan2, atan, pi
import matplotlib.pyplot as plt

# ====================================================================================================

def name(): return 'node'+str(randint(1,1000))

class Env(Node):

# initialisation--------------------------------------------------------------
    # frequency: how many often (in seconds) the spin_once is invoked, or the publisher is publishing to the /cmd_vel
    def __init__(self, name=name(), 
                 freq=1/20, n=28, 
                 speed=.5, θspeed=pi/5, 
                 rewards=[30, -10, 0, -1],
                 verbose=False):
        super().__init__(name)
        
        self.freq = freq
        self.n = n
        
        self.speed = speed
        self.θspeed = round(θspeed,2)
        
        self.robot = Twist()
        self.rewards = rewards
        self.verbose = verbose

        # do not change----------------------------------------------------
        self.x = 0 # initial x position
        self.y = 0 # initial y position
        self.θ = 0 # initial θ angle
        self.scans = np.zeros(60) # change to how many beams you are using
        self.t = 0
        
        self.tol = .6  # meter from goal as per the requirement (tolerance)
        self.goals =  [[2.0, 2.0], [-2.0, -2.0]]
        # -----------------------------------------------------------------
        
        self.controller = self.create_publisher(Twist, '/cmd_vel', 0)
        self.timer = self.create_timer(self.freq, self.control)

        self.scanner = self.create_subscription( LaserScan, '/scan', self.scan, 0)
        self.odometr = self.create_subscription( Odometry, '/odom', self.odom, 0)
        
        self.range_max = 3.5
        self.range_min = .28               # change as you see fit
       

        # establish a reset client 
        self.reset_world = self.create_client(Empty, '/reset_world')
        while not self.reset_world.wait_for_service(timeout_sec=2.0):
            print('world client service...')


        # compatibility----------------------------------------------
        nturns = 15 # number of turns robot takes to complete a full circle
        resol = speed/2
        
        θresol = 2*pi/nturns
        dims = [4,4]
        self.xdim = dims[0]  # realted to the size of the environment
        self.ydim = dims[1]  # realted to the size of the environment
        
        self.resol = round(resol,2)
        self.θresol = round(θresol,2)
        
        self.cols  = int(self.xdim//self.resol) +1   # number of grid columns, related to linear speed
        self.rows  = int(self.ydim//self.resol) +1   # number of grid rows,    related to linear speed
        self.orts  = int(2*pi//self.θresol)     +1   # number of angles,       related to angular speed

        self.nC = self.rows*self.cols              # Grid size
        self.nS = self.rows*self.cols*self.orts # State space size
        self.nA = 3
        

        self.Vstar = None # for compatibility
        # --------------------------------------------------------------- 
        # self.rate = self.create_rate(30)
        self.reset()
        
        print('speed  = ', self.speed)
        print('θspeed = ', self.θspeed)
        print('freq   = ', self.freq)

# sensing--------------------------------------------------------------
    # odometry (position and orientation) readings
    def odom(self, odoms):
        self.x = round(odoms.pose.pose.position.x, 1)
        self.y = round(odoms.pose.pose.position.y, 1)
        self.θ = round(self.yaw(odoms.pose.pose.orientation),2) 
        self.odom = np.array([self.x, self.y, self.θ])
        if self.verbose: print('odom = ',  self.odom )
    
    # laser scanners readings
    def scan(self, scans):
        self.scans = np.array(scans.ranges)
        self.scans[scans==Inf] = self.range_max
        # if self.verbose: print('scan = ', self.scans[:10].round(2))
        if self.verbose: print('scan = ', np.r_[self.scans[-5:], self.scans[:5]].round(2))
        
    # converting to the quaternion self.z to Euler
    # see https://www.allaboutcircuits.com/technical-articles/dont-get-lost-in-deep-space-understanding-quaternions/#
    # see https://eater.net/quaternions/video/intro
    
    def yaw(self, orient):
        x, y, z, w = orient.x, orient.y, orient.z, orient.w
        yaw = atan2(2.0*(x*y + w*z), w*w + x*x - y*y - z*z)
        return yaw if yaw>0 else yaw + 2*pi # in radians, [0, 2pi]
    
    # angular distance of robot to a goal.............................................
    def θgoal(self, goal):
        xgoal, ygoal = self.goals[goal] 
        x, y  = self.x, self.y
        θgoal = atan2(abs(xgoal-x), abs(ygoal-y)) # anglegoal
        # if θgoal<=0  θgoal += 2*pi
        return round(θgoal, 2) # in radians, [0, 2pi]
    
    # Eucleadian distance of robot to nearest goal......................................   
    def distgoal(self):
        dists = [Inf, Inf]        # distances of robot to the two goals
        for goal, (xgoal, ygoal) in enumerate(self.goals):
            dists[goal] = (self.x - xgoal)**2 + (self.y - ygoal)**2
            
        dist = min(dists)         # nearest goal distance
        goal = dists.index(dist)  # nearest goal index
        
        if self.verbose: print('seeking goal ____________________', goal)
        return round(dist**.5, 2), goal
    
    # robot reached goal ...............................................................
    def atgoal(self):
        tol, x, y = self.tol,  self.x, self.y
        atgoal = False
        for xgoal, ygoal in self.goals:
            atgoal = xgoal + tol > x > xgoal - tol  and  \
                     ygoal + tol > y > ygoal - tol
            
            if atgoal: print('Goal has been reached woohoooooooooooooooooooooooooooooo!!'); break
        return atgoal

    # robot hits a wall...................................................................
    def atwall(self, rng=5):
        # check only 2*rng front scans for collision, given the robot does not move backward
        return np.r_[self.scans[-rng:], self.scans[:rng]].min() < self.range_min 
        #return self.scans.min()<self.range_min
        
    # reward function to produce a suitable policy..........................................
    def reward(self, a, imp=2):
        stype = [self.atgoal(), self.atwall(), a==1, a!=1].index(True)
                
        dist, goal = self.distgoal()
        θgoal = self.θgoal(goal)

        # get angular distance to reward/penalise robot relative to its orientation towards a goal
        θdist = abs(self.θ - θgoal)
        if goal==1: θdist -= pi
        θdist = round(abs(θdist),2)

        reward = self.rewards[stype] 
        if stype: reward -= imp*(dist+θdist) 
        
        if self.verbose: 
            print('reward components=', 
                  'Total reward=', reward, 
                  'state reward=', self.rewards[stype],
                  'goal dist=', dist, 
                  '|θ-θgoal|=', θdist)
                #   'θrobot=', self.θ, 
                #   'θgoal =', θgoal, 
        
        # reset without restarting an episode if the robot hits a wall
        if stype==1: self.reset() 

        return reward, stype==0, stype==1

# State representation-------------------------------------------------
   # change this to generate a suitable state representation
    def s_(self):
        
        self.xi = int((self.x+self.xdim/2)//self.resol)     # x index = col, assuming the grid middle is (0,0)
        self.yi = int((self.y+self.ydim/2)//self.resol)     # y index = row, assuming the grid middle is (0,0)
        
        # pi/2 to be superficially resilient to slight angle variation to keep θi unchanged
        self.θi = int((self.θ+pi/2)%(2*pi)//self.θresol)
        
        self.si = self.xi + self.yi*self.cols     # position state in the grid
        self.s = self.nC*(self.θi) + self.si      # position state with orientation
        if self.verbose: print('grid cell= ', self.si, 'state = ', self.s)
        return self.s 


# Control--------------------------------------------------------------    
    def spin_n(self, n):
        for _ in range(n): ros.spin_once(self)
            
    def control(self): 
        self.controller.publish(self.robot) 
        
    # move then stop to get a defined action
    def step(self, a=1, speed=None, θspeed=None):
        if speed is None: speed = self.speed
        if θspeed is None: θspeed = self.θspeed

        self.t +=1
        if self.verbose: print('step = ', self.t)
        
        if  a==-1: self.robot.linear.x  = -speed  # backwards
        elif a==1: self.robot.linear.x  =  speed  # forwards
        elif a==0: self.robot.angular.z =  θspeed # turn left
        elif a==2: self.robot.angular.z = -θspeed # turn right

        # Now move and stop so that we can have a well defined actions  
        self.spin_n(self.n) if a==1 else self.spin_n(self.n//2)
        self.stop()

        reward, done, wall = self.reward(a)
        return self.s_(), reward, done, {}
        
    def stop(self):
        self.robot.linear.x = .0
        self.robot.angular.z = .0
        #  spin less so that we have smoother actions
        self.spin_n(self.n//8)

# reseting--------------------------------------------------------------
    def reset(self):
        print('resetting world..........................................')
        # to ensure earlier queued actions are flushed, there are better ways to do this
        for _ in range(1): self.reset_world.call_async(Empty.Request())
        for _ in range(2): self.step(a=1, speed=0.001)  # move slightly forward to update the odometry to prevent repeating an episode unnecessary
        for _ in range(1): self.reset_world.call_async(Empty.Request())
        
        return self.s_()

    # for compatibility, do not delete
    def render(self, **kw):
        pass

