
<!-- By: Abdulrahman Altahhan, Feb 2025 -->

#  Welcome to Reinforcement Learning and Robotics!

Welcome to the Reinforcement Learning and Robotics module, where you learn the fundamentals of reinforcement learning(RL) with various simulaitons including robotics. The module focuses on RL as a general robust framework that allows us to deal with autonomous agent learning. 

In each unit, you will learn essential RL ideas and algorithms and see how these can be implemented in simplified environments. Each lesson involves some reading material, along with some videos explaining these lessons. This will then be followed by running and experimenting with the Jupyter notebooks we provide, which implement the main ideas you have read and show you the RL algorithms you studied in action. It is essential that you read the material and engage with these notebooks in the way you see fit. This includes studying and running the provided code to gain insight into the covered algorithms, observing the algorithm convergence behaviour, experimenting with the hyperparameter and their effect on the agent's behaviour, and taking a turn to implement some concepts that were left out for you. 
 
There are also some lessons that familiarise you with robotics as an application domain, which covers some simple concepts in robotics without delving deep into classical robotics, which is outside the scope of this module. Our units conclude with a simple, practical tutorial on utilising a simulated robot. You will use the provided code to control a simulated mobile robot called TurtleBot. The lessons are meant to gradually build your ability to deal with atonomous agents in a simplified environment. The final project will focus on comparing different RL solutions to solve a simulated robot navigation problem. We will provide you access to an Azure Ubuntu virtual machine already set up with ROS2 to run the sheets. You do not need to set up your own VM.

Reinforcement Learning (RL) is dedicated to acquiring an optimal policy to enhance agent performance. Traditionally, RL involves the agent seeking an optimal policy to maximise cumulative discounted rewards garnered by navigating various environmental states. While the agent is commonly perceived as a physical entity interacting with its surroundings, it can also encompass abstract systems, with states representing system configurations or settings. Notably, recent RL advancements have ventured into novel territories, such as optimising language models like LLMs for perplexity or other metrics.

In this module, our focus primarily revolves around simulated agents, aligning with the nature of our programme. However, the underlying principles remain universal and applicable across diverse scenarios. The concept of guiding learning through rewards is deeply ingrained in biological organisms, from complex human brains to single-cell organisms like amoebas, all driven by the innate urge to maximise survival and proliferation.

While RL offers tremendous efficacy when configured appropriately, it is susceptible to spectacular failure when its conditions are unmet. Its inherent stochasticity adds another layer of complexity, contributing to its volatile nature. Nonetheless, this volatility serves as a catalyst for researchers to delve deeper into understanding the governing rules of RL processes.

Our module adopts a pragmatic approach, aiming to provide you with a solid theoretical grounding in RL without overly delving into intricate mathematical details. Simultaneously, we will attempt to equip you with practical skills and techniques to harness RL's benefits effectively. This balanced approach ensures that you grasp RL's essence while gaining valuable real-world application tools.

<video width="470" height="200" controls>
  <source src="/videos/Wecome Video.mp4" type="video/mp4">
  Wecome Video
</video>

<!-- <iframe src="https://leeds365-my.sharepoint.com/personal/scsaalt_leeds_ac_uk/_layouts/15/embed.aspx?UniqueId=2efc8e37-2e42-4b77-b694-b68994652e23&embed=%7B%22ust%22%3Atrue%2C%22hv%22%3A%22CopyEmbedCode%22%7D&referrer=StreamWebApp&referrerScenario=EmbedDialog.Create" width="640" height="360" frameborder="0" scrolling="no" allowfullscreen title="Wecome Video.mkv"></iframe> -->




## RL in context: advantages and challenges
In our RL coverage, you will notice that we do not do classical robotics and we believe it is not the way to go except possibly for industrial robots. So, we do not need motion planning or accurate trajectory calculations/kinematics to allow the robot to execute a task, we simply let it interact with its environment and learn by itself how to solve the task and this is what reinforcement learning is about. In our coverage we also do not supervise or directly teach the robot, this type of interference is called imitation. 

This is all great but what about challenges? Obviously, we still have several challenges to this approach. One is the number of experiments required to learn the task which can be numerous, which exposes the physical agents (real robots) to the risk of wear and tear due to repetition and can be very lengthy and tedious for the human that is supervising the task. This can be partially overcome by starting in simulation and then moving to a real robot with a technique called [sim-to-real](https://ai.googleblog.com/2021/06/toward-generalized-sim-to-real-transfer.html) where we can employ GANs(generative adversarial neural networks).  The other challenge is the time required for training regardless whether it is in simulation or in real scenarios. 


The origin of these problems is actually the exploitation/exploration needed by RL algorithms which is in the heart of what we will be exploring in all of our RL coverage. Reducing the amount of training required is important and remains an active area for research. One approach is via experience replay and the other is via eligibility traces as we shall see later.

## Textbook
<!-- The accompanying textbook of this and consecutive units is the Sutton Barto book Introduction to RL available online [here](http://incompleteideas.net/book/RLbook2020.pdf). You do not need to read the book's corresponding chapters, but they provide a further reading for you to understand the maetrial further and delve deeper into the subject.
Please note that we explain the ideas of RL from a practical perspective and not from a theoretical perspective which is already covered in the textbook. -->
The primary textbook for this unit and the following ones is Introduction to Reinforcement Learning by Sutton and Barto, available online [here](http://incompleteideas.net/book/RLbook2020.pdf). While reading the corresponding chapters is not required, they serve as valuable supplementary material to enhance your understanding and explore the subject in greater depth.

**list of symbols**

- $v0$: denotes an initial value
- $θ$: denotes a threshold
- $nS$: denotes state space dimension 
- $nA$: denotes actions space dimension 
- $nR$: denotes rewards space dimension
- $nU$: denotes the number of updates
- goal: a terminal state

- $r$: current step reward
- $s$: current step state
- $a$: current step action
- $rn$: next step reward
- $sn$: next step state
- $an$: next step action

- $α$: learning rate
- $ε$: exploration rate
- $dα$: decay factor for α
- $dε$: decay factor for ε
- $G(t+1,t+n)$: the return between time step t+1 and t+n

- Rs: is the sum of rewards of an episode
- Ts: is the steps of a set of an episode
- Es: is the errors (RMSE) of an episode


## Module Plan
We cover the tabular solution methods in the first three units, while approximate solution methods will be covered in subsequent units.

Tabular and approximate solution methods fall under two types of RL methods that we will attempt to deal with 

1. Prediction methods: AKA Policy Evaluation Methods that attempt to find the best estimate for the value-function $V$ or action-value function $Q$ for a policy $\pi$.
2. Control methods: AKA Policy Improvement Methods that attempt to find the best policy $\pi_*$, often by starting from an initial policy and then moving into a better and improved policy.

Control methods, or policy improvement methods, in turn, falls under two categories:

1. Methods that improve the policy via improving the action-value function $Q$
2. Methods that improve the policy directly $\pi$

We start by assuming that the policy is fixed. This will help us develop algorithms predicting the state‘s value function (expected return). Then, we will move to the policy improvement methods, i.e., methods that help us compare our policy with other policies and move to a better policy when necessary. We then move to seamlessly integrating both for control case (policy and value iteration methods). Finally, we cover policy gradient methods that improve the policy directly.

Note that the guarantee from the policy-improvement theorem no longer applies when we move from the table representation of the value function for small state space to the parametric function approximation representation for large state space. This will encourage us to move to direct policy-improvement methods instead of improving the policy via improving the value function.


## Table of Contents
**Unit 1**

1. [Tabular Methods](unit1/lesson1/lesson1.md)  
2. [K-Arm Bandit](unit1/lesson2/lesson2.md)  
3. [MDP](unit1/lesson3/lesson3.md)  
4. [ROS](unit1/lesson4/lesson4.md)  

**Unit 2**

5. [Dynamic Programming](unit2/lesson5/lesson5.md)  
6. [Monte Carlo](unit2/lesson6/lesson6.md)  
7. [Mobile Robots](unit2/lesson7/lesson7.md)  

**Unit 3**

8. [Temporal Difference](unit3/lesson8/lesson8.md)  
9.  [n-Step Methods](unit3/lesson9/lesson9.md)  
10. [Planning in RL (optional)](unit3/lesson10/lesson10.md)  
11. [Localisation and SLAM](unit3/lesson11/lesson11.md)  

**Unit 4**

12. [Function Approximation Methods](unit4/lesson12/lesson12.md)  
13. [Linear Approximation for Prediction](unit4/lesson13/lesson13.md)  
14. [Linear Approximation for Control](unit4/lesson14/lesson14.md)  

**Unit 5**

15. [Linear Approximation with Eligibility Traces (Prediction and Control)](unit5/lesson15/lesson15.md)  
16. [Nonlinear Approximation for Control](unit5/lesson16/lesson16.md)  
17. [Application on Robot Navigation](unit5/lesson17/lesson17.md)  

**Unit 6**

18. [Application on Games (optional)](unit6/lesson18/lesson18.md)  

## Code Structure and Notebooks Dependecies

<!-- Worksheets can be cloned and launched in GitHub Codespaces (repo is: AltahhanAi/RL-worksheets)
[![Open in GitHub Codespaces-](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?repository=AltahhanAi/RL-worksheets) -->

We have provided you with two [RL libraries](/RLlibrary.zip){:download="RLlibrary.zip"} designed for this module. One has bespoke environments and one that has base RL classes that makes working with algorithms very easy and as close as it can be to just provide an update rule.

**Important note: Please place all worksheets in one folder, and inside this folder you must have the downloaded libraries folders (env and rl) to allow the imports to work appropriately**.

## Installing other libraries that will be needed later
<!-- general -->
```
!pip install --upgrade pip
!pip install opencv-python
!pip install scikit-learn
!pip install matplotlib
!pip install tqdm 
```

<!-- Better Readability -->
```
!pip install jupyterlab
!pip install jupyterthemes

!jt -t solarizedl -T -N  # -T, -N keeps the toolbar and header
```

### available themes:
- oceans16 
- grade3 
- chesterish 
- solarizedl 
- solarizedd 
- gruvboxl
- !jt -r # resets back to the default theme

## Better Readability and Audibility
For better readability and experience, please use Jupyter Lab or Vcode(if you are using Azure VM) to navigate between the different notebooks easily. If you want to use Jupyter Notebooks and not Jupyter Lab, we recommend increasing the cells' width for a better experience. We provided a function that increase your notebook width which is envoked automatically when you import an environment (grid particularly). You may want to utilise also the table of contents button in Jupetr Lab.
For better audibility of the provided [videos](https://leeds365-my.sharepoint.com/:f:/g/personal/scsaalt_leeds_ac_uk/EsNA0S_Tb95Mtdr5Q00OOoIBpCdgaV_UYSpa1GEESlJXMg?e=rjJsDB) please click on the **'noise supression'** button, you may want to speed up as per your need.
 