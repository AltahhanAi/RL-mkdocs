<script>
  window.MathJax = {
    tex: {
      tags: "ams",  // Enables equation numbering
    //   displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>

# Lesson 6 - Tabular Methods: Monte Carlo

**Learning outcomes**

By the end of this lesson, you will be able to: 

1. Understand the difference between learning the expected return and computing it via dynamic programming.
2. Understand the strengths and weaknesses of Monte Carlo (MC) methods.
3. Appreciate that MC methods need to wait until the end of the task to obtain their estimate of the expected return.
4. Compare MC methods with dynamic programming methods.
5. Understand the implication of satisfying and not satisfying the explore-start requirement for MC control and how to mitigate it via the reward function.
6. Understand how to move from prediction to control by extending the \( V \) function to a \( Q \) function and make use of the idea of generalized policy iteration (GPI).
7. Understand how policy gradient methods work and appreciate how they differ from value function methods.

## Overview

In this lesson, we develop the ideas of Monte Carlo methods. Monte Carlo methods are powerful and widely used in settings other than reinforcement learning (RL). You may have encountered them in a previous module where they were mainly used for sampling. We will also use them here to sample observations and average their expected returns. Because they average the returns, Monte Carlo methods have to wait until *all* trajectories are available to estimate the return. Later, we will find out that Temporal Difference methods do not wait until the end of the episode to update their estimate and outperform MC methods.

Note that we have now moved to *learning* instead of *computing* the value function and its associated policy. This is because we expect our agent to learn from *interacting with the environment* instead of using the dynamics of the environment, which is usually hard to compute except for a simple lab-confined environment.

Remember that we are dealing with *expected return*, and we are either finding an *exact solution for this expected return* as when we solve the set of Bellman equations or finding an *approximate solution for the expected return* as in DP or MC.  
Also, remember that the expected return for a state is the future cumulative discounted rewards, given that the agent follows a specific policy.

One pivotal observation that summarizes the justification for using MC methods over DP methods is that it is often the case that we are able to interact with the environment instead of obtaining its dynamics due to its complexity and intractability. In other words, interacting with the environment is often more direct and easier than obtaining the model of the environment. Therefore, we say that MC methods are model-free.

## Plan

As usual, in general, there are two types of RL problems that we will attempt to design methods to deal with:

1. **Prediction problems**  
   For these problems, we will design Policy Evaluation Methods that attempt to find the best estimate for the value function given a policy.

2. **Control problems**  
   For these problems, we will design Value Iteration methods that utilize Generalized Policy Iteration. These methods attempt to find the best policy by estimating an action-value function for the current policy and then moving to a better and improved one by often choosing a greedy action. They minimize an error function to improve their value function estimate, which is used to deduce a policy.  
   We will then move to Policy Gradient methods that directly estimate a useful policy for the agent by maximizing its value function.

We start by assuming that the policy is fixed. This will help us develop an algorithm that predicts the state space's value function (expected return). Then we will move to the policy improvement methods, i.e., these methods that help us compare and improve our policy with respect to other policies and move to a better policy when necessary. Finally, we move to the control case (policy iteration methods).


<!-- ### Bellman Equations(reminder)


# Lesson 6 - Tabular Methods: Monte Carlo

**Learning outcomes**

By the end of this lesson, you will be able to: 

1. Understand the difference between learning the expected return and computing it via dynamic programming.
2. Understand the strengths and weaknesses of Monte Carlo (MC) methods.
3. Appreciate that MC methods need to wait until the end of the task to obtain their estimate of the expected return.
4. Compare MC methods with dynamic programming methods.
5. Understand the implication of satisfying and not satisfying the explore-start requirement for MC control and how to mitigate it via the reward function.
6. Understand how to move from prediction to control by extending the \( V \) function to a \( Q \) function and make use of the idea of generalized policy iteration (GPI).
7. Understand how policy gradient methods work and appreciate how they differ from value function methods. -->

## Bellman Equations (reminder)

As we have seen in a previous lesson, the Bellman equations form the foundation of many RL methods. They define the relationship between the value of a state and the values of successor states.

For a given policy \( \pi \), the Bellman Equation for the state-value function is:

$$
V^\pi(s) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s \right]
$$

For the action-value function \( Q^\pi(s, a) \):

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma Q^\pi(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a \right]
$$

### Dynamic Programming is Model-Based

As we have seen in the last lesson, Dynamic Programming uses Bellman equations as update rules to iteratively compute value functions and policies using two key steps:

1. **Policy Evaluation**: Uses the Bellman equation to obtain an evaluation for a given policy.
2. **Policy Improvement**: Uses the Bellman optimality equation to update the policy towards a better policy.

However, DP has a major limitation: it requires complete knowledge of the dynamics model \( p(s',r | s, a) \), or transition model \( p(s'| s, a) \), which is often unavailable in real-world scenarios. In other words, it is a model-based method.

## Monte Carlo: Model-Free Methods

Monte Carlo (MC) methods are a class of RL algorithms used to estimate value functions and optimize policies by using **sampled episodes** instead of full models of the environment. Unlike Dynamic Programming (DP), which requires a known dynamics model \( p(s',r | s, a) \), Monte Carlo methods learn **directly from experience** by averaging observed rewards.

Monte Carlo (MC) methods estimate value functions without requiring a dynamics model. Instead, they rely on sampled episodes of experience. The key idea is to approximate value functions by averaging observed returns over multiple episodes.

The fundamental principle behind MC methods is that averaging samples from a distribution approximates its expectation. Since state-value and action-value functions are defined as expected returns, averaging returns over a sufficiently large number of episodes provides a reliable estimate of these functions.

There are some technical conditions that must be met for this approximation to hold, which we will reference when necessary.

### Return Definition

For an episode consisting of states, actions, and rewards:

\[
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots
\]

The Monte Carlo estimate of \( V(s) \) is the average return from all episodes where state \( s \) was visited.

\[
V(s) \approx \frac{1}{N} \sum_{i=1}^{N} G_i(s)
\]

Where:

- \(i\) refers to the index over the episodes, not the time steps.
- \( G_i(s) \) is the return (sum of discounted rewards) observed from episode \(i\), starting at state \(s\).
- \(N\) is the total number of episodes used to average the returns.

## First-Visit Monte Carlo Policy Evaluation (Prediction)

Because MC methods depend entirely on experience, a natural way to approximate the cumulative future discounted reward is by taking their average once they become available through experience. So we must collect the cumulative *future* discounted reward once this experience has elapsed. In other words, we need to take the sum *after* the agent has finished an episode for all the rewards obtained from the current state to the end of the episode. Then we average those returns over all of the available episodes. Note that MC methods only apply for episodic tasks, which is one of its limitations, in addition to having to wait until the episode is finished.

Note also that the agent can visit the same state more than once inside the same episode. One question that arises from the above averaging is: which visit should be counted? 

We can take the sum starting from the first visit, or every visit, each yielding a different algorithm. The first-visit algorithm is more suitable for tabular methods, while the every-visit algorithm is more suitable when using function approximation methods (such as neural networks).

One could argue that since we want to estimate the value of a state based on the full horizon of rewards obtained until the end of an episode, we should include only the *first visit of the state* in the average. This is the basis of the First-visit Monte Carlo (MC) policy evaluation method. First-visit MC estimates the value of a state by averaging the returns from the first time that state is encountered in each episode. Below we show the pseudocode for this algorithm.

\(
\begin{array}{ll}
\textbf{Algorithm: }  \text{First-Visit Monte Carlo Policy Evaluation} \\
\textbf{Input: } \text{Episodes generated under policy } \pi \\
\textbf{Initialize: }  V(S) \leftarrow 0, N(S) \leftarrow 0, \forall S \in \mathcal{S} \\
\textbf{For each episode: } & \\
\quad \text{Generate an episode: } (S_0, A_0, R_1, S_1, \dots, S_T) & \\
\quad G \leftarrow 0 & \\
\quad \textbf{For each step } t \textbf{ from } T-1 \textbf{ to } 0: & \\
\quad \quad G \leftarrow \gamma G + R_{t+1} & \\
\quad \quad \text{If } S_t \text{ appears first in the episode:} & \\
\quad \quad \quad N(S_t) \leftarrow N(S_t) + 1 & \\
\quad \quad \quad V(S_t) \leftarrow V(S_t) + \frac{1}{N(S_t)}(G - V(S_t)) & \\
\textbf{Return: } V(S), \forall S \in \mathcal{S} \\
\end{array}
\)

### Random Walk Problem

The Random Walk problem is a simple **Markov Reward Process (MRP)** used to illustrate value estimation methods. It consists of a finite, linear chain of states, where an agent moves randomly left or right until reaching one of the two terminal states.

#### Problem Setup

- Typically, there are five non-terminal states labeled \( A, B, C, D, E \) on a 1-d grid world. Another variation uses 21 states.
- Two terminal states exist at both ends.
- The agent starts in the center and moves *randomly* left or right with equal probability.
- The episode ends when the agent reaches a terminal state.
- A *reward of +1* is received upon reaching the right terminal state, while the left terminal state gives a *0 reward*.

This random walk problem is often used to demonstrate **Monte Carlo** and **Temporal-Difference (TD)** learning methods for estimating state-value functions. Below we show this problem.


![png](output_9_0.png)  
  
            A     B     C     D     E

Let’s now run the MC1st algorithm with a useful visualization. The plot will display the true state values of the random walk as black points connected by a black line. The agent’s estimated values are represented by blue points and a blue line, allowing us to visually assess how closely the learned values for states A–D match their actual values. This visualization is achieved by setting `plotV=True`.

Next to this, an error plot tracks the total error per episode, providing insight into the learning process. Each episode starts from the middle state (C) and ends upon reaching either the far-left or far-right terminal states. These terminal states are not labeled since they do not have values to estimate. This tracking of errors is achieved by setting `plotE=True`.


```python
mc = MC1st(env=randwalk(), episodes=100, plotV=True, plotE=True, seed=1).interact()
```
![png](output_40_0.png)

We use **demoV to implicitly pass plotE=True, plotV=True, animate=True, whenever we want to demo a prediction algorithm.


## Policies
Before we move into control, we need to briefly discuss types of policies that balance exploration and exploitation.

### $\epsilon$-Greedy Policy

The $\epsilon$-greedy policy is a popular action selection strategy that balances exploration and exploitation. The policy chooses the action with the highest estimated value most of the time, but with probability \(\epsilon\), it selects an action randomly to encourage exploration.

Mathematically, the $\epsilon$-greedy policy can be defined as:


\[
\pi(a|s) = 
\begin{cases} 
\frac{\epsilon}{|A|}, & \text{with probability } \epsilon \\
1 - \epsilon + \frac{\epsilon}{|A|}, & \text{for the action with the highest value} \\
0, & \text{for all other actions}
\end{cases}
\]

Where:
- \(\epsilon\) is the probability of exploring (random action).
- \(|A|\) is the total number of actions available.
- The action with the highest value \(Q(s,a)\) is selected with probability \(1-\epsilon + \frac{\epsilon}{|A|}\).

### Softmax Policy

The softmax policy selects actions based on a probability distribution that is a function of the action values. The policy assigns a higher probability to actions with higher expected returns, and the probability of selecting action \(a\) in state \(s\) is proportional to the exponential of its action-value \(Q(s, a)\).

Mathematically, the softmax policy is given by:

\[
\pi(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_{b \in A} e^{Q(s,b)/\tau}}
\]

Where:
- \(\tau\) is the temperature parameter that controls the level of exploration. A high \(\tau\) encourages more exploration (more uniform distribution), and a low \(\tau\) leads to more exploitation (choosing the highest value action).
- \(Q(s,a)\) is the action-value function.

In this way, the softmax policy ensures that actions with higher values are more likely to be chosen, but there is always a chance to explore other actions.

## First-visit MC control
Now let us extend our first-visit MC prediction to control by updating the Q action-value function instead of the state-value function V.


\(
\begin{array}{ll}
\textbf{Algorithm: }  \text{First-Visit Monte Carlo Control (Exploring Starts)} \\
\textbf{Input: } \text{Episodes generated under an exploring-starts policy} \\
\textbf{Initialize: }  Q(S, A) \leftarrow 0, N(S, A) \leftarrow 0, \forall S \in \mathcal{S}, A \in \mathcal{A}(S), \pi(S) \leftarrow \text{arbitrary policy}, \forall S \in \mathcal{S} \\
\textbf{For each episode: } & \\
\quad \text{Generate an episode: } (S_0, A_0, R_1, S_1, A_1, \dots, S_T) & \\
\quad G \leftarrow 0 & \\
\quad \textbf{For each step } t \textbf{ from } T-1 \textbf{ to } 0: & \\
\quad \quad G \leftarrow \gamma G + R_{t+1} & \\
\quad \quad \text{If } (S_t, A_t) \text{ appears first in the episode:} & \\
\quad \quad \quad N(S_t, A_t) \leftarrow N(S_t, A_t) + 1 & \\
\quad \quad \quad Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)}(G - Q(S_t, A_t)) & \\
\quad \quad \text{Update policy: } \pi(S_t) \leftarrow \arg\max_A Q(S_t, A) & \\
\textbf{Return: } Q(S, A), \pi(S), \forall S \in \mathcal{S}, A \in \mathcal{A}(S) \\
\end{array}
\)


### Applying MC on a control problem

Similar to what we did for prediction, we get help from a dictionary that stores a set of useful configurations that we use often. In the case of control, the most useful is plotting the number of steps the agent took to reach a terminal state in each episode or the sum of rewards the agent collected in each episode. Each one of these plots can be useful for certain tasks. Bear in mind that if the reward is given only for reaching the goal location or terminal state, the sum of the rewards plot would be a constant line that does not convey useful information. Below we show each.

Unfortunately, applying the MC control algorithm with the default reward function will not yield a useful policy. This is because the explore-start condition is not satisfied (refer to section 5.4 of our book). In addition, averaging solutions may not perform well because they do not track a changing policy well for non-stationary problems (most of the control problems are non-stationary). To see this, uncomment the lines in the cell below and run it. (Note that we have set up the priorities of the actions in a way that will show this issue (right comes before left and down before up). demoQ is a dictionary that passes visualization values to the MDP control algorithm.



```python
mc = MC1stControl(env=grid(), γ=1, episodes=200,  seed=10, **demoQ()).interact()
```
![png](output_67_0.png)

### Exploration-Exploitation and Exploring Starts

- Exploration-Exploitation Tradeoff: In reinforcement learning, the agent must balance exploration (trying new actions to discover better long-term rewards) with exploitation (choosing known actions that yield high rewards based on current knowledge).

In exploration, the agent tries actions that it has not explored enough yet. This is necessary to gather more information about the environment. In exploitation, the agent uses its current knowledge to select actions that maximize the immediate reward based on its existing value estimates. Both of these need to happen for the agent to learn effectively. $\epsilon$-Greedy and softmax policies have built-in exploration-exploitation capabilities.


<!-- - $\epsilon$-Greedy Policy: A commonly used policy to handle exploration-exploitation:
    - With probability \( \epsilon \), the agent chooses a random action (exploration).
    - With probability \( 1 - \epsilon \), the agent chooses the action that maximizes the value estimate (exploitation).

- Softmax Policy: Instead of choosing actions randomly, the softmax policy selects actions based on a probability distribution derived from the action values. Actions with higher values are more likely to be chosen, but there is always a non-zero chance of selecting less optimal actions (exploration). -->

- Exploring Starts: This is a method used to ensure complete exploration of the state-action space. In exploring starts, each episode begins with a random state and a random action. This guarantees that over many episodes, every state-action pair will eventually be visited, ensuring unbiased value estimation. This condition dictates that all states must be randomly started with to guarantee convergence. It is needed for First-visit MC Control (1stMCC) since it relies on getting enough coverage for all of the states.

Exploring starts guarantees initial exploration and helps ensure all state-action pairs are visited, making it easier for the agent to discover good policies without biases. Exploration-exploitation policies (like $\epsilon$-greedy or softmax) are methods to balance exploration and exploitation during the agent's learning process. These policies allow for exploration to continue throughout the agent's learning, which is particularly useful when the environment is unknown. Exploration-exploitation balance plays an important role when we move into online methods. But they are still important to achieve the end of the episode, and $\epsilon$-greedy is useful to avoid the exploring-start condition.

In short, exploring starts ensures full exploration at the beginning of episodes, while exploration-exploitation policies control the balance between exploration and exploitation throughout the learning process.

### The role of the discount factor $\gamma$ for delayed reward
**Important Note**
It is always the case that when we use a *delayed reward* (which is the default reward for our Grid class), the discount factor $\gamma$ **must not be set to 1**. This is because the sum of the discounted rewards of each visited state will be equal to the delayed reward itself, which will not give any particular advantage to follow a shorter path, yielding a useless policy. Therefore, we can solve this issue:  
1. either by providing a discounted value for $\gamma$ that is less than 1.  
2. or by changing the reward to have intermediate steps reward, which, when accumulated, will provide distinguished sums for the different paths and hence help distinguish the shortest path or the policy that will yield an optimal reward.


### Solution 1
Below we show how we can simply reduce $\gamma$ to solve this issue.


```python
mc = MC1stControl(env=grid(), γ=.99, episodes=30, seed=10, **demoTR()).interact()
``` 
![png](output_70_0.png)

### Solution 2
Also, we can compensate for the above issue by setting up a reward function that allows the agent to quickly realise when it is stuck in a non-useful policy. One way to do this is by designing a reward structure that gives immediate feedback for the agent's actions, guiding it towards more efficient policies. This can be done by:

1. **Providing Intermediate Rewards:** Instead of only rewarding the agent when it reaches the terminal state, we can give small rewards for progressing towards the goal. This helps the agent understand the immediate consequences of its actions and prevents it from being stuck in a loop without rewards.

2. **Penalising Suboptimal Actions:** Introducing penalties for actions that lead to non-productive states can guide the agent away from unhelpful policies. For example, moving further away from the goal could result in a small negative reward, helping the agent learn to avoid those actions.

3. **Shaping the Reward Function:** A reward shaping approach can give the agent additional clues about how good or bad its actions are in a more fine-grained manner. This can be especially useful in environments where the final reward is too delayed or sparse to provide clear guidance.

4. **Negative Rewards for Unproductive States:** If an agent repeatedly takes actions that lead to dead ends or unhelpful paths, it can be penalised with negative rewards, encouraging it to explore more efficient alternatives.

With these adjustments, the agent can more effectively learn from its interactions with the environment and avoid getting stuck in poor policies. By incorporating some or all of these strategies into the reward function, we give the agent more guidance and the ability to identify and correct non-optimal behaviours.

Below we show approach 1 (intermediate rewards) since we are dealing with a simple environment.

```python
env1 = grid(reward='reward_1')
mcc = MC1stControl(env=env1, episodes=30, seed=0, **demoTR()).interact()
```    
![png](output_72_0.png)


Compare the above policy with the one produced by the DP solution in Lesson 2. You will notice that the MC solution does not provide a comprehensive solution from all states because the starting position is fixed. In contrast, Dynamic Programming (DP) assumes access to a complete model of the environment and typically considers all possible state-action pairs during its computation. This allows DP to evaluate and improve policies more systematically across the entire state space.

The exploration nature of the MC policy, however, allows the agent to develop an understanding of the environment through its Q function, which helps the agent figure out the best action to take when it finds itself in a particular state. The Markovian property ensures that the agent’s decisions are based on the current state, and the agent can use this information to guide its learning process effectively, even if the start state is fixed. In this way, the agent gradually builds a policy by sampling trajectories and averaging the returns over time.

One thing you might have noticed is that, despite the simplicity of the task, the agent occasionally took detours from the most straightforward path to the goal. This happens because we are using an *ε-greedy policy* by default, which means that there is a 10% chance of the agent taking random exploratory actions at each step. This randomness can cause the agent to deviate from the optimal path, even though the policy is generally headed in the right direction. 

However, this should not prevent the *maxQ policy* (the one that chooses the action with the highest action-value) from guiding the agent towards the goal. The detours happen because of the inherent nature of Monte Carlo methods, which rely on sampling and averaging across multiple episodes. The stochastic nature of MC methods means that, although the agent’s policy points towards the goal on average, individual episodes might show some variation due to exploration.

In the next section, we will explore how we can mitigate these detours and overcome the inherent randomness of the ε-greedy policy. One possible approach is to fine-tune the exploration rate, but this requires a fair amount of trial and error and is not a straightforward process. While exploring the balance between exploration and exploitation is a key part of reinforcement learning, achieving a fully optimal policy will involve careful consideration of how exploration is managed and possibly modifying the reward function or policy to limit unnecessary exploration.


We can play with the exploration but that needs lots of trail and is not straightforward.

```python
mc = MC1stControl(env=grid(), γ=.97, episodes=50, ε=.5, dε=.99, seed=20, **demoTR()).interact()
```

![png](output_76_0.png)


```python
mcc = MC1stControl(env=grid(reward='reward_1'), γ=.97, episodes=100, ε=0.9, dε=.999, seed=20, **demoTR()).interact()
```
    
![png](output_78_0.png)
    
## Every-visit MC Prediction
In every-visit Monte Carlo, the value of a state is updated based on the average of returns observed from *all* visits to that state, not just the first visit. For a state \( s \), let \( G_j(s) \) be the return from the \( j \)-th occurrence of \( s \) in some episode. The every-visit MC estimate of \( V^\pi(s) \) is given by: 

\[
V^\pi(s) \approx \frac{1}{N_s} \sum_{j=1}^{N_s} G_j(s)
\]

where:  
- \( N_s \) is the total number of times state \( s \) was visited across all episodes.  
- \( G_j(s) \) is the return obtained from the \( j \)-th visit to \( s \), computed as the sum of rewards from that visit to the end of the episode.  

### First-Visit vs. Every-Visit MC
|   | **First-Visit MC** | **Every-Visit MC** |
|---|------------------|------------------|
| **Update Rule** | Uses the return from the first visit to a state in each episode | Uses returns from **all** visits to a state in each episode |
| **Bias/Variance** | Lower bias but higher variance | Higher bias but lower variance |
| **Suitability** | Works well when visits to a state within an episode are correlated | Suitable when visits are independent and representative of overall state behavior |

<!-- ### When to Use Which? -->
<!-- - First-Visit MC is generally preferred when state occurrences within an episode are highly correlated, as it prevents bias due to repeated visits with varying returns.
- Every-Visit MC is computationally simpler and can work well if state visits are independent and uncorrelated.   -->

## Constant-α MC: Incremental Every-Visit MC Prediction

We now extend the every-visit Monte Carlo (MC) prediction algorithm by making it incremental. Incrementality is crucial in reinforcement learning, as it allows an algorithm to interact with the environment and learn continuously, rather than waiting until the end of the experience to make updates. However, since MC methods require computing the return \( G_t \), we must still wait until the end of each episode to obtain the full sequence of rewards. The key difference is that instead of averaging all returns at once, we update the estimates incrementally.

### Incremental Update Rule

To achieve this, we adjust how averaging is performed over time. Suppose we observe state \( s \) at time step \( t \) with return \( G_t \), and the number of times \( s \) has been visited is \( N_s \). We aim to update the value estimate \( V^\pi(s) \) based on our previous estimate \( V^\pi_k(s) \). Since \( N_s - 1 \) represents the number of times \( s \) was visited before, the updated estimate is given by:

\[
    \begin{align}
        V^\pi_{k+1}(s) &= \frac{1}{N_s} \left(\sum_{j=1}^{N_s-1} G_j(s) + G_t(s)\right) \nonumber \\
                        &= \frac{1}{N_s} \left((N_s - 1)V^\pi_k(s) + G_t(s)\right) \nonumber \\
        V^\pi_{k+1}(s) &= V^\pi_k(s) + \frac{1}{N_s} \left(G_t -  V^\pi_k(s) \right)  \label{eq:main} %\tag{1}
    \end{align}
\]

This approach mirrors the concept used in the simple bandit algorithm. In fact this equation \(\eqref{eq:main}\) is the base for any incremental method in RL and is the most pivotal equation in all of the RL field and yet it is usually not talked about in this form. We will be using a variation of this equation in several of the techniques that we will be covering.

### Addressing the Diminishing Update Problem

One issue with standard averaging is that as \( N_s \) increases, the influence of newly observed returns \( G_t \) diminishes because \( \frac{1}{N_s} \) becomes very small. To counteract this, we replace \( \frac{1}{N_s} \) with a fixed constant \( \alpha \). This ensures that the agent continues to adjust its estimates meaningfully over time, which is particularly beneficial when the policy is evolving (i.e., in non-stationary environments).

The incremental update rule then becomes:

\[
V(S_t) \leftarrow V(S_t) + \alpha \left( G_t - V(S_t) \right)
\]

where:

- \( V(s) \) is the estimated value of state \( s \).
- \( G_t \) is the return (sum of discounted rewards) obtained from state \( s \) onwards.
- \( \alpha \) is the constant step-size parameter (\( 0 < \alpha \leq 1 \)).

This approach allows updates to be made incrementally and ensures that the agent continues refining its estimates as more experience is gathered. It is particularly effective in scenarios where multiple visits to the same state provide valuable learning opportunities.

### Algorithm: Incremental Constant-α Monte Carlo Prediction

Below is the full pseudocode for the algorithm:

\[
\begin{array}{ll}
\textbf{Algorithm: }  \text{Incremental Constant-}\alpha \text{ Monte Carlo Prediction} \\
\textbf{Input: } \text{Episodes generated under policy } \pi \\
\textbf{Initialize: }  V(s) \leftarrow 0, \forall s \in \mathcal{S}, 0 < \alpha \leq 1 \\
\textbf{For each episode: } & \\
\quad \text{Generate an episode: } (S_0, R_1, S_1, \dots, S_T) & \\
\quad G \leftarrow 0 & \\
\quad \textbf{For each step } t \textbf{ from } T-1 \textbf{ to } 0: & \\
\quad \quad G \leftarrow \gamma G + R_{t+1} & \\
\quad \quad \text{Update state-value estimate:} & \\
\quad \quad \quad V(S_t) \leftarrow V(S_t) + \alpha (G - V(S_t)) & \\
\textbf{Return: } V(s), \forall s \in \mathcal{S} \\
\end{array}
\]

On important advantage for this methods is that it ensures that the agent continuously learns from its experiences while adapting to changes in the environment (assuming we are updating the estimates between episodes).

<!-- ### Monte Carlo Control with \(\epsilon\)-Greedy Exploration

To optimize policies, we extend MC to **control**, using the **Generalized Policy Iteration (GPI)** framework. The key idea is:

1. **Policy Evaluation:** Estimate \( Q^\pi(s, a) \) using MC.
2. **Policy Improvement:** Use an **\(\epsilon\)-greedy** strategy to gradually improve the policy.

### Algorithm:

\[
\begin{array}{ll}
\textbf{Algorithm:} & \text{Monte Carlo Control with } \epsilon \text{-greedy policy} \\
\textbf{Initialize:} & Q(s, a) \leftarrow 0, N(s, a) \leftarrow 0, \forall s, a \\
\textbf{For each episode:} & \\
\quad \text{Generate an episode using } \pi_{\epsilon} & \\
\quad G \leftarrow 0 & \\
\quad \textbf{For each step t from T to 1:} & \\
\quad \quad G \leftarrow \gamma G + r_{t+1} & \\
\quad \quad \text{If } (s_t, a_t) \text{ appears first in episode:} & \\
\quad \quad \quad N(s_t, a_t) \leftarrow N(s_t, a_t) + 1 & \\
\quad \quad \quad Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \frac{1}{N(s_t, a_t)}(G - Q(s_t, a_t)) & \\
\quad \quad \text{Update policy } \pi \text{ using } \epsilon \text{-greedy:} & \\
\quad \quad \quad \pi(s) \leftarrow \arg\max_a Q(s, a) \text{ with probability } (1-\epsilon) & \\
\quad \quad \quad \text{Choose random action with probability } \epsilon & \\
\textbf{Return:} & Q(s, a), \pi(s) \forall s, a \\
\end{array}
\] -->



## MRP, MDP, and PG Classes

To evaluate the effectiveness of prediction methods, we apply them to a random walk problem. This setup isolates the prediction aspect of an algorithm, focusing solely on estimating the state-value function without involving decision-making. By doing so, we can assess whether a given update rule or algorithm is effective in the prediction setting.

Once we grasp the prediction process, we can extend these methods to control by modifying the update rule to incorporate action-value estimates (\( Q \)-values). This transition is typically done within the Markov Decision Process (MDP) framework and is a crucial step in value-based reinforcement learning methods.

Beyond value-based approaches, another class of methods exists, known as policy-based or *policy gradient (PG)* methods. These methods optimise the policy directly instead of relying on a value function. Unlike value-based methods, policy gradient approaches are typically applied to control problems, such as grid-world mazes, rather than random walk problems.

We define three parent classes to structure our reinforcement learning algorithms:

- **MRP (Markov Reward Process) Class:** Used for any model-free prediction method that has access to \( V \). The MRP class does not include a policy, as it operates under an arbitrary policy.
- **MDP (Markov Decision Process) Class:** Used for any model-free control method, which has access to both \( Q \) (action-value function) and an \( \epsilon \)-greedy policy. This allows the agent to make decisions based on value estimates.
- **PG (Policy Gradient) Class:** Used for any policy-gradient method. The PG class has access to both \( V \) and \( Q \), as well as a softmax policy for selecting actions.

By structuring our approach in this way, we create a clear distinction between prediction and control methods, ensuring a systematic transition from estimating values to learning optimal policies.


### Incremental Constant-\(\alpha\) MC (Prediction) with Python

Below, we present a direct interpretation of the previously discussed pseudocode for the model-free prediction method into Python. This implementation follows an incremental approach, applying updates using a constant step-size parameter \( \alpha \). 

To maintain a structured and modular design, we define this implementation as a subclass of the *MRP* class. This ensures that the algorithm focuses purely on state-value estimation under a fixed policy, without incorporating decision-making or action-value functions.

By leveraging the MRP parent class, we maintain consistency across different prediction methods and enable seamless extension to more complex reinforcement learning algorithms.


```python
class MC(MRP):
    def init(self):
        self.store = True # stores the trajectory of the episode, always needed for offline
    # ------------------ 🌘 offline, MC learning: end-of-episode learning ------------------
    def offline(self): # called at the end of the latest episode
        Gt = 0
        for t in range(self.t, -1, -1):          # go throgh experience backwards as per Gt in update
            s, rn = self.s[t], self.r[t+1]       # retrieve the state and reward for past step t 
            Gt = self.γ*Gt + rn                  # calculate the return for past time step t
            self.V[s] += self.α*(Gt - self.V[s]) # update the state-value function
```

This type of algorithmic design is more flexible and is generally preferred in reinforcement learning. Instead of requiring the storage of cumulative sums or averages, the incremental approach allows for real-time updates, making it more adaptable to dynamic environments.

By using this approach, we can efficiently estimate the state-value function without the computational overhead of storing past returns. This is particularly useful in large-scale problems where maintaining full episode histories would be impractical.

Now, let’s apply our newly implemented prediction algorithm to the random walk problem and observe how well it estimates the state values.

```python
mc = MC( α=.02, episodes=50, **demoV()).interact()
``` 
![png](output_101_0.png)
    
As can be seen, the algorithm behaves as expected, gradually converging to an optimal solution—albeit slowly. This is a general characteristic of Monte Carlo (MC) algorithms, as they rely on sampling and require numerous episodes to achieve accurate estimates. 

## MC vs DP: Computational and Convergence Speed
Despite their slower convergence compared to dynamic programming methods and other methods that we cover later (such as TD), MC algorithms are advantageous in environments where the model is unknown, making them well-suited for model-free reinforcement learning. Monte Carlo (MC) methods are generally slower than Dynamic Programming (DP) in terms of convergence (not). When we take computaitonal complexity into consideration, the comaprison is more nuanced.

### Convergence Speed in Monte Carlo (MC)
- MC methods are slower to converge compared to DP due to their reliance on sampling and waiting until the end of each episode to compute returns.  
  - MC algorithms update state-value estimates incrementally after each episode. The updates rely on the sampled trajectories, and each state’s estimate is refined slowly across multiple episodes. This means that for large state spaces or long episodes, it can take a lot of episodes for MC to converge to an accurate value.
  - The rate of convergence in MC is also influenced by the exploration strategy (such as \(\epsilon\)-greedy). If exploration is high, the agent may not follow the optimal path immediately, requiring more episodes to refine its policy and reach convergence. This is especially important in problems where exploring new actions is key to learning optimal values.

### Convergence Speed in Dynamic Programming (DP)
- DP methods tend to converge faster per iteration than MC. This is because DP directly uses a full model of the environment (transition probabilities, rewards, etc.) and updates the value function across the entire state space in a single pass.  
  - Since DP updates all states at once, each iteration provides more substantial progress towards convergence. As long as the state space is not prohibitively large, DP can refine the value function quickly in a relatively small number of iterations.
  - Despite the faster convergence per iteration, DP requires multiple iterations to converge to the optimal value function. While each individual update is quick, the overall process of achieving convergence may still take several iterations depending on the size and complexity of the environment.

### Computational Complexity of MC
- MC is computationally more efficient per episode because each update only requires sampling from the environment during the episode. This means that during each episode, MC methods only need to process the steps taken and compute returns after the episode ends. 
  - The time complexity of each episode depends on the number of states and actions, but the updates themselves are relatively simple. Thus, per episode, the computational cost can be considered \(O(T)\), where \(T\) is the number of time steps in the episode.
  - Since MC needs multiple episodes to converge, the total computational cost is proportional to the number of episodes and the length of each episode. If the environment is large or the episodes are long, MC’s total complexity will be \(O(N \cdot T)\), where \(N\) is the number of episodes, and \(T\) is the number of steps in each episode.
  
### Computational Complexity of DP
- DP is computationally more expensive per iteration because it requires processing the entire state space to update all states in each pass. Each update involves iterating over all states and performing calculations based on the known model of the environment.
  - The time complexity of one iteration of DP depends on the number of states \(S\) in the environment. The update rule for each state requires examining all possible actions and computing the expected returns based on the transition model and reward function, which results in a per-iteration complexity of \(O(S \cdot A)\), where \(A\) is the number of actions.
  - Since DP requires multiple iterations, the total complexity depends on the number of iterations \(K\) required for convergence. If the environment has a large state space, the computational cost can become significant, resulting in \(O(K \cdot S \cdot A)\), where \(K\) is the number of iterations needed to converge.

<!-- ### Key Differences in Computational Complexity
**MC**

  - Computationally cheaper per update: MC methods update values at the end of an episode, and each update involves relatively simple computations. The total complexity depends on the number of episodes and steps within each episode, but each step in the episode is handled individually.
  - Potentially slower overall: MC might require a large number of episodes, meaning the total computational cost could still be significant in larger environments, especially if the episodes are long or the environment is complex.

**DP**

  - Computationally expensive per iteration: DP requires updating the entire state space in each iteration, leading to higher per-iteration complexity, especially in large environments with many states and actions. This makes DP computationally more expensive than MC in terms of per-iteration cost.
  - Fixed number of iterations: DP usually converges in fewer iterations compared to MC. However, even with fewer iterations, the overall cost per iteration is higher than that of MC, especially if the state-action space is large. -->

### Summary on Convergence Speed and Computational Complexity
- MC is slower to converge because it is based on sampling and episodic updates. It requires many episodes to converge to an optimal solution, and its computational complexity scales with the number of episodes and steps within each episode. While MC can be computationally cheaper per episode, it may require many episodes to converge, resulting in high total computational cost.
  
- DP converges faster per iteration, but it requires processing the entire state space during each iteration, making it computationally more expensive per update. However, DP typically requires fewer iterations to converge, which can make it faster overall in terms of convergence speed when the environment model is known. The computational cost can still become prohibitive for large state spaces due to the higher cost per iteration.

In summary, MC is computationally lighter per update but slower in convergence due to its reliance on sampling and multiple episodes. DP converges faster per iteration but is more computationally expensive per update, especially as the state-action space grows. The trade-off between convergence speed and computational efficiency depends on the environment and whether the environment model is available.



## Incremental MCC: Every-visit MC Control
**Incremental MC Control (Every-visit Monte Carlo Control)** is an approach used for estimating optimal policies through interaction with the environment. It incrementally updates both the state-action value function \( Q \) and the policy using every visit to a state-action pair. The method involves updating the action-value function based on the observed returns, and it uses a constant step-size parameter \( \alpha \) to control the magnitude of updates:

\[
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left( G_t - Q(S_t, A_t) \right)
\]

Where:

- \( Q(S, A) \) is the action-value function,
- \( G_t \) is the return from the state-action pair \( (S, A) \),
- \( \alpha \) is the constant step-size parameter.

This method is applied iteratively to improve the policy, choosing actions greedily with respect to the estimated \( Q \)-values. Below we show the pseudocode for this algorithm.

\[
\begin{array}{ll}
\textbf{Algorithm: }  & \text{Incremental Constant-}\alpha \text{ Monte Carlo Control} \\
\textbf{Input: } & \text{Episodes generated under an } \varepsilon\text{-greedy policy } \pi \\
\textbf{Initialize: } & Q(S, A) \leftarrow 0, \forall S \in \mathcal{S}, A \in \mathcal{A}(S), 0 < \alpha \leq 1 \\
\textbf{For each episode: } & \\
\quad \text{Generate an episode: } & (S_0, A_0, R_1, S_1, A_1, \dots, S_T) \\
\quad G \leftarrow 0 & \\
\quad \textbf{For each step } t \textbf{ from } T-1 \textbf{ to } 0: & \\
\quad \quad G \leftarrow \gamma G + R_{t+1} & \\
\quad \quad \text{Update action-value estimate:} & \\
\quad \quad \quad Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (G - Q(S_t, A_t)) & \\
\quad \quad \text{Update policy: } & \pi(S_t) \leftarrow \arg\max_A Q(S_t, A) \text{ (with } \varepsilon\text{-greedy exploration)} \\
\textbf{Return: } & Q(S, A), \pi(S), \forall S \in \mathcal{S}, A \in \mathcal{A}(S) \\
\end{array}
\]


### Incremental constant-α MC with Python

Below we show a direct *interpretation* of the above pseudocode into Python code. We use the MDP parent class for this control model-free method.

```python
class MCC(MDP()):
    def init(self):
        self.store = True
    # ------------------ 🌘 offline, MC learning: end-of-episode learning 🧑🏻‍🏫 ---------------
    def offline(self):  # called at the end of the latest episode
        Gt = 0
        for t in range(self.t, -1, -1):
            s, a, rn = self.s[t], self.a[t], self.r[t+1] # retrieve the state, action and reward for past step t 
            Gt = self.γ*Gt + rn                          # update Gt incrementally
            self.Q[s,a] += self.α*(Gt - self.Q[s,a])     # update the action-value function
```


```python
mcc = MCC(env=grid(reward='reward1'), α=.2, episodes=1, seed=0, **demoQ()).interact()
```
    
![png](output_105_0.png)
    

As we can see, although we solved the issue of tracking a non-stationary policy by using a constant learning rate \( \alpha \), and we attempted to use a reward function that provides immediate feedback at each step instead of a delayed reward, the performance is still not as good as expected. This is due to our final issue: the action precedence that we set up, which prefers left over right. If we change this precedence, it will help the agent find the goal more quickly. However, we set it up this way to make the problem more challenging. Consider changing this precedence to observe the effect.


## REINFORCE: MC for Policy Gradient

So far, we have only seen how to estimate a value function to deduce a policy from this value function and then improve the policy by preferring a greedy action with a bit of exploration (as in the ε-greedy policy). When we allow the agent to act according to this new policy, its value function might change, so we must re-estimate the value function. We go into iterations of this process until the policy and value function are both stable (converge). We also saw that we could integrate both operations seamlessly into one iteration, as in the value-iteration algorithm in Dynamic Programming. We can even do both stages in one *step* as in Q-learning or Sarsa, as we shall see in the next lesson. The policy improvement theorem and the Generalised Policy Iteration process guarantee all of this. 

The primary approach we took to achieve learning for an *action-value* method is to *minimise an error function* between our estimate of a value function and the actual value function. Since the real value function is unavailable, we replaced it with some samples (unbiased as in MC and biased as in TD that we will see later).

**Policy gradient** algorithms, on the other hand, attempt to *maximise an objective function* instead of minimising an error function. Can you think of a function that, if we maximise, will help us solve the RL problem? Pause for a moment and think.

As you might have guessed, the value function can be used as an objective function. The objective here is to change the policy to maximise the value function. Directly estimating the policy means we are not using a value function to express the policy, as in the ε-greedy method. Instead, we are using the value function to learn the policy directly. So, our algorithm does not need to learn the value function explicitly; it can learn a set of parameters that will maximise the value function without knowing what the value function is. It will come as a consequence of learning a policy. In the same way that we did not need to learn a policy in the value-function approach, we learned a value function, and as a consequence of minimising the error, we can deduce a policy from the learned value function. This is the fundamental difference between value-function approaches and policy-gradient approaches.

Estimating the policy directly means we do not need to restrict the policy parameters to value-function estimates and their ranges. The policy parameters that represent the preferences to select an action are free to take on any range of values, as long as they comparatively form a cohesive policy that maximises the value function by dictating which action to choose in a specific state. This is a major advantage because the value function is strictly tied to the sum of reward values, while a policy need not have this coupling. This will give us more freedom in using classification architectures when we use function approximation, which excels in deducing the best action for a state, instead of using a regression architecture to regress a value function, which is usually more prone to initial condition issues and is harder to train.

The best policy representation in a policy-gradient method is the action-selection softmax policy we came across in our last few lessons. This is a smooth function that, unlike ε-greedy, allows the changes in the probabilities to be continuous and integrates very well with policy-gradient methods. One of the significant advantages of policy-gradient methods (the policy is differentiable everywhere, unlike stepwise ε-greedy functions) is that it provides better guarantees of convergence than ε-greedy due to this smoothness (ε-greedy can change abruptly due to small changes in the action-value functions, while softmax just smoothly increases or decreases the probability of selecting an action when its action-value function changes).

We start our coverage of policy-gradient methods with an offline method: REINFORCE. REINFORCE is an algorithm that takes a *policy gradient* approach instead of an action-value function approach. The idea is simple: given that an episode provides a sample of returns for the visited states, at the end of an episode, we will take the values of the states and use them to guide our search to find the optimal policy that maximises the value function.

<!-- **Note** that policy gradient sections in this lesson, and the next, are based on chapter 13 of our book. They can be read as they appear in the notebook or delayed until the end of lesson 9. -->

<!-- ### Policy Gradient Class -->
The softmax is the default policy selection procedure for Policy Gradient methods. \( \tau \) acts like an exploration factor (more on that later), and we need one-hot encoding for the actions.

Now we are ready to define our REINFORCE algorithm. This algorithm and other policy-gradient algorithms always have two updates: one for \( V \) and one for \( Q \). In other words, the action-value function update will be guided by the state-value update. We usually call the first update that deals with \( V \), the critic, and the second update that deals with \( Q \), the actor. Below we show the Python code for this algorithm.


```python
class REINFORCE(PG()):
    def init(self):
        self.store = True
    # -------------------- 🌘 offline, REINFORCE: MC for policy gradient methdos ----------------------
    def offline(self):
        π, γ, α, τ = self.π, self.γ, self.α, self.τ
        # obtain the return for the latest episode
        Gt = 0
        γt = γ**self.t                           # efficient way to calculate powers of γ backwards
        for t in range(self.t, -1, -1):          # backwards
            s, a, rn = self.s[t], self.a[t], self.r[t+1]
            Gt = γ*Gt + rn                        # update Gt incrementally
            δ = Gt - self.V[s]                    # obtain the error
            self.V[s]   += α*δ                    # update V as per the error
            self.Q[s,a] += α*δ*(1 - π(s,a))*γt/τ  # update Q as per the erro with complement of the policy π
            γt /= γ

```

## The Role of Discount Factor \( \gamma \) in Policy Gradient Methods

\( \gamma \) seems to play a more important role in policy-gradient methods than in action-value methods. The next few examples show how \( \gamma \) can make the difference between convergence and divergence.

The main issue is, as usual, whether the *reward* is delayed or there is an intermediate reward. If the reward is delayed, we would need to assign \( \gamma \) values that are less than 1 so that the sum of the rewards is discounted, which helps the agent differentiate between longer and shorter paths. However, \( \gamma \) also plays a role in convergence when the reward is not delayed. It complements the role that \( \tau \) plays in the SoftMax policy.

Therefore, instead of tuning \( \tau \), we can reduce \( \gamma \) specifically when the goal reward is 0, and the intermediate reward is -1 (reward_0) function. Let us see some examples:

Below, we increase the value of \( \tau \) to deal with this issue of divergence.


```python
reinforce = REINFORCE(env=grid(reward='reward0'), α=.1, τ=2, γ=1, episodes=100, seed=10 , **demoQ()).interact()
```
    
![png](output_135_0.png)
    

As we can see REINFORCE converged when we increase $\tau$ which helped the values in SoftMax to become appropriatly smaller to help the algorithm to converge.

Let us now decrease the value of $\gamma<1$ and keep $\tau=1$


```python
reinforce = REINFORCE(env=grid(reward='reward0'), α=.1, τ=1, γ=.98, episodes=100, seed=10, **demoQ()).interact()
```
    
![png](output_140_0.png)
    


As we can see, decreasing \( \gamma \) helped REINFORCE immensely to converge. Although the reward that we used is intermediate reward ('reward_1'), which is not delayed, discounting the return helped the value function to be more meaningful for the problem at hand. This, in turn, helped the policy to be more appropriate for the problem.

Let us now increase \( \tau \) and keep \( \gamma < 1 \). This will reveal another role for \( \tau \).


```python
reinforce = REINFORCE(env=grid(reward='reward0'), α=.1, τ=2, γ=.98, episodes=100, seed=10, **demoQ()).interact()
```
    
![png](output_144_0.png)
    


As we can see, increasing \( \tau \) while using \( \gamma < 1 \) did not help. We will mostly therefore use \( \gamma < 1 \) for our policy gradient methods.

Note how exploration led to a fully covered environment but slower convergence.

## Conclusion
In this lesson, we studied the properties of Monte Carlo algorithms for prediction and control. We started by covering a basic first-visit MC method that averages the returns, similar to what we did in lesson 1, this time for the associative problem (i.e., when we have states that we select specific actions for, un-associated problems do not have states and have been studied in lesson 1). We then created an incremental MC algorithm that allows us to average the returns in a step-by-step manner. To that end, we developed an essential MRP class that will carry the step-by-step and episode-by-episode interaction with an MRP environment and then added a useful set of visualization routines. We further inherited the MRP class in an MDP class that defines policies that depend on the Q function to obtain a suitable policy for an agent (i.e., control).

We noted that MC needed to wait until the episode was finished to carry out updates. In the next unit, we will study full online algorithms that mitigate this shortcoming of MC with the cost of bootstrapping. We will be using the MRP and MDP classes that we developed here.

Monte Carlo methods provide a powerful alternative to Dynamic Programming for RL problems where the environment’s transition model is unknown. They estimate value functions from sampled episodes and improve policies using exploration strategies like \( \epsilon \)-greedy. We covered two types of methods:

- Policy Evaluation: Use MC to estimate state values.
- Policy Control: Improve policies using MC-based action-value estimates.

### Advantages
- Model-free: No need to know the environment’s transition probabilities.  
- Simple and intuitive: Works by averaging sampled returns.  
- Works well for episodic tasks.  

### Limitations
- Requires complete episodes, which may not always be feasible.  
- Convergence can be slow compared to Temporal-Difference (TD) methods.  

In the next unit, we will explore Temporal-Difference (TD) learning methods, which update state-value estimates without requiring complete episodes. Instead, TD methods rely on one-step updates, much like Value Iteration incrementally improves policies. We will also introduce TD-based action-value methods for control, including SARSA and Q-learning, which use the Q action-value function to learn optimal policies.

### Reading:
For further reading, you can consult chapter 5 from the Sutton and Barto [book](http://incompleteideas.net/book/RLbook2020.pdf). The policy gradient sections in this lesson, and the next are based on chapter 13 of our book. They can be read as they appear in the notebook or delayed until the end of lesson 9.

## Your turn
Now it is time to experiment further and interact with the code in [worksheet6](../../workseets/worksheet6.ipynb).
