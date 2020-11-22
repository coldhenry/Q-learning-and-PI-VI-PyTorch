# Tabular Methods of Q-learning and Policy Iteraion/ Value Iteration

<img src="https://github.com/coldhenry/Q-learning-and-PI-VI-PyTorch/blob/main/pic/openai-pytorch.jpg" width="450" height="150" >

## Model-Based Method: Policy Iteration & Value Iteration

### Problem Formulation
Given a Markov Decision Process described by S, A, R ,P, γ, where S ∈ Rn is the
state-space, A ∈ Rm is the action space, R : Rm × Rn × Rm → R is the reward
function, P : Rm × Rn × Rm → [0, 1] is the transition probability and γ is the discount
factor. Starting with teh deﬁntion of a value function, show that for a deterministic
policy π(s), the value function v(s) can be expressed as:

v(s) = the sum of (p(s′|s, a)[r(s, a, s′) + γv(s′)])

where p(s′|s, a) ∈ P and r(s, a, s′) ∈ R. Assume that the state and action spaces are
discrete.

### Policy Iteration Pseudocode [1]

<img src="https://github.com/coldhenry/Q-learning-and-PI-VI-PyTorch/blob/main/pic/PI.png" width="506" height="370" >

### Value Iteration Pseudocode [1]

<img src="https://github.com/coldhenry/Q-learning-and-PI-VI-PyTorch/blob/main/pic/VI.png" width="506" height="234" >

### Comparison of PI and VI [2]

<img src="https://github.com/coldhenry/Q-learning-and-PI-VI-PyTorch/blob/main/pic/PIVI.png" width="800" height="331" >

## Model-Free Method: Q-learning

### Pseudocode [3]
<img src="https://github.com/coldhenry/Q-learning-and-PI-VI-PyTorch/blob/main/pic/Q learning.png" width="697" height="325" >

## Experiement Environment & Results

Open AI Env: Frozen-Lake-v0

<img src="https://github.com/coldhenry/Q-learning-and-PI-VI-PyTorch/blob/main/pic/Frozen-Lake.png" width="421" height="401" >

### PI and VI

<img src="https://github.com/coldhenry/Q-learning-and-PI-VI-PyTorch/blob/main/pic/PI_res.png" width="400" height="400" ><img src="https://github.com/coldhenry/Q-learning-and-PI-VI-PyTorch/blob/main/pic/VI_res.png" width="400" height="400" >

### Q-learning

We will discuss how performance changes with 2 factors changed: Learning rate and discount factor

The following graph demonstrates the performance using different discount factor and a fixed learning rate (gamma = 0.99) 

<img src="https://github.com/coldhenry/Q-learning-and-PI-VI-PyTorch/blob/main/pic/gamma-0.99-correct.png" width="400" height="400" >

The following graph demonstrates the performance using different learning and a fixed learning rate (gamma = 0.99) 


## Reference
[1] UCSD ECE 276C course slides

[2] Sutton and Barto's book: Reinforcement Learning: An Introduction.

[3] [Lei Mao's Log book](https://leimao.github.io/blog/RL-On-Policy-VS-Off-Policy/)
