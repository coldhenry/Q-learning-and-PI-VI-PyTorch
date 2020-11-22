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

### Policy Iteration





## Model-Free Method: Q-learning


## Experiement Environment & Results


## Reference
