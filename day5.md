# Policy Gradient

We mostly use semi-gradient, which is bad

## Convergence Result

Policy gradient is a stochastic gradient.
J is not convex, converges asymptotically to a stationary point or local minimum

If dynamics are linear then we reach global convergence.

Non-convexity of a loss function.
Unnatural policy parametrization - paramters far in Euclidean space can describe the same policy
Naive stochastic exploration. 
Variance of gradient increases with the length of the horizon.

Follow the greedy policy with soft-update. No suboptimal stationary points.

## Actor Critic

REINFORCE is unbiased but still high variance with monte-carlo estimate

Compatible function approximation
	
Sample efficiency with variance-reduced gradient estimator

Policy performance lemma, distribution times advantage of the other policy

Pinskers inequality


# Generalization in Reinforcement Learning

What is generalization? 
Capacity to wait longer for the preffered rewards seems to develop markedly only at about ages 3-4
Increase discount factor when more data of interest is gathered. 

## Best Practices RL

How to benchmark?

Combining model-based and model-free via abstract representations.
Combined reinforcement learning via abstract representation - paper

Transfer learning


