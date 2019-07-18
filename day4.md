# Part III Google Brain Value Func Approx.

Tabular RL - complexity is polynomial, doesn't scale up

How to handle large state-action spaces in memory?
How to generalise over state-action spaces?

Tabular case is a special case of linear value function approximation. 

Neural nets, 2 choices:
	For discrete acction space, as many outputs as actions for Q, case for big 	state space
	
	Learn mapping Q(s,a) -> q
	
## Policy Evaluation

We mostly use TD(0) estimate
We can use TD(lambda) estimate
We can use monte-carlo estimate

Multiplication of correlated things causes the variance term to appear

Lest Square TD Solution to projection problem, closed form.


## Control

## Reducing variance

Subtract baseline


## Real-World RL, Google Dude

	Training Regime
	Environmental Constraints
	Evaluation Metrics except reward

	Safety Environments
	Constrained MDPs
	Do not tweak reward function, specify constraints
	
	Soccer player visualization, per dimension reward
	Embedding actions for large action spaces, ~13000
	
	Exploration bottlenecks solved through expert data, imitation learning.

	DDPGfD, better than hand-written controller

	PlaNet - latent space for mujoco, deep planning network
	
	Model-based for meta learing for domain adaptation.
	Model-based can be interpretable.

	What needs to be tried
		Epistemic uncertainty
		CEM very unefficient
		policy caching (combining model-free and model-based)
		dynamic constraints
		goal-defined policies

	MAML










