# Recap

## Lai and Robins

## Cesa-Bianchi
	Worst case

## kl-UCB vs UCB
	KL better

# Bayesian Approach
	
	Work of Thomson on clinical trials beginning of bandit research, first bandit algorithm 1933.

## Bayesian Bandit Problem
	
	Frequentist tools - estimators of the mean with MLE, confidence intervals
	Bayesian tools - no confidence intervals because parameters random, instead compute posterior.

	There is an optimal solution unlike the frequentist case.
	Posterior distribution is summarized by matrix containing number of observed ones and zeros.
	
### DP

	Optimal solution is solution to dynamic programming equations.
	Hard to implement DP solution because the state space can be large, intractable.

### Gittins Index 
	
	Compute Gittin index, index policy. 
	First index policy.
	Looks like kl-UCB, with a different exploration rate.

### Bayes-UCB

	Q(\alpha, \pi) quantile ofr order alpha of the distributuon pi.
	The order is 1/(t lnt).
	Gaussian, Beta, Poisson have a conjugate posterior (The posterior remains in the same family)
	
	Bayes UCB is asymptotically optimal for Bernoulli rewards
	
### Thomson Sampling

	First regret analysis 2012
	Randomized algorithm
	Select arm at random proportional to the probability of the arm being the best
	Draw possible bandit model from posterior distribution, act as if the sampled model is the right model.

	Have ability to sample from posterior belief.
	
## Other Randomized Algorithms
	
	The other algorithms are optimal if you know what reward you are facing.
	Universal bandit algorithm.

### BESA - Best Empirical Sub-sampling Average
	
	Twist follow the leader algorithm.
	Follow the FAIR leader.	
	Play each arm once.
	
	Limitation it is not completely understood from theory.
	Work in progress.


# Lecture 2 - Structural Bandits

## Linear Bandits
LAME

# Applied Bandits















	



### Thomson Sampling




	

