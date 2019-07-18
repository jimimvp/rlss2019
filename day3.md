# Markov Decision Processes
	Control of dynamical systems under uncertainty
	

## Finite Horizon Optimal Control
	
	Infinite horizon is a bit simpler.
	Policy (feedback control law)
	
	Assumptions:
		Disturbance does not depend on the past values, Markov property
		
	Policies can be:
		History dependend vs Markov
		Stationary vs. non-stationary
		Random vs. deterministic

		Smallest space is markov, stationary deterministic 

	In MDP, we only need to consider Markov deterministic policies

	Estimate value by monte-carlo simulation, but the problem is that it is approximate
	

### Value Iteration

	Recursive value function approximation

	Value iteration converges asymptotically to the optimal value. It converges at a linear rate.	

## Infinite Horizon Optimal Control

	The system is stationary, homogenuous merkov chains
	gamma makes infinite sum finite
	gamma connected to interest rates in economics
	If gamma=1 we need a termination state that is going to be reached with probability 1

	Why is this simpler? In finite horizon, the number of steps is important. The horizon is stationary in the infinite horizon problem.
	
	Whatever policy you play in tetris you are going to lose in finite time.
	There is a squence of pieces that kills you
	
	Discount factor related to rate of inflation.
	
	
### Contraction Property

	Bellmann T operators are gamma contraction mappings. For all pair of functions, the max norm of ||T_u - T_v|| \le \gamma || v- u ||_\inf	
	Important result because we can use Banach fixed point theorem. These operators have 1 and only fix point. 

	A stationary policy is optimal if and only if it obtains the maximum in the Bellman optimality equation. 
	

### Policy Iteration

	When the MDP is finite, convergence occurs in a finite number of iterations. 


	
# Big Problems

We would like the policy to act greedily with respect to the previous value function.

Pointwise estimation through samples.

You want to bound the difference between q start and current q. It is bounded by a constant multiplied by estimation error.

It can be shown that the bound cannot be improved without assumptions.





# Small Problems

	Unknown model
	Complexity of policy iteration, open problem even when pi is deterministic


# Large Problems

	Least Square Policy Iteration, linear approximation of the value function
	Sensitivity to noise is better if you use non-stationary policies
	
	Cool algorithms, Conservative Policy Iteration and Policy Search by Dynamic Porgramming. Best guarantees. 




# Bandits for Recommender Systems

Ranking comments
Optimizing displays

Bandits in production?

Counterfactual estimations. Keep propensity, probability of selecting action in context
Estimate reward? Direct approach
Indirect approach - use propensity scores

Tip: pull 10 times all new arms and then be greedy
Tip: control pi_0 to be large enough


# Open Problems

	Do counterfactual learning
	

















