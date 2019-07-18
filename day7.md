# Exploration

Posterior sampling is basically the same idea as Thomson sampling.
While not knowledge keep executing.
J-stage contraction of Bellman operator

# Exploration Part 2

Build accurate estimators, evaluate uncertainty of your predictions, define mechanism to combine estimation and uncertainty.

Practical limitations
Have to compute confidence intervals for UCRL, optimism in face of uncertainty


Count-based exploration, estimate proxy for the number of visits, run any DeepRL algorithm over it.
Just modifying the reward can be still a good way to go.
Instead of discretizing the state space, use locality-sensitive hashing. Project with random matrix and use sign to discretize. SimHash. Leverage autoencoders to find a better compression.  You cannot update your hashing function too often - problematic. 

Convolutional random projection for Atari games?

Use density estimation for count-based exploration. 

Prediction based exploration.
	Stochsticity bad
	Model misspecification bad 
	Learning dynamics bad

	Fix target network and use prediction network to predict output of the target network at each state. There is no randomness.

	Extrinsic and intrinsic rewards should be in the same range.
	Use different discount factors for intrinsic and extrinsic rewards.
	

## Random Exploration in Deep RL
Bootstrap DQN, keep k different Q values. Sample Q on the beginning of each episode. Thomson samples at each step.

Noisy networks. Do factored noise to make MC less expensive.


## Florian Strub NLP

Question teacher forcing bad if used all of the time. 
WMT dataset
Check your baseline if it doesn't work
Increase batch size because of high variance
Use softmax temperature.


Grounding/ground simbol













