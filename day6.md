# Bandits for Optimization

Probability of error, bounds on error


# Oriol Talk

Learn end to end as much as possible
Ape-X DQN
Behind every great agent there is a great reward!
FTW agent - state of the art, planning

Point and click games. 

## AlphaStar
Massive scale
RTS are a nice challenge for the community
Problem with starcraft: convergence to local optimum, many strategies exist. No policy improvement theorem.
Pointer networks necessary for Alpha-star

Exploration has been solved thanks to imitation learning

Questions? 
HRL not proven to work on large scale with complicated models.
Or model based?

Do you want to convert RL to supervised learning?
It might be harder to simulate the real world than to solve it.
Transfer learning is a big interest/problem.

modular neural networks


## Advanced Deep RL

Sometimes discounting used in the episodic setting

Deadly triad: off-policy learning, bootstrapping, function approximation. Can lead to divergence.
Soft divergence concept - parameters first diverge, then converge.
Target networks reduce the bad effects of soft-divergence, rduced possibility in feedback loop, reduces non-stationarity of the problem.
In practice people combine double Q-learning and target networks.

By bootstrapping sooner you increase the bias. Multi-step increases the variance of the update in case of stochastic environments.


### Architectures with Good Inductive Biases for DL with RL

Dueling networks example of inductive bias, decomposition of Q into V + A(advantage)

Reward scaling.
Magnitude of updates scales directly with the magnitude/frequency of the rewards. We cannot normalize the loss, since it is non-stationary.
PopArt - use normalized and unnormalized values for the gradient normalizing, learning values across many orders of magnitude

### Planning and Model-based RL

Don't trust fully the model. 

Predictron architecture.
Architecture designed for reinforcement learning.

### State Representation

Incremental state representation
Design architecture that are recurrent that exploit specific propoerties of the environment.
Distributional RL / DQN

### Scalable Deep RL

Parallelism provides the same diversity as replay buffer.
Actor-learner decomposition, IMPALA
Meta-gradient RL

Combining learning and search













