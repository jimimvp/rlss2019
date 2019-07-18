# Inequality Crawl

# Supervised Learning and decision processes

blabla, reinforcement learning includes all

## Supervised Learning

	Takes data and outputs a predictor $\widehat{f}$
	
## Decision Theory
	Abraham Wald formulated
	Predictor/hypothesis/decision function
	
	What does it mean to make a good decision?
	f(x) spits out something close to y (regression) loss function
	f(x) is an action that incurs the smallest possible cost if seen y, reward/cost

### Risk

		R(f) = \mathbb{E}[l(f(X), Y)]
	
### Target Function
		
		R(f^*) = \inf_{f \in A^X}

### Conditional Risk
		
		R(a | x) = \mathbb{E}[l(a, Y) | X = x] = \int l(a. y) dP_{Y|X}(y|x)
		
### Least Square Regression

		l(a,y) = (a-y)^2
		R((f) = \mathbb{E}[(f(X)-y)^2]
		let ftilde(X) = \mathbb{E}[Y|X]

### Linear Regression
	
	Design matrix
	X^TXw = X^Ty , normal equations
	
### Polynomial Regression Overfitting

	Particular case of linear regression where all polynomial coefficients are in w
	When you increase complexity, risk overfitting but deep learning does it anyway?
	
### Regularization

	Tikhonov regularization
	$\min_{f \in S} R_n(f) + \lambda ||f||^2$
	If R_n is convex

### Complexity

	Bias-variance trade-off, approximation-estimation trade-off



# Bandit Problems

Why bandits? 
Clinical trials where the agent is a doctor trying treatments for example
Black box optimization







	
