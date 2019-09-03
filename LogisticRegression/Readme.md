# Logistic Regression

Consider the objective function in logistic regression problem l($\theta$) = $\Sigma$(𝑦<sub>i</sub>log$\sigma$($\theta$<sup>T</sup>𝑥<sub>i</sub>log(1 − $\sigma$($\theta$<sup>T</sup>𝑥<sub>i</sub>))), where $\sigma$(z) = (1 + e<sup>-z</sup>)<sup>-1</sup> is the logistic function.

1. the files LR.dat contains the inputs 𝑥<sub>i</sub> ∈ 𝑅<sup>2</sup> and outputs 𝑦<sub>i</sub> ∈ {0,1} respectively for a binary classification problem, with one training example per row.

2. Implement the gradient descent method for optimizing 𝑙($\theta$), and apply it to fit a logistic regression model to the data. Initialize gradient descent method with $\theta$ = 0 (the vector of all zeros).

3. Implement Newton's method to maximize 𝑙($\theta$), and compare the overall running time and number of iterations needed to converge to the same precision.

4. Plot the training data (your axes should correspond to the two coordinates of the inputs, and you should use a different symbol for each point plotted to indicate whether that example had label 1 or 0). Also plot on the same figure the decision boundary fit by logistic regression. (i.e., this should be a straight line showing the boundary separating the region where $\sigma$($\theta$<sup>T</sup>𝑥) > 0.5 from where $\sigma$($\theta$<sup>T</sup>𝑥) < 0.5.)
