# Linear Discriminant Analysis

Let X<sub>p</sub> and X<sub>n</sub> be the positive and negative training data, respectively, as

```python
Xp = [
    [4, 2, 2, 3, 4, 6, 3, 8],
	[1, 4, 3, 6, 4, 2, 2, 3],
	[0, 1, 1, 0, -1, 0, 1, 0]
]

Xn = [
    [9, 6, 9, 8, 10],
	[10, 8, 5, 7, 8],
	[1, 0, 0, 1, -1]
]
```

1. Write code to compute the class specific means of data matrices X<sub>p</sub> and X<sub>n</sub>. Return the mean as a {\bf column} vector.

2. Compute the mean of matrices X<sub>p</sub> and X<sub>n</sub> given in the problem setting.

3. Compute the class specific covariance matrices of data
   matrices X<sub>p</sub> and X<sub>n</sub>.

4. Compute the class specific covariance matrices of data
   matrices X<sub>p</sub> and X<sub>n</sub> given in the problem setting.
5. Compute between class scattering matrix S<sub>b</sub>.

6. Compute S<sub>b</sub> for data given in the problem setting.

7. Compute the within class scattering matrix S<sub>w</sub>.

8. Compute S<sub>w</sub> for data given in the problem setting.

9. write code to compute the LDA projection by solving the generalized eigenvalue decomposition problem.

10. Compute the LDA projection for the data given in the problem setting. You should see that the second eigenvalue is zero.

11. Collect all previous steps, write a function with name mybLDA_train to perform binary LDA, which takes inputs of two data matrices X<sub>p</sub> and X<sub>n</sub> assuming column data, and return the optimal LDA projection direction as a unit vector.

12. Write a function with name mybLDA_classify which takes a data matrix X and a projection direction v, returns a row vector r that has size as the number of rows in X, and r<sub>i</sub> =+1 if the ith column of X is from the class as in X<sub>p</sub>, and r<sub>i</sub> = -1 if the i<sup>th</sup> column in X is from the class as in X<sub>n</sub>.

13. Run your function, mybLDA_train, on the data given in the problem setting, and then use the obtained projection direction and your function `mybLDA_classify` to classify the following data set 𝑋 

```python
[
    [1.3, 2.4, 6.7, 2.2, 3.4, 3.2],
	[8.1, 7.6, 2.1, 1.1, 0.5, 7.4],
	[-1, 2, 3, 2, 0, 2]
]
```