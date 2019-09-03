# Principal Component Analysis

Using the following set of 3D data written as a matrix x

```python
[
    [-2, 1, 4, 6, 5, 3, 6, 2],
    [9, 3, 2, -1, -4, -2, -4, 5],
    [0, 7, -5, 3, 2, -3, 4, 6]
]
```

1. Write command to compute the mean of the data matrix X use func-tion mean. Your code have to return the mean in terms of a column vector.

2. Compute the mean of matrix X as given in the problem setting.

3. Write code to center data matrix X, you cannot use any loop com-mand. Use variable X1 for the resulting centered matrix.

4. Compute the centered data matrix X as given in the problem setting.

5. write code to compute unnormalized covariance matrix of the cen-tered data matrix X1. Use variable C for the resulting covariance matrix.

6. Compute the covariance matrix of matrix X as given in the problem setting.

7. write code to compute the first principal component (corresponding to the maximum eigenvalue of C).

8. Compute the first principal component and its corresponding princi-pal value for matrix X as given in the problem setting.

9. Write code to compute the best 1D representation of data matrix X.

10. Compute the best 1D representation of matrix X as given in the problem setting.

11. collect all previous steps, write a function with name mypca, which take inputs of a data matrix assuming column data, and return the best $k$-dimensional representation. Your function should use the following declaration: function [rep, pc, pv] = mypca(X, k), where rep contains the optimal k dimensional representation, pc contains k top principal components, and pv contains the top k principal values.
