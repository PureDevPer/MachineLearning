import numpy as np

X = np.array([ 
	[-2, 1, 4, 6, 5, 3, 6, 2],
	[9, 3, 2, -1, -4, -2, -4, 5],
	[0, 7, -5, 3, 2, -3, 4, 6]
	 ])
'''
1. write command to compute the mean of the data matrix X use function mean. 
Your code have to return the mean in terms of a bf column vector.
2. Use your code, compute the mean of matrix X as given in the problem setting
'''
X_mean = np.mean(X, axis=1)

'''
3. Write code to center data matrix X, you can't use any loop command
Use variable X1 for the resulting centered matrix.
4. Use your code, compute the centered data matrix X as given in the problem setting.
'''
X1 = np.zeros( (len(X), len(X[0]) ))
for i in range(len(X[0])):
	X1[:,i] = X[:,i] - X_mean


'''
5. write code to compute unnormalized covariance matrix of the centered data matrix X1. 
Use variable C for the resulting covariance matrix.
6. use your code, compute the covariance matrix of matrix X as given in the problem setting.
'''
C = X1.dot(X1.T)
print("covariance matrix\n", C)

'''
7. write code to compute the first principal component.
(corresponding to the maximum eigenvalue of C).
8. use your code, compute the first principal component 
and its corresponding principal value for matrix X as given in the problem setting.
'''
eigenvalue_C, eigenvector_C = np.linalg.eig(C)
print("Principal Component\n", eigenvector_C)

'''
9. write code to compute the best 1D representation of data matrix X. 
[hint: don't forget to add back the data mean].
10. use your code, compute the best 1D representation of matrix X as given in the problem setting.
'''
arrNum = np.argsort(eigenvalue_C)[::-1]
eigenvector = eigenvector_C[:,arrNum]
eigenvalue = eigenvalue_C[arrNum]


bestRep = eigenvalue[0] / np.sum(eigenvalue_C)
bestRepVector = eigenvector[:,:1]
bestRepMatrix = np.dot(bestRepVector.T, X)


'''
11. collect all previous steps, 
write a function with name mypca, which take inputs of a data matrix assuming column data, 
and return the best $k$-dimensional representation. 
Your function should use the following declaration: 
function [rep, pc, pv] = mypca(X, k), 
where rep contains the optimal k dimensional representation, 
pc contains k top principal components, 
and pv contains the top k principal values.
'''
def mypca(X, k):
	X_mean = np.mean(X, axis=1)
	X1 = np.zeros( (len(X), len(X[0]) ))
	for i in range(len(X[0])):
		X1[:,i] = X[:,i] - X_mean
	C = X1.dot(X1.T)
	eigenvalue_C, eigenvector_C = np.linalg.eig(C)
	arrNum = np.argsort(eigenvalue_C)[::-1]
	eigenvector = eigenvector_C[:,arrNum]
	eigenvalue = eigenvalue_C[arrNum]

	sum = 0
	for i in range(k):
		sum += eigenvalue[i]
		
	pv = sum / np.sum(eigenvalue_C)
	pc = eigenvector[:,:k]
	rep = np.dot(pc.T, X)

	return np.array([rep, pc, pv])