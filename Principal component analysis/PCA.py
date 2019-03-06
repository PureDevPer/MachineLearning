import numpy as np

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



def main():
	X = np.array([ 
	[-2, 1, 4, 6, 5, 3, 6, 2],
	[9, 3, 2, -1, -4, -2, -4, 5],
	[0, 7, -5, 3, 2, -3, 4, 6]
	])
	rep1, pc1, pv1 = mypca(X, 1)
	rep2, pc2, pv2 = mypca(X, 2)
	rep3, pc3, pv3 = mypca(X, 3)

	print("Optimal 1 dimensional representation\n", rep1)
	print("1 top principal component\n", pc1)
	print("top 1 principal value: ", pv1)

	print("\n\nOptimal 2 dimensional representation\n", rep2)
	print("2 top principal component\n", pc2)
	print("top 2 principal value: ", pv2)

	print("\n\nOptimal 3 dimensional representation\n", rep3)
	print("3 top principal component\n", pc3)
	print("top 3 principal value: ", pv3)

if __name__ == "__main__":
	main()