import numpy as np


def getXp():
	Xp = np.array([
	[4, 2, 2, 3, 4, 6, 3, 8],
	[1, 4, 3, 6, 4, 2, 2, 3],
	[0, 1, 1, 0, -1, 0, 1, 0]
	])

	return Xp

def getXn():
	Xn = np.array([
	[9, 6, 9, 8, 10],
	[10, 8, 5, 7, 8],
	[1, 0, 0, 1, -1]
	])

	return Xn

def getX():
	X = np.array([
	[1.3, 2.4, 6.7, 2.2, 3.4, 3.2],
	[8.1, 7.6, 2.1, 1.1, 0.5, 7.4],
	[-1, 2, 3, 2, 0, 2]
	])

	return X

def mybLDA_train(Xp, Xn):
	Xp_mean = np.mean(Xp, axis=1)
	Xn_mean = np.mean(Xn, axis=1)
	
	Xp1 = np.zeros( (len(Xp), len(Xp[0]) ))
	for i in range(len(Xp[0])):
		Xp1[:,i] = Xp[:,i] - Xp_mean

	Xn1 = np.zeros( (len(Xn), len(Xn[0]) ))
	for i in range(len(Xn[0])):
		Xn1[:,i] = Xn[:,i] - Xn_mean

	Xp_Cov = Xp1.dot(Xp1.T)
	Xn_Cov = Xn1.dot(Xn1.T)

	tmp = np.append(Xp, Xn, axis=1)
	total_Mean = np.mean(tmp, axis=1)
	Sb_Xp = len(total_Mean)*((Xp_mean.reshape(len(Xp_mean),1) - total_Mean.reshape(len(total_Mean),1)).dot((Xp_mean.reshape(len(Xp_mean),1) - total_Mean.reshape(len(total_Mean),1)).T))
	Sb_Xn = len(total_Mean)*((Xn_mean.reshape(len(Xn_mean),1) - total_Mean.reshape(len(total_Mean),1)).dot((Xn_mean.reshape(len(Xn_mean),1) - total_Mean.reshape(len(total_Mean),1)).T))
	Sb = Sb_Xp + Sb_Xn

	Sw = Xp_Cov + Xn_Cov

	eigenvalue, eigenvector = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

	arrNum = np.argsort(eigenvalue)[::-1]
	eigenvector_sort = eigenvector[:,arrNum]
	eigenvalue_sort = eigenvalue[arrNum]

	bestRep = eigenvalue_sort[0] / np.sum(eigenvalue)
	bestRepVector = eigenvector_sort[:,:1]
	# Choose best 1 eigenvectors
	projectionDirection = eigenvector_sort[:, :1]

	return projectionDirection

def mybLDA_classify(X, v):
	Xp_lda = (getXp().T).dot(v)
	Xn_lda = (getXn().T).dot(v)
	X_lda = (X.T).dot(v)
	mean = (np.mean(Xp_lda) + np.mean(Xn_lda))/2
	r = np.zeros((len(X_lda),1))

	for i in range(len(X_lda)):
		if X_lda[i] < mean:
			r[i] = 1
		else:
			r[i] = -1

	return r.T


def main():
	Xp = getXp()
	Xn = getXn()
	X = getX()
	v = mybLDA_train(Xp, Xn)
	r = mybLDA_classify(X, v)
	print(r)



if __name__ == "__main__":
	main()