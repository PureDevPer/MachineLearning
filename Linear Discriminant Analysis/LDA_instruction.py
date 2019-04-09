import numpy as np

Xp = np.array([
	[4, 2, 2, 3, 4, 6, 3, 8],
	[1, 4, 3, 6, 4, 2, 2, 3],
	[0, 1, 1, 0, -1, 0, 1, 0]
	])

Xn = np.array([
	[9, 6, 9, 8, 10],
	[10, 8, 5, 7, 8],
	[1, 0, 0, 1, -1]
	])


'''
1. write code to compute the class specific means of data matrices Xp and Xn. 
Your code needs to return the mean as a {\bf column} vector.
2. use your code, compute the mean of matrices Xp and Xn given in the
problem setting.
'''
Xp_mean = np.mean(Xp, axis=1)
Xn_mean = np.mean(Xn, axis=1)

#Xp_mean = np.mean(Xp, axis=0)
#Xn_mean = np.mean(Xn, axis=0)


'''
3. write code to compute the class specific covariance matrices of data
matrices Xp and Xn.
4. use your code, compute the class specific covariance matrices of data
matrices Xp and Xn given in the problem setting.
'''
Xp1 = np.zeros( (len(Xp), len(Xp[0]) ))
for i in range(len(Xp[0])):
	Xp1[:,i] = Xp[:,i] - Xp_mean

Xn1 = np.zeros( (len(Xn), len(Xn[0]) ))
for i in range(len(Xn[0])):
	Xn1[:,i] = Xn[:,i] - Xn_mean

Xp_Cov = Xp1.dot(Xp1.T)
Xn_Cov = Xn1.dot(Xn1.T)


'''
5. write code to compute between class scattering matrix Sb.
6. use your code, compute Sb for data given in the problem setting.
'''
# Using mean of Xp and Xn
# So, combine Xp and Xn to calculate mean
tmp = np.append(Xp, Xn, axis=1)
total_Mean = np.mean(tmp, axis=1)
Sb_Xp = len(total_Mean)*((Xp_mean.reshape(len(Xp_mean),1) - total_Mean.reshape(len(total_Mean),1)).dot((Xp_mean.reshape(len(Xp_mean),1) - total_Mean.reshape(len(total_Mean),1)).T))
Sb_Xn = len(total_Mean)*((Xn_mean.reshape(len(Xn_mean),1) - total_Mean.reshape(len(total_Mean),1)).dot((Xn_mean.reshape(len(Xn_mean),1) - total_Mean.reshape(len(total_Mean),1)).T))
Sb = Sb_Xp + Sb_Xn


'''
7. write code to compute the within class scattering matrix Sw.
8. use your code, compute Sw for data given in the problem setting.
'''
Sw = Xp_Cov + Xn_Cov


'''
9. write code to compute the LDA projection by solving the generalized
eigenvalue decomposition problem. [use function numpy.linalg.eig]
10. use your code, compute the LDA projection for the data given in the problem setting. 
You should see that the second eigenvalue is zero.
'''
eigenvalue, eigenvector = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

# Sorting eigenvalue and eigenvector from highest to lowest
arrNum = np.argsort(eigenvalue)[::-1]
eigenvector_sort = eigenvector[:,arrNum]
eigenvalue_sort = eigenvalue[arrNum]

v = eigenvector_sort[:, :1]

print("eigenvalue: ", eigenvalue_sort)
print("Second eigenvalue: ", eigenvalue_sort[1])
print("We can see that the second eigenvalue is almost zero like given instruction")



'''
11. collect all previous steps, write a function with name mybLDA_train to perform binary LDA, 
which takes inputs of two data matrices Xp and Xn assuming column data, 
and return the optimal LDA projection direction as a unit vector.
'''
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


'''
12. write a function with name mybLDA_classify 
which takes a data matrix X and a projection direction v, 
returns a row vector r that has size as the number of rows in X, 
and r_i =+1 if the ith column of X is from the class as in Xp, 
and r_i =-1 if the ith column in X is from the class as in Xn.
13. Run your function, mybLDA_train, on the data given in the problem setting, 
and then use the obtained projection direction 
and your function mybLDA_classify to classify the following data set𝑋
'''
X = np.array([
	[1.3, 2.4, 6.7, 2.2, 3.4, 3.2],
	[8.1, 7.6, 2.1, 1.1, 0.5, 7.4],
	[-1, 2, 3, 2, 0, 2]
	])
v = eigenvector_sort[:, :1]

def mybLDA_classify(X, v):
	Xp_lda = (Xp.T).dot(v)
	Xn_lda = (Xn.T).dot(v)
	X_lda = (X.T).dot(v)
	mean = (np.mean(Xp_lda) + np.mean(Xn_lda))/2
	r = np.zeros((len(X_lda),1))

	for i in range(len(X_lda)):
		if X_lda[i] < mean:
			r[i] = 1
		else:
			r[i] = -1

	return r


