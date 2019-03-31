import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def add_intercept(x_data):
        return np.c_[np.ones((x_data.shape[0],1)),x_data]

def sigmoid(z):
        return 1 / (1 + np.exp(-z))

def cost(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

X = np.loadtxt('hw6x.dat')
y = np.loadtxt('hw6y.dat')

dataX = add_intercept(X)
theta = np.zeros(dataX.shape[1])

x_train = np.transpose(dataX)
y_train = np.array([y])
m = len(y_train[0])
#thetaT = np.transpose(theta)

pos = np.flatnonzero(y_train == 1)
neg = np.flatnonzero(y_train == 0)

plt.plot(x_train[1, pos], x_train[2, pos], 'ro')
plt.plot(x_train[1, neg], x_train[2, neg], 'bo')  

for x in range(0, 10):
    h = sigmoid(theta.T.dot(x_train))
    error = h - y_train
    tmp = (-1)*y_train*np.log(h) - (1-y_train)*np.log((1-h))
    J = np.sum(tmp)/m;
    J /=3
    
    #calculate H
    H = (h*(1-h)*(x_train)).dot(x_train.T)/m
    #calculate dJ
    dJ = np.sum(error*x_train, axis=1)/m
    #gradient = H-1.dJ
    grad = inv(H).dot(dJ)
    #update theta
    theta = theta - (np.array([grad])).T
    print(J)

    
print(theta[:,0])


plot_x = [np.ndarray.min(x_train[1:]), np.ndarray.max(x_train[1:])]
plot_y = np.subtract(np.multiply(-(theta[2][0]/theta[1][0]), plot_x), theta[0][0]/theta[1][0])
plt.plot(plot_x, plot_y, 'g-')

plt.show()
