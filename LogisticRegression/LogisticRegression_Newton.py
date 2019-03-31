import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

class LogisticRegression:
    def __init__(self, learning_rate=0.01, threshold=0.01, iterations=10, train_intercept=True):
        self._learning_rate = learning_rate  # learning rate
        self._iterations = iterations  # number of iteration
        self._threshold = threshold  # threshold
        self._train_intercept = train_intercept  # whether to use intercept
        

    # return theta coefficient
    def get_theta(self):
        return self._theta

    # intercept
    def add_intercept(self, x_data):
        return np.c_[np.ones((x_data.shape[0],1)),x_data]

    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def train(self, x_data, y_data):
        totalNum, totalCol = np.shape(x_data)

        if self._train_intercept:
            x_data = self.add_intercept(x_data)

        
        self._theta = np.zeros(x_data.shape[1])
        x_data = np.transpose(x_data)
        y_data = np.array(y_data)
        

        for i in range(self._iterations):
            hypothesis = self.sigmoid(np.dot(np.transpose(self._theta), x_data))

            # difference between hypothesis and real data
            diff = hypothesis - y_data

            # cost function
            cost = self.cost(hypothesis, y_data)
            H = np.dot(hypothesis*(1 - hypothesis)*(x_data), x_data.T) / totalNum
            dJ = np.sum(diff*x_data, axis=1) / totalNum
            gradient = np.dot(inv(H), dJ)

            # Update theta
            self._theta = self._theta - np.transpose(np.array([gradient]))

            print('cost :', cost, '\t theta: ', self._theta[:, 0])


    def plotGraph(self, x_data, y_data, theta):
        plt.plot(x_data[np.flatnonzero(y_data == 1), 0], x_data[np.flatnonzero(y_data == 1), 1], 'ro')
        plt.plot(x_data[np.flatnonzero(y_data == 0), 0], x_data[np.flatnonzero(y_data == 0), 1], 'bo')
        plotX = [np.ndarray.min(x_data), np.ndarray.max(x_data)]
        plotY = np.subtract(np.multiply(-(theta[2][0]/theta[1][0]), plotX), theta[0][0]/theta[1][0])
        plt.plot(plotX, plotY, 'g-')
        plt.title("Logistic Regression - Newton's method")
        plt.show()
    


if __name__ == "__main__":
    X = np.loadtxt('hw6x.dat')
    y = np.loadtxt('hw6y.dat')

    # Training
    model = LogisticRegression(learning_rate=0.1, threshold=0.01, iterations=10)
    model.train(X, y)
    print(model.get_theta()[:,0])
    model.plotGraph(X, y, model.get_theta())
