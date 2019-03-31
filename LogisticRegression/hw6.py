import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

class LogisticRegression:
    def __init__(self, learning_rate=0.01, threshold=0.01, iterations=100000, train_intercept=True, verbose=False):
        self._learning_rate = learning_rate  # 학습 계수
        self._iterations = iterations  # 반복 횟수
        self._threshold = threshold  # 학습 중단 계수
        self._train_intercept = train_intercept  # 절편 사용 여부를 결정
        self._verbose = verbose  # 중간 진행사항 출력 여부

    # theta(W) 계수들 return
    def get_theta(self):
        return self._theta

    # 절편 추가
    def add_intercept(self, x_data):
        return np.c_[np.ones((x_data.shape[0],1)),x_data]

    # 시그모이드 함수(로지스틱 함수)
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def train(self, x_data, y_data):
        totalNum, totalCol = np.shape(x_data)

        if self._train_intercept:
            x_data = self.add_intercept(x_data)

        # weights initialization
        self._theta = np.zeros(x_data.shape[1])

        for i in range(self._iterations):
            hypothesis = self.sigmoid(np.dot(x_data, self._theta))

            # 실제값과 예측값의 차이
            diff = hypothesis - y_data

            # cost 함수
            cost = self.cost(hypothesis, y_data)

            # cost 함수의 편미분 : transposed X * diff / n
            # 증명 : https://stats.stackexchange.com/questions/278771/how-is-the-cost-function-from-logistic-regression-derivated
            gradient = np.dot(np.transpose(x_data), diff) / totalNum

            # gradient에 따라 theta 업데이트
            self._theta -= self._learning_rate * gradient

            # 판정 임계값에 다다르면 학습 중단
            if cost < self._threshold:
                return False

            # 100 iter 마다 cost 출력
            if (self._verbose == True and i % 100 == 0):
                print('cost :', cost, '\t theta: ', self._theta)

    def predict_prob(self, x_data):
        if self._train_intercept:
            x_data = self.add_intercept(x_data)

        return self.sigmoid(np.dot(x_data, self._theta))

    def predict(self, x_data):
        # 0,1 에 대한 판정 임계값은 0.5 -> round 함수로 반올림
        return self.predict_prob(x_data).round()

    def plotGraph(self, x_data, y_data, theta):
    	plt.plot(x_data[np.flatnonzero(y_data == 1), 0], x_data[np.flatnonzero(y_data == 1), 1], 'ro')
    	plt.plot(x_data[np.flatnonzero(y_data == 0), 0], x_data[np.flatnonzero(y_data == 0), 1], 'bo')
    	plotX = [np.ndarray.min(x_data), np.ndarray.max(x_data)]
    	plotY = np.subtract(np.multiply(-(theta[2]/theta[1]), plotX), theta[0]/theta[1])
    	plt.plot(plotX, plotY, 'g-')
    	plt.show()


if __name__ == "__main__":
    X = np.loadtxt('hw6x.dat')
    y = np.loadtxt('hw6y.dat')

    # 학습 implementation
    model = LogisticRegression(learning_rate=0.1, threshold=0.01, iterations=10000, verbose=True)
    model.train(X, y)
    preds = model.predict(X)
    print((preds == y).mean())
    print(model.get_theta())
    model.plotGraph(X, y, model.get_theta())


#parameters = fit(X, dataY_transpose, theta)
'''
x_values = [np.min(dataX[:, 0] - 5), np.max(dataX[:, 1] + 5)]
y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]
plt.plot(x_values, y_values, label='Decision Boundary')
plt.legend()
plt.show()
'''