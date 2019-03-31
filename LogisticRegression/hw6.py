import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

'''
def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cost(h, y, totalNum):
	return (-y * np.log(h) - (1 - y) * np.log(1 - h)) / totalNum

def add_intercept(dataX):
	return np.c_[np.ones((dataX.shape[0],1)),dataX]

dataX = np.loadtxt('hw6x.dat')
dataY = np.loadtxt('hw6y.dat')

#print(dataX)
#print(dataY)

#we find all indices that make y=1 and y=0
pos = np.flatnonzero(dataY == 1)
neg = np.flatnonzero(dataY == 0)

plt.plot(dataX[pos, 0], dataX[pos, 1], 'ro')
plt.plot(dataX[neg, 0], dataX[neg, 1], 'bo') 
plt.show()


#dataX_transpose = np.transpose(dataX)
#dataY_transpose = np.transpose(dataY)
#theta = np.array(np.zeros((len(dataX[0]), 1)))

#X = np.c_[np.ones((dataX.shape[0],1)),dataX]


learning_rate = 0.1
threshold = 0.01
max_iterations = 100000
totalNum = len(dataY)

# intercept
dataX_intercept = add_intercept(dataX)

theta = np.zeros((len(dataX_intercept[0]), 1))

for i in range(max_iterations):
	z = np.dot(dataX_intercept, theta)
	hypothesis = sigmoid(z)

	diff = hypothesis - dataY

	cost = cost(hypothesis, dataY, totalNum)

	gradient = np.dot(dataX_intercept.transpose(), diff) / totalNum

	theta -= learning_rate * gradient

	if cost < threshold:
		break

	if (i%100 == 0):
		print('cost: ', cost)
'''


class LogisticRegression:
    def __init__(self, learning_rate=0.01, threshold=0.01, max_iterations=100000, fit_intercept=True, verbose=False):
        self._learning_rate = learning_rate  # 학습 계수
        self._max_iterations = max_iterations  # 반복 횟수
        self._threshold = threshold  # 학습 중단 계수
        self._fit_intercept = fit_intercept  # 절편 사용 여부를 결정
        self._verbose = verbose  # 중간 진행사항 출력 여부

    # theta(W) 계수들 return
    def get_coeff(self):
        return self._theta

    # 절편 추가
    def add_intercept(self, x_data):
        return np.c_[np.ones((x_data.shape[0],1)),x_data]

    # 시그모이드 함수(로지스틱 함수)
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, x_data, y_data):
        num_examples, num_features = np.shape(x_data)

        if self._fit_intercept:
            x_data = self.add_intercept(x_data)

        # weights initialization
        self._theta = np.zeros(x_data.shape[1])

        for i in range(self._max_iterations):
            hypothesis = self.sigmoid(np.dot(x_data, self._theta))

            # 실제값과 예측값의 차이
            diff = hypothesis - y_data

            # cost 함수
            cost = self.cost(hypothesis, y_data)

            # cost 함수의 편미분 : transposed X * diff / n
            # 증명 : https://stats.stackexchange.com/questions/278771/how-is-the-cost-function-from-logistic-regression-derivated
            gradient = np.dot(np.transpose(x_data), diff) / num_examples

            # gradient에 따라 theta 업데이트
            self._theta -= self._learning_rate * gradient

            # 판정 임계값에 다다르면 학습 중단
            if cost < self._threshold:
                return False

            # 100 iter 마다 cost 출력
            if (self._verbose == True and i % 100 == 0):
                print('cost :', cost)

    def predict_prob(self, x_data):
        if self._fit_intercept:
            x_data = self.add_intercept(x_data)

        return self.sigmoid(np.dot(x_data, self._theta))

    def predict(self, x_data):
        # 0,1 에 대한 판정 임계값은 0.5 -> round 함수로 반올림
        return self.predict_prob(x_data).round()


if __name__ == "__main__":
    X = np.loadtxt('hw6x.dat')
    y = np.loadtxt('hw6y.dat')

    # 학습 implementation
    model = LogisticRegression(learning_rate=0.1, threshold=0.01, max_iterations=10000, verbose=True)
    model.fit(X, y)
    preds = model.predict(X)
    print((preds == y).mean())
    print(model.get_coeff())


#parameters = fit(X, dataY_transpose, theta)
'''
x_values = [np.min(dataX[:, 0] - 5), np.max(dataX[:, 1] + 5)]
y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]
plt.plot(x_values, y_values, label='Decision Boundary')
plt.legend()
plt.show()
'''