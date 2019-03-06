import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv


def plotGraph(x, y, x_draw, y_draw):
	plt.plot(x, y, '.', label='Raw Data')
	plt.plot(x_draw, y_draw, '-', label='Regression')
	plt.legend(loc='best')


def degree1(x, y, x_np, ones_arr_np):
	A = np.concatenate((ones_arr_np.T, x_np.T), axis=1)
	C = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y)
	x_draw = np.linspace(-10, 10, 10000)
	y_draw, sum_x = 0, 0
	for i in range(2):
		y_draw += C[i]*(x_draw**i)
		sum_x += C[i]*(x**i)

	sum_square = np.sum((y-sum_x)**2)
	print("Coefficient(order-1):", C)
	print("Residual: ", sum_square, "\nOptimal fit error: ", sum_square/np.size(x))
	
	plt.title("Order: 1")
	plotGraph(x, y, x_draw, y_draw)
	plt.savefig("Order_1.eps", format='eps', dpi=500)
	plt.show()
	


def degree3(x, y, x_np, ones_arr_np):
	A = np.concatenate((ones_arr_np.T, x_np.T, (x_np**2).T, (x_np**3).T), axis=1)
	C = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y)
	x_draw = np.linspace(-10, 10, 10000)
	y_draw, sum_x = 0, 0
	for i in range(4):
		y_draw += C[i]*(x_draw**i)
		sum_x += C[i]*(x**i)

	sum_square = np.sum((y-sum_x)**2)
	print("\nCoefficient(order-3): ", C)
	print("Residual: ", sum_square, "\nOptimal fit error: ", sum_square/np.size(x))
	
	plt.title("Order: 3")
	plotGraph(x, y, x_draw, y_draw)
	plt.savefig("Order_3.eps", format='eps', dpi=500)
	plt.show()
	

def degree5(x, y, x_np, ones_arr_np):
	A = np.concatenate((ones_arr_np.T, x_np.T, (x_np**2).T, (x_np**3).T, (x_np**4).T, (x_np**5).T), axis=1)
	C = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y)
	x_draw = np.linspace(-10, 10, 10000)
	y_draw, sum_x = 0, 0
	for i in range(6):
		y_draw += C[i]*(x_draw**i)
		sum_x += C[i]*(x**i)

	sum_square = np.sum((y-sum_x)**2)
	print("\nCoefficient(order-5): ", C)
	print("Residual: ", sum_square, "\nOptimal fit error: ", sum_square/np.size(x))

	plt.title("Order: 5")
	plotGraph(x, y, x_draw, y_draw)
	plt.savefig("Order_5.eps", format='eps', dpi=500)
	plt.show()


def degree7(x, y, x_np, ones_arr_np):
	A = np.concatenate((ones_arr_np.T, x_np.T, (x_np**2).T, (x_np**3).T, (x_np**4).T, (x_np**5).T, (x_np**6).T, (x_np**7).T), axis=1)
	C = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y)
	x_draw = np.linspace(-10, 10, 10000)
	y_draw, sum_x = 0, 0
	for i in range(8):
		y_draw += C[i]*(x_draw**i)
		sum_x += C[i]*(x**i)
	
	sum_square = np.sum((y-sum_x)**2)
	print("\nCoefficient(order-7): ", C)
	print("Residual: ", sum_square, "\nOptimal fit error: ", sum_square/np.size(x))
	
	plt.title("Order: 7")
	plotGraph(x, y, x_draw, y_draw)
	plt.savefig("Order_7.eps", format='eps', dpi=500)
	plt.show()



def main():
	data = np.loadtxt('LLS.dat', delimiter=' ')

	x, y = data[:,0], data[:,1]
	data_len = np.size(x)

	x_np, y_np = np.array([x]), np.array([y])
	ones_arr = np.ones(data_len)
	ones_arr_np = np.array([ones_arr])
	
	degree1(x, y, x_np, ones_arr_np)
	degree3(x, y, x_np, ones_arr_np)
	degree5(x, y, x_np, ones_arr_np)
	degree7(x, y, x_np, ones_arr_np)
	


if __name__ == "__main__":
	main()

