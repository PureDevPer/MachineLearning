import matplotlib.pyplot as plt
import numpy as np
import random

def plotGraph(data, k_number, colors, cluster, distance_data, center1_arr):
	for i in range(k_number):
		new_data = np.array( [ data[j] for j in range(len(data)) if cluster[j] == i  ]  )
		plt.scatter(new_data[:, 0], new_data[:,1], label=i+1, color=colors[i])
		plt.scatter(center1_arr[:, 0], center1_arr[:,1], marker='*', s=100, color='k')

	title = "cluster - " + str(k_number)
	plt.title(title)
	plt.legend(loc='best')
	plt.show()
	print("Total distance\nk = ", k_number,": ", np.sum(distance_data))	


def kMeans(data, k_number, colors):
	init_center = random.sample(range(len(data)), k_number)
	center = [0 for raw in range(k_number)]
	for i in range(k_number):
		center[i] = data[init_center[i]]
		 
	cluster = [0 for raw in range(len(data))]

	while True:
		new = [[] for raw in range(k_number)]
		count = [0 for raw in range(k_number)]
		distance_data = [[] for raw in range(len(data))]
			 
		for i in range(len(data)):
			for j in range(k_number):
				distance_data[i].append(np.square(np.linalg.norm(center[j]-data[i], 2)))
			
		for i in range(len(data)):
			cluster[i] = np.argmin(distance_data[i])
			new[ cluster[i] ].append(data[i])
			count[ cluster[i] ] += 1 
			
		center1 = [[] for raw in range(k_number)]
		
		for i in range(k_number):
			tmp = [ data[j] for j in range(len(data)) if cluster[j] == i ]
			center1[i] = np.mean(tmp, 0)
		
		center1_arr = np.array(center1)
		if np.all(center1_arr == center): 
			break 
		center = center1_arr 

	plotGraph(data, k_number, colors, cluster, distance_data, center1_arr)
	


def main():
	data = np.loadtxt('K-means.dat', delimiter=' ')
	colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
	kMeans(data, 2, colors)
	kMeans(data, 3, colors)
	kMeans(data, 4, colors)
	kMeans(data, 5, colors)


if __name__ == "__main__":
	main()
 
