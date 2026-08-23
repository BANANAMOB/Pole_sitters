import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
import os
import pickle

filepath = "/home/lha43/Desktop/Lucas/pareto_points/Oct_24_2025_GOOD_z/pareto_230.pkl" 

with open(filepath, "rb") as f:
	pareto = pickle.load(f)

x = []
y = []
x_conv = []
y_conv = []
x_not_conv = []
y_not_conv = []

print(len(pareto))

# Each pareto point is saved like so:
# pareto_point = [converged,Umax,period,target_z,Traj,theta]
# ensure that the correct index is called for the corresponding axis

for i in range(len(pareto)):
	x.append(pareto[i][3])
	y.append(pareto[i][1])
	if not pareto[i][0]:
		x_not_conv.append(pareto[i][3])
		y_not_conv.append(pareto[i][1])
	else:
		x_conv.append(pareto[i][3])
		y_conv.append(pareto[i][1])

#plt.scatter(x,y)
plt.scatter(x_conv,y_conv, c='green', marker='o')
plt.scatter(x_not_conv,y_not_conv, c='red', marker='x')
plt.xlabel('Target_z')
#plt.xlabel('Period')
plt.ylabel('Umax')
#plt.title('target_z = 0.3')
plt.title('period = 2.085034838884136')
plt.grid()
plt.show()
