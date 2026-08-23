import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
import os
import pickle

filepath1 = "/home/lha43/Desktop/Lucas/pareto_points/Nov_22_2025_GOOD_z/pareto_+_fine_pareto_3D_fine_side_0.05_all.pkl"
filepath2 = "/home/lha43/Desktop/Lucas/pareto_points/Nov_22_2025_GOOD_z/pareto_+_fine_pareto_3D_fine_top_0.05_all.pkl"
filepath3 = "/home/lha43/Desktop/Lucas/pareto_points/Nov_22_2025_GOOD_z/front_pareto_3D_fine_side_0.05_all.pkl"
filepath4 = "/home/lha43/Desktop/Lucas/pareto_points/Nov_22_2025_GOOD_z/front_pareto_3D_fine_top_0.05_all.pkl"
# pareto_+_fine_pareto_3D_fine_side_0.05_all
# pareto_+_fine_pareto_3D_fine_top_0.05_all
# filepath5 = "/home/lha43/Desktop/Lucas/pareto_points/Oct_15_2025_5/pareto_all.pkl"
# filepath6 = "/home/lha43/Desktop/Lucas/pareto_points/Oct_15_2025_6/pareto_all.pkl"
# filepath7 = "/home/lha43/Desktop/Lucas/pareto_points/Oct_15_2025_7/pareto_10.pkl"
# filepath8 = "/home/lha43/Desktop/Lucas/pareto_points/Oct_15_2025_8/pareto_50.pkl"
# filepath9 = "/home/lha43/Desktop/Lucas/pareto_points/pareto_15.pkl"

paths = [filepath1,filepath2,filepath3,filepath4] #filepath5,filepath6,filepath7,filepath8,filepath9]

#x = []
#y = []
x_conv = []
y_conv = []
x_not_conv = []
y_not_conv = []

x_front_side = []
y_front_side = []

x_front_top = []
y_front_top = []

j=0
for path in paths:

	with open(path, "rb") as f:
		pareto = pickle.load(f)
	if j<2:
		for i in range(len(pareto)):
			#x.append(pareto[i][3])
			#y.append(pareto[i][1])
			if not pareto[i][0]:
				x_not_conv.append(pareto[i][3])
				y_not_conv.append(pareto[i][1])
			else:
				x_conv.append(pareto[i][3])
				y_conv.append(pareto[i][1])
	else:
		if j == 2:
			for i in range(len(pareto)):
				x_front_side.append(pareto[i][3])
				y_front_side.append(pareto[i][1])
		else:
			for i in range(len(pareto)):
				x_front_top.append(pareto[i][3])
				y_front_top.append(pareto[i][1])
	j=j+1

#plt.scatter(x,y)
plt.scatter(x_conv,y_conv, c='#7745FF', marker='o')
plt.scatter(x_not_conv,y_not_conv, c='#EF9C41', marker='x')
plt.plot(x_front_side,y_front_side, c='#D81B60', marker='o')
plt.plot(x_front_top,y_front_top, c='#004D40', marker='o')
plt.ylabel('Umax')
plt.xlabel('Target_z')
plt.title(f'period = 0.05 ')
#plt.legend(['Converged', 'Not converged',"Front from the side","Front from the top"])
plt.minorticks_on()
plt.grid(which='major',linewidth=1)
plt.grid(which='minor',linestyle=':',linewidth=0.5)
plt.show()
