import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
import os
import pickle

#filepath = "/home/lha43/Desktop/Lucas/pareto_points/Oct_21_2025_4_GOOD_z/pareto_all.pkl" #146 all
#filepath = "/home/lha43/Desktop/Lucas/pareto_points/Oct_23_2025_1_GOOD_z/pareto_140.pkl" #140
#filepath = "/home/lha43/Desktop/Lucas/pareto_points/Oct_23_2025_2_GOOD_period/pareto_1232.pkl" #1232
#filepath_p1 = "/home/lha43/Desktop/Lucas/pareto_points/coarse1-side_30.pkl" #230
filepath = "/home/lha43/Desktop/Lucas/pareto_points/Nov_22_2025_GOOD_z/pareto_+_fine_pareto_3D_fine_top_3_all.pkl"
# pareto_+_fine_pareto_3D_fine_side_2_all

#ith open(filepath_p1, "rb") as f:
#	pareto1 = pickle.load(f)
with open(filepath, "rb") as f:
	pareto = pickle.load(f)

# filepath_f1 = "/home/lha43/Desktop/Lucas/pareto_points/fine_params_coarse1-side_7.pkl" #230
# filepath_f2 = "/home/lha43/Desktop/Lucas/pareto_points/fine_params_coarse1-side_7.pkl" #230

# with open(filepath_f1, "rb") as f:
# 	fine1 = pickle.load(f)
# with open(filepath_f2, "rb") as f:
# 	fine2 = pickle.load(f)

# x = []
# y = []
x_conv = []
y_conv = []
x_not_conv = []
y_not_conv = []

print(len(pareto))
#print(len(pareto1))

# Each pareto point is saved like so:
# pareto_point = [converged,Umax,period,target_z,Traj,theta]
# ensure that the correct index is called for the corresponding axis

# Desicion Variables # 
x_axis = 'target_z'
##########

match x_axis:
	case 'target_z':
		for i in range(len(pareto)):
			# x.append(pareto[i][3])
			# y.append(pareto[i][1])
			if not pareto[i][0]:
				x_not_conv.append(pareto[i][3])
				y_not_conv.append(pareto[i][1])
			else:
				x_conv.append(pareto[i][3])
				y_conv.append(pareto[i][1])

		plt.xlabel('Target_z')
		period = pareto[0][2]
		plt.title(f'period = {period}')
	case 'period':
		for i in range(len(pareto)):
			x.append(pareto[i][2])
			y.append(pareto[i][1])
			if not pareto[i][0]:
				x_not_conv.append(pareto[i][2])
				y_not_conv.append(pareto[i][1])
			else:
				x_conv.append(pareto[i][2])
				y_conv.append(pareto[i][1])

		plt.xlabel('Period')
		z = pareto1[0][3]
		plt.title(f'target_z = {z}')


# Traj_data_m = [converged_m,Umax,period,target_z,TrajI_m,theta,min_alt,dV_m,time_cpu_m,rand_flag,f_m]
def theta(inputlist):
	return [sublist[5] for sublist in inputlist if sublist[5]]

#plt.scatter(x,y) #  c='#7745FF',
sc = plt.scatter(x_conv,y_conv, marker='o',c=theta(pareto),cmap='plasma' , vmin=min(theta(pareto)),vmax=max(theta(pareto))) 
plt.scatter(x_not_conv,y_not_conv, c='#EF9C41', marker='x') 
plt.ylabel('Umax')
#plt.legend(['Converged', 'Not converged'])
plt.minorticks_on()
plt.colorbar(sc)
plt.grid(which='major',linewidth=1)
plt.grid(which='minor',linestyle=':',linewidth=0.5)
plt.show()
