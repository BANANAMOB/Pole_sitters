import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.animation as animation
import matplotlib as mpl
import scipy
import os
import pickle
from scipy.io import savemat
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
colorama_init()
import random
from math import inf
import math
from rich import print as rprint
import time
import seaborn as sns
import alphashape
from descartes import PolygonPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# mass ratio for earth-moon system 
mu_star =  0.01215059   # Constant for CR3BP
max_target_z = 0.406089144456503 

# Choose x and y s.t. it's a pole sitter
target_x = 1 - mu_star
target_y = 0

def get_cone_angle(Traj): 

    Traj_array = np.array(Traj)
    max_theta = 0

    for i in range(Traj_array.shape[0]):
        x = Traj_array[i,0] - (1-mu_star)
        y = Traj_array[i,1]
        z = Traj_array[i,2]

        r = np.linalg.norm([x,y])
        theta = np.arctan(r/z)
        if theta > max_theta:
            max_theta = theta

    return max_theta

def plot_cone(ax,cone_ang):

    #ax = plt.subplot(111, projection='3d')
    # set parameters    
    height = max_target_z*np.cos(cone_ang)
    radius = max_target_z*np.sin(cone_ang)
    n = 50

    # draw cone
    theta = np.linspace(0, 2*np.pi, n)
    z = np.linspace(0, height, 2)
    T, Z_cone = np.meshgrid(theta, z)

    R = radius * (Z_cone/height)
    
    X_cone = R * np.cos(T) + 1-mu_star
    Y_cone = R * np.sin(T)

    ax.plot_surface(X_cone, Y_cone, Z_cone, color='b', alpha=0.1, edgecolor='k', linewidth=0.2)
    
    # draw cap
    phi = np.linspace(0, cone_ang, n) # polar angle (0 = north pole, pi = south pole)
    theta = np.linspace(0, 2*np.pi, n)
    Phi, Theta = np.meshgrid(phi, theta)

    X_cap = max_target_z*np.sin(Phi)*np.cos(Theta) + 1-mu_star
    Y_cap = max_target_z*np.sin(Phi)*np.sin(Theta)
    Z_cap = max_target_z*np.cos(Phi)

    ax.plot_surface(X_cap, Y_cap, Z_cap, color='b', alpha=0.1, edgecolor='k', linewidth=0.2)

def rmse(pred,obs):
    pred = np.array(pred)
    obs = np.array(obs)

    return np.sqrt(((pred-obs)**2).mean())

def plot_error(folder,params):

    const,const_val,_,_ = params
    
    file_side = f"{folder}/pareto_+_fine_pareto_3D_fine_side_{const_val}_all.pkl"
    file_side_e = f"{folder}/pareto_+_fine_pareto_3D_fine_side_{const_val}_e_all.pkl"

    file_top = f"{folder}/pareto_+_fine_pareto_3D_fine_top_{const_val}_all.pkl"
    file_top_e = f"{folder}/pareto_+_fine_pareto_3D_fine_top_{const_val}_e_all.pkl"

    paths = [[file_side,file_side_e],[file_top,file_top_e]]

    pos_errors = []
    u_errors = []
    for path in paths:

        with open(path[0], "rb") as f:
            mass_opt = pickle.load(f)
        with open(path[1], "rb") as f:
            ener_opt = pickle.load(f)

        for i in range(len(mass_opt)):
            if mass_opt[i][0] and (type(ener_opt[i][4]) != int):
                x_m = np.array(mass_opt[i][4][:,0])
                y_m = np.array(mass_opt[i][4][:,1])
                z_m = np.array(mass_opt[i][4][:,2])

                ux_m = np.array(mass_opt[i][4][:,7])
                uy_m = np.array(mass_opt[i][4][:,8])
                uz_m = np.array(mass_opt[i][4][:,9])

                m_size = np.shape(mass_opt[i][4][:,6])[0]
                e_size = np.shape(ener_opt[i][4][:,6])[0]

                x_e = np.array([ener_opt[i][4][int(j * e_size/m_size),0] for j in range(m_size)])
                y_e = np.array([ener_opt[i][4][int(j * e_size/m_size),1] for j in range(m_size)])
                z_e = np.array([ener_opt[i][4][int(j * e_size/m_size),2] for j in range(m_size)])

                ux_e = np.array([ener_opt[i][4][int(j * e_size/m_size),7] for j in range(m_size)])
                uy_e = np.array([ener_opt[i][4][int(j * e_size/m_size),8] for j in range(m_size)])
                uz_e = np.array([ener_opt[i][4][int(j * e_size/m_size),9] for j in range(m_size)])

                pos_error_vec = np.array([x_m - x_e, y_m - y_e, z_m - z_e]).T
                u_error_vec = np.array([ux_m - ux_e, uy_m - uy_e, uz_m - uz_e]).T

                pos_errors.append(np.linalg.norm(pos_error_vec, axis=1))
                u_errors.append(np.linalg.norm(u_error_vec, axis=1))

                #print(np.shape(pos_errors[len(pos_errors)-1])[0]-m_size)

    for pos_error in pos_errors:
        timerange = np.linspace(0,const_val,len(pos_error))
        plt.plot(timerange,pos_error,color='blue',alpha=0.2)
    plt.xlabel('Time in TU')
    plt.ylabel('Norm of error vector')
    plt.title(f'Error between mass and energy optimal for {const} = {const_val}')

    plt.minorticks_on()
    plt.grid(which='major',linewidth=1)
    plt.grid(which='minor',linestyle=':',linewidth=0.5)

# Just the top and side front lines
def plot_front(folder,params,color):

    const,const_val,_,_ = params

    front_side = f"{folder}/front_pareto_3D_fine_side_{const_val}_all.pkl"
    front_top = f"{folder}/front_pareto_3D_fine_top_{const_val}_all.pkl"

    match const:
        case 'target_z':
            x_ind = 2
            plt.xlabel('period')
        case 'period':
            x_ind = 3
            plt.xlabel('target_z')

    x_front_side = []
    y_front_side = []

    x_front_top = []
    y_front_top = []

    paths = [front_side,front_top]
    j = 0
    for path in paths:
        with open(path, "rb") as f:
            front = pickle.load(f)

        if not j:
            for i in range(len(front)):
                x_front_side.append(front[i][x_ind])
                y_front_side.append(front[i][1])
        
        else:
            for i in range(len(front)):
                x_front_top.append(front[i][x_ind])
                y_front_top.append(front[i][1])

        j = j+1


    plt.plot(x_front_side,y_front_side, c=color, marker='o',label=f'Front, T = {const_val} [TU]')
    plt.plot(x_front_top,y_front_top, c=color, marker='o',label='_nolabel_')
    plt.ylabel('Umax')
    plt.title(f'{const} = {const_val}')

    plt.minorticks_on()
    plt.grid(which='major',linewidth=1)
    plt.grid(which='minor',linestyle=':',linewidth=0.5)

def graph(formula, x_range,color):
    x = np.array(x_range)
    y = formula(x)
    plt.plot(x,y,c=color,lw=2,ms=3,label = f'Analytical hovering solution')

def hover_eq(z):
    return (((mu_star-1)+((1-mu_star)/(1+z**2)**(3/2)))**2+((1-mu_star)*(z/(1+z**2)**(3/2))+mu_star*(1/z**2))**2)**0.5

# Aphashape
def plot_full_front(points_conv,const_val,color):
    #alpha = 0.95 * alphashape.optimizealpha(points_conv)
    alpha = 3.01
    hull = alphashape.alphashape(points_conv,alpha)
    hull_pts = hull.exterior.coords.xy
    hull_pts = np.array(hull_pts).T
    
    min_y_of_max_x = min(hull_pts, key=lambda p: (-p[0], p[1]))
    min_x_of_max_y = min(hull_pts, key=lambda p: (-p[1], p[0]))

    mask = []
    tol_x = 0.01
    tol_y = 0.1
    for i in range(len(hull_pts)):
        if hull_pts[i][0] >= max(hull_pts[:,0])-tol_x and not np.array_equal(hull_pts[i],min_y_of_max_x):
            mask.append(i)
        if hull_pts[i][1] >= max(hull_pts[:,1])-tol_y and not np.array_equal(hull_pts[i],min_x_of_max_y):
            mask.append(i)        

    front = np.delete(hull_pts, mask, axis=0).T

    # print(front.T)
    front = front.T[np.sort(np.unique(front.T,axis=0,return_index=True)[1])].T

    #Rotate
    i1 = np.argmin(np.linalg.norm(front.T - min_x_of_max_y, axis=1))
    front = np.roll(front, -i1-1, axis=1)

    plt.plot(front[0],front[1],c=color,marker='o',ms=1.5,lw=1.5, path_effects=[pe.Stroke(linewidth=2.5, foreground='w'), pe.Normal()])
    # lw=2 or default, ms=3 or 1, # label = f'Front, $P$ = {const_val} [TU]'
    # path_effects=[pe.Stroke(linewidth=2.5, foreground='w'), pe.Normal()],

def plot_heatmap(folder,params,fcolor):

    const,const_val,plot_rand,poi = params

    file_side = f"{folder}/pareto_+_fine_pareto_3D_fine_side_{const_val}_all.pkl"
    file_top = f"{folder}/pareto_+_fine_pareto_3D_fine_top_{const_val}_all.pkl"

    paths = [file_side,file_top]

    match const:
        case '$z_0$':
            x_ind = 2
            plt.xlabel('$P$ [TU]')
        case '$P$':
            x_ind = 3
            plt.xlabel('$z_0$ [DU]')

    x_conv = []
    y_conv = []
    x_not_conv = []
    y_not_conv = []
    converged_points = []
    paths = [file_side,file_top]
    for path in paths:

        with open(path, "rb") as f:
            pareto = pickle.load(f)

        for i in range(len(pareto)):

            if plot_rand or not pareto[i][9]:
            #if (i % 6) != 0:
                if not pareto[i][0]:
                    x_not_conv.append(pareto[i][x_ind])
                    y_not_conv.append(pareto[i][1])
                else:
                    x_conv.append(pareto[i][x_ind])
                    y_conv.append(pareto[i][1])
                    converged_points.append(pareto[i])


    c_list = [sublist[poi]*(180/np.pi) for sublist in converged_points] # if sublist[poi]]
    
    # c_list = [get_cone_angle(sublist[poi])*(180/np.pi) for sublist in converged_points] # if sublist[poi]]
    
    # c_list *(180/np.pi) for cone angle
    # c_list *2*np.pi/(const_val*1000)  for dV [km/s]
    # draw full front
    points_conv = list(tuple(zip(x_conv,y_conv)))
    #plot_full_front(points_conv,const_val,fcolor)

    # draw converged
    #sc = plt.scatter(x_conv,y_conv, marker='o',s=1,c=c_list,cmap='plasma' , vmin=min(c_list),vmax=max(c_list),label='Converged') 
    #plt.colorbar()

    # draw heatmap 
    #vmin, vmax = math.floor(min(c_list)), math.ceil(max(c_list)) 
    #print(vmin,vmax)

    # HERE
    levels = np.linspace(0,90,91) # levels=levels # angle - np.linspace(65,91,53) , dV = np.linspace(0,15, 45 + 1) / np.geomspace(0.4,10, 45 + 1)
    #norm = mpl.colors.LogNorm(vmin=0.4,vmax=10)
    #print(min(c_list),max(c_list))
    hb = plt.tricontourf(x_conv,y_conv,c_list,levels=levels,cmap='plasma',zorder=1) # vmin=min(c_list),vmax=max(c_list) ,  dV -> norm=norm
    
    #plt.colorbar(format='%.2f',label='Cone angles $\phi$ [deg]') # label='Cone angles $\phi$ [rad]' label='$\Delta V$ [m/s]'
    
    # draw non converged
    #plt.scatter(x_not_conv,y_not_conv, s=4,c='#004D40', marker='x', label='Not Converged') 
    
    # draw utopia points
    #plt.scatter(min(x_conv),min(y_conv),c=fcolor,marker='x',s=100,label=f'Utopia point, $P$ = {const_val} [TU]')
    #print(min(x_conv),min(y_conv))

    plt.ylabel('$u_{max}$ [DU/TU$^2$]')
    plt.title(f'{const} = {const_val} [TU]')
    #plt.title('Pareto fronts for constant period')
    #plt.title(f'$\Delta v$ [DU/TU] over constant period, T = {const_val} [TU]')
    #plt.title(f'Cone angles [rad] over constant period, T = {const_val} [TU]')

    plt.xlim(0,0.4)
    plt.ylim(0,2.5)

    plt.minorticks_on()
    plt.grid(which='major',linewidth=.3,zorder=0)
    plt.grid(which='minor',linestyle=':',zorder=0,linewidth=0.2)
    ax = plt.gca()
    ax.set_axisbelow(True)

    #return 1
    return hb

def plot_search(folder,params,fcolor):

    const,const_val,plot_rand,poi = params

    file_side = f"{folder}/pareto_+_fine_pareto_3D_fine_side_{const_val}_all.pkl"
    file_top = f"{folder}/pareto_+_fine_pareto_3D_fine_top_{const_val}_all.pkl"

    paths = [file_side,file_top]

    match const:
        case '$z_0$':
            x_ind = 2
            plt.xlabel('$P$ [TU]')
        case '$P$':
            x_ind = 3
            plt.xlabel('$z_0$ [DU]')

    x_conv = []
    y_conv = []
    x_not_conv = []
    y_not_conv = []
    converged_points = []
    paths = [file_side,file_top]
    for path in paths:

        with open(path, "rb") as f:
            pareto = pickle.load(f)

        for i in range(len(pareto)):

            if plot_rand or not pareto[i][9]:
            #if (i % 6) != 0:
                if not pareto[i][0]:
                    x_not_conv.append(pareto[i][x_ind])
                    y_not_conv.append(pareto[i][1])
                else:
                    x_conv.append(pareto[i][x_ind])
                    y_conv.append(pareto[i][1])
                    converged_points.append(pareto[i])


    c_list = [sublist[poi]*(180/np.pi) for sublist in converged_points] # if sublist[poi]]
    
    # c_list = [get_cone_angle(sublist[poi])*(180/np.pi) for sublist in converged_points] # if sublist[poi]]
    
    # c_list *(180/np.pi) for cone angle
    # c_list *2*np.pi/(const_val*1000)  for dV [km/s]
    # draw full front
    points_conv = list(tuple(zip(x_conv,y_conv)))
    plot_full_front(points_conv,const_val,fcolor)

    # draw converged
    sc = plt.scatter(x_conv,y_conv, marker='o',s=4,c=c_list,cmap='plasma' , vmin=min(c_list),vmax=max(c_list),label='Converged') 
    plt.colorbar()

    # draw heatmap 
    #vmin, vmax = math.floor(min(c_list)), math.ceil(max(c_list)) 
    #print(vmin,vmax)

    # HERE
    #levels = np.linspace(0,90,91) # levels=levels # angle - np.linspace(65,91,53) , dV = np.linspace(0,15, 45 + 1) / np.geomspace(0.4,10, 45 + 1)
    #norm = mpl.colors.LogNorm(vmin=0.4,vmax=10)
    #print(min(c_list),max(c_list))
    
    # plt.colorbar(format='%.2f',label='Cone angles $\phi$ [deg]') # label='Cone angles $\phi$ [rad]' label='$\Delta V$ [m/s]'
    
    # draw non converged
    plt.scatter(x_not_conv,y_not_conv, s=4,c='#004D40', marker='x', label='Not Converged') 
    
    # draw utopia points
    #plt.scatter(min(x_conv),min(y_conv),c=fcolor,marker='x',s=100,label=f'Utopia point, $P$ = {const_val} [TU]')
    #print(min(x_conv),min(y_conv))

    plt.ylabel('$u_{max}$ [DU/TU$^2$]')
    plt.title(f'{const} = {const_val} [TU]')
    #plt.title('Pareto fronts for constant period')
    #plt.title(f'$\Delta v$ [DU/TU] over constant period, T = {const_val} [TU]')
    #plt.title(f'Cone angles [rad] over constant period, T = {const_val} [TU]')

    plt.xlim(0,0.4)
    plt.ylim(0,2.5)

    plt.minorticks_on()
    plt.grid(which='major',linewidth=.3,zorder=0)
    plt.grid(which='minor',linestyle=':',zorder=0,linewidth=0.2)
    ax = plt.gca()
    ax.set_axisbelow(True)

    return 1

def plot_orbit(Traj,cone_ang): # ,IG):

    DU = 384400000.0000000
    TU = 2.360584684800000E+06/(2*np.pi)

    Traj_array = np.array(Traj).T
    # IG_array = np.array(IG)

    # if len(IG_array) == max(np.shape(IG_array)):
    #     IG_array = IG_array.T

    #Thrust_Energy_Mag = DU/TU**2 * np.array([[np.sqrt(IG_array[7,i]**2 + IG_array[8,i]**2 + IG_array[9,i]**2)] for i in range(max(np.shape(IG_array)))])
    Thrust_Mass_Mag = DU/TU**2 * np.array([[np.sqrt(Traj_array[7,i]**2 + Traj_array[8,i]**2 + Traj_array[9,i]**2)] for i in range(max(np.shape(Traj_array)))])

    fig = plt.figure()
    ax0 = plt.subplot(421)
    ax1 = plt.subplot(423)
    ax2 = plt.subplot(425)
    ax3 = plt.subplot(427)
    ax4 = plt.subplot(122, projection='3d')

    ax0.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[7], color=[9/255,83/255,186/255])  # plots u_x vs time
    #ax0.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[7], color=[252/255, 186/255, 3/255])
    ax1.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[8], color=[9/255,83/255,186/255])  # plots u_y vs time
    #ax1.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[8], color=[252/255, 186/255, 3/255])
    ax2.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[9], color=[9/255,83/255,186/255])  # plots u_z vs time
    #ax2.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[9], color=[252/255, 186/255, 3/255])
    ax3.plot(TU/86400 * Traj_array[6], Thrust_Mass_Mag, color=[9/255,83/255,186/255])     # plots u mag vs time
    #ax3.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), Thrust_Energy_Mag, color=[252/255, 186/255, 3/255])     # plots u mag vs time

    # plot the energy optimal trajectory
    #ax4.plot(IG_array[0], IG_array[1], IG_array[2], color=[252/255, 186/255, 3/255])
    # plot the mass optimal trajectory
    ax4.plot(Traj_array[0],Traj_array[1],Traj_array[2], color=[9/255,83/255,186/255])
    # plot the moon
    ax4.scatter(1-mu_star,0,0, color=[130/255,130/255,130/255], s=20)
    # plot the cone 
    #plot_cone(ax4,cone_ang) 

    ax0.grid(True)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)

    ax0.set_ylabel(r'$U_x$')
    ax1.set_ylabel(r'$U_y$')
    ax2.set_ylabel(r'$U_z$')
    ax3.set_ylabel(r'$||U||$ [m/s$^2$]')
    ax3.set_xlabel(r't [days]')

    ax4.set_xlabel(r'$X$')
    ax4.set_ylabel(r'$Y$')
    ax4.set_zlabel(r'$Z$')
    fig.set_size_inches(10.5, 5.5, forward=True)
    
    fig.set_tight_layout(True)

    return ax4

# not ready
def plot_all_orbits(Traj,IG):

    DU = 384400000.0000000
    TU = 2.360584684800000E+06/(2*np.pi)

    Traj_array = np.array(Traj).T
    IG_array = np.array(IG)

    if len(IG_array) == max(np.shape(IG_array)):
        IG_array = IG_array.T

    Thrust_Energy_Mag = DU/TU**2 * np.array([[np.sqrt(IG_array[7,i]**2 + IG_array[8,i]**2 + IG_array[9,i]**2)] for i in range(max(np.shape(IG_array)))])
    Thrust_Mass_Mag = DU/TU**2 * np.array([[np.sqrt(Traj_array[7,i]**2 + Traj_array[8,i]**2 + Traj_array[9,i]**2)] for i in range(max(np.shape(Traj_array)))])

    fig = plt.figure()
    ax0 = plt.subplot(421)
    ax1 = plt.subplot(423)
    ax2 = plt.subplot(425)
    ax3 = plt.subplot(427)
    ax4 = plt.subplot(122, projection='3d')

    ax0.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[7], color=[9/255,83/255,186/255])  # plots u_x vs time
    ax0.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[7], color=[252/255, 186/255, 3/255])
    ax1.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[8], color=[9/255,83/255,186/255])  # plots u_y vs time
    ax1.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[8], color=[252/255, 186/255, 3/255])
    ax2.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[9], color=[9/255,83/255,186/255])  # plots u_z vs time
    ax2.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[9], color=[252/255, 186/255, 3/255])
    ax3.plot(TU/86400 * Traj_array[6], Thrust_Mass_Mag, color=[9/255,83/255,186/255])     # plots u mag vs time
    ax3.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), Thrust_Energy_Mag, color=[252/255, 186/255, 3/255])     # plots u mag vs time

    # plot the energy optimal trajectory
    ax4.plot(IG_array[0], IG_array[1], IG_array[2], color=[252/255, 186/255, 3/255])
    # plot the mass optimal trajectory
    ax4.plot(Traj_array[0],Traj_array[1],Traj_array[2], color=[9/255,83/255,186/255])
    # plot the moon
    ax4.scatter(1-mu_star,0,0, color=[130/255,130/255,130/255], s=20)
    # plot the cone 
    plot_cone(ax4) 

    ax0.grid(True)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)

    ax0.set_ylabel(r'$U_x$')
    ax1.set_ylabel(r'$U_y$')
    ax2.set_ylabel(r'$U_z$')
    ax3.set_ylabel(r'$||U||$ [m/s$^2$]')
    ax3.set_xlabel(r't [days]')

    ax4.legend(["Reference", "Intermediate Soln", "Final Soln"], loc="upper right")
    ax4.set_xlabel(r'$X$')
    ax4.set_ylabel(r'$Y$')
    ax4.set_zlabel(r'$Z$')
    fig.set_size_inches(10.5, 5.5, forward=True)
    
    fig.set_tight_layout(True)
    #plt.show()

if __name__ == "__main__":

    # CHANGE YOUR PATH HERE
    folder_parent = "/home/lha43/Desktop/Lucas/pareto_points_2026"

    style_path = "/home/lha43/Desktop/Lucas/Generated_figures/paper.mplstyle"
    plt.style.use(style_path)

    # Color duo
    # success: #7745FF || failure: #EF9C41

    # Color quintet
    # #D81B60 - #1E88E5 - #FFC107 - #004D40 - #965081

    # CHANGE TO YOUR WANTED FOLDER HERE
    folder = f"{folder_parent}/Feb_9_2026_z_99%"

    # Traj_data = [converged_m,Umax,period,target_z,TrajI_m,theta,min_alt,dV_m,-time_cpu_m,rand_flag,f_m]
    const = '$P$' # $P$ or $z_0$
    const_val = 0.05 # period value
    plot_rand = 0 # whether or not to plot random points
    poi = 5 # Theta
    params = [const,const_val,plot_rand,poi]
    
    #region This plotted the pole sitting search 

    # #plt.suptitle('Heatmaps of $\Delta v$ [DU/TU] over constant period')
    # #plt.suptitle('Heatmaps of cone angles [rad] over constant period')
    # #plt.suptitle('Cone angles [rad] over constant period')

    fig, ax = plt.subplots(2,2,constrained_layout=True)
    (ax00,ax01), (ax10,ax11) = ax

    # graph(hover_eq,np.linspace(0.01,0.5,1000),'#D81B60') # #D81B60

    # #plt.subplot(2,2,1)
    plt.sca(ax00)
    fcolor = '#000000' # '#000000' #'#1E88E5'
    const_val = 0.05
    params = [const,const_val,plot_rand,poi]
    plot_search(folder,params,fcolor)
    #plot_heatmap(folder,params,fcolor)
    #plt.legend()
    #plt.show()

    #plt.subplot(2,2,2)
    plt.sca(ax01)
    fcolor = '#000000' # '#000000' #'#FFC107' 
    const_val = 2
    params = [const,const_val,plot_rand,poi]
    plot_search(folder,params,fcolor)
    #plot_heatmap(folder,params,fcolor)
    #plt.legend()
    #plt.show()

    # #plt.subplot(2,2,3)
    plt.sca(ax10)
    fcolor = '#000000' #'#000000' #'#004D40'
    const_val = 4
    params = [const,const_val,plot_rand,poi]
    plot_search(folder,params,fcolor)
    #plot_heatmap(folder,params,fcolor)
    #plt.legend()
    #plt.show()

    # #plt.subplot(2,2,4)
    plt.sca(ax11)
    fcolor = '#000000' # '#000000' #'#965081'
    const_val = 6
    params = [const,const_val,plot_rand,poi]
    hb = plot_search(folder,params,fcolor)
    #hb = plot_heatmap(folder,params,fcolor)
    #plt.legend()
    #plt.show()

    #cbar = fig.colorbar(hb, ax=[ax00,ax01,ax10,ax11],orientation='horizontal',format='%.2f',location='bottom',fraction=.1,aspect=40)
    #cbar.set_label('Cone angles $\phi_{max}$ [deg]') # '$\Delta V$ [km/s]' # 'Cone angles $\phi$ [deg]'

    #plt.xlim(0,0.42)
    #plt.ylim(0,2.6)

    # # plot_error(folder,params)
    # plt.legend()
    # # plt.axis('equal')
    # # plt.tight_layout()
    plt.show()

    #endregion

    #region plot family

    # # Traj_data = [converged_m,Umax,period,target_z,TrajI_m,theta,min_alt,dV_m,-time_cpu_m,rand_flag,f_m]
    # const = '$P$' # $P$ or $z_0$
    # const_val = 6 # period value
    # plot_rand = 0 # whether or not to plot random points
    # poi = 5 # Theta
    # params = [const,const_val,plot_rand,poi]

    # const,const_val,plot_rand,poi = params

    # file_side = f"{folder}/pareto_+_fine_pareto_3D_fine_side_{const_val}_all.pkl"
    # file_top = f"{folder}/pareto_+_fine_pareto_3D_fine_top_{const_val}_all.pkl"

    # paths = [file_side,file_top]

    # match const:
    #     case '$z_0$':
    #         x_ind = 2
    #         #plt.xlabel('$P$ [TU]')
    #     case '$P$':
    #         x_ind = 3
    #         #plt.xlabel('$z_0$ [DU]')

    # with open(paths[0], "rb") as f:
    #     pareto = pickle.load(f)

    # umax_search = 2
    # z_search = [0.08, 0.12]

    # for i in range(len(pareto)):

    #     if pareto[i][1] >  umax_search and z_search[0] < pareto[i][3] and pareto[i][3] < z_search[1]:
    #     #if (i % 6) != 0:
    #         if pareto[i][0]:
    #             Traj = pareto[i][4]
    #             cone_ang = pareto[i][5]
    #             break

    # ax = plot_orbit(Traj,cone_ang)
    # plot_cone(ax,cone_ang)
    # ax.legend(["Traj", "Body", "Cone"], loc="upper right")
    # ax.axis('equal')
    # print(cone_ang*180/np.pi)
    # plt.show()

    #endregion
