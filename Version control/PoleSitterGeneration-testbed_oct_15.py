import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
import os
import pickle
from scipy.io import savemat

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

# setting canonical distance unit
DU = 384400000.0000000

# setting canonical time unit
TU = 2.360584684800000E+06/(2*np.pi)

# mass ratio for earth-moon system 
mu_star =  0.01215059   # Constant for CR3BP

# maximum contol authority magnitude
Umax = 0.0184*100  # DU/TU^2

# REFERENCE TRAJECTORY STATE AND PERIOD
tf  = 2.085034838884136 # TU, 1 full orbital period

# minimum control profile set to aid ASSET convergence 
Umin = 1.0e-8

####################### LUCAS #######################

max_target_z = 0.406089144456503 # DU, SOI earth-moon 
max_cone_angle = np.pi/4 # rad
cone_angle_range = np.linspace(np.pi/3, np.pi/15, 50) # rad
real_u_max = 0.00244 # potential reasonable value

# u - thrust; p - period; z - initial z; theta - cone_angle 
pareto_points = [] # [converged?,traj,u,p,z,theta]

## TODO
# add params to title of plot
# add path constraint for z DONE
# add path param for conic section + SOI DONE
# Use stepping to find converging value for pareto

#####################################################

# Garrison+Lucas: play around with these parameters and see how they change solutions
# ASSET parameters
optType = "LGL5"
E_or_T = 1
numKnots = 200
numThreads = 8
MeshTol = 1.0e-10
EControl = 1.0e-12



# thrust and fuel limited (non-linear) dynamics for a satellite in the synodic frame
class CR3BP_Thrust_Dynamics(oc.ODEBase):

    def __init__(self,mu_star):
        Xvars = 6
        Uvars = 3
        ####################################################
        XtU = oc.ODEArguments(Xvars,Uvars)

        x,y,z,xdot,ydot,zdot = XtU.XVec().tolist()
        ux,uy,uz = XtU.UVec().tolist()

        R_31 = ((x+mu_star)**2 + y**2 + z**2)**(0.5)
        R_32 = ((x-1+mu_star)**2 + y**2 + z**2)**(0.5)

        xddot = 2*ydot + x - (1-mu_star)*(x+mu_star)/R_31**3 - mu_star*(x-1+mu_star)/R_32**3 + ux
        yddot = -2*xdot + y - (1-mu_star)*y/R_31**3 - mu_star*y/R_32**3 + uy
        zddot = -(1-mu_star)*z/R_31**3 - mu_star*z/R_32**3 + uz
    
        ode = vf.stack([xdot,ydot,zdot,xddot,yddot,zddot])
        ####################################################
        super().__init__(ode,Xvars,Uvars)

def plot_cone():  # LUCAS

    ax = plt.subplot(111, projection='3d')
    # set parameters    
    height = max_target_z*np.cos(max_cone_angle)
    radius = max_target_z*np.sin(max_cone_angle)
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
    phi = np.linspace(0, max_cone_angle, n) # polar angle (0 = north pole, pi = south pole)
    theta = np.linspace(0, 2*np.pi, n)
    Phi, Theta = np.meshgrid(phi, theta)

    X_cap = max_target_z*np.sin(Phi)*np.cos(Theta) + 1-mu_star
    Y_cap = max_target_z*np.sin(Phi)*np.sin(Theta)
    Z_cap = max_target_z*np.cos(Phi)

    ax.plot_surface(X_cap, Y_cap, Z_cap, color='b', alpha=0.1, edgecolor='k', linewidth=0.2)

def get_cone_angle(Traj): # Lucas

    Traj_array = np.array(Traj)
    max_theta = 0

    for i in range(Traj_array.shape[0]):
        x = Traj_array[i,0]
        y = Traj_array[i,1]
        z = Traj_array[i,2]

        r = np.linalg.norm([x,y])
        theta = np.arctan(r/z)
        if theta > max_theta:
            max_theta = theta

    return max_theta

def plot_opt_family(Traj,IG,ref_state):

    DU = 384400000.0000000
    TU = 2.360584684800000E+06/(2*np.pi)

    Traj_array = np.array(Traj).T
    IG_array = np.array(IG)

    if len(IG_array) == max(np.shape(IG_array)):
        IG_array = IG_array.T

    # Thrust_Energy_Mag = DU/TU**2 * np.array([[np.sqrt(IG_array[7,i]**2 + IG_array[8,i]**2 + IG_array[9,i]**2)] for i in range(max(np.shape(IG_array)))])
    # Thrust_Mass_Mag = DU/TU**2 * np.array([[np.sqrt(Traj_array[7,i]**2 + Traj_array[8,i]**2 + Traj_array[9,i]**2)] for i in range(max(np.shape(Traj_array)))])

    # fig = plt.figure()
    # ax0 = plt.subplot(421)
    # ax1 = plt.subplot(423)
    # ax2 = plt.subplot(425)
    # ax3 = plt.subplot(427)
    ax4 = plt.subplot(111, projection='3d')

    # ax0.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[7], color=[9/255,83/255,186/255])  # plots u_x vs time
    # ax0.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[7], color=[252/255, 186/255, 3/255])
    # ax1.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[8], color=[9/255,83/255,186/255])  # plots u_y vs time
    # ax1.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[8], color=[252/255, 186/255, 3/255])
    # ax2.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[9], color=[9/255,83/255,186/255])  # plots u_z vs time
    # ax2.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[9], color=[252/255, 186/255, 3/255])
    # ax3.plot(TU/86400 * Traj_array[6], Thrust_Mass_Mag, color=[9/255,83/255,186/255])     # plots u mag vs time
    # ax3.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), Thrust_Energy_Mag, color=[252/255, 186/255, 3/255])     # plots u mag vs time

    # plot the reference trajectory
    ax4.plot(ref_state[:,0], ref_state[:,1], ref_state[:,2], color=[0/255,0/255,0/255])
    # plot the energy optimal trajectory
    ax4.plot(IG_array[0], IG_array[1], IG_array[2], color=[252/255, 186/255, 3/255])
    # plot the mass optimal trajectory
    ax4.plot(Traj_array[0],Traj_array[1],Traj_array[2], color=[9/255,83/255,186/255])
    # plot the moon
    ax4.scatter(1-mu_star,0,0, color=[130/255,130/255,130/255], s=20)

    # ax0.grid(True)
    # ax1.grid(True)
    # ax2.grid(True)
    # ax3.grid(True)
    ax4.grid(True)

    # ax0.set_ylabel(r'$U_x$')
    # ax1.set_ylabel(r'$U_y$')
    # ax2.set_ylabel(r'$U_z$')
    # ax3.set_ylabel(r'$||U||$ [m/s$^2$]')
    # ax3.set_xlabel(r't [days]')

    # ax4.legend(["Reference", "Intermediate Soln", "Final Soln"], loc="upper right")
    ax4.set_xlabel(r'$X$')
    ax4.set_ylabel(r'$Y$')
    ax4.set_zlabel(r'$Z$')

    # fig.set_size_inches(10.5, 5.5, forward=True)
    
    # fig.set_tight_layout(True)
    # plt.show()

def Full_Plot(Traj,IG,ref_state):

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

    # plot the reference trajectory
    ax4.plot(ref_state[:,0], ref_state[:,1], ref_state[:,2], color=[0/255,0/255,0/255])
    # plot the energy optimal trajectory
    ax4.plot(IG_array[0], IG_array[1], IG_array[2], color=[252/255, 186/255, 3/255])
    # plot the mass optimal trajectory
    ax4.plot(Traj_array[0],Traj_array[1],Traj_array[2], color=[9/255,83/255,186/255])
    # plot the moon
    ax4.scatter(1-mu_star,0,0, color=[130/255,130/255,130/255], s=20)
    # plot the cone LUCAS
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
    plt.show()


# constrain the initial and terminal position and velocities to be the same after one period of the forced-periodic trajectory 
def FrontBackEqCon():
    X_0, t_0, X_f, t_f = Args(14).tolist([(0,6), (6,1), (7,6), (13,1)])
    eq1 = X_0[3:] - X_f[3:]
    return eq1


def plot_traj(Traj_array):
    fig = plt.figure()
    ax0 = plt.subplot(421)
    ax1 = plt.subplot(423)
    ax2 = plt.subplot(425)
    ax3 = plt.subplot(426)

    # plot control efforts undertaken vs time 

    ax0.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[7])  # plots u_x vs time

    ax1.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[8])  # plots u_y vs time

    ax2.plot(TU/86400 * Traj_array[6], DU/TU**2 * Traj_array[9])  # plots u_z vs time

    ax3.plot(TU / 86400 * Traj_array[6], DU / TU ** 2 * np.linalg.norm(Traj_array[7:10], axis=0))


    ax0.grid(True)
    ax1.grid(True)
    ax2.grid(True)  


    ax0.set_ylabel(r'$U_x$')
    ax1.set_ylabel(r'$U_y$')
    ax2.set_ylabel(r'$U_z$')
    ax3.set_ylabel(r'$Control Magnitude$')

    fig.set_size_inches(10.5, 5.5, forward=True)


    fig.set_tight_layout(True)

# code for computing the forced periodic trajectories. this example computes forced periodic trajectories that travel over the moon in the synodic frame "pole sitters"
# it computes these trajectories for a variety of elevations over the pole of the moon
#


def pareto_wifi_search():
    print("running pole sitter computation, just for you handsome")
    # target_z = [0.5, 0.21052631] # picked out two particularly interesting trajectories that DO CONVERGE

    ode = CR3BP_Thrust_Dynamics(mu_star)

    # Reference trajectory computation
    InitStepSize = 0.001 # TU
    RefInteg = ode.integrator("DOPRI87", InitStepSize)
    RefInteg.setAbsTol(1.0e-16)
    RefInteg.setRelTol(1.0e-14)
    
    pareto = []

    # enumerating the different altitudes
    #for _, targ in enumerate(target_z):

    target_x = 1 - mu_star
    target_y = 0

    # Decision variables tf, umax, z are safe
    mid_period = 2.085034838884136
    max_period = mid_period*1.5
    min_period = mid_period*0.5

    mid_Umax = 0.0184*100
    max_Umax = mid_Umax*1.5
    min_Umax = mid_Umax*0.18

    max_target_z = 0.406089144456503
    mid_target_z = max_target_z*2/3
    min_target_z = max_target_z*0
    # target_z = np.linspace(max_target_z, 0, 20)
    # tf_mult = np.linspace(1.5, 0, 20)  # tf = 2.085034838884136 

    # pareto wifi search
    n = 10000 # loop limit
    ds = 1/2000 # small step size multiplier 
    s = 1/100 # dynamic step size multiplier 
    wifi_search_threshold = 10 # steps for wifi search
    wifi_dist_x = ds*10 # convergence distance to warrant a wifi sweep
    wifi_flag = 0 # when a wifi sweep is triggered on next non-convergence
    wifi_density_rad = 8 # decity of sweep of "bars"
    wifi_density_conc = 6 # amount of "bars"

    conv_num = 0 # number of sequential convergences
    conv_num_max = 5 # number of sequential convergences to warrant change in step

    not_conv_num = 0 # number of sequential non-convergences
    not_conv_num_max = 3 # number of allowed sequential non-convergences

    # Start values 
    target_z = max_target_z-0.3071891
    first_z = target_z
    tf = mid_period
    Umax = min_Umax
    l = 1
    k = 1
    for j in range(n):

        print(f"Point: {j},Umax {Umax} ,Period {tf},z: {target_z}")

        if target_z < 0:
            target_z = last_conv
            k = k + round (( 1 - (target_z/(max_target_z-0.3071891)) )/ds )
            conv_num = 0


        if j % 5 == 0:
            filepath = f"/home/lha43/Desktop/Lucas/pareto_points/pareto_{j}.pkl"

            with open(filepath,"wb") as f:
                pickle.dump(pareto,f)

            savemat(f'/home/lha43/Desktop/Lucas/pareto_points/pareto_{j}.mat', {f'pareto_{j}': pareto})

        try: 
            ref_state_integrated = np.array(RefInteg.integrate_dense(np.array([1.06315768, 0.000326952322, -0.200259761, 0.000361619362, -0.176727245, -0.000739327422, 0.0, 0.0, 0.0, 0.0]),tf))

            # giving an initial guess that is obtained by multiplying the reference state by 3, to enforce convergence to a non-halo orbit
            # many different initial guesses can be given, often resulting in different pole-sitter trajectories
            IG = [[ref_state_integrated[i,0] * 3, ref_state_integrated[i,1] * 3, ref_state_integrated[i,2] * 3, ref_state_integrated[i,3] * 3, ref_state_integrated[i,4] * 3, ref_state_integrated[i,5] * 3, tf*i/(max(np.shape(ref_state_integrated))), 1e-6, 1e-6, 1e-6] for i in range(max(np.shape(ref_state_integrated)))]


            # Garrison+Lucas: change the value of tf in BoundaryLast (e.g., tf*0.8) and see how that changes the solution space 
            #               : also try out different values for targ or target_z        
            BoundaryFirst = list([target_x, target_y, target_z]) + [0]
            BoundaryLast =  list([target_x, target_y, target_z]) + [tf] # LUCAS

            phase1 = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, 0, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z)
            IG = phase1.returnTraj()
            phase2 = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z)
            Traj = phase2.returnTraj()


            dV, f = reintegration(phase2, tf,Traj,Umax)

            print("Z magnitude:                 ",target_z," DU")
            print("Period:                      ",tf, " TU") 
            print("Passed tests (1) or not (0): ",f)
            print("Delta V:                     ",dV," m/s")

            # Full_Plot(Traj,IG,ref_state_integrated)
            #plot_opt_family(Traj,IG,ref_state_integrated)
            IG = Traj

            # add point to front
            theta = get_cone_angle(Traj)
            converged = 1

            pareto.append([converged,Umax,tf,target_z,Traj,theta])
        except:
            # add point to front
            converged = 0
            pareto.append([converged,Umax,tf,target_z,0,0])

        # Algorithm
        if converged: 
            not_conv_num = 0
            conv_num = conv_num + 1
            last_conv = target_z
            if wifi_dist_x < (first_z-target_z):
                wifi_flag = 1
            if conv_num > conv_num_max:
                s_mult = conv_num - conv_num_max 
                target_z = (max_target_z-0.3071891)*((1-(ds*k))-s*s_mult)
                k = k + round (( 1 - (target_z/(max_target_z-0.3071891)) )/ds )
            else:
                target_z = (max_target_z-0.3071891)*(1-(ds*k))
                k = k + 1
        else: 
            not_conv_num = not_conv_num + 1
            conv_num = 0
            if not_conv_num <= not_conv_num_max: 
                target_z = (max_target_z-0.3071891)*(1-(ds*k))
                k = k + 1
            elif (last_conv-target_z) > ds*5 :
                target_z = (last_conv-target_z)/2 + target_z
                k = k + round (( 1 - (target_z/(max_target_z-0.3071891)) )/ds )
                not_conv_num = 0
            else:
                target_z = last_conv
                if wifi_flag:
                    # SWEEP
                    pareto = wifi_sweep(pareto,target_z,first_z,Umax,wifi_density,tf)
                wifi_flag = 0
                Umax = min_Umax*(1+(ds*l))
                l = l + 1
                first_z = target_z        


        def wifi_sweep(pareto,target_z,first_z,Umax,wifi_density,tf):
            base_z = target_z
            base_Umax = Umax
            angles = np.linspace(0,-np.pi/2,wifi_density_rad)
            bars = np.linspace(0,target_z-first_z,wifi_density_conc)
            for angle in angles:

                for bar in bars:

                    target_z = base_z + bar*np.cos(angle)
                    Umax = baseUmax + bar*np.sin(angle)

                    try: 
                        ref_state_integrated = np.array(RefInteg.integrate_dense(np.array([1.06315768, 0.000326952322, -0.200259761, 0.000361619362, -0.176727245, -0.000739327422, 0.0, 0.0, 0.0, 0.0]),tf))

                        # giving an initial guess that is obtained by multiplying the reference state by 3, to enforce convergence to a non-halo orbit
                        # many different initial guesses can be given, often resulting in different pole-sitter trajectories
                        IG = [[ref_state_integrated[i,0] * 3, ref_state_integrated[i,1] * 3, ref_state_integrated[i,2] * 3, ref_state_integrated[i,3] * 3, ref_state_integrated[i,4] * 3, ref_state_integrated[i,5] * 3, tf*i/(max(np.shape(ref_state_integrated))), 1e-6, 1e-6, 1e-6] for i in range(max(np.shape(ref_state_integrated)))]


                        # Garrison+Lucas: change the value of tf in BoundaryLast (e.g., tf*0.8) and see how that changes the solution space 
                        #               : also try out different values for targ or target_z        
                        BoundaryFirst = list([target_x, target_y, target_z]) + [0]
                        BoundaryLast =  list([target_x, target_y, target_z]) + [tf] # LUCAS

                        phase1 = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, 0, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z)
                        IG = phase1.returnTraj()
                        phase2 = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z)
                        Traj = phase2.returnTraj()

                        # add point to front
                        theta = get_cone_angle(Traj)
                        converged = 1
                        pareto.append([converged,Umax,tf,target_z,Traj,theta])
                    except:
                        # add point to front
                        converged = 0
                        pareto.append([converged,Umax,tf,target_z,0,0])
            return pareto

        
    filepath = f"/home/lha43/Desktop/Lucas/pareto_points/pareto_all.pkl"

    with open(filepath,"wb") as f:
        pickle.dump(pareto,f)

    savemat('/home/lha43/Desktop/Lucas/pareto_points/pareto_all.mat', {'pareto_all': pareto})

    return pareto

def reintegration(phase2,tf,Traj,Umax):
    # REINTEGRATION STEP (VERY IMPORTANT!!!)
    try:
        Tab = phase2.returnTrajTable()

        integTab = ode.integrator(0.00001,Tab)
        integTab.setAbsTol(1.0e-16)
        integTab.setRelTol(1.0e-14)

        TrajI   = np.array(integTab.integrate_dense(Traj[0],tf)) 

        DU = 384400000.0000000
        TU = 2.360584684800000E+06/(2*np.pi)

        # Below is one method to calculate dV
        Thrust_Mag = (np.array([[np.sqrt(TrajI[i,7]**2 + TrajI[i,8]**2 + TrajI[i,9]**2)] for i in range(max(np.shape(TrajI)))])) # in DU/TU**2
        # dV = sum(np.array([[(Thrust_Mag[i]+Thrust_Mag[i-1])/2*(TrajI[i,6]-TrajI[i-1,6])] for i in range(1,max(np.shape(TrajI)))])) # in DU/TU
        # dV = dV*DU/TU
        # Here, I use Simpson's rule to approximate the integral over time (agrees to exact to <0.1% on all cases thus far)
        dV = DU/TU*scipy.integrate.simpson(Thrust_Mag, x=np.reshape(TrajI[:,6],(len(Thrust_Mag),1)), axis=0)[-1]

        if np.linalg.norm(TrajI[-1][:6] - TrajI[0][:6]) < MeshTol*10 and max(Thrust_Mag) < Umax:
            f = 1
        else:
            f = 0

    except:
        dV = 0.0
        f = 0

    return dV, f

def run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z):
    # try:
    phase = ode.phase(optType, IG, numKnots)
    # except:
        # plot the cone LUCAS
        # plot_cone() 
        # plt.show()
    #Fix first state and time
    phase.addBoundaryValue("First", range(0,3), BoundaryFirst[0:3]) #Q: Does this have constraints on the dimensionality? A: 1d array if this doesnt work
    #Fix last state and time
    phase.addBoundaryValue("Last", range(0,3), BoundaryLast[0:3])

    # adding the time boundary conditions
    phase.addBoundaryValue("First", [6], BoundaryFirst[3])
    phase.addBoundaryValue("Last", [6], BoundaryLast[3])


    # enforce perioodicity
    phase.addEqualCon("FirstandLast", FrontBackEqCon(), range(0,7), [], [])

    # Bound control forces
    phase.addLUNormBound("Path",[7,8,9],Umin,Umax) #Q: how to write this line? A: LUNorm - don't let it go to 0

    # Bound z to be 0<z<max_x
    phase.addUpperVarBound("Path",[2],max_target_z) # LUCAS

    # Bound xyz to be within cone
    def coneIneqCon():
        xyz = Args(3)
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]

        r = z/np.tan(max_cone_angle)
        # h = np.sqrt(x*x +y*y)*np.tan(cone_angle)

        return (x*x + y*y)**(1/2) - (z/np.tan(cone_angle))

    # phase.addInequalCon("Path",coneIneqCon(),[0,1,2]) # LUCAS


    # Produce a mass-optimal result by integrating over a norm() applied to the thrust vector
    if E_or_T == 0:
        phase.addIntegralObjective(Args(3).squared_norm(),[7,8,9]) 
    else:
        phase.addIntegralObjective(Args(3).norm(),[7,8,9])


    XtUVars = [2]
    OPVars = []
    SPVars = []


    def inequalCon():
        x3 = Args(1).tolist()
        return x3

 
    # bound the object below the line z = 0 
    PhaseRegion = "Path"
    VarIndex = 3 
    upperBound = 0 
    scale = 1
    # phase.addUpperVarBound(PhaseRegion, VarIndex, upperBound, scale)

    
    # here, we are adding adaptive meshing
    phase.setAdaptiveMesh(True)  #Enable Adaptive mesh for all following solve/optimize calls
    #phase.setMeshErrorEstimator('deboor')     #default
    ## Set Error tolerance on mesh (epsilon)
    phase.setMeshTol(MeshTol)  #default = 1.0e-6 
    ## Make sure to set optimizer EContol to be the same as or smaller than MeshTol
    phase.optimizer.set_EContol(EControl)
    ## Set Max number of mesh iterations:
    #phase.setMaxMeshIters(10)  #default = 10
     
    phase.setThreads(numThreads,numThreads) #Q: what does this do? A:Parallelization
    phase.optimizer.set_PrintLevel(1)
    phase.optimize()

    return phase



if __name__ == "__main__":
 
    #compute_trajectories()
    pareto = pareto_wifi_search()

    # plot the cone LUCAS
    # plot_cone() 
    # plt.show()
