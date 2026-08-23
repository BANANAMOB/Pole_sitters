import numpy as np
import numpy.linalg as lin
import asset_asrl as ast
import matplotlib.pyplot as plt
import scipy
import os
import pickle
from scipy.io import savemat
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
colorama_init()
import random
from math import inf, sin, cos, sqrt
from rich import print as rprint
import time
from scipy.integrate import solve_ivp
import h5py
# All in m, kg, s, rad

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

# ASSET parameters - GLOBAL
optType = "LGL3"
E_or_T = 1
numKnots = 200
numThreads = 16
MeshTol = 1.0e-6 # 1.0e-10
EControl = 1.0e-8 # 1.0e-11

# setting canonical distance unit
#DU = 384400000.0000000

# setting canonical time unit
#TU = 2.360584684800000E+06/(2*np.pi)

# earth 
mu = 398600.49 # km # 398600.49e9 in m

# minimum control profile set to aid ASSET convergence 
Umin = 1.0e-8


# MEE Gauss variational equations
class MEE_gve(oc.ODEBase):

    def __init__(self,mu,Isp):
        Xvars = 7
        Uvars = 3
        ####################################################
        XtU = oc.ODEArguments(Xvars,Uvars)

        a, f, g, h, k, L, m = XtU.XVec().tolist()
        f_r, f_t, f_h = XtU.UVec().tolist()

        p = a * (1 - f**2 - g**2)
        q = 1 + f*vf.cos(L) + g*vf.sin(L)

        a_dot = 2 * vf.sqrt(a**3 / (mu*(1-f**2-g**2))) * \
                ( (f*vf.sin(L) - g*vf.cos(L))*f_r + q*f_t )
        f_dot = 1/q * vf.sqrt(p/mu) * \
                ( (q*vf.sin(L))*f_r + ((q+1)*vf.cos(L)+f)*f_t + -g*(h*vf.sin(L)-k*vf.cos(L))*f_h )
        g_dot = 1/q * vf.sqrt(p/mu) * \
                ( (-q*vf.cos(L))*f_r + ((q+1)*vf.sin(L)+g)*f_t + f*(h*vf.sin(L)-k*vf.cos(L))*f_h )
        h_dot = 1/q * vf.sqrt(p/mu) * \
                ( 0.5*(1+h**2+k**2)*vf.cos(L)*f_h )
        k_dot = 1/q * vf.sqrt(p/mu) * \
                ( 0.5*(1+h**2+k**2)*vf.sin(L)*f_h ) 
        L_dot = 1/q * vf.sqrt(p/mu) * \
                ( (h*vf.sin(L)-k*vf.cos(L))*f_h ) + vf.sqrt(mu*p) * (q/p)**2
        
        g = 9.81
        m_dot = -(m*vf.sqrt(f_r**2 + f_t**2 + f_h**2)) / (g*Isp)
    
        ode = vf.stack([a_dot, f_dot, g_dot, h_dot, k_dot, L_dot, m_dot])
        ####################################################
        super().__init__(ode,Xvars,Uvars)

def find_traj(mee_0, mee_T, time_0, time_max, Umax, IG, ode):

    a_0, f_0, g_0, h_0, k_0, L_0 = mee_0
    a_T, f_T, g_T, h_T, k_T, _ = mee_T

    start_time = time.process_time()

    try: 
        BoundaryFirst = list([a_0, f_0, g_0, h_0, k_0, L_0]) + [time_0]
        BoundaryLast =  list([a_T, f_T, g_T, h_T, k_T, 0]) + [time_max]

        phase1 = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, 0, Umax, Umin, numKnots, numThreads, MeshTol, EControl)
        Traj1 = phase1.returnTraj()
        dV_e, f_e, TrajI_e = reintegration(ode,phase1,Traj1,Umax,mee_T)

        time_cpu_e = time.process_time()-start_time

        # Save result
        if f_e == 1:
            converged_e = 1
            Traj_data_e = [converged_e,TrajI_e,mee_0,mee_T,dV_e,time_cpu_e,f_e]
        elif f_e == -1:
            converged_e = 0
            Traj_data_e = [converged_e,TrajI_e,mee_0,mee_T,0,time_cpu_e,f_e]
        else:
            converged_e = 0
            Traj_data_e = [converged_e,TrajI_e,mee_0,mee_T,0,time_cpu_e,f_e]

        IG = Traj1
        phase2 = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl)
        Traj2 = phase2.returnTraj()
        dV_m, f_m, TrajI_m = reintegration(ode,phase2,Traj2,Umax,mee_T)

        time_cpu_m = time.process_time()-start_time

        # Save result
        if f_m == 1:
            converged_m = 1
            Traj_data_m = [converged_m,TrajI_m,mee_0,mee_T,dV_m,time_cpu_m,f_m]
        elif f_m == -1:
            converged_m = 0
            Traj_data_m = [converged_m,TrajI_m,mee_0,mee_T,0,time_cpu_m,f_m]
        else:
            converged_m = 0
            Traj_data_m = [converged_m,TrajI_m,mee_0,mee_T,0,time_cpu_m,f_m]
    
    except Exception as e:
        # save result
        print(Fore.RED + "Error")
        print(e)
        converged_e = 0
        converged_m = 0
        time_cpu = start_time-time.process_time()
        Traj_data_e = [converged_e,0,mee_0,mee_T,0,time_cpu,-2]
        Traj_data_m = [converged_m,0,mee_0,mee_T,0,time_cpu,-2]

    return Traj_data_e, converged_e, Traj_data_m, converged_m

def reintegration(ode,phase2,Traj,Umax,mee_T):
    # REINTEGRATION STEP (VERY IMPORTANT!!!)
    try:
        Tab = phase2.returnTrajTable()

        integTab = ode.integrator(0.00001,Tab)
        integTab.setAbsTol(1.0e-16)
        integTab.setRelTol(1.0e-14)

        traj_time = Traj[-1][6]

        TrajI = np.array(integTab.integrate_dense(Traj[0],traj_time)) 

        # DU = 384400000.0000000
        # TU = 2.360584684800000E+06/(2*np.pi)

        # Below is one method to calculate dV
        Thrust_Mag = (np.array([[np.sqrt(TrajI[i,7]**2 + TrajI[i,8]**2 + TrajI[i,9]**2)] for i in range(max(np.shape(TrajI)))])) # in DU/TU**2
        # dV = sum(np.array([[(Thrust_Mag[i]+Thrust_Mag[i-1])/2*(TrajI[i,6]-TrajI[i-1,6])] for i in range(1,max(np.shape(TrajI)))])) # in DU/TU
        # dV = dV*DU/TU
        # Here, I use Simpson's rule to approximate the integral over time (agrees to exact to <0.1% on all cases thus far)
        dV = scipy.integrate.simpson(Thrust_Mag, x=np.reshape(TrajI[:,6],(len(Thrust_Mag),1)), axis=0)[-1] # DU/TU *

        if np.linalg.norm(TrajI[-1][:5] - mee_T) < MeshTol*10 and max(Thrust_Mag) < Umax:
            f = 1
        else:
            f = 0

    except Exception as e:
        # save result
        print(Fore.RED + "Error")
        print(e)
        dV = 0.0
        f = -1

    return dV, f, TrajI

def run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl):
    # try:
    phase = ode.phase(optType, IG, numKnots)
    # except:
        # plot the cone 
        # plot_cone() 
        # plt.show()
    #Fix first state and time
    phase.addBoundaryValue("First", range(0,6), BoundaryFirst[0:6]) #Q: Does this have constraints on the dimensionality? A: 1d array if this doesnt work
    #Fix last state and time
    phase.addBoundaryValue("Last", range(0,5), BoundaryLast[0:5])

    # adding the time boundary conditions
    phase.addBoundaryValue("First", [6], BoundaryFirst[6])
    #phase.addBoundaryValue("Last", [6], BoundaryLast[3])
    #phase.addLUVarBound("Last",[6], BoundaryFirst[6].item()+1, BoundaryLast[6].item())
    #phase.addBoundaryValue("Last", [6], BoundaryLast[6].item()*10)

    # Singularity account
    #phase.addUpperVarBound("Path", [3], 1e10)
    #phase.addUpperVarBound("Path", [4], 1e10)

    # constrain the initial and terminal position and velocities to be the same after one period of the forced-periodic trajectory 
    def FrontBackEqCon():
        X_0, t_0, X_f, t_f = Args(14).tolist([(0,6), (6,1), (7,6), (13,1)])
        eq1 = X_0[3:] - X_f[3:]
        return eq1

    # enforce perioodicity
    # phase.addEqualCon("FirstandLast", FrontBackEqCon(), range(0,7), [], [])

    # Bound control forces
    phase.addLUNormBound("Path",[7,8,9],Umin,Umax) #Q: how to write this line? A: LUNorm - don't let it go to 0

    # Bound z to be 
    # phase.addLUVarBound("Path",[2],0.0001,max_target_z) # 0<z<max_x
    # phase.addLowerVarBound("Path",[2],0.0001) # 0<z
    # phase.addLowerVarBound("Path",[2],BoundaryFirst[2]*0.99)


    # Bound xyz to be within cone
    #def coneIneqCon():
    #    xyz = Args(3)
    #    x = xyz[0]
    #    y = xyz[1]
    #    z = xyz[2]
    #
    #    r = z/np.tan(max_cone_angle)
    #
    #    return (x*x + y*y)**(1/2) - r

    # phase.addInequalCon("Path",coneIneqCon(),[0,1,2]) 

    # Produce a mass-optimal result by integrating over a norm() applied to the thrust vector
    U = 1e5
    if E_or_T == 0:
        phase.addIntegralObjective(Args(3).squared_norm(), [7,8,9]) 
    else:
        phase.addIntegralObjective(Args(3).norm(), [7,8,9])

    #XtUVars = [2]
    #OPVars = []
    #SPVars = []
 
    # bound the object below the line z = 0 
    #PhaseRegion = "Path"
    #VarIndex = 3 
    #upperBound = 0 
    #scale = 1
    # phase.addUpperVarBound(PhaseRegion, VarIndex, upperBound, scale)

    # here, we are adding adaptive meshing
    phase.setAdaptiveMesh(True)  #Enable Adaptive mesh for all following solve/optimize calls
    #phase.setMeshErrorEstimator('deboor')     #default
    ## Set Error tolerance on mesh (epsilon)
    phase.setMeshTol(MeshTol)  #default = 1.0e-6 
    ## Make sure to set optimizer EContol to be the same as or smaller than MeshTol
    phase.optimizer.set_EContol(EControl)
    ## Set Max number of mesh iterations:
    phase.setMaxMeshIters(15)  # 5 is too low, try  7 default = 10
     
    phase.setThreads(numThreads,numThreads) #Q: what does this do? A:Parallelization
    phase.optimizer.set_PrintLevel(1) # Verbosity
    phase.optimize()

    return phase 

## Utils

# CHANGE YOUR PATH HERE
def save_pkl_mat(filename,data):
    filepath = f"/home/lha43/Desktop/Manifold/DATA/{filename}.pkl"
    with open(filepath,"wb") as f:
        pickle.dump(data,f)
    savemat(f'/home/lha43/Desktop/Manifold/DATA/{filename}.mat', {f'{filename}': data})

def plot_orbit(Traj,col): # ,IG):

    fig = plt.figure()
    ax0 = plt.subplot(421)
    ax1 = plt.subplot(423)
    ax2 = plt.subplot(425)
    ax3 = plt.subplot(427)
    ax4 = plt.subplot(122, projection='3d')

    DU = 384400000.0000000
    TU = 2.360584684800000E+06/(2*np.pi)

    Traj_array = np.array(Traj).T
    # IG_array = np.array(IG)

    # if len(IG_array) == max(np.shape(IG_array)):
    #     IG_array = IG_array.T

    #Thrust_Energy_Mag = DU/TU**2 * np.array([[np.sqrt(IG_array[7,i]**2 + IG_array[8,i]**2 + IG_array[9,i]**2)] for i in range(max(np.shape(IG_array)))])
    Thrust_Mass_Mag = np.array([[np.sqrt(Traj_array[7,i]**2 + Traj_array[8,i]**2 + Traj_array[9,i]**2)] for i in range(max(np.shape(Traj_array)))])

    ax0.plot(Traj_array[6], Traj_array[7], color=col,lw=2)  # plots u_x vs time
    #ax0.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[7], color=[252/255, 186/255, 3/255])
    ax1.plot(Traj_array[6], Traj_array[8], color=col,lw=2)  # plots u_y vs time
    #ax1.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[8], color=[252/255, 186/255, 3/255])
    ax2.plot(Traj_array[6], Traj_array[9], color=col,lw=2)  # plots u_z vs time
    #ax2.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), DU/TU**2 * IG_array[9], color=[252/255, 186/255, 3/255])
    ax3.plot(Traj_array[6], Thrust_Mass_Mag, color=col,lw=2)     # plots u mag vs time
    #ax3.plot(TU/86400 * np.linspace(0,Traj_array[6][-1],max(np.shape(IG_array))), Thrust_Energy_Mag, color=[252/255, 186/255, 3/255])     # plots u mag vs time

    # plot the energy optimal trajectory
    #ax4.plot(IG_array[0], IG_array[1], IG_array[2], color=[252/255, 186/255, 3/255])
    # plot the mass optimal trajectory
    if col == '#956081':
        ax4.plot(Traj_array[0],Traj_array[1],Traj_array[2], linewidth=2,color=col, label="$\mathbf{r}_0$ constraint")
    else:
        ax4.plot(Traj_array[0],Traj_array[1],Traj_array[2], linewidth=2,color=col, label="$\mathbf{x}_0$ constraint")
    # plot the moon
    #ax4.scatter(1-mu_star,0,0, color=[130/255,130/255,130/255], s=20)
    # plot the cone 
    #plot_cone(ax4,cone_ang) 

    ax0.grid(True)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)

    ax0.set_ylabel(r'$u_x$ [DU/TU$^2$]')
    ax1.set_ylabel(r'$u_y$ [DU/TU$^2$]')
    ax2.set_ylabel(r'$u_z$ [DU/TU$^2$]')
    ax3.set_ylabel(r'$u$ [DU/TU$^2$]')
    ax3.set_xlabel(r'$t$ [TU]')

    ax4.set_xlabel(r'$x$ [DU]', labelpad=15)
    ax4.set_ylabel(r'$y$ [DU]', labelpad=15)
    ax4.set_zlabel(r'$z$ [DU]', labelpad=-35)

    fig.set_size_inches(10.5, 5.5, forward=True) 
    fig.set_tight_layout(True) 

    return ax4

if __name__ == "__main__":

    Isp = 3100 # 2000
    ode = MEE_gve(mu,Isp)

    case = "A"

    ### Extract Q-law initial guess
    fileparent = f"/home/lha43/Desktop/Qlaw/IG_DATA"

    filepath = f"{fileparent}/q_law_case_{case}_mee_traj_mee_toPy_eta0.jld2"
    with h5py.File(filepath, 'r') as f: 

        n = len(f["m_traj"])
        
        mee_IG = np.array(f["mee_traj"])
        u_IG = np.array(f["u_traj"])
        m_IG = np.array(f["m_traj"]).reshape(n,1)
        t_IG = np.array(f["time"]).reshape(n,1)

    mee_IG[:,0] = mee_IG[:,0]/1000 # to KM)

    filepath = f"{fileparent}/q_law_case_{case}_mee_traj_rv_toPy_eta0.jld2"
    with h5py.File(filepath, 'r') as f: 
        
        r_IG_full = np.array(f["r_traj"])
        v_IG_full = np.array(f["v_traj"])

    # Assemble IG (remove the last row - it's a duplicate)
    Traj_IG_full = np.block([mee_IG,t_IG,u_IG,m_IG])[:-1,:]

    ###########################################################

    # Set up Figure
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # plot Q-law reference
    #ax.scatter(0, 0, 0, color='blue', label='Earth')

    perc_of_traj = 0.0
    idx = int(n*perc_of_traj)
    Traj_test = np.block([r_IG_full,v_IG_full,t_IG,u_IG,m_IG])[:-1,:]
    Traj_test = Traj_test[idx:,:]
    plot_orbit(Traj_test,'#956081')
    plt.show()
    #ax.plot(r_IG.T[0], r_IG.T[1], r_IG.T[2], color='black',linewidth=0.5, label='Case E')
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')
    #ax.axis("equal")
    #ax.legend()
    #plt.show()

    ###########################################################

    # Piece together an optimized phase
    perc_of_traj = 0.998 # 0.998
    idx = int(n*perc_of_traj)

    mee_0 = mee_IG[idx]
    mee_T = mee_IG[-1] 
    
    time_0 = t_IG[idx]
    time_max = t_IG[-1]

    Umax = 3/1000 # in km # 2 N

    # set all 0 control to Umin
    Traj_IG = Traj_IG_full[idx:,:]
    Unorm = lin.norm(Traj_IG[:,7:10],axis=1)
    Umask = Unorm < Umin
    Traj_IG[Umask, 7:10] = Umin * Traj_IG[Umask, 3:6]

    # f_vals = Traj_IG[:, 1] 
    # g_vals = Traj_IG[:, 2] 
    # print("max f²+g²:", np.max(f_vals**2 + g_vals**2)) 
    # print(np.any(np.isnan(Traj_IG)))
    # print(np.any(np.isinf(Traj_IG)))
    # input()

    r_IG=r_IG_full[idx:,:]

    # ax.plot(r_IG.T[0], r_IG.T[1], r_IG.T[2], color='black',linewidth=0.5, label='Case E')
    # plt.show()

    Traj_data_e, converged_e, Traj_data_m, converged_m = find_traj(mee_0, mee_T, time_0, time_max, Umax, Traj_IG, ode)

    print(Fore.BLACK + 'Results:')
    print(Traj_data_e)
    print(converged_e)
    print(Traj_data_m)
    print(converged_m)

    Traj_sol = Traj_data_m[1]
    plot_orbit(Traj_sol,'#1e81eb')
    plt.show()


    #print(Traj_data_e)
    #print(Traj_data_m)

