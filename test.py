import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
from rich import print as rprint
import time


vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

# ASSET parameters - GLOBAL
optType = "LGL3"
E_or_T = 1
numKnots = 200
numThreads = 16
MeshTol = 1.0e-8
EControl = 1.0e-10

# setting canonical distance unit
DU = 384400000.0000000

# setting canonical time unit
TU = 2.360584684800000E+06/(2*np.pi)

# mass ratio for earth-moon system 
mu_star =  0.01215059   # Constant for CR3BP

# Choose x and y s.t. it's a pole sitter
target_x = 1 - mu_star
target_y = 0

# minimum control profile set to aid ASSET convergence 
Umin = 1.0e-8

# Legacy contol authority and period values
Umax = 0.0184*100  # DU/TU^2
period  = 2.085034838884136 # TU, 1 full orbital period

####################### LUCAS #######################

max_target_z = 0.406089144456503 # DU, SOI earth-moon 
max_cone_angle = np.pi/4 # rad
real_u_max = 0.00244 # potential reasonable value

#####################################################

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

def find_traj(target_x,target_y,target_z,period,Umax,IG,rand_flag,ode):

    try: 
        BoundaryFirst = list([target_x, target_y, target_z]) + [0]
        BoundaryLast =  list([target_x, target_y, target_z]) + [period] # LUCAS

        phase1, _, _ = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, 0, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z)
        IG = phase1.returnTraj()
        phase2, runtime, last_iter = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z)
        Traj = phase2.returnTraj()

        dV, f, TrajI = reintegration(ode,phase2,period,Traj,Umax)

        if f == 1:
            # save result
            theta = get_cone_angle(TrajI)
            min_alt = get_min_altitude(TrajI)
            converged = 1
            Traj_data = [converged,Umax,period,target_z,TrajI,theta,min_alt,dV,rand_flag,runtime,last_iter]
        elif f == -1:
            converged = 0
            Traj_data = [converged,Umax,period,target_z,0,0,0,0,rand_flag,runtime,last_iter]
        else:
            converged = 0
            Traj_data = [converged,Umax,period,target_z,0,0,0,0,rand_flag,runtime,last_iter]
    except:
        # save result
        converged = 0
        Traj_data = [converged,Umax,period,target_z,0,0,0,0,rand_flag,runtime,last_iter]

    return Traj_data, converged

def reintegration(ode,phase2,period,Traj,Umax):
    # REINTEGRATION STEP (VERY IMPORTANT!!!)
    try:
        Tab = phase2.returnTrajTable()

        integTab = ode.integrator(0.00001,Tab)
        integTab.setAbsTol(1.0e-16)
        integTab.setRelTol(1.0e-14)

        TrajI   = np.array(integTab.integrate_dense(Traj[0],period)) 

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
        f = -1

    return dV, f, TrajI

def run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z):
    # try:
    phase = ode.phase(optType, IG, numKnots)
    # except:
        # plot the cone 
        # plot_cone() 
        # plt.show()
    #Fix first state and time
    phase.addBoundaryValue("First", range(0,3), BoundaryFirst[0:3]) #Q: Does this have constraints on the dimensionality? A: 1d array if this doesnt work
    #Fix last state and time
    phase.addBoundaryValue("Last", range(0,3), BoundaryLast[0:3])

    # adding the time boundary conditions
    phase.addBoundaryValue("First", [6], BoundaryFirst[3])
    phase.addBoundaryValue("Last", [6], BoundaryLast[3])

    # constrain the initial and terminal position and velocities to be the same after one period of the forced-periodic trajectory 
    def FrontBackEqCon():
        X_0, t_0, X_f, t_f = Args(14).tolist([(0,6), (6,1), (7,6), (13,1)])
        eq1 = X_0[3:] - X_f[3:]
        return eq1

    # enforce perioodicity
    phase.addEqualCon("FirstandLast", FrontBackEqCon(), range(0,7), [], [])

    # Bound control forces
    phase.addLUNormBound("Path",[7,8,9],Umin,Umax) #Q: how to write this line? A: LUNorm - don't let it go to 0

    # Bound z to be 0<z<max_x
    # phase.addLUVarBound("Path",[2],0.0001,max_target_z) # LUCAS
    phase.addLowerVarBound("Path",[2],0.0001)

    # Bound xyz to be within cone
    def coneIneqCon():
        xyz = Args(3)
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]

        r = z/np.tan(max_cone_angle)

        return (x*x + y*y)**(1/2) - r

    # phase.addInequalCon("Path",coneIneqCon(),[0,1,2]) 

    # Produce a mass-optimal result by integrating over a norm() applied to the thrust vector
    if E_or_T == 0:
        phase.addIntegralObjective(Args(3).squared_norm(),[7,8,9]) 
    else:
        phase.addIntegralObjective(Args(3).norm(),[7,8,9])

    XtUVars = [2]
    OPVars = []
    SPVars = []
 
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
    phase.setMaxMeshIters(8)  # 5 is too low, try  7 default = 10
     
    phase.setThreads(numThreads,numThreads) #Q: what does this do? A:Parallelization
    phase.optimizer.set_PrintLevel(5) # Verbosity
    phase.optimize()

    runtime = phase.optimizer.LastTotalTime 
    last_iter = phase.optimizer.LastIterNum

    return phase, runtime, last_iter

def get_cone_angle(Traj): 

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

def get_min_altitude(Traj):

    Traj_array = np.array(Traj)
    min_alt = inf

    for i in range(Traj_array.shape[0]):
        x = Traj_array[i,0]
        y = Traj_array[i,1]
        z = Traj_array[i,2]

        alt = np.linalg.norm([x,y,z])
        if alt < min_alt:
            min_alt = alt

    return min_alt


def save_pkl_mat(filename,data):
    filepath = f"/home/lha43/Desktop/Lucas/pareto_points/{filename}.pkl"
    with open(filepath,"wb") as f:
        pickle.dump(data,f)
    savemat(f'/home/lha43/Desktop/Lucas/pareto_points/{filename}.mat', {f'{filename}': data})


if __name__ == "__main__":

    ode = CR3BP_Thrust_Dynamics(mu_star)
    # Reference trajectory computation
    InitStepSize = 0.001 # TU
    RefInteg = ode.integrator("DOPRI87", InitStepSize)
    RefInteg.setAbsTol(1.0e-16)
    RefInteg.setRelTol(1.0e-14)

    ref_state_integrated = np.array(RefInteg.integrate_dense(np.array([1.06315768, 0.000326952322, -0.200259761, 0.000361619362, -0.176727245, -0.000739327422, 0.0, 0.0, 0.0, 0.0]),period))

    # giving an initial guess that is obtained by multiplying the reference state by 3, to enforce convergence to a non-halo orbit
    # many different initial guesses can be given, often resulting in different pole-sitter trajectories
    IG = [[ref_state_integrated[i,0] * 3, ref_state_integrated[i,1] * 3, ref_state_integrated[i,2] * 3, ref_state_integrated[i,3] * 3, ref_state_integrated[i,4] * 3, ref_state_integrated[i,5] * 3, period*i/(max(np.shape(ref_state_integrated))), 1e-6, 1e-6, 1e-6] for i in range(max(np.shape(ref_state_integrated)))]
    
    IG_Umax = 2.5
    IG_period = 2
    IG_target_z = 0.407
    rand_flag = 0

    start_time_wall = time.time()
    start_time_cpu = time.process_time()

    Traj_IG, _ = find_traj(target_x,target_y,IG_target_z,IG_period,IG_Umax,IG,rand_flag,ode)

    end_time_cpu = time.process_time()
    end_time_wall = time.time()

    rprint("[black] yo")
    print(Traj_IG)
