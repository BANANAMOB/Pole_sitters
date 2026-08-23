import numpy as np
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
from math import inf
from rich import print as rprint
import time
from scipy.integrate import solve_ivp


vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

# ASSET parameters - GLOBAL
optType = "LGL3"
E_or_T = 1
numKnots = 200
numThreads = 16
MeshTol = 1.0e-8 # 1.0e-10
EControl = 1.0e-10 # 1.0e-11

# setting canonical distance unit
DU = 384400000.0000000

# setting canonical time unit
TU = 2.360584684800000E+06/(2*np.pi)

# mass ratio for earth-moon system 
mu_star =  0.012150585609624   # Constant for CR3BP

# minimum control profile set to aid ASSET convergence 
Umin = 1.0e-8

# Legacy contol authority and period values
Umax = 0.0184*100  # DU/TU^2
period  = 2.085034838884136 # TU, 1 full orbital period

####################### LUCAS #######################

max_target_z = 0.406089144456503 # DU, SOI earth-moon / sun
max_cone_angle = np.pi/4 # rad
real_u_max = 0.00244 # potential reasonable value
MISSING = object()

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

# North pole sitter - position
def find_traj(target_x,target_y,target_z,period,Umax,IG,rand_flag,ode):

    start_time = time.process_time()

    try: 
        BoundaryFirst = list([target_x, target_y, target_z]) + [0]
        BoundaryLast =  list([target_x, target_y, target_z]) + [period] 

        phase1 = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, 0, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z)
        Traj1 = phase1.returnTraj()
        dV_e, f_e, TrajI_e = reintegration(ode,phase1,period,Traj1,Umax)

        time_cpu_e = time.process_time()-start_time

        # Save result
        if f_e == 1:
            theta = get_cone_angle(TrajI_e)
            min_alt = get_min_altitude(TrajI_e)
            converged_e = 1
            Traj_data_e = [converged_e,Umax,period,target_z,TrajI_e,theta,min_alt,dV_e,time_cpu_e,rand_flag,f_e]
        elif f_e == -1:
            converged_e = 0
            Traj_data_e = [converged_e,Umax,period,target_z,0,0,0,0,time_cpu_e,rand_flag,f_e]
        else:
            converged_e = 0
            Traj_data_e = [converged_e,Umax,period,target_z,0,0,0,0,time_cpu_e,rand_flag,f_e] 

        IG = Traj1
        phase2 = run_optimizer(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z)
        Traj2 = phase2.returnTraj()
        dV_m, f_m, TrajI_m = reintegration(ode,phase2,period,Traj2,Umax)

        time_cpu_m = time.process_time()-start_time

        # Save result
        if f_m == 1:
            theta = get_cone_angle(TrajI_m)
            min_alt = get_min_altitude(TrajI_m)
            converged_m = 1
            Traj_data_m = [converged_m,Umax,period,target_z,TrajI_m,theta,min_alt,dV_m,time_cpu_m,rand_flag,f_m]
        elif f_m == -1:
            converged_m = 0
            Traj_data_m = [converged_m,Umax,period,target_z,0,0,0,0,time_cpu_m,rand_flag,f_m]
        else:
            converged_m = 0
            Traj_data_m = [converged_m,Umax,period,target_z,0,0,0,0,time_cpu_m,rand_flag,f_m]
    
    except:
        # save result
        converged_e = 0
        converged_m = 0
        time_cpu = start_time-time.process_time()
        Traj_data_e = [converged_e,Umax,period,target_z,0,0,0,0,time_cpu,rand_flag,-2]
        Traj_data_m = [converged_m,Umax,period,target_z,0,0,0,0,time_cpu,rand_flag,-2]

    return Traj_data_e, converged_e, Traj_data_m, converged_m

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

    # Bound z to be 
    # phase.addLUVarBound("Path",[2],0.0001,max_target_z) # 0<z<max_x
    # phase.addLowerVarBound("Path",[2],0.0001) # 0<z
    phase.addLowerVarBound("Path",[2],BoundaryFirst[2]*0.99)


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
    phase.setMaxMeshIters(15)  # 5 is too low, try  7 default = 10
     
    phase.setThreads(numThreads,numThreads) #Q: what does this do? A:Parallelization
    phase.optimizer.set_PrintLevel(5) # Verbosity
    phase.optimize()

    return phase 

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

def get_min_altitude(Traj):

    Traj_array = np.array(Traj)
    min_alt = inf

    for i in range(Traj_array.shape[0]):
        x = Traj_array[i,0] - (1-mu_star)
        y = Traj_array[i,1]
        z = Traj_array[i,2] 

        alt = np.linalg.norm([x,y,z])
        if alt < min_alt:
            min_alt = alt

    return min_alt

# South pole sitter - position
def find_traj_s(target_x,target_y,target_z,period,Umax,IG,rand_flag,ode):

    start_time = time.process_time()

    try: 
        BoundaryFirst = list([target_x, target_y, target_z]) + [0]
        BoundaryLast =  list([target_x, target_y, target_z]) + [period] 

        phase1 = run_optimizer_s(ode, IG, BoundaryFirst, BoundaryLast, optType, 0, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z)
        Traj1 = phase1.returnTraj()
        dV_e, f_e, TrajI_e = reintegration(ode,phase1,period,Traj1,Umax)

        time_cpu_e = time.process_time()-start_time

        # Save result
        if f_e == 1:
            theta = get_cone_angle(TrajI_e)
            min_alt = get_min_altitude(TrajI_e)
            converged_e = 1
            Traj_data_e = [converged_e,Umax,period,target_z,TrajI_e,theta,min_alt,dV_e,time_cpu_e,rand_flag,f_e]
        elif f_e == -1:
            converged_e = 0
            Traj_data_e = [converged_e,Umax,period,target_z,0,0,0,0,time_cpu_e,rand_flag,f_e]
        else:
            converged_e = 0
            Traj_data_e = [converged_e,Umax,period,target_z,0,0,0,0,time_cpu_e,rand_flag,f_e] 

        IG = Traj1
        phase2 = run_optimizer_s(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z)
        Traj2 = phase2.returnTraj()
        dV_m, f_m, TrajI_m = reintegration(ode,phase2,period,Traj2,Umax)

        time_cpu_m = time.process_time()-start_time

        # Save result
        if f_m == 1:
            theta = get_cone_angle(TrajI_m)
            min_alt = get_min_altitude(TrajI_m)
            converged_m = 1
            Traj_data_m = [converged_m,Umax,period,target_z,TrajI_m,theta,min_alt,dV_m,time_cpu_m,rand_flag,f_m]
        elif f_m == -1:
            converged_m = 0
            Traj_data_m = [converged_m,Umax,period,target_z,0,0,0,0,time_cpu_m,rand_flag,f_m]
        else:
            converged_m = 0
            Traj_data_m = [converged_m,Umax,period,target_z,0,0,0,0,time_cpu_m,rand_flag,f_m]
    
    except:
        # save result
        converged_e = 0
        converged_m = 0
        time_cpu = start_time-time.process_time()
        Traj_data_e = [converged_e,Umax,period,target_z,0,0,0,0,time_cpu,rand_flag,-2]
        Traj_data_m = [converged_m,Umax,period,target_z,0,0,0,0,time_cpu,rand_flag,-2]

    return Traj_data_e, converged_e, Traj_data_m, converged_m

def run_optimizer_s(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z):
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

    # Bound z to be 
    # phase.addLUVarBound("Path",[2],0.0001,max_target_z) # 0<z<max_x
    # phase.addLowerVarBound("Path",[2],0.0001) # 0<z
    phase.addUpperVarBound("Path",[2],BoundaryFirst[2]*0.99)


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
    phase.setMaxMeshIters(15)  # 5 is too low, try  7 default = 10
     
    phase.setThreads(numThreads,numThreads) #Q: what does this do? A:Parallelization
    phase.optimizer.set_PrintLevel(5) # Verbosity
    phase.optimize()

    return phase 

# North pole sitter - full state
def find_traj_f(state,period,Umax,IG,rand_flag,ode):

    start_time = time.process_time()
    target_x, target_y, target_z, vx, vy, vz = state

    try: 
        BoundaryFirst = list([target_x, target_y, target_z,vx,vy,vz]) + [0]
        BoundaryLast =  list([target_x, target_y, target_z,vx,vy,vz]) + [period] 

        phase1 = run_optimizer_f(ode, IG, BoundaryFirst, BoundaryLast, optType, 0, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z)
        Traj1 = phase1.returnTraj()
        dV_e, f_e, TrajI_e = reintegration(ode,phase1,period,Traj1,Umax)

        time_cpu_e = time.process_time()-start_time

        # Save result
        if f_e == 1:
            theta = get_cone_angle(TrajI_e)
            min_alt = get_min_altitude(TrajI_e)
            converged_e = 1
            Traj_data_e = [converged_e,Umax,period,target_z,TrajI_e,theta,min_alt,dV_e,time_cpu_e,rand_flag,f_e]
        elif f_e == -1:
            converged_e = 0
            Traj_data_e = [converged_e,Umax,period,target_z,0,0,0,0,time_cpu_e,rand_flag,f_e]
        else:
            converged_e = 0
            Traj_data_e = [converged_e,Umax,period,target_z,0,0,0,0,time_cpu_e,rand_flag,f_e] 

        IG = Traj1
        phase2 = run_optimizer_f(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z)
        Traj2 = phase2.returnTraj()
        dV_m, f_m, TrajI_m = reintegration(ode,phase2,period,Traj2,Umax)

        time_cpu_m = time.process_time()-start_time

        # Save result
        if f_m == 1:
            theta = get_cone_angle(TrajI_m)
            min_alt = get_min_altitude(TrajI_m)
            converged_m = 1
            Traj_data_m = [converged_m,Umax,period,target_z,TrajI_m,theta,min_alt,dV_m,time_cpu_m,rand_flag,f_m]
        elif f_m == -1:
            converged_m = 0
            Traj_data_m = [converged_m,Umax,period,target_z,0,0,0,0,time_cpu_m,rand_flag,f_m]
        else:
            converged_m = 0
            Traj_data_m = [converged_m,Umax,period,target_z,0,0,0,0,time_cpu_m,rand_flag,f_m]
    
    except:
        # save result
        converged_e = 0
        converged_m = 0
        time_cpu = start_time-time.process_time()
        Traj_data_e = [converged_e,Umax,period,target_z,0,0,0,0,time_cpu,rand_flag,-2]
        Traj_data_m = [converged_m,Umax,period,target_z,0,0,0,0,time_cpu,rand_flag,-2]

    return Traj_data_e, converged_e, Traj_data_m, converged_m

def run_optimizer_f(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z):
    # try:
    phase = ode.phase(optType, IG, numKnots)
    # except:
        # plot the cone 
        # plot_cone() 
        # plt.show()
    #Fix first state and time
    phase.addBoundaryValue("First", range(0,3), BoundaryFirst[0:3]) #Q: Does this have constraints on the dimensionality? A: 1d array if this doesnt work
    phase.addBoundaryValue("First", range(3,6), BoundaryFirst[3:6])
    #Fix last state and time
    phase.addBoundaryValue("Last", range(0,3), BoundaryLast[0:3])
    phase.addBoundaryValue("Last", range(3,6), BoundaryLast[3:6])

    # adding the time boundary conditions
    phase.addBoundaryValue("First", [6], BoundaryFirst[6])
    phase.addBoundaryValue("Last", [6], BoundaryLast[6])

    # constrain the initial and terminal position and velocities to be the same after one period of the forced-periodic trajectory 
    def FrontBackEqCon():
        X_0, t_0, X_f, t_f = Args(14).tolist([(0,6), (6,1), (7,6), (13,1)])
        eq1 = X_0[3:] - X_f[3:]
        return eq1

    # enforce perioodicity
    phase.addEqualCon("FirstandLast", FrontBackEqCon(), range(0,7), [], [])

    # Bound control forces
    phase.addLUNormBound("Path",[7,8,9],Umin,Umax) #Q: how to write this line? A: LUNorm - don't let it go to 0

    # Bound z to be 
    # phase.addLUVarBound("Path",[2],0.0001,max_target_z) # 0<z<max_x
    # phase.addLowerVarBound("Path",[2],0.0001) # 0<z
    phase.addLowerVarBound("Path",[2],BoundaryFirst[2]*0.99)


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
    phase.setMaxMeshIters(15)  # 5 is too low, try  7 default = 10
     
    phase.setThreads(numThreads,numThreads) #Q: what does this do? A:Parallelization
    phase.optimizer.set_PrintLevel(5) # Verbosity
    phase.optimize()

    return phase 

# South pole sitter - full state
def find_traj_fs(state,period,Umax,IG,rand_flag,ode):

    start_time = time.process_time()
    target_x, target_y, target_z, vx, vy, vz = state

    try: 
        BoundaryFirst = list([target_x, target_y, target_z,vx,vy,vz]) + [0]
        BoundaryLast =  list([target_x, target_y, target_z,vx,vy,vz]) + [period] 

        phase1 = run_optimizer_fs(ode, IG, BoundaryFirst, BoundaryLast, optType, 0, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z)
        Traj1 = phase1.returnTraj()
        dV_e, f_e, TrajI_e = reintegration(ode,phase1,period,Traj1,Umax)

        time_cpu_e = time.process_time()-start_time

        # Save result
        if f_e == 1:
            theta = get_cone_angle(TrajI_e)
            min_alt = get_min_altitude(TrajI_e)
            converged_e = 1
            Traj_data_e = [converged_e,Umax,period,target_z,TrajI_e,theta,min_alt,dV_e,time_cpu_e,rand_flag,f_e]
        elif f_e == -1:
            converged_e = 0
            Traj_data_e = [converged_e,Umax,period,target_z,0,0,0,0,time_cpu_e,rand_flag,f_e]
        else:
            converged_e = 0
            Traj_data_e = [converged_e,Umax,period,target_z,0,0,0,0,time_cpu_e,rand_flag,f_e] 

        IG = Traj1
        phase2 = run_optimizer_fs(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z)
        Traj2 = phase2.returnTraj()
        dV_m, f_m, TrajI_m = reintegration(ode,phase2,period,Traj2,Umax)

        time_cpu_m = time.process_time()-start_time

        # Save result
        if f_m == 1:
            theta = get_cone_angle(TrajI_m)
            min_alt = get_min_altitude(TrajI_m)
            converged_m = 1
            Traj_data_m = [converged_m,Umax,period,target_z,TrajI_m,theta,min_alt,dV_m,time_cpu_m,rand_flag,f_m]
        elif f_m == -1:
            converged_m = 0
            Traj_data_m = [converged_m,Umax,period,target_z,0,0,0,0,time_cpu_m,rand_flag,f_m]
        else:
            converged_m = 0
            Traj_data_m = [converged_m,Umax,period,target_z,0,0,0,0,time_cpu_m,rand_flag,f_m]
    
    except:
        # save result
        converged_e = 0
        converged_m = 0
        time_cpu = start_time-time.process_time()
        Traj_data_e = [converged_e,Umax,period,target_z,0,0,0,0,time_cpu,rand_flag,-2]
        Traj_data_m = [converged_m,Umax,period,target_z,0,0,0,0,time_cpu,rand_flag,-2]

    return Traj_data_e, converged_e, Traj_data_m, converged_m

def run_optimizer_fs(ode, IG, BoundaryFirst, BoundaryLast, optType, E_or_T, Umax, Umin, numKnots, numThreads, MeshTol, EControl,max_target_z):
    # try:
    phase = ode.phase(optType, IG, numKnots)
    # except:
        # plot the cone 
        # plot_cone() 
        # plt.show()
    #Fix first state and time
    phase.addBoundaryValue("First", range(0,3), BoundaryFirst[0:3]) #Q: Does this have constraints on the dimensionality? A: 1d array if this doesnt work
    phase.addBoundaryValue("First", range(3,6), BoundaryFirst[3:6])
    #Fix last state and time
    phase.addBoundaryValue("Last", range(0,3), BoundaryLast[0:3])
    phase.addBoundaryValue("Last", range(3,6), BoundaryLast[3:6])

    # adding the time boundary conditions
    phase.addBoundaryValue("First", [6], BoundaryFirst[6])
    phase.addBoundaryValue("Last", [6], BoundaryLast[6])

    # constrain the initial and terminal position and velocities to be the same after one period of the forced-periodic trajectory 
    def FrontBackEqCon():
        X_0, t_0, X_f, t_f = Args(14).tolist([(0,6), (6,1), (7,6), (13,1)])
        eq1 = X_0[3:] - X_f[3:]
        return eq1

    # enforce perioodicity
    phase.addEqualCon("FirstandLast", FrontBackEqCon(), range(0,7), [], [])

    # Bound control forces
    phase.addLUNormBound("Path",[7,8,9],Umin,Umax) #Q: how to write this line? A: LUNorm - don't let it go to 0

    # Bound z to be 
    # phase.addLUVarBound("Path",[2],0.0001,max_target_z) # 0<z<max_x
    # phase.addLowerVarBound("Path",[2],0.0001) # 0<z
    phase.addUpperVarBound("Path",[2],BoundaryFirst[2]*0.99)


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
    phase.setMaxMeshIters(15)  # 5 is too low, try  7 default = 10
     
    phase.setThreads(numThreads,numThreads) #Q: what does this do? A:Parallelization
    phase.optimizer.set_PrintLevel(5) # Verbosity
    phase.optimize()

    return phase 

## Utils

# CHANGE YOUR PATH HERE
def save_pkl_mat(filename,data):
    filepath = f"/home/lha43/Desktop/Manifold/DATA/{filename}.pkl"
    with open(filepath,"wb") as f:
        pickle.dump(data,f)
    savemat(f'/home/lha43/Desktop/Manifold/DATA/{filename}.mat', {f'{filename}': data})

def cr3bp_equa(t, state):
    mu = mu_star
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)
   
    ax = 2*vy + x - (1 - mu)*(x + mu)/r1**3 - mu*(x - (1 - mu))/r2**3
    ay = -2*vx + y - (1 - mu)*y/r1**3 - mu*y/r2**3
    az = - (1 - mu)*z/r1**3 - mu*z/r2**3
    return [vx, vy, vz, ax, ay, az]

def prop_orbit_from_IC(state0, t_span,n_span=MISSING):
    t_eval = None if (n_span is MISSING) else np.linspace(t_span[0], t_span[1], n_span)
    sol = solve_ivp(cr3bp_equa, t_span, state0, t_eval=t_eval,method='RK45', rtol=1e-12, atol=1e-12)
    return sol.y, sol.t

def plot_orbit(Traj,ax0,ax1,ax2,ax3,ax4,col): # ,IG):

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

    #### IG for pole sitter
    ode = CR3BP_Thrust_Dynamics(mu_star)

    InitStepSize = 0.001 # TU
    RefInteg = ode.integrator("DOPRI87", InitStepSize)
    RefInteg.setAbsTol(1.0e-16)
    RefInteg.setRelTol(1.0e-14)

    ref_state_integrated = np.array(RefInteg.integrate_dense(np.array([1.06315768, 0.000326952322, -0.200259761, 0.000361619362, -0.176727245, -0.000739327422, 0.0, 0.0, 0.0, 0.0]),period))

    # giving an initial guess that is obtained by multiplying the reference state by 3, to enforce convergence to a non-halo orbit
    # many different initial guesses can be given, often resulting in different pole-sitter trajectories
    IG = [[ref_state_integrated[i,0] * 3, ref_state_integrated[i,1] * 3, ref_state_integrated[i,2] * 3, ref_state_integrated[i,3] * 3, ref_state_integrated[i,4] * 3, ref_state_integrated[i,5] * 3, period*i/(max(np.shape(ref_state_integrated))), 1e-6, 1e-6, 1e-6] for i in range(max(np.shape(ref_state_integrated)))]
    
    print(np.array(IG).shape)
    input()

    # Compute a better initial guess
    IG_Umax = 2
    IG_period = 2
    target_x = 1 - mu_star
    target_y = 0
    IG_target_z = 0.1
    rand_flag = 0
    _, _, Traj_IG, _ = find_traj(target_x,target_y,IG_target_z,IG_period,IG_Umax,IG,rand_flag,ode)
    IGps = Traj_IG[4]
    
    
    ####################################################

    #### Use a periodic orbit - Found via differential correction - SAME AS IN GENERATION SCRIPT
    IG_corr = [8.24841202e-01, -6.87587042e-02, 1.80883634e-01, 1.38486538e+00]

    #### Propogate orbit
    state0, T = [IG_corr[0],0,IG_corr[1],0,IG_corr[2],0], IG_corr[3]*2
    t_span = [0, T]  # Integrate for n normalized time units
    ref, times = prop_orbit_from_IC(state0, t_span)

    # UNPACK
    filepath = f"DATA/Manifold_transfer_max_z.pkl"
    with open(filepath, 'rb') as f:
        man_trans_max = pickle.load(f)

    filepath = f"DATA/Manifold_transfer_min_z.pkl"
    with open(filepath, 'rb') as f:
        man_trans_min = pickle.load(f)    

    # [prop,times[idx_max_z],idx_max_z] or [prop,times[idx_min_z],idx_min_z]
    prop_max, time_max, idx_max_z = man_trans_max
    prop_min, time_min, idx_min_z = man_trans_min

    top_state = prop_max.T[idx_max_z]
    low_state = prop_min.T[idx_min_z]

    ####################################################

    #### Pole sitter from manifold
    
    des_period = 2
    des_umax = 2

    # initial positions
    #_, _, Traj_top, _ = find_traj(top_state[0],top_state[1],top_state[2],des_period,des_umax,IGps,0,ode)
    _, _, Traj_low, _ = find_traj_s(low_state[0],low_state[1],low_state[2],des_period,des_umax,IGps,0,ode)
    # full state
    #_, _, Traj_top_full, _ = find_traj_f(top_state,des_period,des_umax,IGps,0,ode)
    _, _, Traj_low_full, _ = find_traj_fs(low_state,des_period,des_umax,IGps,0,ode)

    dv_low, dv_low_full = Traj_low[7] *2*np.pi/(des_period*1000), Traj_low_full[7] *2*np.pi/(des_period*1000)

    Traj_low, Traj_low_full = Traj_low[4], Traj_low_full[4]
    #Traj_top, Traj_top_full = Traj_top[4], Traj_top_full[4]

    # print(Fore.GREEN + f"{np.shape(np.array(Traj_top))}")
    # print(np.shape(np.array(Traj_top_full)))
    # print(np.shape(np.array(Traj_low)))
    # print(np.shape(np.array(Traj_low_full)))
    # input()
    
    ####################################################

    style_path = "/home/lha43/Desktop/Lucas/Generated_figures/paper.mplstyle"
    plt.style.use(style_path) 

    #### Plot
    fig = plt.figure()
    ax0 = plt.subplot(421)
    ax1 = plt.subplot(423)
    ax2 = plt.subplot(425)
    ax3 = plt.subplot(427)
    ax4 = plt.subplot(122, projection='3d')

    col = 'teal'
    #plot_orbit(Traj_top,ax0,ax1,ax2,ax3,ax4,col)

    col = 'blue'
    #plot_orbit(Traj_top_full,ax0,ax1,ax2,ax3,ax4,col)

    col = '#956081'
    plot_orbit(Traj_low,ax0,ax1,ax2,ax3,ax4,col)

    col = '#FFC107'
    plot_orbit(Traj_low_full,ax0,ax1,ax2,ax3,ax4,col)

    # Plot bodies
    # ax.scatter(-mu, 0, 0, color='blue', label='Earth')
    

    #ax4.plot(prop_max[0], prop_max[1], prop_max[2], color='lightblue', label='Manifold for highest point')
    ax4.plot(prop_min[0], prop_min[1], prop_min[2], color='#1E88E5',linewidth=2, label='Manifold')

    ax4.plot(ref[0], ref[1], ref[2], linewidth=2, color='black', label='Natural orbit')

    #ax4.scatter(prop_max[0][idx_max_z], prop_max[1][idx_max_z], prop_max[2][idx_max_z], color='Blue', label='Highest point')
    ax4.scatter(prop_min[0][idx_min_z], prop_min[1][idx_min_z], prop_min[2][idx_min_z], color='red', label='Anchor')

    ax4.scatter(1 - mu_star, 0, 0, color='gray', label='Moon')

    print(Fore.GREEN + f'Data')
    print(dv_low, dv_low_full)
    print(np.linalg.norm(np.array(Traj_low)[0][3:6]-np.array(Traj_low_full)[0][3:6]))

    ax4.set_xlim(0.5,1.5)
    plt.legend()
    plt.show()

    ####################################################

