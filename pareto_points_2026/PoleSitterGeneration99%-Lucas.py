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
MeshTol = 1.0e-8 # 1.0e-10
EControl = 1.0e-10 # 1.0e-11

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

max_target_z = 0.406089144456503 # DU, SOI earth-moon / sun
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
    phase.setMaxMeshIters(7)  # 5 is too low, try  7 default = 10
     
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

# CHANGE YOUR PATH HERE
def save_pkl_mat(filename,data):
    filepath = f"/home/lha43/Desktop/Lucas/pareto_points_2026/{filename}.pkl"
    with open(filepath,"wb") as f:
        pickle.dump(data,f)
    savemat(f'/home/lha43/Desktop/Lucas/pareto_points_2026/{filename}.mat', {f'{filename}': data})

def pareto_3D_search(s_params_side,opt_params_side,s_params_top,opt_params_top,z_range,n_save,main_run_name,tol,fine_non_conv,IG,ode):
    
    # z is used here is to refer to the z axis 

    for z in z_range:
        run_name = main_run_name+f'_coarse_side_{z}'
        target_z, Umax, period, decision_params = opt_params_side
        period = z
        opt_params_side = [target_z, Umax, period, decision_params]
        pareto, fine_params, pareto_e = pareto_2D_coarse_search(s_params_side, opt_params_side,n_save,run_name,IG,ode)

        run_name = main_run_name+f'_fine_side_{z}'
        _ , front, _ = pareto_2D_fine_search(pareto,fine_params,tol,fine_non_conv,n_save,run_name,ode,pareto_e)

        run_name = main_run_name+f'_coarse_top_{z}'
        target_z, Umax, period, decision_params = opt_params_top
        period = z
        opt_params_top = [target_z, Umax, period, decision_params]
        pareto, fine_params, pareto_e = pareto_2D_coarse_search(s_params_top, opt_params_top,n_save,run_name,IG,ode)

        run_name = main_run_name+f'_fine_top_{z}'
        _ , front, _ = pareto_2D_fine_search(pareto,fine_params,tol,fine_non_conv,n_save,run_name,ode,pareto_e)

def pareto_2D_fine_search(pareto,fine_params,tol,max_non_conv,n_save,run_name,ode,pareto_e):

    front = []
    n_front = 0
    not_conv_num = 0
    rand_flag = 0
    for fine_param in fine_params:
        # unpack params
        target_z, Umax, period, decision_params, IG, interval = fine_param

        x = fine_param[decision_params.index(0)]
        y = fine_param[decision_params.index(1)]
        c = fine_param[decision_params.index(2)]

        bottom = interval[0]
        top = interval[1]
        x = (top - bottom)/2 + bottom
        cur_tol = (tol/100) * y
        converged = 0
        num_iter = 0
        while ((top - bottom) > tol) or (not converged):
            target_z = c if (decision_params[0] == 2) else y if (decision_params[0] == 1) else x 
            Umax = c if (decision_params[1] == 2) else y if (decision_params[1] == 1) else x
            period = c if (decision_params[2] == 2) else y if (decision_params[2] == 1) else x
            Traj_e_data, _, Traj_data, converged = find_traj(target_x,target_y,target_z,period,Umax,IG,rand_flag,ode)
            pareto.append(Traj_data) 
            pareto_e.append(Traj_e_data) 

            n_point = len(pareto)
            print(f"""{Fore.BLACK} Point: {n_point}, Umax: {Umax}, Period: {period}, z: {target_z} \n
            {f'{Fore.GREEN} YAY! Converged' if converged else f'{Fore.RED} Oof, not converged'} \n""")

            # Save data for every n_save points
            if n_point % n_save == 0:
                filename = f'pareto_+_fine_{run_name}_{n_point}'
                save_pkl_mat(filename,pareto)

            if converged:
                top = x
                x = (top - bottom)/2 + bottom
                IG = Traj_data[4]
                not_conv_num = 0
            else:
                not_conv_num = not_conv_num + 1
                if not_conv_num < max_non_conv:
                    x = x-(x-bottom)/100
                else:
                    not_conv_num = 0
                    bottom = x
                    x = (top - bottom)/2 + bottom

            if num_iter > 50:
                break
            num_iter = num_iter + 1

        front.append(Traj_data)

        n_front = len(front)
        filename = f"front_{run_name}_{n_front}"
        save_pkl_mat(filename,front)

    filename = f"pareto_+_fine_{run_name}_all"
    save_pkl_mat(filename,pareto)

    filename = f"pareto_+_fine_{run_name}_e_all"
    save_pkl_mat(filename,pareto_e)

    filename = f"front_{run_name}_all"
    save_pkl_mat(filename,front)  

    return pareto, front, pareto_e

def pareto_2D_coarse_search(s_params, opt_params,n_save,run_name,IG,ode):
    print(f"{Fore.BLACK}Running pole sitter alg, just for you handsome")

    # unpack params
    ls, ds, max_non_conv, max_non_conv_clu, n_rand = s_params
    target_z, Umax, period, decision_params = opt_params

    max_x = opt_params[decision_params.index(0)]
    y_range = opt_params[decision_params.index(1)]
    c = opt_params[decision_params.index(2)]

    first_x_conv_traj = IG

    # Start values 
    pareto = []
    pareto_e = [] 
    fine_params = []
    rand_flag = 0
    for p_y in y_range:
        not_conv_clu = 0
        not_conv_num = 0
        p_x = max_x
        # prev_p_x is the previous converged x
        prev_p_x = p_x
        IG = first_x_conv_traj
        first_x_conv = 1
        while max_non_conv_clu > not_conv_clu:

            # Retain p_x > 0
            i = 1
            while p_x <= 0.0001:
                p_x = prev_p_x - ls/(2.1*i)
                i = i+1

            # Generate trajectory
            target_z = c if (decision_params[0] == 2) else p_y if (decision_params[0] == 1) else p_x 
            Umax = c if (decision_params[1] == 2) else p_y if (decision_params[1] == 1) else p_x
            period = c if (decision_params[2] == 2) else p_y if (decision_params[2] == 1) else p_x
            Traj_e_data, _, Traj_data, converged = find_traj(target_x,target_y,target_z,period,Umax,IG,rand_flag,ode) 
            pareto.append(Traj_data)
            pareto_e.append(Traj_e_data)
            
            n_point = len(pareto)
            print(f"""{Fore.BLACK} Point: {n_point}, Umax: {Umax}, Period: {period}, z: {target_z} \n
            {f'{Fore.GREEN} YAY! Converged' if converged else f'{Fore.RED} Oof, not converged'} \n""")

            # Save data for every n_save points
            if n_point % n_save == 0:
                filename = f"{run_name}_{n_point}"
                save_pkl_mat(filename,pareto)

            # Generate trajectory for random point
            if n_point % (n_rand+1) == 0:
                rand_flag = 1
                target_z_rand = c if (decision_params[0] == 2) else random.uniform(min(y_range),max(y_range)) if (decision_params[0] == 1) else random.uniform(0.0001,max_x)
                Umax_rand = c if (decision_params[1] == 2) else random.uniform(min(y_range),max(y_range)) if (decision_params[1] == 1) else random.uniform(0.0001,max_x)
                period_rand = c if (decision_params[2] == 2) else random.uniform(min(y_range),max(y_range)) if (decision_params[2] == 1) else random.uniform(0.0001,max_x)
                Traj_e_random, _, Traj_random, _ = find_traj(target_x,target_y,target_z_rand,period_rand,Umax_rand,IG,rand_flag,ode) 
                pareto.append(Traj_random)
                pareto_e.append(Traj_e_random)
                rand_flag = 0

            # Algorithm
            if converged: # step 1 ls to the left
                not_conv_num = 0
                not_conv_clu = 0
                prev_p_x = p_x
                p_x = p_x - ls
                IG = Traj_data[4]
                if first_x_conv:
                    first_x_conv_traj = Traj_data[4]
                    first_x_conv=0
            else: 
                not_conv_num = not_conv_num + 1
                if max_non_conv > not_conv_num: # take ds step to the left
                    p_x = p_x - ds
                else: # take a half step back
                    not_conv_num = 0
                    not_conv_clu = not_conv_clu + 1
                    if max_non_conv_clu > not_conv_clu:
                        p_x = prev_p_x - ls/(2*not_conv_clu)
                    else:
                        if first_x_conv == 0:
                            fine_param = [target_z, Umax, period, decision_params, IG, [p_x+(ds*(max_non_conv-1)),prev_p_x]]
                            fine_params.append(fine_param)
                            
                            # save fine search range
                            filename = f"fine_params_{run_name}_{n_point}"
                            save_pkl_mat(filename,fine_params)

    # Save all pareto
    filename = f"{run_name}_all"
    save_pkl_mat(filename,pareto)

    filename = f"{run_name}_e_all"
    save_pkl_mat(filename,pareto_e)

    # save fine search range
    filename = f"fine_params_{run_name}_all"
    save_pkl_mat(filename,fine_params)

    return pareto, fine_params, pareto_e

if __name__ == "__main__":
    
    ode = CR3BP_Thrust_Dynamics(mu_star)

    # Start: INITIAL GUESS #
    InitStepSize = 0.001 # TU
    RefInteg = ode.integrator("DOPRI87", InitStepSize)
    RefInteg.setAbsTol(1.0e-16)
    RefInteg.setRelTol(1.0e-14)

    ref_state_integrated = np.array(RefInteg.integrate_dense(np.array([1.06315768, 0.000326952322, -0.200259761, 0.000361619362, -0.176727245, -0.000739327422, 0.0, 0.0, 0.0, 0.0]),period))

    # giving an initial guess that is obtained by multiplying the reference state by 3, to enforce convergence to a non-halo orbit
    # many different initial guesses can be given, often resulting in different pole-sitter trajectories
    IG = [[ref_state_integrated[i,0] * 3, ref_state_integrated[i,1] * 3, ref_state_integrated[i,2] * 3, ref_state_integrated[i,3] * 3, ref_state_integrated[i,4] * 3, ref_state_integrated[i,5] * 3, period*i/(max(np.shape(ref_state_integrated))), 1e-6, 1e-6, 1e-6] for i in range(max(np.shape(ref_state_integrated)))]
    
    # Compute a better initial guess
    IG_Umax = 2.5
    IG_period = 2
    IG_target_z = 0.407
    rand_flag = 0
    _, _, Traj_IG, _ = find_traj(target_x,target_y,IG_target_z,IG_period,IG_Umax,IG,rand_flag,ode)
    IG = Traj_IG[4]
    # End: INITIAL GUESS #


    # Search params side
    ls = 0.02 # small step size multiplier
    ds = 0.0005 # large step size multiplier 
    max_non_conv = 3 # maximum number of non convergences
    max_non_conv_clu = 2 # maximum number of non convergence clusters
    n_rand = 5 # Generate random traj within limit every n_rand points
    s_params_side = [ls,ds,max_non_conv,max_non_conv_clu, n_rand]

    # Opt params side
    target_z = 0.407 # altitude at the start of the orbit 
    Umax = np.linspace(0.05,2.5,20) # Maximum control authority
    period = 1 # 2.085034838884136 # Period
    decision_params = [0,1,2] # indices correspond to target_z, Umax, period respectively
                              # number within correspondes to the following:
                              # [x_axis (max scalar), y-axis (range), hold constant (scalar)]
                              # [1,0,2] means Umax is x-axis, target_z is iterating through y-axis, and period is constant
    opt_params_side = [target_z, Umax, period, decision_params] # preserve order

    # Search params top
    ls = 0.2 #0.005 # small step size multiplier 
    ds = 0.005 #0.0005 # large step size multiplier 
    max_non_conv = 3 # 3 # maximum number of non convergences
    max_non_conv_clu = 2 # maximum number of non convergence clusters
    n_rand = 5 # Generate random traj within limit every n_rand points
    s_params_top = [ls,ds,max_non_conv,max_non_conv_clu, n_rand]

    # Opt params top
    target_z = np.linspace(0.407,0.005,25)  # altitude at the start of the orbit 
    Umax = 2.5 # Maximum control authority
    period = 1 # 2.085034838884136 # Period
    decision_params = [1,0,2] # indices correspond to target_z, Umax, period respectively
                              # number within correspondes to the following:
                              # [x_axis (max scalar), y-axis (range), hold constant (scalar)]
                              # [1,0,2] means Umax is x-axis, target_z is iterating through y-axis, and period is constant
    opt_params_top = [target_z, Umax, period, decision_params] # preserve order

    n_save = 100 # Save data for every n_save points
    run_name = 'pareto_3D'
    tol = 1 # fine error percetage 
    fine_non_conv = 2 # fine consecutive non convergences

    z_range = [0.05,2,4,6] # Range of z-axis values considered - currently mapped to period

    pareto_3D_search(s_params_side,opt_params_side,s_params_top,opt_params_top,z_range,n_save,run_name,tol,fine_non_conv,IG,ode)
