import numpy as np
import numpy.linalg as lin
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pickle

MISSING = object()

# CR3BP parameters for Earth-Moon system
mu = 0.012150585609624  #  (Moon)/ (Earth + Moon)

def cr3bp_equa(t, state):
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)
    
    ax = 2*vy + x - (1 - mu)*(x + mu)/r1**3 - mu*(x - (1 - mu))/r2**3
    ay = -2*vx + y - (1 - mu)*y/r1**3 - mu*y/r2**3
    az = - (1 - mu)*z/r1**3 - mu*z/r2**3
    return [vx, vy, vz, ax, ay, az]

def jac_cr3bp_equa(state):

    x, y, z, _, _, _ = state

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)

    dxddot_dx = 1 - (1-mu)/r1**3 - mu/r2**3 + 3*(1-mu)*(x+mu)**2/r1**5 + 3*mu*(x-(1-mu))**2/r2**5 
    dyddot_dx = 3*(1 - mu)*y*(x+mu)/r1**5 + 3*mu*y*(x-(1-mu))/r2**5
    dzddot_dx = 3*(1 - mu)*z*(x+mu)/r1**5 + 3*mu*z*(x-(1-mu))/r2**5
    dxddot_dy = dyddot_dx
    dyddot_dy = 1 - (1-mu)/r1**3 - mu/r2**3 + 3*(1-mu)*y**2/r1**5 + 3*mu*y**2/r2**5 
    dzddot_dy = 3*(1 - mu)*z*y/r1**5 + 3*mu*z*y/r2**5
    dxddot_dz = dzddot_dx
    dyddot_dz = dzddot_dy
    dzddot_dz = - (1 - mu)/r1**3 - mu/r2**3 + 3*(1-mu)*z**2/r1**5 + 3*mu*z**2/r2**5 

    A = np.array([[dxddot_dx,dxddot_dy,dxddot_dz],
                  [dyddot_dx,dyddot_dy,dyddot_dz],
                  [dzddot_dx,dzddot_dy,dzddot_dz]])
    B = np.array([[0,2,0],[-2,0,0],[0,0,0]])
    zero3x3, I3x3 = np.zeros((3,3)), np.identity(3)

    return np.block([[zero3x3, I3x3], [A, B]])

def prop_orbit_from_IC(state0, t_span,n_span=MISSING):
    t_eval = None if (n_span is MISSING) else np.linspace(t_span[0], t_span[1], n_span)
    sol = solve_ivp(cr3bp_equa, t_span, state0, t_eval=t_eval,method='RK45', rtol=1e-12, atol=1e-12)
    return sol.y, sol.t

def STM(t,M_param):
    state = M_param[:6]
    M = M_param[6:].reshape(6,6)

    # Jacobian @ time t
    state_dot = cr3bp_equa(t,state)
    J = jac_cr3bp_equa(state)
    M_dot = J @ M

    return np.concatenate([state_dot, M_dot.flatten()])

def periodic_from_IG(IG,tol=1e-8,max_iter=1000):
    # propogate guess
    i, err = 0, 10
    while err > tol and i < max_iter:
        state0 = [IG[0],0,IG[1],0,IG[2],0,IG[3]]
        M0 = np.eye(6).flatten()
        init_M = np.concatenate([state0[:6], M0])
        # Propogate orbit
        sol = solve_ivp(STM, [0, state0[6]], init_M,t_eval=None,method='RK45',rtol=1e-12,atol=1e-12)

        # Get final state and deviation
        state_f = sol.y.T[-1][:6]
        state_f_IG = np.array([state_f[1],state_f[3],state_f[5]])
        err = lin.norm(state_f_IG)

        # Find correction - Newton's method
        PHI = sol.y.T[-1][6:].reshape(6,6)
        PHI_cor = np.array([[PHI[1, 0], PHI[1, 2], PHI[1, 4]],
                            [PHI[3, 0], PHI[3, 2], PHI[3, 4]],
                            [PHI[5, 0], PHI[5, 2], PHI[5, 4]]])
        
        state_f_dot = cr3bp_equa(0,state_f)
        d_final = np.zeros([3, 4])
        d_final[:, 0:3], d_final[:,3] = PHI_cor, np.array([state_f_dot[1],state_f_dot[3],state_f_dot[5]]).T

        d_final_neg1 = d_final.T @ lin.inv(d_final @ d_final.T) 
        IG = IG - d_final_neg1 @ state_f_IG.T

        #if i % 10 == 0:
        #    print(i)
        #    print(f"Error = {err}")

        i = i+1

    return IG

#### Forced
def cr3bp_equa_f(t, state):
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)
    
    ax = 2*vy + x - (1 - mu)*(x + mu)/r1**3 - mu*(x - (1 - mu))/r2**3 + acc[0]
    ay = -2*vx + y - (1 - mu)*y/r1**3 - mu*y/r2**3 + acc[1]
    az = - (1 - mu)*z/r1**3 - mu*z/r2**3 + acc[2] # KEY TERM
    return [vx, vy, vz, ax, ay, az]

def STM_f(t,M_param):
    state = M_param[:6]
    M = M_param[6:].reshape(6,6)

    # Jacobian @ time t
    state_dot = cr3bp_equa_f(t,state)
    J = jac_cr3bp_equa(state)
    M_dot = J @ M

    return np.concatenate([state_dot, M_dot.flatten()])

def prop_orbit_from_IC_f(state0, t_span,n_span=MISSING):
    t_eval = None if (n_span is MISSING) else np.linspace(t_span[0], t_span[1], n_span)
    sol = solve_ivp(cr3bp_equa_f, t_span, state0, t_eval=t_eval,method='RK45', rtol=1e-12, atol=1e-12)
    return sol.y, sol.t

def periodic_from_IG_f(IG,tol=1e-8,max_iter=1000):
    # propogate guess
    i, err = 0, 10
    while err > tol and i < max_iter:
        state0 = [IG[0],0,IG[1],0,IG[2],0,IG[3]]
        M0 = np.eye(6).flatten()
        init_M = np.concatenate([state0[:6], M0])
        # Propogate orbit
        sol = solve_ivp(STM_f, [0, state0[6]], init_M,t_eval=None,method='RK45',rtol=1e-12,atol=1e-12)

        # Get final state and deviation
        state_f = sol.y.T[-1][:6]
        state_f_IG = np.array([state_f[1],state_f[3],state_f[5]])
        err = lin.norm(state_f_IG)

        # Find correction - Newton's method
        PHI = sol.y.T[-1][6:].reshape(6,6)
        PHI_cor = np.array([[PHI[1, 0], PHI[1, 2], PHI[1, 4]],
                        [PHI[3, 0], PHI[3, 2], PHI[3, 4]],
                        [PHI[5, 0], PHI[5, 2], PHI[5, 4]]])
        
        state_f_dot = cr3bp_equa_f(0,state_f)
        d_final = np.zeros([3, 4])
        d_final[:, 0:3], d_final[:,3] = PHI_cor, np.array([state_f_dot[1],state_f_dot[3],state_f_dot[5]]).T

        d_final_neg1 = d_final.T @ lin.inv(d_final @ d_final.T) 
        IG = IG - d_final_neg1 @ state_f_IG.T

        #if i % 10 == 0:
        #    print(i)
        #    print(f"Error = {err}")

        i = i+1

    return IG

if __name__ == "__main__":
    
    #### Set up Figure 
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot bodies
    # ax.scatter(-mu, 0, 0, color='blue', label='Earth')
    ax.scatter(1 - mu, 0, 0, color='gray', label='Moon')
    
    ################# Step 1: find orbit, propogate manifolds #################

    #### Find a periodic orbit
    # IG in the form of [x, z, vy, T/2]

    #DEFAULT
    #T = 2.7464
    #IG = [0.8234,0.0224,0.1343,T/2]

    T = 1.3743*2
    IG = [1.0118,0.1739,-0.0799, T/2]

    IG_corr = periodic_from_IG(IG)

    #### Propogate orbit
    state0, T = [IG_corr[0],0,IG_corr[1],0,IG_corr[2],0], IG_corr[3]*2 
    t_span = [0, T]  # Integrate for n normalized time units
    ref, times = prop_orbit_from_IC(state0, t_span)

    ax.plot(ref[0], ref[1], ref[2], '--', color="black", label='Reference orbit')

    ################# Step 2: Pole sitter #################

    #Default
    #T = 4
    #IG = [0.8234,0.2,0.1343,T/2]
    #acc = [0,0,0.5]

    T = 4
    IG = [0.8234,0.2,0.1343,T/2]
    acc = [0,0,0.35]

    IG_corr = periodic_from_IG_f(IG)
    
    #### Propogate orbit
    state0, T = [IG_corr[0],0,IG_corr[1],0,IG_corr[2],0], IG_corr[3]*2 
    t_span = [0, T]  # Integrate for n normalized time units
    ref_pole, times = prop_orbit_from_IC_f(state0, t_span)
    
    ax.plot(ref_pole[0], ref_pole[1], ref_pole[2], color="black", label='Pole sitting orbit')

    #################

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis("equal")
    ax.legend()
    #plt.show()

    ################# Step 3: Pole sitter, family search #################

    #T = 4
    #IG = [0.8234,0.2,0.1343,T/2]
    #acc = [0,0,0.35]
    T0, Tf = 0.05, 6
    T_range = np.linspace(T0, Tf, 60) # 60
    z0, zf = 0.1, 0.5 # 2.5
    z_range = np.linspace(z0, zf, 60) # 60
    acc_z0, acc_zf = 0.0, 2.5
    acc_z_range = np.linspace(acc_z0, acc_zf, 60) # 60

    ctr, ctr_succ = 0, 0 

    orbits_all, orbits_xlim = [], []

    num = len(T_range)*len(z_range)*len(acc_z_range)
    print(num)
    for T in T_range:
        for z in z_range:
            for acc_z in acc_z_range:
                ctr = ctr + 1
                if ctr % 100 == 0:
                    print(f"Iteration {ctr} / {num}")

                try:
                    IG = [0.8234, z, 0.1343,T/2]
                    acc = [0,0,acc_z]
                    IG_corr = periodic_from_IG_f(IG)
                    state0, T = [IG_corr[0],0,IG_corr[1],0,IG_corr[2],0], IG_corr[3]*2 
                    t_span = [0, T]  # Integrate for n normalized time units
                    ref_pole, times = prop_orbit_from_IC_f(state0, t_span)
                    orbits_all.append([ref_pole,state0,T,acc_z])
                    if all(x > 0.1 and x < 5 for x in ref_pole[0]) and T > 0.01:

                        ctr_succ = ctr_succ + 1

                        #fig = plt.figure(figsize=(8,6))
                        #ax = fig.add_subplot(111, projection='3d')
                        ax.plot(ref_pole[0], ref_pole[1], ref_pole[2], alpha=0.8 ,marker='o',markersize=2)# label=f"{ctr}")#, alpha=0.8) ,marker='o',
                        #orbits_xlim.append([ref_pole,state0,T,acc_z])

                        #ax.set_title(f"T={T:.3f}, z={state0[2]:.3f}, acc_z={acc_z:.3f}")
                        #ax.set_xlabel('X')
                        #ax.set_ylabel('Y')
                        #ax.set_zlabel('Z')
                        #ax.axis("equal")

                except Exception as e:
                    print(f"Failed for T={T}, z={z}, acc_z={acc_z}") # pass
    
    print(f"Total successful plots: {ctr_succ}")

    ## Save to pickle
    filepath_all = f"Unoptimized_all_T={T0}-{Tf}_z={z0}-{zf}_acc_z={acc_z0}-{acc_zf}.pkl"
    with open(filepath_all, 'wb') as f:
        pickle.dump(orbits_all, f)

    #filepath_xlim = f"Unoptimized_xlim_T={T0}-{Tf}_z={z0}-{zf}_acc_z={acc_z0}-{acc_zf}.pkl"
    #with open(filepath_xlim, 'wb') as f:
    #    pickle.dump(orbits_xlim, f)
    #################

    ax.set_title(f"Total successful plots: {ctr_succ} || T={T0}-{Tf}, z={z0}-{zf}, acc_z={acc_z0}-{acc_zf}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


