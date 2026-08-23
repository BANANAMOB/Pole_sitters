import numpy as np
import numpy.linalg as lin
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

MISSING = object()

# CR3BP parameters for Earth-Moon system
mu = 0.012150585609624  #  (Moon)/ (Earth + Moon)
SOI = 0.406089144456503 #  ( (Moon+earth)/ SUN )^(2/5) *r_earthmoonRELsun

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

def fsolve_eq(IG, tang, IG_guess):
    state0, T = [IG[0],0,IG[1],0,IG[2],0], IG[3]
    t_span = [0, T]  # Integrate for n normalized time units
    prop, times = prop_orbit_from_IC(state0, t_span)
    state_f_IG = np.array([prop[-1][1],prop[-1][3],prop[-1][5]])

    tang_new = tang.T @ (IG-IG_guess)
    sys = np.append(state_f_IG,tang_new)

    return sys

#### Utils

def save_pkl(filename,data): 
    filepath = f"/home/lha43/Desktop/Manifold/DATA/{filename}.pkl"
    with open(filepath,"wb") as f:
        pickle.dump(data,f)

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

    # 1. IC = [1.0110350588, 0, -0.1731500000, 0, -0.0780141199, 0, 1.3632096570/2] # L2
    # 2. IC = [0.8234, 0, 0.0224, 0, 0.1343, 0, 2.7464/2] # L1
    # 3. IC = [1.0118, 0, 0.1739, 0, -0.0799, 0, 1.3743] # L2 Butterfly
    # 4. IC = [0.9624690577, 0, 0, 0, 0.7184165432, 0, 0.2230147974/2] # DRO 

    # From Garrison:
    # L1 IC = [9.1279120120516621E-1, 0, 2.0716544281164431E-1, 0, 1.5441078879207529E-1, 0, 1.83171291183 / 2]
    # L2 IC = [1.011035058929108, 0, -0.173149999840112, 0, -0.078014276336041, 0, 1.3632096570/2]

    # Generate 10 orbits for each IC case
    N_L1, N_L2 = 40, 60

    T_L1 = 1.83171291183
    IG_L1 = [9.1279120120516621E-1,2.0716544281164431E-1,1.5441078879207529E-1,T_L1/2]

    T_L2 = 1.3632096570
    IG_L2 = [1.011035058929108,-0.173149999840112,-0.078014276336041,T_L2/2]

    Ns = [N_L1,N_L2]
    IGs = [IG_L1,IG_L2]

    # Manifold params
    prop_time = 6
    eps = 4e-1

    # Capture transfers via manifold 
    man_trans_max = [[0 for _ in range(Ns[i])] for i in range(len(Ns))]
    man_trans_min = [[0 for _ in range(Ns[i])] for i in range(len(Ns))]

    ctr = 0

    # For each type of orbit (IG) and correspondng # of nat orbits via continuation
    for k in range(len(IGs)):
        IG, N = IGs[k], Ns[k]

        z, step = np.array([0,0,0,1]), 1e-2

        # Naturally periodic orbits
        for j in range(N):

            print(IG)
            IG_corr = periodic_from_IG(IG)

            #### Propogate orbit
            state0, T = [IG_corr[0],0,IG_corr[1],0,IG_corr[2],0], IG_corr[3]*2
            t_span = [0, T]  # Integrate for n normalized time units
            ref, times = prop_orbit_from_IC(state0, t_span)
            #### Manifold
            # Compute monodromy matrix for state0
            M0 = np.eye(6).flatten()
            init_M = np.concatenate([state0, M0])
            sol = solve_ivp(STM, [0, T], init_M,t_eval=None,method='RK45',rtol=1e-12,atol=1e-12)
            # Extract final monodromy matrix
            M = sol.y.T[-1][6:].reshape(6,6)

            t_span_unstable = [0, prop_time]
            #t_span_stable = [prop_time, 0]

            max_z, min_z = -np.inf, np.inf

            print(len(sol.t))

            for i in range(len(sol.t)):
                ctr = ctr + 1
                if ctr % 10 == 0:
                    print(ctr)

                if i % 100 == 0:
                    # get current monodromy matrix
                    state0 = sol.y.T[i][:6]
                    STM_i = sol.y.T[i][6:].reshape(6,6)
                    M_i = STM_i @ M @ lin.inv(STM_i)
                    eig_val_i, eig_vec_i = lin.eig(M_i)  

                    idx_unstable = np.argmax(eig_val_i)
                    unstable_vec = eig_vec_i[:,idx_unstable]
                    #idx_stable = np.argmin(eig_val_i)
                    #stable_vec = eig_vec_i[:,idx_stable]

                    state0_unstable_i, state0_unstable_neg_i = state0 + eps*unstable_vec.real, state0 - eps*unstable_vec.real
                    #state0_stable_i, state0_stable_neg_i = state0 + eps*stable_vec.real, state0 - eps*stable_vec.real

                    prop0, times0 = prop_orbit_from_IC(state0_unstable_i,t_span_unstable)
                    #prop1, times = prop_orbit_from_IC(state0_stable_i,t_span_stable)
                    prop2, times2 = prop_orbit_from_IC(state0_unstable_neg_i,t_span_unstable)
                    #prop3, times = prop_orbit_from_IC(state0_stable_neg_i,t_span_stable)

                    # check if in SOI and get max_z and min_z
                    offset0 = np.tile([1-mu,0,0],(len(times0),1)).T
                    mask_SOI = lin.norm(prop0[0:3] - offset0, axis=0) <= SOI
                    prop0_in_SOI = prop0[:,mask_SOI]

                    #print(np.shape(prop0_in_SOI))
                    #print(np.shape(prop0))

                    idx_max_z = np.argmax(prop0_in_SOI[2])
                    if prop0_in_SOI[2,idx_max_z] > max_z:
                        max_z = prop0_in_SOI[2,idx_max_z]
                        man_trans_max[k][j] = [prop0,times0[idx_max_z],idx_max_z]

                    idx_min_z = np.argmin(prop0_in_SOI[2])
                    if prop0_in_SOI[2,idx_min_z] < min_z:
                        min_z = prop0_in_SOI[2,idx_min_z]
                        man_trans_min[k][j] = [prop0,times0[idx_min_z],idx_min_z]

                    offset2 = np.tile([1-mu,0,0],(len(times2),1)).T
                    mask_SOI = lin.norm(prop2[0:3]- offset2, axis=0) <= SOI
                    prop2_in_SOI = prop2[:,mask_SOI]
                    
                    idx_max_z = np.argmax(prop2_in_SOI[2])
                    if prop2_in_SOI[2,idx_max_z] > max_z:
                        max_z = prop2_in_SOI[2,idx_max_z]
                        man_trans_max[k][j] = [prop2,times2[idx_max_z],idx_max_z]

                    idx_min_z = np.argmin(prop2_in_SOI[2])
                    if prop2_in_SOI[2,idx_min_z] < min_z:
                        min_z = prop2_in_SOI[2,idx_min_z]
                        man_trans_min[k][j] = [prop2,times2[idx_min_z],idx_min_z]

            print(f"New Orbit, N={j}")
            # Get next periodic orbit, get new IG, pseudo-inverse optimization
            sol_p = IG + z*step
            fsolve_out = fsolve(fsolve_eq, IG, args=(z,sol_p), full_output=True, xtol=1e-12)
            IG = fsolve_out[0]
            Q, Rs = fsolve_out[1]['fjac'], fsolve_out[1]['r'] 
            R = np.zeros((4,4))
            idx, col = np.triu_indices(4, k=0)
            R[idx,col] = Rs
            J = Q.T @ R

            z = lin.inv(J) @ z
            z = z / lin.norm(z)

    # Save all transfers
    save_pkl("Manifold_transfer_max_z",man_trans_max)
    save_pkl("Manifold_transfer_min_z",man_trans_min)

    #ax.plot(prop0[0], prop0[1], prop0[2], color='red', label='Unstable manifold')
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')
    #ax.axis("equal")
    #ax.legend()
    #plt.show()

    #input()