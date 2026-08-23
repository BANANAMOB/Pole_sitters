import numpy as np
import numpy.linalg as lin
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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

        if i % 10 == 0:
            print(i)
            print(f"Error = {err}")

        i = i+1

    return IG

#### Forced
def cr3bp_equa_f(t, state):
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)
   
    ax = 2*vy + x - (1 - mu)*(x + mu)/r1**3 - mu*(x - (1 - mu))/r2**3
    ay = -2*vx + y - (1 - mu)*y/r1**3 - mu*y/r2**3
    az = - (1 - mu)*z/r1**3 - mu*z/r2**3 + 0.5 # KEY TERM
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

        if i % 10 == 0:
            print(i)
            print(f"Error = {err}")

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

    #### Manifold
    # Compute monodromy matrix for state0
    M0 = np.eye(6).flatten()
    init_M = np.concatenate([state0, M0])
    sol = solve_ivp(STM, [0, T], init_M,t_eval=None,method='RK45',rtol=1e-12,atol=1e-12)
    # Extract final monodromy matrix
    M = sol.y.T[-1][6:].reshape(6,6)
    eig_val, eig_vec = lin.eig(M)

    # Propogate manifolds
    prop_time = 2
    t_span_unstable = [0, prop_time]
    t_span_stable = [prop_time, 0]
    eps = 1e-1
    print(len(sol.t))
    for i in range(len(sol.t)):
        # get current monodromy matrix
        if i % 10 == 0:
            state0 = sol.y.T[i][:6]
            STM_i = sol.y.T[i][6:].reshape(6,6)
            M_i = STM_i @ M @ lin.inv(STM_i)
            eig_val_i, eig_vec_i = lin.eig(M_i)  

            idx_unstable = np.argmax(eig_val_i)
            unstable_vec = eig_vec_i[:,idx_unstable]
            idx_stable = np.argmin(eig_val_i)
            stable_vec = eig_vec_i[:,idx_stable]

            state0_unstable_i, state0_unstable_neg_i = state0 + eps*unstable_vec.real, state0 - eps*unstable_vec.real
            state0_stable_i, state0_stable_neg_i = state0 + eps*stable_vec.real, state0 - eps*stable_vec.real

            prop0, times = prop_orbit_from_IC(state0_unstable_i,t_span_unstable)
            prop1, times = prop_orbit_from_IC(state0_stable_i,t_span_stable)
            if i == 0:
                ax.plot(prop0[0], prop0[1], prop0[2], color='red', label='Unstable manifold')
                ax.plot(prop1[0], prop1[1], prop1[2], color='green', label='Stable manifold')
            else:
                ax.plot(prop0[0], prop0[1], prop0[2], color='red')
                ax.plot(prop1[0], prop1[1], prop1[2], color='green')

            prop2, times = prop_orbit_from_IC(state0_unstable_neg_i,t_span_unstable)
            prop3, times = prop_orbit_from_IC(state0_stable_neg_i,t_span_stable)            
            ax.plot(prop2[0], prop2[1], prop2[2], color='red')
            ax.plot(prop3[0], prop3[1], prop3[2], color='green')

    ################# Step 2: Pole sitter #################

    #Default
    T = 4
    IG = [0.8234,0.2,0.1343,T/2]

    #T = 1.3743*2
    #IG = [1.0118,0.1739,-0.0799, T/2]

    IG_corr = periodic_from_IG_f(IG)
   
    #### Propogate orbit
    state0, T = [IG_corr[0],0,IG_corr[1],0,IG_corr[2],0], IG_corr[3]*2
    t_span = [0, T]  # Integrate for n normalized time units
    ref_pole, times = prop_orbit_from_IC_f(state0, t_span)

    #### Manifold
    # Compute monodromy matrix for state0
    M0_pole = np.eye(6).flatten()
    init_M_pole = np.concatenate([state0, M0_pole])
    sol_pole = solve_ivp(STM_f, [0, T], init_M_pole,t_eval=None,method='RK45',rtol=1e-12,atol=1e-12)
    # Extract final monodromy matrix
    M_pole = sol_pole.y.T[-1][6:].reshape(6,6)
    eig_val_pole, eig_vec_pole = lin.eig(M_pole)

    # Propogate manifolds
    prop_time = 2
    t_span_unstable = [0, prop_time]
    t_span_stable = [prop_time, 0]
    eps = 1e-3
    print(len(sol_pole.t))
    for i in range(len(sol_pole.t)):
        # get current monodromy matrix
        if i % 10 == 0:
            state0 = sol_pole.y.T[i][:6]
            STM_i = sol_pole.y.T[i][6:].reshape(6,6)
            M_i = STM_i @ M_pole @ lin.inv(STM_i)
            eig_val_i, eig_vec_i = lin.eig(M_i)  

            idx_unstable = np.argmax(eig_val_i)
            unstable_vec = eig_vec_i[:,idx_unstable]
            idx_stable = np.argmin(eig_val_i)
            stable_vec = eig_vec_i[:,idx_stable]

            state0_unstable_i, state0_unstable_neg_i = state0 + eps*unstable_vec.real, state0 - eps*unstable_vec.real
            state0_stable_i, state0_stable_neg_i = state0 + eps*stable_vec.real, state0 - eps*stable_vec.real

            prop0, times = prop_orbit_from_IC_f(state0_unstable_i,t_span_unstable)
            prop1, times = prop_orbit_from_IC_f(state0_stable_i,t_span_stable)
            if i == 0:
                ax.plot(prop0[0], prop0[1], prop0[2], color='salmon', label='Unstable manifold')
                ax.plot(prop1[0], prop1[1], prop1[2], color='palegreen', label='Stable manifold')
            else:
                ax.plot(prop0[0], prop0[1], prop0[2], color='salmon')
                ax.plot(prop1[0], prop1[1], prop1[2], color='palegreen')

            prop2, times = prop_orbit_from_IC_f(state0_unstable_neg_i,t_span_unstable)
            prop3, times = prop_orbit_from_IC_f(state0_stable_neg_i,t_span_stable)            
            ax.plot(prop2[0], prop2[1], prop2[2], color='salmon')
            ax.plot(prop3[0], prop3[1], prop3[2], color='palegreen')
   
    ax.plot(ref_pole[0], ref_pole[1], ref_pole[2], color="black", label='Pole sitting orbit')
    ax.plot(ref[0], ref[1], ref[2], '--', color="black", label='Reference orbit')
   
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis("equal")
    ax.legend()
    plt.show()




