# Solve the household problem using EGM

import numpy as np
import numba as nb
from consav import linear_interp
import quantecon as qe

#######
# EGM #
#######

@nb.njit(parallel=True)
def solve(h,sol,par):
    """ solve backwards with c_plus from previous iteration """

    # unpack policy solution vector
    c = sol.c[h]
    a = sol.a[h]
    v = sol.v[h]

    # intertemporal values
    c_plus = sol.c[h+1] # next period solution
    v_plus = sol.v[h+1] # next period value

    # fixed states
    for i_beta in nb.prange(par.Nbeta):
        beta = par.beta_grid[i_beta]
        
        for i_alpha_l in nb.prange(par.Nalpha_l):
            alpha_l = par.alpha_l_grid[i_alpha_l]
            
            for i_alpha_s in nb.prange(par.Nalpha_s):
                alpha_s = par.alpha_s_grid[i_alpha_s]

                # idiosyncratic states
                for i_z in nb.prange(par.Nz):

                    # a. post-decision marginal value of cash, can be written without the loop (?)
                    q_vec = np.zeros(par.Na)
                    for i_z_plus in range(par.Nz):
                        q_vec += par.z_trans[i_z,i_z_plus]*c_plus[i_beta,i_alpha_l,i_alpha_s,i_z_plus]**(-par.sigma) # transition probabilties should not be changed

                    # b. implied consumption function
                    c_vec = (beta*(1+par.r)*q_vec)**(-1.0/par.sigma)
                    m_vec = par.a_grid + c_vec

                    # c. interpolate from (m,c) to (a_lag,c)
                    for i_a_lag in nb.prange(par.Na):
                        
                        m = (1+par.r)*par.a_grid[i_a_lag] + par.w*par.z_grid[i_z] + alpha_l + alpha_s*h # From def. of assets and cash-on-hand, added the alphas
                        
                        if m <= m_vec[0]: # constrained (lower m than choice with a = 0)
                            c[i_beta,i_alpha_l,i_alpha_s,i_z,i_a_lag] = m
                            a[i_beta,i_alpha_l,i_alpha_s,i_z,i_a_lag] = 0.0 # from definition of being constrained
                        else: # unconstrained
                            c[i_beta,i_alpha_l,i_alpha_s,i_z,i_a_lag] = linear_interp.interp_1d(m_vec,c_vec,m) 
                            a[i_beta,i_alpha_l,i_alpha_s,i_z,i_a_lag] = m-c[i_beta,i_alpha_l,i_alpha_s,i_z,i_a_lag]
                    
                    # intertemporal value of choice
                    expectation = np.zeros(par.Na)
                    for i_z_plus in range(par.Nz): # write without loop

                        # probability weight from transition matrix
                        w = par.z_trans[i_z,i_z_plus]

                        # increment expectation
                        expectation += w*v_plus[i_beta,i_alpha_l,i_alpha_s,i_z,:]

                    v[i_beta,i_alpha_l,i_alpha_s,i_z,:] = c[i_beta,i_alpha_l,i_alpha_s,i_z,:]**(1-par.sigma) / (1-par.sigma) + beta*expectation