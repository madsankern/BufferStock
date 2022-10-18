# Solve the household problem using EGM

import numpy as np
import numba as nb
from consav import linear_interp
import quantecon as qe

#######
# EGM #
#######
# ADD PRANGE WHEN LOOP IS WORKING
# @nb.njit(parallel=True)
def solve(h,sol,par):
    """ solve backwards with c_plus from previous iteration """

    # unpack policy solution vector
    c = sol.c[h]
    a = sol.a[h]
    c_plus = sol.c[h+1] # next period solution

    # fixed states
    for i_beta in range(par.Nbeta):
        beta = par.beta_grid[i_beta]
        
        for i_alpha_l in range(par.Nalpha_l):
            alpha_l = par.alpha_l_grid[i_alpha_l]
            
            for i_alpha_s in range(par.Nalpha_s):
                alpha_s = par.alpha_s_grid[i_alpha_s]

                # idiosyncratic states
                for i_z in range(par.Nz):

                    # a. post-decision marginal value of cash
                    q_vec = np.zeros(par.Na)
                    for i_z_plus in range(par.Nz):
                        q_vec += par.z_trans[i_z,i_z_plus]*c_plus[i_beta,i_alpha_l,i_alpha_s,i_z_plus]**(-par.sigma) # transition probabilties should not be changed

                    # b. implied consumption function
                    c_vec = (beta*(1+par.r)*q_vec)**(-1.0/par.sigma)
                    m_vec = par.a_grid + c_vec # check if this should be changed

                    # c. interpolate from (m,c) to (a_lag,c)
                    for i_a_lag in range(par.Na):
                        
                        m = (1+par.r)*par.a_grid[i_a_lag] + par.w*par.z_grid[i_z] + alpha_l + h*alpha_s # From def. of assets and cash-on-hand, added the alphas
                        
                        if m <= m_vec[0]: # constrained (lower m than choice with a = 0)
                            c[i_beta,i_alpha_l,i_alpha_s,i_z,i_a_lag] = m
                            a[i_beta,i_alpha_l,i_alpha_s,i_z,i_a_lag] = 0.0 # from definition of being constrained
                        else: # unconstrained
                            c[i_beta,i_alpha_l,i_alpha_s,i_z,i_a_lag] = linear_interp.interp_1d(m_vec,c_vec,m) 
                            a[i_beta,i_alpha_l,i_alpha_s,i_z,i_a_lag] = m-c[i_beta,i_alpha_l,i_alpha_s,i_z,i_a_lag]