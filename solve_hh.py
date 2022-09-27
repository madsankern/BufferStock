# Solve the household problem using EGM

import numpy as np
import numba as nb
from consav import linear_interp
import quantecon as qe

#######
# EGM #
#######

@nb.njit(parallel=True)
def egm(h,par,c_plus,c,a):
    """ solve backwards with c_plus from previous iteration """

    for i_beta in nb.prange(par.Nbeta):
        for i_z in nb.prange(par.Nz):

            beta = par.beta_grid[i_beta]

            # a. post-decision marginal value of cash
            q_vec = np.zeros(par.Na)
            for i_z_plus in range(par.Nz):
                q_vec += par.z_trans[i_z,i_z_plus]*c_plus[h+1,i_beta,i_z_plus,:]**(-par.sigma)
            
            # b. implied consumption function
            c_vec = (beta*(1+par.r)*q_vec)**(-1.0/par.sigma)
            m_vec = par.a_grid+c_vec

            # c. interpolate from (m,c) to (a_lag,c)
            for i_a_lag in range(par.Na):
                
                m = (1+par.r)*par.a_grid[i_a_lag] + par.w*par.z_grid[i_z]
                
                if m <= m_vec[0]: # constrained (lower m than choice with a = 0)
                    c[h,i_beta,i_z,i_a_lag] = m - par.b*par.w
                    a[h,i_beta,i_z,i_a_lag] = par.b*par.w
                else: # unconstrained
                    c[h,i_beta,i_z,i_a_lag] = linear_interp.interp_1d(m_vec,c_vec,m) 
                    a[h,i_beta,i_z,i_a_lag] = m-c[h,i_beta,i_z,i_a_lag] 