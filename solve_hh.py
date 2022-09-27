# Solve the household problem using VFI or EGM¨

import numpy as np
import numba as nb
from consav import linear_interp
import quantecon as qe

#######
# VFI #
#######

@nb.njit
def value_of_choice(c,par,i_z,m,vbeg_plus):
    """ value of choice for use in vfi """

    # a. utility
    utility = c[0]**(1-par.sigma)/(1-par.sigma)

    # b. end-of-period assets
    a = m - c[0]

    # c. continuation value     
    vbeg_plus_interp = linear_interp.interp_1d(par.a_grid,vbeg_plus[i_z,:],a)

    # d. total value
    value = utility + par.beta*vbeg_plus_interp
    return value

@nb.njit(parallel=True)        
def solve_hh_backwards_vfi(par,vbeg_plus,c_plus,vbeg,c,a):
    """ solve backwards with v_plus from previous iteration """

    v = np.zeros(vbeg_plus.shape)

    # a. solution step
    for i_beta in nb.prange(par.Nbeta):
        for i_z in nb.prange(par.Nz):
            for i_a_lag in nb.prange(par.Na):

                # i. cash-on-hand and maximum consumption
                m = (1+par.r)*par.a_grid[i_a_lag] + par.w*par.z_grid[i_z]
                c_max = m - par.b*par.w

                # ii. initial consumption and bounds
                c_guess = np.zeros((1,1))
                bounds = np.zeros((1,2))

                c_guess[0] = c_plus[i_z,i_a_lag]
                bounds[0,0] = 1e-8 
                bounds[0,1] = c_max

                # iii. optimize
                results = qe.optimize.nelder_mead(value_of_choice,
                    c_guess, 
                    bounds=bounds,
                    args=(par,i_z,m,vbeg_plus))

                # iv. save
                c[i_z,i_a_lag] = results.x[0]
                a[i_z,i_a_lag] = m-c[i_z,i_a_lag]
                v[i_z,i_a_lag] = results.fun # convert to maximum

    # b. expectation step
    vbeg[:,:] = par.z_trans@v

#######
# EGM #
#######

@nb.njit(parallel=True)
def solve_hh_backwards_egm(par,c_plus,c,a):
    """ solve backwards with c_plus from previous iteration """

    for i_z in nb.prange(par.Nz):

        # a. post-decision marginal value of cash
        q_vec = np.zeros(par.Na)
        for i_z_plus in range(par.Nz):
            q_vec += par.z_trans[i_z,i_z_plus]*c_plus[i_z_plus,:]**(-par.sigma)
        
        # b. implied consumption function
        c_vec = (par.beta*(1+par.r)*q_vec)**(-1.0/par.sigma)
        m_vec = par.a_grid+c_vec

        # c. interpolate from (m,c) to (a_lag,c)
        for i_a_lag in range(par.Na):
            
            m = (1+par.r)*par.a_grid[i_a_lag] + par.w*par.z_grid[i_z]
            
            if m <= m_vec[0]: # constrained (lower m than choice with a = 0)
                c[i_z,i_a_lag] = m - par.b*par.w
                a[i_z,i_a_lag] = par.b*par.w
            else: # unconstrained
                c[i_z,i_a_lag] = linear_interp.interp_1d(m_vec,c_vec,m) 
                a[i_z,i_a_lag] = m-c[i_z,i_a_lag] 