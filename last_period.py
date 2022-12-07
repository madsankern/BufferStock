# Solve the household problem in the last period

import numba as nb

@nb.njit(parallel=True)
def solve(h,sol,par):
    """ solve the household problem in the last period of life """

    # unpack
    c = sol.c[h]
    v = sol.v[h]

    # Loop over states, no need to loop over fixed states
    for i_alpha_l in nb.prange(par.Nalpha_l):
        alpha_l = par.alpha_l_grid[i_alpha_l]
        
        for i_alpha_s in nb.prange(par.Nalpha_s):
            alpha_s = par.alpha_s_grid[i_alpha_s]
            
            for i_z in nb.prange(par.Nz):
                for i_a_lag in nb.prange(par.Na):

                    # Unpack states
                    z = par.z_grid[i_z]
                    a_lag = par.a_grid[i_a_lag]

                    # Define cash on hand
                    m = (1+par.r)*a_lag + par.w*z + alpha_l + h*alpha_s

                    # Optimal to consume everything. Choice does not depend on beta
                    c[:,i_alpha_l,i_alpha_s,i_z,i_a_lag] = m

                    # value of choice
                    v[:,i_alpha_l,i_alpha_s,i_z,i_a_lag] = c[:,i_alpha_l,i_alpha_s,i_z,i_a_lag]**(1-par.sigma) / (1-par.sigma)

