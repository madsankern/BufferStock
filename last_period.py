# Solve the household problem in the last period

import numba as nb

@njit(parallel=True)
def solve(h,sol,par):
    """ solve the household problem in the last period of life """

    # unpack
    v = sol.v[h]
    c = sol.c[h]

    # Loop over states
    # for i_beta in nb.prange(par.Nbeta):
    for i_z in nb.prange(par.Nz):
        for i_a_lag in nb.prange(par.Na):

            # Unpack states
            z = par.z_grid[i_z]
            a_lag = par.a_grid[i_a_lag]

            # Define cash on hand
            m = (1+par.r)*a_lag + par.w*z

            # Optimal to consume everything. Choice does not depend on beta
            c[:,i_z,i_a_lag] = m

            # Value of choice
            v[:,i_z,i_a_lag] = c[:,i_z,i_a_lag]**(1-par.sigma) / (1-par.sigma)

                