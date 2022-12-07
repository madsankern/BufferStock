import numba as nb
import numpy as np

# @nb.njit(parallel=True)
def solve(sol,par):
    """ find optimal alphas in the inital period """
    
    #########################################
    # Compute expectation of value function #
    #########################################
    expectation_exante = np.zeros((par.Nbeta,par.Nalpha_l,par.Nalpha_s))

    # betas are assumed known before choice, so loop over them
    for i_beta in range(par.Nbeta):
    
        # compute beginning of period expectation, using the ergodic distribution of productivity
        for i_z in range(par.Nz):
 
            # probability weight
            w = par.z_ergodic[i_z]

            # expected value, assuming life starts with zero assets
            expectation_exante[i_beta] += w*sol.v[0,i_beta,:,:,i_z,0]

    #############################################
    # Compute penalty term for each alpha_tilde #
    #############################################

    # weights of penalty terms, move to preferences paramters
    w_l = .5
    w_s = .5

    # test
    mean_alpha_l = np.mean(par.alpha_l_grid)
    mean_alpha_s = np.mean(par.alpha_s_grid)

    # loop over values of innate ability
    val_of_choice = np.zeros((par.Nbeta,par.Nalpha_tilde,par.Nalpha_l,par.Nalpha_s)) # initialize

    # loop over all fixed states
    for i_beta in range(par.Nbeta):
        for i_alpha_tilde in range(par.Nalpha_tilde):
            for i_alpha_l in range(par.Nalpha_l):
                for i_alpha_s in range(par.Nalpha_s):

                    # unpack states
                    alpha_l = par.alpha_l_grid[i_alpha_l]
                    alpha_s = par.alpha_s_grid[i_alpha_s]
                    alpha_tilde = par.alpha_tilde_grid[i_alpha_tilde]
                    beta = par.beta_grid[i_beta]

                    # penalty term
                    penalty = -1*(1-beta**(par.H-1))/(1-beta)*(w_l*(alpha_l/mean_alpha_l - alpha_tilde)**2 + w_s*(alpha_s/mean_alpha_s - alpha_tilde)**2)
                    
                    # value of choice of (alpha_l,alpha_s)
                    val_of_choice[i_beta,i_alpha_tilde,i_alpha_l,i_alpha_s] = expectation_exante[i_beta,i_alpha_l,i_alpha_s] + penalty

                    # find optimal choice
                    x = np.unravel_index(np.argmax(val_of_choice[i_beta,i_alpha_tilde],axis=None),val_of_choice[i_beta,i_alpha_tilde].shape) # can this be written better?

                    sol.alpha_l[i_beta,i_alpha_tilde] = x[0] # the name should be changed, as the vector contains the optimal INDICIES
                    sol.alpha_s[i_beta,i_alpha_tilde] = x[1] # the name should be changed, as the vector contains the optimal INDICIES