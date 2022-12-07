# Consumption-Saving model with individual discount factors and heterogeneous income profiles

import time
import numpy as np
import numba as nb

import quantecon as qe

from EconModel import EconModelClass, jit

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst, find_ergodic, choice
from consav.quadrature import log_normal_gauss_hermite
from consav.linear_interp import binary_search, interp_1d
from consav.misc import elapsed

# local modules
import egm
import last_period
import first_period

class FEModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        par = self.par

        # a. discount factor - add more than two values
        par.beta_max = 0.96
        par.beta_min = 0.7
        par.Nbeta = 5 # number of grid points for beta

        # b. individual income states
        par.alpha_l_min = 0.5 # level parameter
        par.alpha_l_max = 1.0
        par.Nalpha_l = 2

        par.alpha_s_min = 0.1 #0.02 # slope parameter
        par.alpha_s_max = 0.5 #0.05
        par.Nalpha_s = 2

        # c. ability parameter
        par.alpha_tilde_min = 0.5
        par.alpha_tilde_max = 1.5
        par.Nalpha_tilde = 10 # don't know how many grid points to use

        # d. preferences
        par.sigma = 2.0 # CRRA coefficient

        # e. income
        par.w = 1.0 # wage level
        
        par.rho_zt = 0.96 # AR(1) parameter
        par.sigma_psi = 0.5 # std. of persistent shock
        par.Nzt = 5 # number of grid points for zt
        
        par.sigma_xi = 0.10 # std. of transitory shock
        par.Nxi = 5 # number of grid points for xi

        # f. saving
        par.r = 0.02 # interest rate
        par.b = -0.0 # borrowing constraint relative to wage

        # g. grids
        par.a_max = 50.0 # maximum point in grid
        par.Na = 500 # number of grid points       

        # h. length of lifecylcle
        par.H = 10 #100

        # i. simulation
        par.simT = 1 # number of periods
        par.simN = 100_000 # number of individuals (mc)

        # j. tolerances
        par.max_iter_solve = 10_000 # maximum number of iterations
        par.max_iter_simulate = 10_000 # maximum number of iterations
        par.tol_solve = 1e-8 # tolerance when solving
        par.tol_simulate = 1e-8 # tolerance when simulating

    def allocate(self):
        """ allocate model """

        par = self.par
        sol = self.sol
        sim = self.sim
        
        # a. transition matrix
        
        # i. persistent
        _out = log_rouwenhorst(par.rho_zt,par.sigma_psi,par.Nzt)
        par.zt_grid,par.zt_trans,par.zt_ergodic,par.zt_trans_cumsum,par.zt_ergodic_cumsum = _out
        
        # ii. transitory
        if par.sigma_xi > 0 and par.Nxi > 1:
            par.xi_grid,par.xi_weights = log_normal_gauss_hermite(par.sigma_xi,par.Nxi)
            par.xi_trans = np.broadcast_to(par.xi_weights,(par.Nxi,par.Nxi))
        else:
            par.xi_grid = np.ones(1)
            par.xi_weights = np.ones(1)
            par.xi_trans = np.ones((1,1))

        # iii. combined
        par.Nz = par.Nxi*par.Nzt
        par.z_grid = np.repeat(par.xi_grid,par.Nzt)*np.tile(par.zt_grid,par.Nxi)
        par.z_trans = np.kron(par.xi_trans,par.zt_trans)
        par.z_trans_cumsum = np.cumsum(par.z_trans,axis=1)
        par.z_ergodic = find_ergodic(par.z_trans)
        par.z_ergodic_cumsum = np.cumsum(par.z_ergodic)
        par.z_trans_T = par.z_trans.T

        # b. discount factor grid
        par.beta_grid = np.linspace(par.beta_min,par.beta_max,par.Nbeta)
        par.beta_cumsum = np.cumsum(np.tile(1./par.Nbeta,par.Nbeta)) #  implicitly discrete uniform disitribution

        # c. alpha grids
        par.alpha_tilde_grid = np.linspace(par.alpha_tilde_min,par.alpha_tilde_max,par.Nalpha_tilde) # innate ability grid
        par.alpha_tilde_cumsum = np.cumsum(np.tile(1./par.Nalpha_tilde,par.Nalpha_tilde))
        par.alpha_l_grid = np.linspace(par.alpha_l_min,par.alpha_l_max,par.Nalpha_l)
        par.alpha_s_grid = np.linspace(par.alpha_s_min,par.alpha_s_max,par.Nalpha_s)

        # b. asset grid
        assert par.b <= 0.0, f'{par.b = :.1f} > 0, should be negative'
        b_min = -par.z_grid.min()/par.r
        if par.b < b_min:
            print(f'parameter changed: {par.b = :.1f} -> {b_min = :.1f}') 
            par.b = b_min + 1e-8

        par.a_grid = par.w*equilogspace(par.b,par.a_max,par.Na)

        # c. solution arrays
        sol_shape = (par.H,par.Nbeta,par.Nalpha_l,par.Nalpha_s,par.Nz,par.Na) # added alphas as states
        sol.c = np.zeros(sol_shape)
        sol.a = np.zeros(sol_shape)
        sol.vbeg = np.zeros(sol_shape) # not used?
        sol.v = np.zeros(sol_shape) # to compute expectation for optimal choice at h=0

        # solution for optimal alphas
        alpha_shape = (par.Nbeta,par.Nalpha_tilde)
        sol.alpha_l = np.zeros(alpha_shape)
        sol.alpha_s = np.zeros(alpha_shape)

        sol.expectation_exante = np.zeros((par.Nbeta,par.Nalpha_l,par.Nalpha_s)) # just a temporary variable

        # hist - check the shapes of these
        sol.pol_indices = np.zeros(sol_shape,dtype=np.int_)
        sol.pol_weights = np.zeros(sol_shape)

        # d. simulation arrays
        sim_shape = (par.H, par.simN) # shape of simulation arrays

        # mc
        sim.a_ini = np.zeros((par.simN,)) # initial assets, which are zero
        sim.p_z_ini = np.zeros((par.simN,)) # productivity
        sim.p_beta = np.zeros((par.simN,)) # discount factor
        sim.p_alpha_tilde = np.zeros((par.simN,)) # innate ability

        sim.c = np.zeros(sim_shape)
        sim.a = np.zeros(sim_shape)
        sim.p_z = np.zeros(sim_shape)
        sim.i_z = np.zeros(sim_shape,dtype=np.int_)

        sim.i_beta = np.zeros(par.simN,dtype=np.int_)
        sim.i_alpha_tilde = np.zeros(par.simN,dtype=np.int_)
        sim.i_alpha_l = np.zeros(par.simN,dtype=np.int_)
        sim.i_alpha_s = np.zeros(par.simN,dtype=np.int_)

        sim.beta = np.zeros(par.simN,)
        sim.alpha_tilde = np.zeros(par.simN,)
        sim.alpha_l = np.zeros(par.simN,)
        sim.alpha_s = np.zeros(par.simN,)

        # hist
        sim.Dbeg = np.zeros((par.simT,*sol.a.shape))
        sim.D = np.zeros((par.simT,*sol.a.shape))
        sim.Dbeg_ = np.zeros(sol.a.shape)
        sim.D_ = np.zeros(sol.a.shape)

    def solve(self,do_print=True):
        """ solve model using value function iteration or egm """

        t0 = time.time()

        with jit(self) as model:

            par = model.par
            sol = model.sol

            # loop backwards over the life cycle
            for h in reversed(range(par.H)):

                # a. last period
                if h == par.H-1: # last period of life => consume everything
                    t0_last = time.time()
                    last_period.solve(h,sol,par)
                    if do_print: print(f'last period solved in {elapsed(t0_last)}')

                # b. all other periods
                else:
                    egm.solve(h,sol,par)

            # find optimal choice in the first period
            t0_first = time.time()
            first_period.solve(sol,par)
            if do_print: print(f'first period problem solved in {elapsed(t0_first)}')
           
        if do_print: print(f'model solved in {elapsed(t0)}')              

    def prepare_simulate(self,algo='mc',do_print=True):
        """ prepare simulation """

        # set timer
        t0 = time.time()

        # extract namespaces
        par = self.par
        sim = self.sim

        # if using monte carlo
        if algo == 'mc':

            # everybody starts with zero initial assets
            sim.a_ini[:] = 0.0

            # draw uniform numbers for stochastic realizations
            sim.p_z_ini[:] = np.random.uniform(size=(par.simN,)) # initial productivity
            sim.p_z[:,:] = np.random.uniform(size=(par.H,par.simN)) # each productivity shift

            sim.p_beta[:] = np.random.uniform(size=(par.simN,)) # discount factor
            sim.p_alpha_tilde[:] = np.random.uniform(size=(par.simN,)) # innate ability

        elif algo == 'hist': # not usable atm

            sim.Dbeg[0,:,0] = par.z_ergodic
            sim.Dbeg_[:,0] = par.z_ergodic

        else:
            
            raise NotImplementedError

        if do_print: print(f'model prepared for simulation in {time.time()-t0:.1f}')

    def simulate(self,algo='mc',do_print=True):
        """ simulate model """

        t0 = time.time()

        with jit(self) as model:

            par = model.par
            sim = model.sim
            sol = model.sol

            # prepare
            if algo == 'hist': find_i_and_w(par,sol)

            # time loop
            for h in range(par.H):
                
                if algo == 'mc':
                    simulate_forwards_mc(h,par,sim,sol)
                elif algo == 'hist': # not usable atm
                    sim.D[h] = par.z_trans.T@sim.Dbeg[h]
                    if h == par.simT-1: continue
                    simulate_hh_forwards_choice(par,sol,sim.D[h],sim.Dbeg[h+1])
                else:
                    raise NotImplementedError

        if do_print: print(f'model simulated in {elapsed(t0)}')
            
    def simulate_hist_alt(self,do_print=True):
        """ simulate model """

        t0 = time.time()


        with jit(self) as model:

            par = model.par
            sim = model.sim
            sol = model.sol

            Dbeg = sim.Dbeg_
            D = sim.D_

            # a. prepare
            find_i_and_w(par,sol)

            # b. iterate
            it = 0 
            while True:

                Dbeg_old = Dbeg.copy()
                simulate_hh_forwards_stochastic(par,Dbeg,D)
                simulate_hh_forwards_choice(par,sol,D,Dbeg)

                max_abs_diff = np.max(np.abs(Dbeg-Dbeg_old))
                if max_abs_diff < par.tol_simulate: 
                    Dbeg = Dbeg_old
                    break

                it += 1
                if it > par.max_iter_simulate: raise ValueError('too many iterations in simulate()')

        if do_print: 
            print(f'model simulated in {elapsed(t0)} [{it} iterations]')

############################
# simulation - monte carlo #
############################

@nb.njit(parallel=True)
def simulate_forwards_mc(h,par,sim,sol):
    """ monte carlo simulation of model. """
    
    # unpack sim choice containers
    c = sim.c
    a = sim.a
    i_z = sim.i_z

    alpha_l = sim.alpha_l
    alpha_s = sim.alpha_s

    i_beta = sim.i_beta
    i_alpha_tilde = sim.i_alpha_tilde
    i_alpha_l = sim.i_alpha_l
    i_alpha_s = sim.i_alpha_s

    # parallel loop over all individuals
    for i in nb.prange(par.simN):

        # do stuff in the first period        
        if h == 0:

            # initial productivity determined by p_z_ini and ergodic distribution
            p_z_ini = sim.p_z_ini[i]
            i_z_lag = choice(p_z_ini,par.z_ergodic_cumsum)
            
            # beta
            p_beta = sim.p_beta[i]
            i_beta[i] = choice(p_beta,par.beta_cumsum)

            # alpha_tilde
            p_alpha_tilde = sim.p_alpha_tilde[i]
            i_alpha_tilde[i] = choice(p_alpha_tilde,par.alpha_tilde_cumsum)

            # find optimal alpha_l & alpha_s given (beta,alpha_tilde)
            i_alpha_l[i] = int(sol.alpha_l[i_beta[i],i_alpha_tilde[i]])
            i_alpha_s[i] = int(sol.alpha_s[i_beta[i],i_alpha_tilde[i]])

            alpha_l[i] = par.alpha_l_grid[i_alpha_l[i]]
            alpha_s[i] = par.alpha_s_grid[i_alpha_s[i]]

            # initial assets
            a_lag = sim.a_ini[i] # initial assets are just zero
        
        else:
            i_z_lag = sim.i_z[h-1,i]
            a_lag = sim.a[h-1,i]

        # b. productivity
        p_z = sim.p_z[h,i]
        i_z_ = i_z[h,i] = choice(p_z,par.z_trans_cumsum[i_z_lag,:])

        # c. consumption
        c[h,i] = interp_1d(par.a_grid,sol.c[h,i_beta[i],i_alpha_l[i],i_alpha_s[i],i_z_,:],a_lag)

        # d. end-of-period assets
        m = (1+par.r)*a_lag + par.w*par.z_grid[i_z_] + alpha_l[i] + alpha_s[i]*h # added all the alphas
        a[h,i] = m-c[h,i]

##########################
# simulation - histogram #
##########################

@nb.njit(parallel=True) 
def find_i_and_w(par,sol):
    """ find pol_indices and pol_weights for simulation """

    i = sol.pol_indices
    w = sol.pol_weights

    for i_z in nb.prange(par.Nz):
        for i_a_lag in nb.prange(par.Na):
            
            # a. policy
            a_ = sol.a[i_z,i_a_lag]

            # b. find i_ such a_grid[i_] <= a_ < a_grid[i_+1]
            i_ = i[i_z,i_a_lag] = binary_search(0,par.a_grid.size,par.a_grid,a_) 

            # c. weight
            w[i_z,i_a_lag] = (par.a_grid[i_+1] - a_) / (par.a_grid[i_+1] - par.a_grid[i_])

            # d. bound simulation
            w[i_z,i_a_lag] = np.fmin(w[i_z,i_a_lag],1.0)
            w[i_z,i_a_lag] = np.fmax(w[i_z,i_a_lag],0.0)

@nb.njit
def simulate_hh_forwards_stochastic(par,Dbeg,D):
    D[:,:] = par.z_trans_T@Dbeg # Take matrix product to compute the expectation over the possible values of z

@nb.njit(parallel=True)   
def simulate_hh_forwards_choice(par,sol,D,Dbeg_plus):
    """ simulate choice transition """

    for i_z in nb.prange(par.Nz):
    
        Dbeg_plus[i_z,:] = 0.0

        for i_a_lag in range(par.Na):
            
            # i. from
            D_ = D[i_z,i_a_lag]
            if D_ <= 1e-12: continue

            # ii. to
            i_a = sol.pol_indices[i_z,i_a_lag]            
            w = sol.pol_weights[i_z,i_a_lag]
            Dbeg_plus[i_z,i_a] += D_*w
            Dbeg_plus[i_z,i_a+1] += D_*(1.0-w)