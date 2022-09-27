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
import solve_hh

class FEModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        par = self.par
        
        # preferences
        par.beta_max = 0.96 # highest beta
        par.beta_min = 0.6 # lowest beta
        par.Nbeta = 5 # number of grid points for beta

        par.sigma = 2.0 # CRRA coefficient

        par.H = 20 # length of lifecycle

        # income
        par.w = 1.0 # wage level
        
        par.rho_zt = 0.96 # AR(1) parameter
        par.sigma_psi = 0.10 # std. of persistent shock
        par.Nzt = 5 # number of grid points for zt
        
        par.sigma_xi = 0.0 #0.10 # std. of transitory shock
        par.Nxi = 1 # number of grid points for xi

        # saving
        par.r = 0.02 # interest rate
        par.b = -0.0 # borrowing constraint relative to wage

        # grid
        par.a_max = 15.0 # maximum point in grid
        par.Na = 500 # number of grid points       

        # simulation
        par.simT = 500 # number of periods
        par.simN = 100_000 # number of individuals (mc)

        # tolerances
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
        
        # persistent
        _out = log_rouwenhorst(par.rho_zt,par.sigma_psi,par.Nzt)
        par.zt_grid,par.zt_trans,par.zt_ergodic,par.zt_trans_cumsum,par.zt_ergodic_cumsum = _out
        
        # transitory
        if par.sigma_xi > 0 and par.Nxi > 1:
            par.xi_grid,par.xi_weights = log_normal_gauss_hermite(par.sigma_xi,par.Nxi)
            par.xi_trans = np.broadcast_to(par.xi_weights,(par.Nxi,par.Nxi))
        else:
            par.xi_grid = np.ones(1)
            par.xi_weights = np.ones(1)
            par.xi_trans = np.ones((1,1))

        # combined
        par.Nz = par.Nxi*par.Nzt
        par.z_grid = np.repeat(par.xi_grid,par.Nzt)*np.tile(par.zt_grid,par.Nxi)
        par.z_trans = np.kron(par.xi_trans,par.zt_trans)
        par.z_trans_cumsum = np.cumsum(par.z_trans,axis=1)
        par.z_ergodic = find_ergodic(par.z_trans)
        par.z_ergodic_cumsum = np.cumsum(par.z_ergodic)
        par.z_trans_T = par.z_trans.T

        # b. discount factor grid
        par.beta_grid = np.linspace(par.beta_min, par.beta_max, par.Nbeta)

        # b. asset grid
        assert par.b <= 0.0, f'{par.b = :.1f} > 0, should be negative'
        b_min = -par.z_grid.min()/par.r
        if par.b < b_min:
            print(f'parameter changed: {par.b = :.1f} -> {b_min = :.1f}') 
            par.b = b_min + 1e-8

        par.a_grid = par.w*equilogspace(par.b,par.a_max,par.Na)

        # c. solution arrays
        sol_shape = (par.H,par.Nbeta,par.Nz,par.Na)
        sol.c = np.zeros(sol_shape)
        sol.a = np.zeros(sol_shape)
        sol.vbeg = np.zeros(sol_shape)

        # hist - check the shapes of these
        sol.pol_indices = np.zeros(sol_shape,dtype=np.int_)
        sol.pol_weights = np.zeros(sol_shape)

        # d. simulation arrays

        # mc
        sim.a_ini = np.zeros((par.simN,))
        sim.p_z_ini = np.zeros((par.simN,))
        sim.c = np.zeros((par.simT,par.simN))
        sim.a = np.zeros((par.simT,par.simN))
        sim.p_z = np.zeros((par.simT,par.simN))
        sim.i_z = np.zeros((par.simT,par.simN),dtype=np.int_)

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
                
                # t0_it = time.time()

                # a. next-period value function
                if h == par.H-1: # last period of life => consume everything

                    m_plus = np.zeros((par.H,par.Nbeta,par.Nz,par.Na)) # added a dimension for betas             
                    m_plus[:,:] = (1+par.r)*par.a_grid[np.newaxis,:] + par.w*par.z_grid[:,np.newaxis] # dimension 0 of m_plus does not matter in the last period
                    c_plus_max = m_plus - par.w*par.b
                    c_plus = 0.99*c_plus_max # arbitary factor
                    v_plus = c_plus**(1-par.sigma)/(1-par.sigma)
                    vbeg_plus = par.z_trans@v_plus # not needed. Check if matmul makes sense at all when dim(v_plus) = 3

                else:

                    vbeg_plus = sol.vbeg.copy()
                    c_plus = sol.c.copy()

                # b. solve this period
                solve_hh.egm(h,par,c_plus,sol.c,sol.a)

                # max_abs_diff = np.max(np.abs(sol.c-c_plus))

                # c. check convergence
                # converged = max_abs_diff < par.tol_solve
                
                # d. break
                # if do_print and (converged or it < 10 or it%100 == 0):
                #     print(f'iteration {it:4d} solved in {elapsed(t0_it):10s}',end='')              
                #     print(f' [max abs. diff. {max_abs_diff:5.2e}]')

                # if converged: break

                # it += 1
                # if it > par.max_iter_solve: raise ValueError('too many iterations in solve()')
        
        if do_print: print(f'model solved in {elapsed(t0)}')              

    def prepare_simulate(self,algo='mc',do_print=True):
        """ prepare simulation """

        t0 = time.time()

        par = self.par
        sim = self.sim

        if algo == 'mc':

            sim.a_ini[:] = 0.0
            sim.p_z_ini[:] = np.random.uniform(size=(par.simN,))
            sim.p_z[:,:] = np.random.uniform(size=(par.simT,par.simN))

        elif algo == 'hist':

            sim.Dbeg[0,:,0] = par.z_ergodic
            sim.Dbeg_[:,0] = par.z_ergodic

        else:
            
            raise NotImplementedError

        if do_print: print(f'model prepared for simulation in {time.time()-t0:.1f} secs')

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
            for t in range(par.simT):
                
                if algo == 'mc':
                    simulate_forwards_mc(t,par,sim,sol)
                elif algo == 'hist':
                    sim.D[t] = par.z_trans.T@sim.Dbeg[t]
                    if t == par.simT-1: continue
                    simulate_hh_forwards_choice(par,sol,sim.D[t],sim.Dbeg[t+1])
                else:
                    raise NotImplementedError

        if do_print: print(f'model simulated in {elapsed(t0)} secs')
            
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
def simulate_forwards_mc(t,par,sim,sol):
    """ monte carlo simulation of model. """
    
    c = sim.c
    a = sim.a
    i_z = sim.i_z

    for i in nb.prange(par.simN):

        # a. lagged assets
        if t == 0:
            p_z_ini = sim.p_z_ini[i]
            i_z_lag = choice(p_z_ini,par.z_ergodic_cumsum)
            a_lag = sim.a_ini[i]
        else:
            i_z_lag = sim.i_z[t-1,i]
            a_lag = sim.a[t-1,i]

        # b. productivity
        p_z = sim.p_z[t,i]
        i_z_ = i_z[t,i] = choice(p_z,par.z_trans_cumsum[i_z_lag,:])

        # c. consumption
        c[t,i] = interp_1d(par.a_grid,sol.c[i_z_,:],a_lag)

        # d. end-of-period assets
        m = (1+par.r)*a_lag + par.w*par.z_grid[i_z_]
        a[t,i] = m-c[t,i]

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