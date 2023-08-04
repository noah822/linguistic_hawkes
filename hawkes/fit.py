from sklearn.neighbors import KernelDensity
from ._internal import (
    normalize,
    get_end_point_mask,
    indep_roll,
    pairwise_difference,
    diagonal_wise_sum,
    definite_integral,
    columnwise_broadcast_div
)
import numpy as np

from typing import Callable
from .stat import get_bounded_pdf_estimator
from .dist import Gaussian
from copy import deepcopy
import json

class DiscreteHawkes:
    def __init__(self,
                 bandwidth: float,
                 init_mu0: float=None,
                 init_A: float=None,
                 eps: float=1e-8,
                 mle_iter_round: int=20,
                 lambda_bar: float=0.999,
                 save_path: str=None):
        self.bandwidth = bandwidth

        self._radius = int((self.bandwidth-1) / 2)

        self.init_mu0, self.init_A = init_mu0, init_A

        # for numerical stability which doing division
        self.eps = eps 

        assert 0 < lambda_bar < 1
        self.lambda_bar = lambda_bar

        self.mle_iter_round = mle_iter_round
        self.save_path = save_path

    def init_params(self, X, mu0=None, A=None):
        # init parameters and un-parameterized distribution according to X
        num_sample = X.shape[0]
        mu0 = 1. if mu0 is None else mu0
        A = 1. if A is None else A

        default_bg_ratio = 0.6
        mu_t = default_bg_ratio * np.array([np.mean(X, axis=0) for _ in range(num_sample)])
        mu_t = normalize(mu_t)

        g_t = DiscreteHawkes._g_initializer(num_sample)
        g_t = normalize(g_t)

        return mu0, A, mu_t, g_t
    
    @staticmethod
    def _g_initializer(size):
        # sample from expoential distribution

        def exponential_dist(x, lambda_=0.1):
            return lambda_ * np.exp(x * -lambda_)
        
        init_g = exponential_dist(np.arange(size))
        # by default, set self & adjacent triggering effect to 0
        init_g[0] = .0
        init_g[1] = .0
        return init_g
    

    def load_model(self,
                   X: np.ndarray,
                   mu0: float,
                   A: float,
                   mu_t: np.ndarray,
                   g_t: np.ndarray,
                   truncated: bool=False) -> Callable:
        _, lamdba_ = self._eval_fn_factory(X, mu0, A, mu_t, g_t, truncated)
        mu_ = self._mu_factory(mu_t)
        g_lag = self._g_lag_factory(X, g_t)
        return LambdaModel(
            lambda_fn=lamdba_,
            bg_fn=lambda x: mu0 * mu_(x),
            excite_fn=lambda x: A * g_lag(x)
        )



    def fit(self,
            X,
            epoch=5,
            truncated=True,
            verbose=True):

        num_sample = X.shape[0]
        occurence = np.argwhere(X == 1).reshape(-1)

        occur_lag = pairwise_difference(occurence)

        # integrate out kernel estimation for each obversation in the region of interest
        smooth_lambda_normalizer = []
        for i in occurence:
            kernel_estimator = Gaussian(mu=i, sigma=self.bandwidth)
            smooth_lambda_normalizer.append(
                definite_integral(kernel_estimator, 0, num_sample-1)
            )
        smooth_lambda_normalizer = np.array(smooth_lambda_normalizer)
        if verbose:
            print('Integration over lambda finishes')

        smooth_lag_normalizer = []
        for delta in occur_lag:
            kernel_estimator = Gaussian(mu=delta, sigma=self.bandwidth)

            smooth_lag_normalizer.append(
                definite_integral(
                    kernel_estimator, 0, num_sample-1
                )
            )
        smooth_lag_normalizer = np.array(smooth_lag_normalizer)
        print('Integration over ∆t finishes')


        mu0, A, mu_t, g_t = self.init_params(X, self.init_mu0, self.init_A)
        if verbose:
            print('[Init]')
            print('mu0: {:.2f}\tA: {:.2f}'.format(mu0, A))
        for idx in range(epoch):
            if verbose:
                print(f'[Epoch {idx}]')
            mu0, A, mu_t, g_t = self._fit_impl(
                X, mu0, A, mu_t, g_t,
                smooth_lambda_normalizer,
                smooth_lag_normalizer,
                truncated,
                verbose
            )
        
        # save progress
        if self.save_path is not None:
            with open(self.save_path, 'w') as handler:
                json.dump({
                    'mu0' : mu0,
                    'A' : A,
                    'mu_t' : list(mu_t.astype(np.float64)),
                    'g_t' : list(g_t.astype(np.float64))
                }, handler)

        return mu0, A, mu_t, g_t
    
    def _g_factory(self, g):
        def _g(x):
            return g[x]
        return _g
    
    def _g_lag_factory(self, X, g):
        assert X.shape == g.shape
        # g -> array of value at different lag time
        num_sample = g.shape[0] 
        # vectorize this 
        def g_lag(idx: np.ndarray) -> np.ndarray:
            if isinstance(idx, int):
                raise ValueError('input should be an iterable')
            idx = np.array(idx)
            
            num_query = idx.shape[0]
            end_point_mask = get_end_point_mask(idx, num_sample, key=lambda x,y: x<y)

            # pretty tricky vecterized operation
            flipped_g = np.repeat(np.flip(g, axis=-1).reshape(1, -1), repeats=num_query, axis=0)
            aligned_g = indep_roll(flipped_g, idx)

            # compute lagging time matrix
            lagging_mask = np.repeat(X.reshape(1, -1), repeats=num_query, axis=0) * end_point_mask
            lagging_matrix = lagging_mask * aligned_g
            return np.sum(lagging_matrix, axis=-1)
        return g_lag
            
    def _mu_factory(self, mu):
        def _mu(x):
            return mu[x]
        return _mu
    
    def _eval_fn_factory(self,
                    X: np.ndarray,
                    mu0: float,
                    A: float,
                    mu: np.ndarray,
                    g: np.ndarray,
                    truncated: bool=True) -> Callable:
        g_lag = self._g_lag_factory(X, g)
        # when evaluating lambda(x) it can be the case that 
        # lambda is greater than 1
        # if truncated option is set, value larger than 1 will be truncated to lambda_bar


        # wrap mu, g into callable
        def _lambda(x):
            # x is idx 
            lambda_res = mu0 * mu[x] + A * g_lag(x) + 1e-8
            if not truncated:
                return lambda_res
            else:
                lambda_res[lambda_res >= 1] = self.lambda_bar
            return lambda_res

        
        def _phi(x):
            return mu0 * mu[x] / _lambda(x)
        return _phi, _lambda
    
    def pairwise_kernel_est(self, num_sample, occurence):
        # current impl using Gaussian kernel
        stacked_occur = np.repeat(occurence[None,:], repeats=num_sample, axis=0) # (sample, occur)
        est = 1 / (np.sqrt(2*np.pi) * self.bandwidth) \
            * np.exp(
            (-(stacked_occur-np.arange(num_sample)[:,None]) ** 2 / (2 * self.bandwidth**2))
        )
        return est


    
    def _fit_impl(self,
                  X: float,
                  mu0: float,
                  A: float, 
                  mu_: Callable,
                  g_: Callable,
                  smooth_lambda_normalizer: np.ndarray,
                  smooth_lag_normalizer: np.ndarray,
                  truncated: bool=True,
                  verbose: bool=False):
        # expression of lambda
        # lambda = mu0 * mu + A \sum g


        # estimate of lambda_t

        # E step:
        # iterate through all points 
        # -> estimate g(t)
        # -> estimate pho_t in explicit form 

        updated_mu, updated_g = self._E_step(
            X, mu0, A, mu_, g_,
            smooth_lambda_normalizer,
            smooth_lag_normalizer
        )
        # wrap mu, g into callable

        updated_mu0, updated_A = self._M_step(
            X, mu0, A, 
            updated_mu,
            updated_g,
            self.mle_iter_round,
            truncated
        )

        if verbose:
            print('mu0: {:.2f}\tA: {:.2f}'.format(updated_mu0, updated_A))

        return updated_mu0, updated_A, updated_mu, updated_g
    
    
    def _E_step(self,
                X: np.ndarray,
                mu0: float,
                A: float, 
                mu: np.ndarray,
                g: np.ndarray,
                smooth_lambda_normalizer: np.ndarray,
                smooth_lag_normalizer: np.ndarray):
        occurence = np.argwhere(X == 1).reshape(-1)
        occur_lag = pairwise_difference(occurence) # 1-d array

        num_sample = X.shape[0]

        _, eval_lambda = self._eval_fn_factory(X, mu0, A, mu, g, truncated=False)
        eval_g = self._g_factory(g)
        eval_mu = self._mu_factory(mu)

        # compute kernel estimate at a given time stamp for all
        # observation on-the-fly
        
        Z_lambda = self.pairwise_kernel_est(num_sample, occurence)
        updated_mu = np.sum(
            eval_mu(occurence) / eval_lambda(occurence) * Z_lambda / smooth_lambda_normalizer,
            axis=-1
        )

        # index j < i, rho[ij] denotes g(ti-tj) / lambda(ti)
        Z_g = self.pairwise_kernel_est(num_sample, occur_lag)
        num_occur = len(occurence)

        flattened_occur = np.repeat(occurence[None,:], repeats=num_occur, axis=0)[np.triu_indices(num_occur, k=1)]
        rho_ij = eval_g(occur_lag) / eval_lambda(flattened_occur)

        # t_i < T - t

        # for a given lag, we count the number of valid occurence
        # (pair_cnt, )
        valid_occur_count = np.sum(
            np.repeat(occurence[None,:], repeats=num_sample, axis=0) \
          < (num_sample - np.arange(num_sample)[:,None]),
            axis=-1
        )

        updated_g = np.sum(
            rho_ij * Z_g / smooth_lag_normalizer,
            axis=-1
        ) / (valid_occur_count + self.eps)

        updated_mu = normalize(updated_mu)

        # force g(0) = g(1) = 0

        # updated_g[0] = updated_g[1] = 0
        updated_g = normalize(updated_g)

        return updated_mu, updated_g
    

    def _M_step(self,
                X: np.ndarray,
                mu0: float,
                A: float, 
                mu: Callable,
                g: Callable,
                iter_round: int,
                truncated: bool=True):
        
        occurence = np.argwhere(X == 1).reshape(-1)
        non_occur = np.argwhere(X == 0).reshape(-1)

        # wrap mu, g into Callable
        eval_mu = self._mu_factory(mu)
        g_lag = self._g_lag_factory(X, g)

        # iteratively optimize mu0 & A
        for _ in range(iter_round):
            # update mu0
            eval_phi, eval_lambda = self._eval_fn_factory(X, mu0, A, mu, g, truncated)

            mu0 = sum(eval_phi(occurence)) / \
                  sum(eval_mu(non_occur) / (1 - eval_lambda(non_occur)))
            
            # print('sum of phi: {:.2f}'.format(sum(eval_phi(occurence))))
            # print(sum(eval_mu(non_occur) / (1 - eval_lambda(non_occur))))
            
            # udpate A
            eval_phi, eval_lambda = self._eval_fn_factory(X, mu0, A, mu, g)

            # print(sum(1-eval_phi(occurence)))
            A = sum(1 - eval_phi(occurence)) / \
                sum(g_lag(non_occur) / (1-eval_lambda(non_occur)))
        
        return mu0, A


    @staticmethod
    def _integral_estimate(fn: Callable,
                           lvalue: float,
                           rvalue: float,
                           resolution: int=10000) -> float:
        # Riemann integral approximation of integral
        X = np.linspace(lvalue, rvalue, (rvalue-lvalue)*resolution)
        res = 1/resolution * sum(fn(X))
        return res

class LambdaModel:
    def __init__(self,
                 lambda_fn: Callable,
                 bg_fn: Callable,
                 excite_fn: Callable):
        self.lambda_fn = lambda_fn
        self.bg_fn, self.excite_fn = bg_fn, excite_fn

    def __call__(self, x):
        return self.lambda_fn(x)

    def excitation(self, x):
        return self.excite_fn(x)
    def background(self, x):
        return self.bg_fn(x)
