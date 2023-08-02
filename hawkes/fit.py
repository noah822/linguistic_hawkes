from sklearn.neighbors import KernelDensity
from ._internal import (
    two_way_bisect,
    normalize,
    get_end_point_mask,
    indep_roll
)
import numpy as np

from typing import Callable


class DiscreteHawkes:
    def __init__(self,
                 bandwidth: float,
                 delta: float,
                 init_mu0: float=None,
                 init_A: float=None,
                 mle_iter_round: int=20,
                 lambda_bar: float=0.999):
        self.bandwidth = bandwidth

        self._radius = int((self.bandwidth-1) / 2)
        self.delta = delta

        self.init_mu0, self.init_A = init_mu0, init_A

        assert 0 < lambda_bar < 1
        self.lambda_bar = lambda_bar

        self.mle_iter_round = mle_iter_round

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

        def exponential_dist(x, lambda_=0.001):
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
        # lambda_pdf = KernelDensity('gaussian', self.bandwidth).fit(X)
        # lambda_t = lambda_pdf.score_samples(X) # (#sample, )

        # lambda_t = X

        mu0, A, mu_t, g_t = self.init_params(X, self.init_mu0, self.init_A)
        if verbose:
            print('[Init]')
            print('mu0: {:.2f}\tA: {:.2f}'.format(mu0, A))
        for idx in range(epoch):
            if verbose:
                print(f'[Epoch {idx}]')
            mu0, A, mu_t, g_t = self._fit_impl(
                X, mu0, A, mu_t, g_t,
                truncated,
                verbose
            )

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
    
    def _fit_impl(self,
                  X: float,
                  mu0: float,
                  A: float, 
                  mu_: Callable,
                  g_: Callable,
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
            X, mu0, A, mu_, g_
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
                g: np.ndarray):
        num_sample = X.shape[0]

        _, eval_lambda = self._eval_fn_factory(X, mu0, A, mu, g, truncated=False)
        eval_g = self._g_factory(g)

        # g of size (num_sample, ), where sample=0..N-1
        # we assume that g(0) = 0
        updated_mu = np.zeros_like(mu)
        updated_g = np.zeros_like(g)
        for idx in range(num_sample):
            l = 0 if idx < self._radius else (idx - self._radius)
            r = num_sample - 1 if idx > num_sample-1-self._radius else (idx + self._radius)

            interval = range(l, r+1)

            updated_mu[idx] = sum((mu[interval] / eval_lambda(interval)) * X[interval])
            updated_g[idx] = sum((eval_g(interval) / eval_lambda(interval)) * X[interval]) + 1e-4

        updated_mu = normalize(updated_mu)
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