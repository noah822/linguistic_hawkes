import bisect
import concurrent.futures 
import json
import numpy as np
import os
import pickle
from typing import (
    Callable,
    Tuple
)

import dask.array as da


from ._jit_hotspot import bundled_g_lag_compute, kernel_est_on_window
from ._internal import (
    normalize,
    pairwise_difference,
    definite_integral,
)
from .dist import Gaussian
from utils.visual import GifConverter

from typing import Dict


class DiscreteHawkes:
    def __init__(self,
                 bandwidth: float,
                 init_mu0: float=None,
                 init_A: float=None,
                 init_exp_lambda: float=None,
                 eps: float=1e-8,
                 mle_iter_round: int=20,
                 lambda_bar: float=0.999,
                 system_profile: Dict=None,
                 chunk_config: Dict=None,
                 use_file_buffer: bool=False,
                 num_internal_worker: int=None,
                 g_truncate_bound: int=None,
                 kernel_window: int=None,
                 save_as_gif_path: str=None,
                 serialize: bool=False,
                 save_path: str=None):
        self.bandwidth = bandwidth

        self._radius = int((self.bandwidth-1) / 2)

        self.init_mu0, self.init_A = init_mu0, init_A
        self.init_exp_lambda = init_exp_lambda

        # for numerical stability when doing division
        self.eps = eps 

        assert 0 < lambda_bar < 1
        self.lambda_bar = lambda_bar

        self.mle_iter_round = mle_iter_round
        self.save_path = save_path

        self._save_progress = save_as_gif_path is not None
        self.save_as_gif_path = save_as_gif_path

        self.use_file_buffer = use_file_buffer

        self._system_profile = system_profile

        self._overriding_chunk_config = chunk_config

        # configure workers for internal concurrent operation
        if num_internal_worker is None:
            num_internal_worker = os.cpu_count()
        self.num_internal_worker = num_internal_worker

        self.serialize = serialize

        # if this option is turned on
        # when updating g, value at time stamp larger then threshold `g_truncate_bound`
        # will not be computed and default to self._xi 
        self._g_truncate_bound = g_truncate_bound
        self._xi = 1e-150

        self._optimized_kernel_compute = False # currently deprecated
        self._kernel_window = kernel_window
    

    
    @property
    def chunk_config(self):
        _chunk_config = {
            'X' : 1000,
            'kernel' : 1000,
            'occur_lag' : 1000
        }
        if self._overriding_chunk_config is not None:
            assert all(
                [v in _chunk_config.keys() for v in self._overriding_chunk_config]
            )
            for k, v in self._overriding_chunk_config.items():
                _chunk_config[k] = v
        return _chunk_config


    def init_params(self, X, mu0=None, A=None, init_exp_lambda=None):
        # init parameters and un-parameterized distribution according to X
        num_sample = X.shape[0]
        mu0 = 1. if mu0 is None else mu0
        A = 1. if A is None else A
        init_exp_lambda = 0.02 if init_exp_lambda is None else init_exp_lambda

        default_bg_ratio = 0.6
        mu_t = default_bg_ratio * np.array([np.mean(X, axis=0) for _ in range(num_sample)])
        mu_t = normalize(mu_t, divide_by_mean=True)

        g_t = DiscreteHawkes._g_initializer(num_sample, init_exp_lambda)
        g_t = normalize(g_t)

        return mu0, A, mu_t, g_t
    
    @staticmethod
    def _g_initializer(size, init_exp_lambda):
        # sample from expoential distribution

        def exponential_dist(x, lambda_=init_exp_lambda):
            return lambda_ * np.exp(x * -lambda_)
        
        init_g = exponential_dist(np.arange(size))
        # by default, set self & adjacent triggering effect to 0
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
        if self._save_progress:
            # initialize frame holder for each trackable varibale
            self.gif_converter_dict = {
                'mu_t' : GifConverter(),
                'g_t' : GifConverter()
            }

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

            smooth_lag_normalizer.append(1.)

        smooth_lag_normalizer = np.array(smooth_lag_normalizer)
        print('Integration over ∆t finishes')


        smooth_lag_normalizer = da.from_array(
            smooth_lag_normalizer,
            chunks=(self.chunk_config['X'], )
        )
        smooth_lambda_normalizer = da.from_array(
            smooth_lambda_normalizer,
            chunks=(self.chunk_config['X'], )
        )


        mu0, A, mu_t, g_t = self.init_params(X, self.init_mu0, self.init_A)
        if verbose:
            print('[Init]')
            print('mu0: {:.7f}\tA: {:.7f}'.format(mu0, A))

        if self._save_progress:
            # log init condition
            self.gif_converter_dict['mu_t'].push(
                range(num_sample), mu_t,
                'Init'
            )
            self.gif_converter_dict['g_t'].push(
                range(num_sample), g_t, 
                'Init'
            )
        
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

            if self._save_progress:
                # record the evolution of mu_t and g_t as mp4
                self.gif_converter_dict['mu_t'].push(
                    range(num_sample), mu_t,
                    f'Epoch {idx}'
                )
                self.gif_converter_dict['g_t'].push(
                    range(num_sample), g_t, 
                    f'Epoch {idx}'
                )
        
        # save fitted parameters
        if self.save_path is not None:
            with open(self.save_path, 'w') as handler:
                json.dump({
                    'mu0' : mu0,
                    'A' : A,
                    'mu_t' : list(mu_t.astype(np.float64)),
                    'g_t' : list(g_t.astype(np.float64))
                }, handler)
        
        # save progress as mp4
        if self._save_progress:
            if not os.path.isdir(self.save_as_gif_path):
                os.makedirs(self.save_as_gif_path, exist_ok=False)
            for param_name, gif in self.gif_converter_dict.items():
                gif.save(
                    os.path.join(self.save_as_gif_path, f'{param_name}.mp4'),
                    save_fps=2
                )

        return mu0, A, mu_t, g_t
    
    def _g_factory(self, g,
                   wrap_into_dask: bool=False):
        def _g(x):
            res = g[x]
            if wrap_into_dask:
                return da.from_array(res, chunks=(self.chunk_config['X'], ))
            return g[x]
        return _g
    
    def _g_lag_factory(self, X, g):
        assert X.shape == g.shape
        # g -> array of value at different lag time
        def _atomic_g_lag(idx: np.ndarray) -> np.ndarray:
            # do not actually need to stack X statically, do it on-the-fly
            def _bundle_query_op(query):
                return bundled_g_lag_compute(
                    g, X, query,
                    window_size=self._g_truncate_bound
                )
            
            if isinstance(idx, int):
                raise ValueError('input should be an iterable')
                
            # everything should be wrapped into dask array
            wrapped_query = da.from_array(idx, chunks=(self.chunk_config['occur_lag'], ))

            res = da.map_blocks(
                _bundle_query_op, wrapped_query, 
                dtype='float64',
                chunks=(self.chunk_config['occur_lag'], )
            )

            return res.compute() # export numpy result
    
        if self.serialize:
            # serialize g_lag query into multiple chunks to prevent OOM
            serialize_num_chunk = 10
            def _serialized_g_lag(idx: np.ndarray):
                res = np.zeros_like(idx)
                self._serialize_scheduler(res, _atomic_g_lag, (idx, ), serialize_num_chunk)
                return res
                
            g_lag = _serialized_g_lag
        else:
            g_lag = _atomic_g_lag
        return g_lag
    
    
    def _mu_factory(self, mu,
                    wrap_into_dask: bool=False):
        def _mu(x):
            res = mu[x]
            if wrap_into_dask:
                return da.from_array(res, chunks=(self.chunk_config['X'], ))
            return res
        return _mu
    
    def _eval_fn_factory(self,
                    X: np.ndarray,
                    mu0: float,
                    A: float,
                    mu: np.ndarray,
                    g: np.ndarray,
                    truncated: bool=True,
                    wrap_into_dask: bool=False) -> Callable:
        g_lag = self._g_lag_factory(X, g)
        # when evaluating lambda(x) it can be the case that 
        # lambda is greater than 1
        # if truncated option is set, value larger than 1 will be truncated to lambda_bar


        # wrap mu, g into callable
        def _lambda(x):
            # x is idx 
            lambda_res = mu0 * mu[x] + A * g_lag(x) + 1e-150
            if truncated:
                lambda_res[lambda_res >= 1] = self.lambda_bar
            if wrap_into_dask:
                return da.from_array(lambda_res, chunks=(self.chunk_config['X'], ))
            return lambda_res

        
        def _phi(x):
            background = mu0 * mu[x]
            if wrap_into_dask:
                background = da.from_array(background, chunks=(self.chunk_config['X'], ))
            return background / _lambda(x)
        return _phi, _lambda
    
    def pairwise_kernel_est(self,
                            num_sample: int,
                            occurence: np.ndarray,
                            repeat_mask: np.ndarray=None
                        ):
        # current impl using Gaussian kernel

        # buffer stacked occurence array
        wrapped_est = da.arange(num_sample, chunks=(self.chunk_config['kernel'], ))
        num_occur = occurence.shape[0]
        def _row_wise_kernel_est(row: np.ndarray):
            # compute elememt-wise gaussian on each row
            if self._optimized_kernel_compute:
                occur_indicator = np.zeros(num_sample)
                occur_indicator[occurence] = 1
                estimate = kernel_est_on_window(
                    row, occur_indicator,
                    self._kernel_window, self.bandwidth
                )
            else:
                estimate = 1 / (np.sqrt(2*np.pi) * self.bandwidth) * np.exp(
                    -(occurence[None,:]-row[:,None])**2 / (2*self.bandwidth**2)
                )
            if repeat_mask is not None:
                estimate = estimate * repeat_mask
            return estimate
        
        est = da.map_blocks(
            _row_wise_kernel_est,
            wrapped_est,
            chunks=(self.chunk_config['kernel'], num_occur), 
            new_axis=[1],
            dtype=np.float64
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
            print('mu0: {:.7f}\tA: {:.7f}'.format(updated_mu0, updated_A))

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

        _, eval_lambda = self._eval_fn_factory(
            X, mu0, A, mu, g,
            truncated=False,
            wrap_into_dask=True
        )
        eval_g = self._g_factory(g, wrap_into_dask=True)
        eval_mu = self._mu_factory(mu, wrap_into_dask=True)

        # compute kernel estimate at a given time stamp for all
        # observation on-the-fly
        
        # wrap this into dask framework for better memory usage
        Z_lambda = self.pairwise_kernel_est(num_sample, occurence)
        updated_mu = da.sum(
            eval_mu(occurence) / eval_lambda(occurence) * Z_lambda / smooth_lambda_normalizer,
            axis=-1
        ).compute()

        # index j < i, rho[ij] denotes g(ti-tj) / lambda(ti)

        # get occurence count
        Z_g = self.pairwise_kernel_est(num_sample, occur_lag)
        num_occur = len(occurence)

        flattened_occur = np.repeat(occurence[None,:], repeats=num_occur, axis=0)[np.triu_indices(num_occur, k=1)]
        rho_ij = eval_g(occur_lag) / eval_lambda(flattened_occur)  # dask array

        updated_g = da.sum(
            rho_ij * Z_g / smooth_lag_normalizer,
            axis=-1
        ).compute()
        updated_mu = normalize(updated_mu, divide_by_mean=True)

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
            
            
            # udpate A
            eval_phi, eval_lambda = self._eval_fn_factory(X, mu0, A, mu, g)

            A = sum(1 - eval_phi(occurence)) / \
                sum(g_lag(non_occur) / (1-eval_lambda(non_occur)))
        
        return mu0, A


    @staticmethod
    def _buffer_stacked_arr(
        base_arr: np.ndarray,
        repeats: int,
        buffer_folder: str,
        bundle_size: int=1,
        dtype: str='float64',
        axis: int=0,
        num_worker: int=4
    ):
        # save stacked array in a format that can be recognized by 
        # dask.from_stack_npy() function
        # this machnism is used for bypass time-consuming 
        # dask arry rechunk/repeat operation

        # it requires that the base array should be able to fit in the RAM

        os.makedirs(buffer_folder, exist_ok=True)
        if len(base_arr.shape) == 1:
            # add new dimension to it if necessary
            base_arr = base_arr[None,:]

        def _save_one_chunk(i: int):
            save_path = os.path.join(buffer_folder, f'{i.npy}')
            np.save(save_path, np.repeat(base_arr, repeats=bundle_size, axis=0))

        # integer bundle, parallel this
        with concurrent.futures.ThreadPoolExecutor(num_worker) as executor:
            for i in range(int(repeats/bundle_size)):
                executor.submit(_save_one_chunk, i)

        # remainer 
        remainer = repeats % bundle_size
        if remainer > 0:
            np.save(
                os.path.join(buffer_folder, f'{int(repeats/bundle_size)}.npy'),
                np.repeat(base_arr, repeats=remainer, axis=0)
            )

        # config meta info
        row_wise_chunk = [bundle_size for _ in range(int(repeats / bundle_size))]
        if remainer > 0:
            row_wise_chunk = row_wise_chunk + [remainer]
        chunks_info = (
            tuple(row_wise_chunk),
            (base_arr.shape[-1], )
        )
        meta_info = {
            'chunks' : chunks_info,
            'dtype' : dtype,
            'axis' : axis
        }
        with open(os.path.join(buffer_folder, 'info'), 'wb') as handler:
            pickle.dump(meta_info, handler)
    
    def _serialize_scheduler(self, 
                             res_container: np.ndarray,
                             func: Callable,
                             args: Tuple,
                             chunk: int):
        # default to be 1d numpy array
        N = len(res_container)
        # divided into chunks to be executed serially
        chunk_size = int((N + chunk - 1) / chunk)
        for i in range(chunk):
            lbound = i * chunk_size
            rbound = lbound + chunk_size
            res_container[lbound:rbound] = func(args[lbound:rbound])


    @staticmethod
    def _stack_then_rechunk(np_arr: np.ndarray,
                            repeats: int, 
                            chunks: Tuple):
        # stack array at zero axis then rechunk it

        # by default will take smallest chunk as the arr to be stacked itself
        wrapped_arr = da.from_array(np_arr, chunks=(np_arr.shape))
        stacked_arr = da.repeat(
            wrapped_arr[None,:], repeats, axis=0
        )
        return stacked_arr.rechunk(chunks)



            
    
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
