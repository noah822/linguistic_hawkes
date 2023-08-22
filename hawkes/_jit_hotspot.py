from numba import njit
import numpy as np
from typing import Callable
# wrapp frequently executed CPU intensive function 
# when fitting Hawkes process into jit

@njit(nogil=True)
def kernel_est_on_window(X: np.ndarray,
                       Y: np.ndarray,
                       window_size: int,
                       bandwidth: int,
                       eps=1e-100
                    ):
    # X/Y: 1-d time-stamp array
    # hacked trick: zero-pad Y on both side
    # current impl
    # -> requires X to be array of time stamps, for example 0,1,2,...[not necessarily consecutive]
    # -> requires Y to be occurence array of full episode of time
    #    1 indicates occur, while 0 indicates non-occur

    num_sample = Y.shape[0]
    radius = int((window_size+1)/2)
    padder = np.zeros(radius, dtype=X.dtype)
    zero_padded_occur = np.concatenate((padder, Y, padder), axis=0)

    # select roi for each time stamp
    l_selected = X[:,None] + radius - np.arange(radius, 0, -1)
    r_selected = X[:,None] + radius + np.arange(0, radius+1)
    selected_time_stamp = np.concatenate((l_selected, r_selected), axis=-1)

    roi = np.take(
        np.concatenate((padder, np.arange(num_sample), padder)),
        selected_time_stamp
    )
    mask = np.take(zero_padded_occur, selected_time_stamp)

    # compute Gaussian Kernel Esimate
    unmasked_estimate = 1 / (np.sqrt(2*np.pi) * bandwidth) * np.exp(
            -(X[:,None]-roi)**2 / (2*bandwidth**2)
    )
    estimate = np.sum(mask * unmasked_estimate, axis=-1) + eps
    return estimate 


@njit(nogil=True)
def bundled_g_lag_compute(
    g: np.ndarray,
    X: np.ndarray,
    query: np.ndarray,
    window_size: int=None
):
    # for two aligned chunk (row), to compute g lag 
    # we need to do the following operation
    # step 1: rotate row of g according to occurence time stamp
    # step 2: element-wise multiply rotated g with X
    # return sum of it 
    num_sample = X.shape[0] # un-stacked X
    num_query = query.shape[0]
    if window_size is not None:
        # hack: double the size of X to accommodate boundary case
        zero_padded_X = np.concatenate((np.zeros_like(X), X), axis=0)
        selected_time_stamp = query[:,None] + num_sample - np.arange(window_size, 0, -1)
        stacked_X = np.take(zero_padded_X, selected_time_stamp)
        rotated_g = np.fliplr(
            np.repeat(g[:window_size], num_query).reshape(-1, num_query).T
        )
        return np.sum(stacked_X * rotated_g, axis=-1)
    else:
        shifts = query
        stacked_X = np.repeat(X, num_query).reshape(-1, num_query).T
        stacked_g = np.repeat(g, num_query).reshape(-1, num_query).T
        rotated_g = indep_2d_roll(np.fliplr(stacked_g), shifts)
        mask = np.arange(num_sample)[None,:] < shifts[:,None]

        return np.sum(stacked_X * rotated_g * mask, axis=-1)

@njit(nogil=True)
def indep_2d_roll(arr, shifts):
    """Apply an independent roll for each dimensions of a single axis.

    Parameters
    ----------
    arr : np.ndarray
        Array of any shape.

    shifts : np.ndarray
        How many shifting to use for each dimension. Shape: `(arr.shape[axis],)`.

    axis : int
        Axis along which elements are shifted. 
    """
    _, col = arr.shape
    col_idcs = np.arange(col)[None,:]

    # Convert to a positive shift
    shifts[shifts < 0] += arr.shape[-1] 
    col_idcs = col_idcs - shifts[:, None]

    result = np.take_along_axis(arr, col_idcs, axis=-1)
    return result

@njit(nogil=True)
def _gaussian_kernel(x, y, bandwidth):
    coef = 1 / (np.sqrt(2*np.pi) * bandwidth)
    estimate = coef * np.exp(
        -(x-y)**2 / (2*bandwidth**2)
    )
    return estimate
    


@njit(nogil=True)
def pairwise_kernel_est(
    X: np.ndarray,
    Y: np.ndarray,
    bandwidth: float,
    mirror_boundary: bool=True,
    l_bound: int=None,
    r_bound: int=None
):
    """
    X : one-d array of shape (m, )
    Y : one-d array of shape (n, )

    Return : two-d array of shape (m, n)
    """
    m = X.shape[0]; n = Y.shape[0]

    # broadcasted pairwise operation
    in_region_est = _gaussian_kernel(
            X[None, :], Y[:,None], bandwidth
        )

    if mirror_boundary:
        l_bound = 0 if l_bound is None else l_bound
        r_bound = m-1 if r_bound is None else r_bound

        cum_est = in_region_est
        l_mirrored_est = _gaussian_kernel(
            X[None,:], (2*l_bound - Y)[:,None], bandwidth
        )
        r_mirrored_est = _gaussian_kernel(
            X[None,:], (2*r_bound - Y)[:,None], bandwidth
        )
        cum_est = cum_est + l_mirrored_est + r_mirrored_est
        return cum_est
    else:
        # no special compansate operation done in the boundary region
        return in_region_est
