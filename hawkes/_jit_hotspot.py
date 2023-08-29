from numba import njit
import numpy as np
from typing import Callable, Any
# wrapp frequently executed CPU intensive function 
# when fitting Hawkes process into jit


@njit(nogil=True)
def kernel_est_on_window(X: np.ndarray,
                        Y: np.ndarray,
                        bandwidth: int,
                        l_bound: int=None,
                        r_bound: int=None,
                        num_neighbor: int=10,
                        weights: np.ndarray=None
                    ):
    """
    Args:
    - X: (m, ) array of time stamps, for example, 0,1,2,... [not necessarily consecutive]
    - Y: (n, ) occurence time stamps
    - l/r_bound: left/right boundary to flip padder of two ends of Y around
    - num_neighbor: number of adjacent points to consider when estimating, default to 10
    - bandwidth: bandwidth of Gaussian Kernel, equivalent to std
    Return:
    - est: (m, n)

    Spec:
    1. When using this optimized kernel estimate function, Y should be a sorted array
    2. To accomondate boundary cases, this function will extend both sides of array Y by
    mirroring first/last $num_neighbor points at two ends of the boundary
    """
    
    m = X.shape[0]
    n = Y.shape[0]
    assert num_neighbor <= m, 'Number of neighbors to look up exceeds size of array Y'
    l_bound = 0 if l_bound is None else l_bound
    r_bound = n - 1 if r_bound is None else r_bound


    # take first/last num_neighbor of elements in Y as padder to handle corner cases
    left_padder = 2*l_bound - X[:num_neighbor]
    right_padder = 2*r_bound - X[-num_neighbor:]

    extended_X = np.concatenate(
        (left_padder, X, right_padder), axis=0
    )
    # bisect Y using X to find the (approximated) center of roi
    center_idcs = np.searchsorted(X, Y, side='left')

    # indices of selected points to estimate
    selected_idcs = np.arange(0, 2*num_neighbor)[None,:] + center_idcs[:,None]

    # select roi for each time stamp
    roi = np.take(extended_X, selected_idcs)
    est = _gaussian_kernel(Y[:,None], roi, bandwidth)

    if weights is not None:
        # pad weight array accordingly
        left_padder = weights[:num_neighbor]
        right_padder = weights[-num_neighbor:]
        extended_w = np.concatenate(
            (left_padder, weights, right_padder), axis=-1
        )
        w_roi = np.take(extended_w, selected_idcs) # no data copy involved
        est = w_roi * est

    return est

@njit(nogil=True)
def stack_2d_jit_array(arr: np.ndarray,
                       repeats: int):
    return arr.repeat(repeats).reshape(-1, repeats).T





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
            (2*l_bound - X)[None,:], Y[:,None], bandwidth
        )
        cum_est += l_mirrored_est
        r_mirrored_est = _gaussian_kernel(
            (2*r_bound - X)[None,:], Y[:,None], bandwidth
        )
        cum_est += r_mirrored_est
        return cum_est
    else:
        # no special compansate operation done in the boundary region
        return in_region_est

@njit(nogil=True)
def jit_argsort_2d(arr: np.ndarray,
                   chunk_upper_bound: int,
                   dtype_max_value: Any=None
                ):
    """
    Args:
    - arr: 2-d array to be sorted
    - chunk_upper_bound: numba does not inherently supports argsort on 2d array
      the work-around adopted is to flatten the input 2-d array into 1d,
      before a penalty should be added to each row to keep the position of each row in 
      the flattened array localized.
      if not specified, the largest value in the input array will be used
    - dtype_max_value: #row * chunk_upper_bound < dtype_max_value
      if this value is not provided, safety of operation will not be guaranteed.
    """
    if chunk_upper_bound is None:
        chunk_upper_bound = np.max(arr)
    num_row, num_col = arr.shape
    if dtype_max_value is not None:
        assert num_row * chunk_upper_bound < dtype_max_value, \
        'Row number of input arr is too large. Dtype overflow will occur. Consider reduce it'
    
    penalized_arr = arr + (np.arange(num_row) * chunk_upper_bound)[:,None]
    flatten_sorted_idcs = penalized_arr.reshape(-1).argsort() 
    revert_sorted_idcs = flatten_sorted_idcs.reshape(num_row, -1) - (np.arange(num_row) * num_col)[:,None]
    return revert_sorted_idcs