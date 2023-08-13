from typing import (
    Callable,
    Tuple,
    Any,
    List,
    Union
)
import dask.array as da

def two_way_bisect(X: List,
                   center: int,
                   key: Callable[[Any, Any], bool]
            ) -> Tuple:
    # key is expected to be a symmetric function
    wrapped_key = lambda x: key(X[center], x)

    left_bound = _left_bisect(X, 0, center, wrapped_key)
    right_bound = _right_bisect(X, center, len(X)-1, wrapped_key)
    return [left_bound, right_bound]


# try to go left in the first place
# if failed, then search right
def _left_bisect(X, l, r, key):
    if l == r:
        return l
    
    while l < r:
        mid = int((l+r)/2)
        if key(X[mid]):
            r = mid
        else:
            l = mid+1

    return r

# try to go right in the first place
# if failed, then search left
def _right_bisect(X, l, r, key):
    if l == r:
        return l
    
    while l < r:
        mid = int((l+r+1)/2)
        if key(X[mid]):
            l = mid
        else:
            r = mid-1
    
    return l

import numpy as np
def normalize(t: np.ndarray, divide_by_mean: bool=False):
    if divide_by_mean:
        return t / t.mean()
    else:
        return t / sum(t)



def indep_2d_roll_da(
        arr,
        shifts: np.ndarray):
    # independent rolling equivalent for dask array
    # specifically impl for 2-D dask array

    # chunk size of arr should be configured properly
    # with chunks=(1, col_num)

    # secure type of shifts array
    if not isinstance(shifts, np.ndarray):
        shifts = np.array(shifts)


    _, col = arr.shape
    shifts[shifts < 0] += col

    def _roll_single_row(row: np.ndarray, block_id=None):
        idx, _ = block_id
        return np.roll(row, shift=shifts[idx])

    res = da.map_blocks(
        _roll_single_row, arr,
        dtype=arr.dtype
    )

    return res


def pairwise_difference(X: np.ndarray):
    num_sample = len(X)
    diff = np.repeat(X[None,:], repeats=num_sample, axis=0) - X[:,None]
    return diff[np.triu_indices(num_sample, k=1)]

import scipy
def definite_integral(f: Callable, l: float, r: float):
    res, _ = scipy.integrate.quad(f, l, r)
    return res


def columnwise_broadcast_div(x, a):
    return (x.T / a[:None]).T
