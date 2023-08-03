from typing import (
    Callable,
    Tuple,
    Any,
    List,
    Union
)

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
def normalize(t: np.ndarray):
    return t / sum(t)

def get_end_point_mask(
                end_points: np.ndarray,
                N: int,
                key: Callable=lambda x, y: x < y):
    num_query = end_points.shape[0]
    mask = np.repeat(np.arange(N).reshape(1, -1), repeats=num_query, axis=0)
    mask = key(mask, end_points[:,None])
    return mask


def indep_roll(arr, shifts, axis=1):
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
    arr = np.swapaxes(arr,axis,-1)
    all_idcs = np.ogrid[[slice(0,n) for n in arr.shape]]

    # Convert to a positive shift
    shifts[shifts < 0] += arr.shape[-1] 
    all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]

    result = arr[tuple(all_idcs)]
    arr = np.swapaxes(result,-1,axis)
    return arr

def pairwise_difference(X: np.ndarray):
    num_sample = len(X)
    diff = np.repeat(X[None,:], repeats=num_sample, axis=0) - X[:,None]
    return diff[np.triu_indices(num_sample, k=1)]

def diagonal_wise_sum(X, k=0, upper=True):
    row, col = X.shape
    assert row == col
    masked = None
    if upper:
        masked = np.triu(X)
    else:
        masked = np.tril(X)
    rotated_x = indep_roll(masked, shifts=row-np.arange(row))
    return np.sum(rotated_x[:,k:], axis=0)