from numba import njit
import numpy as np
from typing import Callable
# wrapp frequently executed CPU intensive function 
# when fitting Hawkes process into jit

@njit(nogil=True)
def bundled_g_compute(
    g: np.ndarray,
    X: np.ndarray,
    shifts: np.ndarray
):
    _, num_sample = X.shape
    rotated_g = indep_2d_roll(np.fliplr(g), shifts)
    # mask = np.repeat(np.arange(num_sample)[None, :], repeats=num_query, axis=0)
    mask = np.arange(num_sample)[None,:] < shifts[:,None]
    return np.sum(X * rotated_g * mask, axis=-1)

# @njit(nogil=True)
# def get_end_point_mask(
#                 end_points: np.ndarray,
#                 N: int,
#                 key: Callable=lambda x, y: x < y):
#     num_query = end_points.shape[0]
#     mask = np.repeat(np.arange(N).reshape(1, -1), repeats=num_query, axis=0)
#     mask = key(mask, end_points[:,None])
#     return mask

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
