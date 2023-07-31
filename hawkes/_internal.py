from typing import (
    Callable,
    Tuple,
    Any,
    List
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
