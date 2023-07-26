from typing import List, Callable, Any

def bisect(a: List,
                num: int,
                left: int,
                right: int,
                key: Callable[[Any, Any], bool]=None) -> int:
    '''
    Bisection find
    - a: List/Random-Accessiable object to be searched
    - num: object to be inserted
    - left: left value of search range
    - right: right value of search range
    - key: Callable, taking to arguments (left-object, num) -> bool
      if num can be inserted after left-object return True,
      otherwise False
      By default:
      a is assumed to have been sorted in reverse order,
      key is default to lambda x, y: x < y
    '''
    if key is None:
        key = lambda x, y: x<y
    lvalue = left; rvalue = right

    mid = int((lvalue+rvalue)/2)
    while lvalue < rvalue:
        if key(a[mid], num): # if can be inserted, move right-ward
            lvalue = mid+1
        else:
            rvalue = mid-1
        mid = int((lvalue+rvalue)/2)
    
    if key(a[mid], num):
        return mid + 1
    else:
        return mid
    