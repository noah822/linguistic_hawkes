from abc import ABC, abstractmethod
from typing import List, Any, Union, Iterable
import numpy as np

from .dist import Gaussian



class Kernel(ABC):
    def __init__(self):
        super(Kernel, self).__init__()
    @abstractmethod
    def estimate(self,
                 center: int,
                 interval: List[Any]) -> float:
    # kernel estimator on a given interval at given point
    # - center: index of point to be evaluated in the interval
    # - interval: interval surrounds central point
        pass

class DiscreteKernel(Kernel):
    def __init__(self):
        super(DiscreteKernel, self).__init__()

    @abstractmethod
    def get_weight(self,
                   distance_list: List) -> List[float]:
        pass


class DiscreteGaussianKernel(DiscreteKernel):
    def __init__(self,
                 radius,
                 mu=0.,
                 sigma=1.):
        super(DiscreteGaussianKernel, self).__init__()
        self.radius = radius

        self.weight_mapping = Gaussian(mu=mu, sigma=sigma)
    
    def estimate(self, center: int, interval: List[Any]) -> float:
        np_interval = np.array(interval)

        dummy = np.array([center for _ in range(len(interval))])
        normed_dist = (np.array(range(len(interval))) - dummy) / self.radius
        w = self.get_weight(normed_dist)

        return sum(w * np_interval)

    def get_weight(self, distance_list: List) -> List[float]:
        return self.weight_mapping(distance_list)


import copy
def coarse_grain_series(
    series: List, 
    interval: int,
    start: int=None,
    end: int=None
):
    if isinstance(series, (list, tuple,)):
        series = np.array(series)

    if start is None:
        start = min(series)
    if end is None:
        end = max(series)
    bin_num = int((end-start) / interval)

    # by default remaining part of series will be discarded
    insterest_region = copy.deepcopy(series[start: start+bin_num*interval])
    counts, bin = np.histogram(insterest_region, bins=bin_num)

    psedo_label = list(range(len(counts)))

    return counts, bin, psedo_label


def pdf_kernel_estimate(X: List,
                        kernel: DiscreteKernel=None,
                        max_bandwidth: int=10,
                        normalize=True) -> np.ndarray:
    '''
    approximate histgram using kernel to sliding through the 
    entire region of insterest
    if normalize is set to True,
    the smoothed weighted histogram will be normalized such that 
    sum of the bar value equals to 1
    '''
    smoothed_estimate = []

    # TODO bandwith annealing to the max value, then decrease again

    # 1000

    # by default, boundary size if half of bandwidth size
    boundary_size = int(max_bandwidth / 2)
    half_bd = int(max_bandwidth / 2)

    if kernel is None:
        kernel = DiscreteGaussianKernel(
            radius=half_bd / 3,
            mu=0.,
            sigma=1.
        )

    region_size = len(X)
    for i in range(region_size):
        if _in_boundary_region(i, boundary_size, region_size):
            if i < boundary_size: # left boundary
                interval = X[:boundary_size]
                focus = i
            else: # right boundary
                interval = X[-boundary_size:]
                focus = i - (region_size - boundary_size)
            
        else: # not in boundary region
            interval = X[i-half_bd:i+half_bd]
            focus = half_bd
        
        smoothed_estimate.append(
            kernel.estimate(focus, interval)
        )
    
    smoothed_estimate = np.array(smoothed_estimate)
    if normalize:
        smoothed_estimate = smoothed_estimate / sum(smoothed_estimate)
        print(sum(smoothed_estimate))
    return smoothed_estimate

def _in_boundary_region(i, boundary_size, size) -> bool:
    if i < boundary_size or i > (size - boundary_size):
        return True
    else:
        return False




class SlideWindow(ABC):
    def __init__(self):
        super(SlideWindow, self).__init__()
    
    @abstractmethod
    def operate(self, x: List[Any]):
        pass

    def map(self, 
            x: Iterable,
            kernel_size: int,
            stride=1) -> List[Any]:
        return self._slide_impl(x, kernel_size, stride)

    def _slide_impl(self,
                    x: Iterable,
                    kernel_size: int,
                    stride: int) -> List[Any]:
        res = []
        for i in range(0, len(x)-(kernel_size-1), stride):
            window = x[i:i+kernel_size]
            print(window)
            res.append(self.operate(window))
        return res




        
        