from abc import ABC, abstractmethod
from typing import List, Any, Union, Iterable
import numpy as np

from ._internal import normalize as normalizer

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

    # determine whether a point to be considered is in region-of-interest
    @abstractmethod
    def in_roi(self, center, other) -> bool:
        pass


class GaussianKernel(Kernel):
    def __init__(self,
                 radius,
                 mu=0.,
                 sigma=1.):
        super(GaussianKernel, self).__init__()
        self.radius = radius

        self.weight_mapping = Gaussian(mu=mu, sigma=sigma)

    def in_roi(self, center, other) -> bool:
        return abs(center-other) < 2 * self.radius
    
    def estimate(self, center: int, interval: List[Any]) -> float:
        np_interval = np.array(interval)

        dummy = np.array([interval[center] for _ in range(len(interval))])
        normed_dist = (dummy - np_interval) / self.radius
        w = self.get_weight(normed_dist)

        return sum(w * np_interval) / sum(w)
    

    def get_weight(self, distance_list: List) -> List[float]:
        return self.weight_mapping(distance_list)

class DiscreteGaussianKernel(DiscreteKernel):
    def __init__(self,
                 radius,
                 mu=0.,
                 sigma=1.):
        super(DiscreteGaussianKernel, self).__init__()
        self.radius = radius

        self.weight_mapping = Gaussian(mu=mu, sigma=sigma)

    def in_roi(self, center, other) -> bool:
        return abs(center-other) < 2 * self.radius
    
    def estimate(self, center: int, interval: List[Any]) -> float:
        np_interval = np.array(range(len(interval)))

        dummy = np.array([center for _ in range(len(interval))])
        normed_dist = (dummy - np_interval) / self.radius
        w = self.get_weight(normed_dist)

        return sum(w * np_interval) / sum(w)
    

    def get_weight(self, distance_list: List) -> List[float]:
        return self.weight_mapping(distance_list)


import copy
from ._internal import two_way_bisect
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
                        kernel: DiscreteKernel) -> np.ndarray:
    '''
    approximate histgram using kernel to sliding through the 
    entire region of insterest
    if normalize is set to True,
    the smoothed weighted histogram will be normalized such that 
    sum of the bar value equals to 1
    '''
    smoothed_estimate = []
    num_sample = len(X)

    # TODO bandwith annealing to the max value, then decrease again

    # 1000
    radius = int(kernel.bandwidth / 2)

    region_size = len(X)
    for i in range(region_size):
        # lvalue, rvalue = two_way_bisect(X, i, kernel.in_roi)
        l = 0 if i < radius else (i - radius)
        r = num_sample-1 if i > num_sample-1-radius else (i + radius)

        interval = X[l:r+1]
        focus = i - l
        
        smoothed_estimate.append(
            kernel.estimate(focus, interval)
        )
    
    smoothed_estimate = np.array(smoothed_estimate)
    return smoothed_estimate



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
    

from sklearn.neighbors import KernelDensity
# we only use gaussian kernel here, provided by sklearn package
def get_bounded_pdf_estimator(
    X: np.ndarray,
    l: Union[float, int],
    r: Union[float, int],
    bandwidth: float,
    normalize: bool=True
):  
    assert len(X.shape) == 1
    X = X.reshape(-1, 1)
    kernel_type = 'gaussian'
    # flip X around boundary axis, specified by l and r value
    left_flipped = 2*l - X[::-1]
    right_flipped = 2*r - X[::-1]
    augmented_X = np.concatenate(
        [left_flipped, X, right_flipped],
        axis=0
    )

    # use sklearn kernel density Gaussian estimator
    kde = KernelDensity(kernel=kernel_type, bandwidth=bandwidth).fit(augmented_X)
    
    def hooked_kde(x: np.ndarray):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        prob = np.exp(kde.score_samples(x))
        if normalize:
            prob = normalizer(prob)
        return prob
    
    return hooked_kde






        
        