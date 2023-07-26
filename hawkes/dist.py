import numpy as np
import matplotlib.pyplot as plt

class Gaussian:
    def __init__(self,
                 mu: float=.0,
                 sigma: float=1.):
        self.mu, self.sigma = mu, sigma
    
    def __call__(self, x):
        if isinstance(x, (list, tuple)):
            x = np.array(x)
        return self._pdf(x)

    def _pdf(self, x):
        coef = 1 / np.sqrt(2 * np.pi * self.sigma**2)
        return coef * np.exp(
            - ((x-self.mu)**2 / 2*self.sigma**2)
        )
    
    def plot(self):
        '''Plot pdf of the distribution
        '''
        X = np.linspace(-4, 4, 5000)
        Y = self._pdf(X)
        plt.plot(X, Y)
        plt.show()


    


