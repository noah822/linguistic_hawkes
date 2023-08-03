from typing import Any
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import numpy as np

class GridCanvas:
    def __init__(self, num_row, num_col, cmap='rainbow'):
        _, self.axs = plt.subplots(num_row, num_col)
        self.num_row, self.num_col = num_row, num_col

        base_cmap = cm.get_cmap(cmap)
        self.color_spectrum = base_cmap(np.linspace(0, 1, num_row*num_col), alpha=.8)
    
    
    def __iter__(self):
        if self.num_row != 1 and self.num_col != 1:
            for row in range(self.num_row):
                for col in range(self.num_col):
                    plot_color = self.color_spectrum[row * self.num_row + col]
                    yield _HookedSubplot(self.axs[row, col], plot_color)
        else:
            for i in range(self.num_row * self.num_col):
                plot_color = self.color_spectrum[i]
                yield _HookedSubplot(self.axs[i], plot_color)
        

    def legend(self):
        plt.legend()
    def show(self):
        plt.show()
    def tight_layout(self, pad):
        plt.tight_layout(pad=pad)
    

class _HookedSubplot:
    def __init__(self,
                 fig,
                 line_color: str):
        self.fig = fig
        self.line_color = line_color
    
    # overwriting plot
    # intercept plot function
    def __getattribute__(self, name: str) -> Any:
        hooked_plot = super(_HookedSubplot, self).__getattribute__('fig')
        line_color = super(_HookedSubplot, self).__getattribute__('line_color')

        method = hooked_plot.__getattribute__(name)
        if name == 'plot':
            def wrapper(*args, **kwargs):
                return method(*args, **kwargs, color=line_color)
            return wrapper
        else:
            return method
