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
        
import matplotlib.animation as animation
import os

class GifConverter:
    def __init__(self,
                 interval: int=800,
                 repeat_delay: int=1000):
        self.frame_queue = []
        self._fig, self._axs = plt.subplots()

        self._frame_cnt = 0
        self.interval, self.repeat_delay = interval, repeat_delay

        self._line, = self._axs.plot([], [])
        self._title_display = self._axs.text(
                0.5, 1.01,
                '',
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=self._axs.transAxes
            )  # persisted text pox


        

    def push(self, x, y, title=None):
        # push next frame into queue
        # currently only support plt.plot
        self.frame_queue.append((x, y, title))
    
    def func_painter_wrap(self, frame_idx):
        x, y, title = self.frame_queue[frame_idx]


        self._line.set_data(x, y)
        self._axs.relim()  # recompute x, y limits according to current artist
        self._axs.autoscale_view()

        if title is not None:
            self._title_display.set_text(title)

        return [self._line, self._title_display]
        

    def display(self):
        ani = animation.FuncAnimation(
            self._fig,
            func=self.func_painter_wrap,
            frames=len(self.frame_queue),
            interval=self.interval,
            repeat_delay=self.repeat_delay
        )
        plt.show()
    
    def save(self,
             path: str,
             save_fps: int=None):
        anim = animation.FuncAnimation(
            self._fig,
            func=self.func_painter_wrap,
            frames=len(self.frame_queue),
            interval=self.interval,
            repeat_delay=self.repeat_delay
        )
        save_fps = int(1000 / self.interval) if save_fps is None else save_fps
        writervideo = animation.FFMpegWriter(save_fps)
        anim.save(path, writer=writervideo)