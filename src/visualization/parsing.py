import matplotlib.pyplot as plt
import numpy as np

def plot_events(trigger, idxs1, idxs2, idxs3, window_size=200, xrange=None):
    strigger = np.convolve(trigger, np.ones(window_size, dtype=int), 'valid')

    n = 1
    fig, ax = plt.subplots(n, 1, figsize=(15, 3), sharex=True)

    ax.plot(trigger * window_size, label="trigger")
    ax.plot(strigger, label="sum")

    colors = ['red', 'orange', 'green']
    for color, idxs in zip(colors, [idxs1, idxs2, idxs3]):
        for idx in idxs:
            ax.axvline(idx[0], color=color)
            ax.axvline(idx[1], color=color)

    ax.legend()

    ax.grid()
    if xrange is not None:
        ax.set_xlim(xrange) 
    plt.show(block=True)
    return fig