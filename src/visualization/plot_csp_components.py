from matplotlib import colormaps as cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt 
import numpy as np 

from mne.viz import plot_topomap

viridisBig = cm.get_cmap('jet')
newcmp = "jet" #ListedColormap(viridisBig(np.linspace(0, 1, 15)))

# топографические карты
def plot_topoplot(X, positions, vmin=None, vmax=None, ch_labels=None, axes=None):
        im, cn = plot_topomap(X, positions,  image_interp='cubic', ch_type='eeg', names =ch_labels,
                size=5, show=False, contours=4, sphere=0.5, 
                cmap=newcmp, extrapolate='head', axes=axes, vlim=[vmin, vmax])
        return im

def plot_components(projForward, xy, ch_labels, gs=None, row_ind=None, idxs=None):
        ims = []
        if idxs is None:
                idxs = [0, 1, 2, 3, -4, -3, -2, -1]
        vmin, vmax = np.min(projForward[:, idxs]), np.max(projForward[:, idxs])
        vmin, vmax = -max(abs(vmin), abs(vmax)), max(abs(vmin), abs(vmax))
        
        for i, idx in enumerate(idxs):
                ax_map = plt.subplot(gs[row_ind, i+1])
                im = plot_topoplot(projForward[:, idx], xy, axes=ax_map, 
                        vmin=vmin, vmax=vmax)
                comp_number = idx if idx >= 0 else len(ch_labels)+idx
                ax_map.set_title(f"CSP #{comp_number+1}")
                ims.append(im)
        
        return ims, vmin, vmax

def plot_eigenvalues(eigvals, ax):
        ax.plot(eigvals, "k")
        ax.scatter(range(len(eigvals)), eigvals, marker="o", s=20)
        ax.set_ylim(0, 1)
        ax.set_title("Eigenvalues")

        ediff = np.diff(eigvals)
        ok_steps = np.where(ediff > np.median(ediff) * 5)[0]
        try:
                ok_evalLow_inds = np.arange(np.max(ok_steps[ok_steps < 10]))
                ax.scatter(ok_evalLow_inds, eigvals[ok_evalLow_inds],  label='ERD OK')
        except Exception as e:
                print(e)

        try:
                ok_evalHigh_inds = np.arange(len(eigvals)-1, np.min(ok_steps[ok_steps > 30]), -1)
                ax.scatter(ok_evalHigh_inds, eigvals[ok_evalHigh_inds], label='ERS OK')
        except Exception as e:
                print(e)

        
def plot_10_csp_components(eigenvals, projForward, xy, component_scores=None):
        fig = plt.figure(figsize=(22, 6))
        n = 6
        gs = gridspec.GridSpec(2, n, height_ratios=[1, 1], width_ratios=[2]+[1]*(n-1), wspace=0.05)
        
        plot_eigenvalues(eigenvals, plt.subplot(gs[:, 0]))

        idxs = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
        n_components = projForward.shape[1]
        titles = [idx if idx >= 0 else n_components + idx for idx in idxs]
        vmin, vmax = np.min(projForward[:, idxs]), np.max(projForward[:, idxs])
        vmin, vmax = -max(abs(vmin), abs(vmax)), max(abs(vmin), abs(vmax))
        for i, ch in enumerate(idxs):
                ax = plt.subplot(gs[int(ch<0), abs(ch)+1*int(ch>=0)])
                im, _ = plot_topomap(
                        projForward[:, ch],
                        xy,
                        size=5,
                        axes=ax,
                        show=False,
                        contours=0,
                        sphere=0.6,
                        cmap=newcmp,
                        extrapolate='head',
                        vlim=(vmin, vmax),
                ) #, names=ch_labels)
                title = f'# {titles[i]}'
                if component_scores is not None:
                        title = (
                                f"{title}\n"
                                f"contra {component_scores['final_score_contra'][i]:.2f}\n"
                                f"ipsi {component_scores['final_score_ipsi'][i]:.2f}"
                        )
                ax.set_title(title)

        cbar = fig.colorbar(im, ax=fig.axes)
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
        cbar.ax.yaxis.set_tick_params(labelsize=10)
        return fig


def plot_CSP_components(eigvals, A, positions, ch_labels, row_idx, gs, fig):
        # первый график: линия eigenvalues

        ax0 = plt.subplot(gs[row_idx, 0])
        plot_eigenvalues(eigvals, ax0)
        
        # топоплоты
        ims, vmin, vmax = plot_components(A, positions, ch_labels, gs, row_idx)
        
        # общий colorbar справа от последнего topomap
        ax_map = plt.subplot(gs[row_idx, len(ims)])
        divider = make_axes_locatable(ax_map)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(ims[-1], cax=cax)
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
        cbar.ax.yaxis.set_tick_params(labelsize=10)
