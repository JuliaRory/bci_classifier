import matplotlib.pyplot as plt
import numpy as np

from src.analysis.spectral_analysis import get_fft_fast

from src.visualization.plot_helpers import get_color_map
newcmp = get_color_map()

def plot_spectrogram(epochs, Fs, fmin=3, fmax=30, baseline=(0, 100), start_shift=500, ch_roi=None, title=None):

    specs = []
    for ep in epochs:
        fft, t, freqs = get_fft_fast(ep, Fs, hop=int(0.1*Fs), window=int(Fs))
        specs.append(fft)

    specs = np.stack(specs) # [n_epochs,  n_freq, n_channels, n_times]
    spec = np.mean(specs, axis=0)  # усреднение по эпохам [n_freq, n_channels, n_times]

    # --- коррекция на бейзлайн ---
    baseline_idx = np.where((t*Fs >= baseline[0]) & (t*Fs <= baseline[1]))[0]
    baseline_mean = spec[:, :, baseline_idx].mean(axis=2, keepdims=True)
    print("MEAN BASELINE", baseline_mean.shape)

    spec = 10*np.log10(spec / (baseline_mean + 1e-12))
    # spec = 10*np.log10(spec) 
    # --- частотный диапазон ---
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[mask]
    spec = spec[mask, :, :]
    
    if ch_roi is None:
        ch_roi = np.arange(spec.shape[1])
    spec = spec[:, ch_roi, :].mean(axis=1).T      # [n_freq, n_times] -> [n_times, n_freq]
    print(spec.shape)
    
    fig, ax = plt.subplots(1, 1, figsize=(18,6))

    # симметричная шкала цвет
    # vlim = np.percentile(np.abs(spec), 98)
    vmin, vmax = -30, 30
    im = ax.imshow(spec, origin="lower", aspect="auto", extent=[t[0], t[-1], freqs[0], freqs[-1]], cmap=newcmp) #, vmin=vmin, vmax=vmax)

    start_idx = np.where(t*Fs >= start_shift)[0][0]
    ax.axvline(t[start_idx], color="black", lw=1)  # линия события

    # подписи осей
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")

    # цветовая шкала
    # --- добавляем отдельную вертикальную colorbar справа от всех графиков ---
    fig.subplots_adjust(right=0.9)  # оставляем место справа
    cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])  # [лево, низ, ширина, высота]
    fig.colorbar(im, cax=cbar_ax, label="ΔPower (dB vs baseline)")
    # fig.colorbar(im, ax=ax.ravel().tolist(), label="ΔPower (dB vs baseline)")

    if title is None:
        title = "Spectrogram"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    plt.show(block=True)
    return fig