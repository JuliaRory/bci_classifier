
from scipy.signal import welch
import numpy as np

def plot_psd(epochs_1, epochs_2, Fs, ch_names=None, ch_idxs=None, labels=["rest", "right"]):
    """
    picks: список индексов каналов (например C3, C4)
    """
    def compute_psd_epochs(epochs):
        psds = []
    
        for ep in epochs:
            f, p = welch(ep, fs=Fs, nperseg=2, axis=0)
            psds.append(p)  # [freqs, channels]

        psds = np.array(psds)  # [n_epochs, freqs, channels]
        return f, psds

    f, psd_epochs_1 = compute_psd_epochs(epochs_1, Fs)
    _, psd_epochs_2 = compute_psd_epochs(epochs_2, Fs)

    psd1 = mean(psds, axis=0)
    psd2 = mean(psds, axis=0)

    freq_mask = (f >= 0) & (f <= 30)
    f = f[freq_mask]
    psd1 = psd1[freq_mask]
    psd2 = psd2[freq_mask]

    if picks is None:
        picks = range(psd1.shape[1])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for ch in picks:
        label1 = f"class1-{ch_names[ch]}" if ch_names else f"class1-ch{ch}"
        label2 = f"class2-{ch_names[ch]}" if ch_names else f"class2-ch{ch}"

        ax.plot(f, 10*np.log10(psd1[:, ch]), label=label1)
        ax.plot(f, 10*np.log10(psd2[:, ch]), linestyle="--", label=label2)

    ax.set_xlim(0, 30)  # 👉 фиксируем ось X
    ax.set_xticks(np.arange(0, 30, 2))

    ax.set_xlabel("Hz")
    ax.set_ylabel("Power (dB)")
    ax.legend()
    ax.grid()

    return fig
