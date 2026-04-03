import pandas as pd
import numpy as np

def find_ch_idx(channel, fl_montage):
    import warnings
    df = pd.read_csv(fl_montage, sep='\t')
    idx = df.loc[df['labels'] == channel, 'Number'].values
    if len(idx) > 1:
        warnings.warn(f"Найдено несколько значений ({len(idx)}). Верну первое.", UserWarning)
    return int(idx[0] - 1)

def get_channel_names(fl_montage):
    df = pd.read_csv(fl_montage, sep='\t')
    return df["labels"].values

def get_ch_idxs(channels, fl_montage):
    idxs = [find_ch_idx(ch, fl_montage) for ch in channels]
    return np.array(idxs)

def get_topo_positions(fl_montage):
    df = pd.read_csv(fl_montage, sep='\t')
    th = np.pi / 180 * np.array(df.theta.values)
    df['y'] = np.round(np.array(df.radius.values) * np.cos(th), 2)
    df['x'] = np.round(np.array(df.radius.values) * np.sin(th), 2)
    return df[['x', 'y']].values

def get_good_channels(fl_montage, radius=0.54):
    df = pd.read_csv(fl_montage, sep='\t')
    return df.loc[df.radius <= radius]["labels"].values