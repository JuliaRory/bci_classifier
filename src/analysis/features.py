import numpy as np

def get_csp_features(epochs_csp):
    """
    epochs_csp: [n_epochs, samples, components]
    """
    feats = []

    for ep in epochs_csp:
        var = np.var(ep, axis=0)
        var /= np.sum(var)
        feats.append(np.log(var))

    return np.array(feats)