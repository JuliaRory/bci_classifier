import numpy as np
import scipy.linalg as la

from src.utils.olivehawkins_robustcov import olivehawkins_robustcov

def compute_csp(epochs1, epochs2, config):
    """
    epochs1, epochs2 : [n_epochs, samples, channels]

    Returns
    -------
    W : spatial filters
    A : spatial patterns (для визуализации)
    eigvals : eigenvalues
    """
    robust=config["robust"]
    concat=config["concat"]

    reg=config["regularization"]
    alpha=config["alpha"]
    
    compute_covariance = lambda data: compute_cov(data) if not robust else compute_robust_cov(data)
    regularize = lambda cov: shrink_cov(cov, alpha=alpha) if reg else cov
    
    calculate_cov = lambda data: epoch_cov(data, compute_covariance, regularize) if not concat else concat_cov(data, compute_covariance, regularize)

    C1 = calculate_cov(epochs1)
    C2 = calculate_cov(epochs2)

    C = C1 + C2

    # whitening
    eigvals, eigvecs = la.eigh(C)
    eigvals[eigvals < 1e-10] = 1e-10

    P = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    S1 = P.T @ C1 @ P

    eigvals, B = la.eigh(S1)

    W = P @ B       # W [channels, components]

    # сортировка
    order = np.argsort(eigvals)
    W = W[:, order]
    eigvals = eigvals[order]    # min - class 1, max - class 2

    # spatial patterns
    A = la.pinv(W).T    # A [channels, components]

    return W, A, eigvals

def epoch_cov(epochs, compute_covariance, regularize):
    covs = []
    for ep in epochs:
        try:
            cov = compute_covariance(ep)
            cov = regularize(cov)
            covs.append(cov)
        except Exception as e:
            print(f"An error occurred: {e}")   
    return np.mean(covs, axis=0)

def concat_cov(epochs, compute_covariance, regularize):
    all_epochs = np.concatenate(epochs, axis=0)
    cov = compute_covariance(all_epochs)
    cov = regularize(cov)
    return cov

def compute_cov(epoch):
    X = epoch - np.mean(epoch, axis=0, keepdims=True)
    C = X.T @ X
    return C / np.trace(C)

def compute_robust_cov(epoch):
    cov = olivehawkins_robustcov(epoch)[0]
    cov /= np.trace(cov)
    return cov

def shrink_cov(C, alpha= 0.1):
    n = C.shape[0]
    mu = np.trace(C) / n
    # маленький α → переобучение
    # большой α → потеря локальности
    return (1 - alpha) * C + alpha * mu * np.eye(n)