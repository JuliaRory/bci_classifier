import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc(y_true, y_proba):
    """
    y_true  : array-like, истинные метки (0 или 1)
    y_proba : array-like, предсказанные вероятности для класса 1
    """
    
    # Вычисляем точки ROC-кривой
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    # Площадь под кривой
    roc_auc = auc(fpr, tpr)
    
    # Строим график
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--')  # случайный классификатор
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc='lower right')
    plt.grid()
    
    plt.show(block=True)
    
    return fpr, tpr, roc_auc

def plot_roc_with_optimal_threshold(y_true, y_proba):
    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # === Находим оптимальный порог ===
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    
    best_threshold = thresholds[best_idx]
    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]
    
    # === График ===
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], linestyle='--')
    
    # Точка оптимального порога
    ax.scatter(best_fpr, best_tpr, s=100, label=f'Best thr = {best_threshold:.3f}')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC-кривая')
    ax.legend(loc='lower right')
    ax.grid()
    
    # plt.show()
    
    return fig, {
        "threshold": best_threshold,
        "tpr": best_tpr,
        "fpr": best_fpr,
        "auc": roc_auc
    }



def plot_proba(y_true, y_proba):
    
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.plot(np.arange(len(y_true)), y_true, label="class", linewidth=5)
    ax.plot(np.arange(len(y_true)), y_proba, label="proba")
    # decision =  get_decision(X, w_lda, b_lda)
    # ax.plot(np.arange(len(y)), decision, label="decision")
    ax.legend()
    ax.grid()
    ax.axhline(0.5, linewidth=0.5, color='darkgrey')
    return fig
        