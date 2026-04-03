

import os
import json
from pathlib import Path
import copy
import pandas as pd
from h5py import File
from numpy import array, where, arange, argsort, zeros, exp
from itertools import combinations
from matplotlib.pyplot import close
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score

from src.utils.montage_processing import get_channel_names

from src.analysis.evaluate_spatial_patterns import score_spatial_patterns_physio, calculate_eigenscore
from src.visualization.ROC_curve import plot_roc, plot_roc_with_optimal_threshold, plot_proba

EEG_CHANNELS = arange(64)
bad_channels = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]
labels = get_channel_names(r"./resources/mks64_standard.ced")
EEG_CHANNELS =  [ch for ch in labels if not(ch in bad_channels)]
# EEG_CHANNELS = array([find_ch_idx(ch, r"./resources/mks64_standard.ced") for ch in labels if not(ch in bad_channels)])


def get_decision(x_window, w, b):
    z =  x_window @ w + b   # линейное предсказание
    dec = zeros(len(x_window))
    dec[z > 0] = 1
    return dec

def predict_proba(x_window, w, b):
    z =  x_window @ w + b   # линейное предсказание
    p = 1 / (1 + exp(-z))  # вероятность класса 1

    # or y_pred = (p > 0.5).astype(int)
    return p

def load_model(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def process_record(full_path, folder_output, models, config):
    with File(full_path, "r") as h5f:
        features = h5f["features"][:]     # [n_channels, n_components]
        labels = h5f["labels"][:].squeeze()
        config_csp = h5f["metadata_csp"][()]
    config_csp = json.loads(config_csp)

    for model in models:
        model_data = load_model(model)
        w_lda = model_data["w_lda"]
        b_lda = model_data["b_lda"]

        X = features[:, model_data["sel_comp"]]
        proba = predict_proba(X, w_lda, b_lda)

        # ROC CURVE
        fig, score = plot_roc_with_optimal_threshold(labels, proba)
        filename  = Path(model).parts[-1]
        
        folder = os.path.join(folder_output, "ROC_curve")
        os.makedirs(folder, exist_ok=True)
        filename_output = os.path.join(folder, filename[:-5] + ".png")
        fig.savefig(filename_output, dpi=300, bbox_inches="tight")
        close()

        # PROBABILITY
        dec = get_decision(X, w_lda, b_lda)
        bal_acc = balanced_accuracy_score(labels, dec)
        fig = plot_proba(labels, proba)
        folder = os.path.join(folder_output, "PROBA")
        os.makedirs(folder, exist_ok=True)
        filename_output = os.path.join(folder, filename[:-5] + ".png")
        fig.suptitle(f"Balanced accuracy = {round(bal_acc, 2)}")
        fig.savefig(filename_output, dpi=300, bbox_inches="tight")
        close()


def process_records(folder_input,  folder_models, records, folder_output, config):
    for record in records:
        print(f"Record {record}")
        rec = record[record.rfind("_")-2:-4]
        models = os.listdir(folder_models)
        models = [os.path.join(folder_models, model) for model in models if model.find(rec) != -1]
        band  = record[record.find("_")+1:record.find("]")]
        
        models = [model for model in models if model.find(band) != -1]
        # folder_output = os.path.join(folder_output, rec)
        # os.makedirs(folder_output, exist_ok=True)
        process_record(full_path=os.path.join(folder_input, record), folder_output=folder_output, models=models, config=config)


project = "pr_Agency_EBCI"
stage = "test"
sessions = ["03_30 Artem"]

config = {
    "sel_comps": [0, 1, 2, -1, -2, -3],
    "n_feat": [2, 3, 4], 
    "classifier": "lda"
}
if __name__ == "__main__":
    for session in sessions:
        folder_input = os.path.join(r"data", project, "features", "csp", stage, session)
        records = os.listdir(folder_input)
        records = [record for record in records if record.find("FEATURES") != -1]
        records = [record for record in records if record.find("calib") != -1]

        folder_models = os.path.join(r"models", project, stage, session)

        folder_output = os.path.join(r"results", project, stage, session)
        os.makedirs(folder_output, exist_ok=True)
        process_records(folder_input, folder_models, records, folder_output, config)

        