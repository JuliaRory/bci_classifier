
import os
import json
from pathlib import Path
import copy
import pandas as pd
from h5py import File
from numpy import array, where, arange, argsort
from itertools import combinations
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from src.utils.montage_processing import get_channel_names

from src.analysis.evaluate_spatial_patterns import score_spatial_patterns_physio, calculate_eigenscore
from scipy.signal import butter

EEG_CHANNELS = arange(64)
bad_channels = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]
labels = get_channel_names(r"./resources/mks64_standard.ced")
EEG_CHANNELS =  [ch for ch in labels if not(ch in bad_channels)]
# EEG_CHANNELS = array([find_ch_idx(ch, r"./resources/mks64_standard.ced") for ch in labels if not(ch in bad_channels)])

def generate_groups(nums=[0, 1, -1, -2], n_features=2):
    def valid(group):
        has_positive = any(x >= 0 for x in group)
        has_negative = any(x < 0 for x in group)
        return has_positive and has_negative

    combs = [g for g in combinations(nums, n_features)]
    
    # combs = [g for g in combinations(nums, n_features) if valid(g)]
    return combs

def get_W(folder, info):
    filename = os.path.join(folder, "MATRIX_" + info + ".hdf")
    with File(filename, "r") as h5f:
        projForward = h5f["projForward"][:]
    return projForward

def save_classifier(folder, folder_output, classifier, sel_comp, info):
    projForward = get_W(folder, info)

    # Веса LDA (для признаков, полученных из CSP)
    w_lda = classifier.coef_[0]  # вектор [компоненты] или [признаки]
    b_lda = classifier.intercept_[0]
    band = info[:info.find("_")]
    
    band = [int(band[1:band.find(",")]), int(band[band.find(",")+2:-1])]
    sos_basic = butter(4, [5, 35], btype="bandpass", output='sos', fs=1000)
    sos = butter(4, band, btype="bandpass", output='sos', fs=1000)

    classifier_data = {
        'spatialW': projForward[:, sel_comp].tolist(),  # веса csp фильтра
        'sos_basic': sos_basic.tolist(),  # коэффициенты SOS фильтра [секции × 6]
        'sos': sos.tolist(),  # коэффициенты SOS фильтра [секции × 6]
        'features_type': "csp",  # тип признаков

        "Cref": None, 
        "inv_sqrt": None, 

        # LDA веса
        'w_lda': w_lda.tolist(),  # веса LDA
        'b_lda': float(b_lda),    # смещение LDA

        "sel_comp": sel_comp, 
        "band": band, 
        'fs': 1000,  # частота дискретизации
        "n_components": len(sel_comp)
    }
    
    # Сохраняем в JSON файл
    output_filename = os.path.join(folder_output, f"feat{sel_comp}_{info}.json")
    with open(output_filename, 'w') as f:
        json.dump(classifier_data, f, indent=4)
    
    print(f"Классификатор сохранен в {output_filename}")

def process_record(full_path, folder_output, config):
    feat_numbers = []
    for n in config["n_feat"]:
        for feats in generate_groups(config["sel_comps"], n):
            feat_numbers.append(feats)
    print(f"Features combinations number is {len(feat_numbers)}.")

    with File(full_path, "r") as h5f:
        features = h5f["features"][:]     # [n_channels, n_components]
        labels = h5f["labels"][:].squeeze()
        config_csp = h5f["metadata_csp"][()]
    config_csp = json.loads(config_csp)
    
    for sel_comp in feat_numbers:
        if config["classifier"] == "lda":
            classifier = LDA()

        classifier.fit(features[:, sel_comp], labels)

        filename = Path(full_path).parts[-1]
        info = filename[filename.find("_")+1:-4]

        save_classifier(os.path.dirname(full_path), folder_output, classifier, sel_comp, info)

def process_records_clf(folder_input, records, folder_output, config):
    for record in records:
        print(f"Record {record}")
        process_record(full_path=os.path.join(folder_input, record), folder_output=folder_output, config=config)
        

project = "pr_Agency_EBCI"
stage = "test"
sessions = ["04_03 Artem"]

project = "pr_Feedback_Quasi"
sessions = ["Julia"]

config = {
    "sel_comps": [0, 1, -1],
    "n_feat": [3], 
    "classifier": "lda"
}

if __name__ == "__main__":
    for session in sessions:
        folder_input = os.path.join(r"data", project, "features", "csp", stage, session)
        records = os.listdir(folder_input)
        records = [record for record in records if record.find("FEATURES") != -1]
        records = [record for record in records if record.find("calib") != -1]

        folder_output = os.path.join(r"models", project, stage, session)
        os.makedirs(folder_output, exist_ok=True)
        process_records_clf(folder_input, records, folder_output, config)
    