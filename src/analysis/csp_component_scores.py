import json
import os
from pathlib import Path

import pandas as pd
from numpy import abs as np_abs
from numpy import arange, array

from src.analysis.evaluate_spatial_patterns import (
    calculate_eigenscore,
    score_spatial_patterns_physio,
)
from src.utils.montage_processing import find_ch_idx, get_channel_names, get_weights


BAD_CHANNELS = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]
MONTAGE_PATH = r"./resources/mks64_standard.ced"

CHANNEL_LABELS = get_channel_names(MONTAGE_PATH)
GOOD_CHANNEL_LABELS = [ch for ch in CHANNEL_LABELS if ch not in BAD_CHANNELS]
EEG_CHANNELS = array([find_ch_idx(ch, MONTAGE_PATH) for ch in GOOD_CHANNEL_LABELS])
PHYSIO_ROI_CHANNELS_CONTRA = ("C3", "CP3", "FC3", "C1", "C5", "CP5", "CP1", "P5", "P3")
PHYSIO_ROI_CHANNELS_IPSI = ("C4", "CP4", "FC4", "C2", "C6", "CP6", "CP2", "P6", "P4")

WEIGHTS, _ = get_weights(CHANNEL_LABELS, EEG_CHANNELS, r"./resources/pattern_weights.json")
WEIGHTS_CONTRA, _ = get_weights(CHANNEL_LABELS, EEG_CHANNELS, r"./resources/pattern_weights_contra.json")
WEIGHTS_IPSI, _ = get_weights(CHANNEL_LABELS, EEG_CHANNELS, r"./resources/pattern_weights_ipsi.json")


def get_selected_component_indices(n_components):
    return [0, 1, 2, 3, 4, n_components - 5, n_components - 4, n_components - 3, n_components - 2, n_components - 1]


def calculate_weighted_score(patterns, weights):
    """
    patterns [n_components, n_channels]
    weights dict {ch_idx: weight}
    """
    w = array([weights[key] for key in weights.keys()])
    return patterns @ w


def build_component_assessment(proj_inverse, evals):
    n_components = proj_inverse.shape[1]
    selected_components = get_selected_component_indices(n_components)
    patterns = np_abs(proj_inverse.T[selected_components, :])
    eigscore = calculate_eigenscore(evals[selected_components])

    score = calculate_weighted_score(patterns, WEIGHTS)
    score_contra = calculate_weighted_score(patterns, WEIGHTS_CONTRA)
    score_ipsi = calculate_weighted_score(patterns, WEIGHTS_IPSI)
    physio_scores_contra = score_spatial_patterns_physio(
        patterns=patterns,
        ch_names=GOOD_CHANNEL_LABELS,
        roi_channels=PHYSIO_ROI_CHANNELS_CONTRA,
    )
    physio_scores_ipsi = score_spatial_patterns_physio(
        patterns=patterns,
        ch_names=GOOD_CHANNEL_LABELS,
        roi_channels=PHYSIO_ROI_CHANNELS_IPSI,
    )
    locality = physio_scores_contra["locality"]
    contrast_contra = physio_scores_contra["contrast"]
    contrast_ipsi = physio_scores_ipsi["contrast"]

    physio_boost_locality = 1 + locality
    physio_boost_contra_contrast = 1 + contrast_contra
    physio_boost_ipsi_contrast = 1 + contrast_ipsi

    physio_boost_contra = physio_boost_locality * physio_boost_contra_contrast
    physio_boost_ipsi = physio_boost_locality * physio_boost_ipsi_contrast
    physio_boost = physio_boost_contra + physio_boost_ipsi

    final_score_contra_basic = score_contra * eigscore
    final_score_ipsi_basic = score_ipsi * eigscore

    final_score_basic = final_score_contra_basic + final_score_ipsi_basic
    final_score_contra = final_score_contra_basic * physio_boost_contra
    final_score_ipsi = final_score_ipsi_basic * physio_boost_ipsi

    final_score = final_score_contra + final_score_ipsi

    return {
        "n_comp": array(selected_components),
        "evals": evals[selected_components],
        "eigscore": np_abs(evals[selected_components] - 0.5),
        "eigscore1": eigscore,
        "score": score,
        "score_contra": score_contra,
        "score_ipsi": score_ipsi,
        "locality": locality,
        "contrast": contrast_contra + contrast_ipsi,
        "contrast_contra": contrast_contra,
        "contrast_ipsi": contrast_ipsi,
        "physio_boost_locality": physio_boost_locality,
        "physio_boost_contra_contrast": physio_boost_contra_contrast,
        "physio_boost_ipsi_contrast": physio_boost_ipsi_contrast,
        "physio_boost": physio_boost,
        "physio_boost_contra": physio_boost_contra,
        "physio_boost_ipsi": physio_boost_ipsi,
        "final_score_contra_basic": final_score_contra_basic,
        "final_score_ipsi_basic": final_score_ipsi_basic,
        "final_score_basic": final_score_basic,
        "final_score_contra": final_score_contra,
        "final_score_ipsi": final_score_ipsi,
        "final_score": final_score,
    }


def extract_record_name(filename, metadata_csp):
    reg_alpha = f'reg{metadata_csp["alpha"]}'
    return filename[filename.find(reg_alpha) + len(reg_alpha) + 1 :]


def build_component_assessment_table(proj_inverse, evals, metadata_csp, session, filename):
    base_row = {
        "session": session,
        "record": extract_record_name(filename, metadata_csp),
    }
    for key, value in metadata_csp.items():
        if key == "bands":
            continue
        base_row[key] = value

    scores = build_component_assessment(proj_inverse, evals)
    rows = []
    for i in range(len(scores["n_comp"])):
        row = base_row.copy()
        for metric, values in scores.items():
            row[metric] = round(values[i], 3)
        rows.append(row)

    df_results = pd.DataFrame(rows)
    df_results["band"] = df_results["band"].apply(json.dumps)
    return df_results


def save_component_assessment_table(df_results, output_dir, matrix_filename):
    output_filename = os.path.join(output_dir, "DATAFRAME_" + matrix_filename[len("MATRIX_") : -4] + ".xlsx")
    df_results.to_excel(output_filename, index=False)
    return output_filename
