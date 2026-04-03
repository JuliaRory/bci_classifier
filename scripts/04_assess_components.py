
import os
import json
from pathlib import Path
import copy
import pandas as pd
from h5py import File
from numpy import array, where, arange, argsort


from src.utils.montage_processing import get_channel_names

from src.analysis.evaluate_spatial_patterns import score_spatial_patterns_physio, calculate_eigenscore


EEG_CHANNELS = arange(64)
bad_channels = ["FT9", "TP9", "T7", "AF7", "AF8", "FT10", "TP10", "T8"]
labels = get_channel_names(r"./resources/mks64_standard.ced")
EEG_CHANNELS =  [ch for ch in labels if not(ch in bad_channels)]
# EEG_CHANNELS = array([find_ch_idx(ch, r"./resources/mks64_standard.ced") for ch in labels if not(ch in bad_channels)])


def process_record(full_path, folder_output, config):
    with File(full_path, "r") as h5f:
        projInverse = h5f["projInverse"][:]     # [n_channels, n_components]
        # projForward = h5f["projForward"][:]
        evals = h5f["evals"][:]
        metadata_csp = h5f['metadata_csp'][()]
    ch_len = projInverse.shape[0]
    sel_comp = [0, 1, 2, 3, 4, ch_len-5, ch_len-4, ch_len-3, ch_len-2, ch_len-1]

    projInverse = projInverse.T                 # (n_components, n_channels)
    scores = score_spatial_patterns_physio(
        patterns=projInverse[sel_comp, :],
        ch_names=EEG_CHANNELS, 
        roi_channels=["FC1", "FC3", "FC5", "C1", "C3", "C5", "CP1", "CP3", "CP5"]
    )
    
    metadata_csp = json.loads(metadata_csp)
    filename = Path(full_path).parts[-1]
    reg_alpha = f"reg{metadata_csp["alpha"]}"
    record = filename[filename.find(reg_alpha)+len(reg_alpha) + 1: ]
    
    results = {
        "session": Path(full_path).parts[-2],
        "record": record
    }
    for key in metadata_csp.keys():
        if key == "bands":
            continue
        results[key] = metadata_csp[key]
    
    eigscore = calculate_eigenscore(evals)
    df_results = []
    for i, comp in enumerate(sel_comp):
        res = results.copy()
        res["comp"] = comp
        res["evals"] = round(evals[sel_comp[i]], 3)
        res['escore'] = eigscore[sel_comp[i]]
        for metric in scores.keys():
            res[metric] = round(scores[metric][i], 3)
        df_results.append(res)

    df_results = pd.DataFrame(df_results)
    df_results["band"] = df_results["band"].apply(json.dumps)
    filename = filename[len("MATRIX_"):]
    output_filename = os.path.join(folder_output, "DATAFRAME_"+filename[:-4]+".xlsx")
    df_results.to_excel(output_filename, index=False)


def process_records(folder_input, records, folder_output, config):
    for record in records:
        print(f"Record {record}")
        process_record(full_path=os.path.join(folder_input, record), folder_output=folder_output, config=config)


project = "pr_Agency_EBCI"
stage = "test"
sessions = ["04_03 Artem"]

config = {
    "sth": "stt"
}
if __name__ == "__main__":
    for session in sessions:
        folder_input = os.path.join(r"data", project, "features", "csp", stage, session)
        records = os.listdir(folder_input)
        records = [record for record in records if record.find("MATRIX") != -1]
        # records = [record for record in records if record.find("4_calib") != -1]


        folder_output = os.path.join(r"data", project, "features", "csp", stage, session)
        os.makedirs(folder_output, exist_ok=True)
        process_records(folder_input, records, folder_output, config)
    