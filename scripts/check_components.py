
import os
import json
from pathlib import Path
import copy
import pandas as pd
from h5py import File
from numpy import array, where, arange, argsort


def process_records(folder_input, records, folder_output, config):
    df_all = []
    for record in records:
        # print(f"Record {record}")
        # process_record(full_path=os.path.join(folder_input, record), folder_output=folder_output, config=config)
        df = pd.read_excel(os.path.join(folder_input, record))
        df_all.append(df)
    df = pd.concat(df_all)

    df_all = []
    for record in df.record.unique():
        df_record = df.loc[df.record == record].copy()
        for band in df_record.band.unique():
            df_band = df_record.loc[df_record.band == band].copy()
            df_all.append(df_band)
            # df_sorted_contra = df_band.sort_values(by="final_score_contra", ascending=False)
            # df_sorted_ipsi = df_band.sort_values(by="final_score_ipsi", ascending=False)
            # print(record, band)
            # print("CONTRA:", df_sorted_contra.n_comp.values[:4], df_sorted_contra.final_score_contra.iloc[:4].values)
            # print("IPSI:", df_sorted_ipsi.n_comp.values[:4], df_sorted_ipsi.final_score_ipsi.iloc[:4].values)
    df_all = pd.concat(df_all, ignore_index=True)
    # band_contra = df_all.loc[df_all.final_score_contra == df_all.final_score_contra.max()]["band"]
    df_sorted = df_all.sort_values(by="final_score_contra", ascending=False)

    print(df_all)
    print("CONTRA", df_sorted.loc[df_sorted.final_score_contra > 1][["band", "n_comp", "final_score_contra"]])
    df_sorted = df_all.sort_values(by="final_score_ipsi", ascending=False)
    print("IPSI", df_sorted.loc[df_sorted.final_score_ipsi > 1][["band", "n_comp", "final_score_ipsi"]])

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
        records = [record for record in records if record.find("DATAFRAME") != -1]
        records = [record for record in records if record.find("1_calib") != -1]


        folder_output = os.path.join(r"data", project, "features", "csp", stage, session)
        os.makedirs(folder_output, exist_ok=True)
        process_records(folder_input, records, folder_output, config)
    