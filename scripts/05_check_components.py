
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

    for record in df.record.unique():
        df_record = df.loc[df.record == record].copy()
        for band in df_record.band.unique():
            df_band = df_record.loc[df_record.band == band].copy()
            df_sorted = df_band.sort_values(by="total", ascending=False)

            print(band, df_sorted.comp.values[:4], df_sorted.total.iloc[:4].mean().round(2))


project = "pr_Agency_EBCI"
stage = "test"
sessions = ["03_30 Artem"]

config = {
    "sth": "stt"
}
if __name__ == "__main__":
    for session in sessions:
        folder_input = os.path.join(r"data", project, "features", "csp", stage, session)
        records = os.listdir(folder_input)
        records = [record for record in records if record.find("DATAFRAME") != -1]
        # records = [record for record in records if record.find("4_calib") != -1]


        folder_output = os.path.join(r"data", project, "features", "csp", stage, session)
        os.makedirs(folder_output, exist_ok=True)
        process_records(folder_input, records, folder_output, config)
    