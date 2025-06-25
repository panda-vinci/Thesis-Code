import pandas as pd
import numpy as np
import os

df = pd.read_csv("/home/u500764/data/all_fmri/adhd200_preprocessed_phenotypics.tsv", sep='\t')

df = df[(df["DX"] != "pending") & (df["QC_NIAK"] == 1.0) & (df["Gender"].notna())]

df["subject_id_run1"] = df["ScanDir ID"].apply(lambda x: f"fmri_X_{x}_session_1_run1")
df["subject_id_run2"] = df["ScanDir ID"].apply(lambda x: f"fmri_X_{x}_session_1_run2")


fmri_dir = "/home/u500764/data/all_fmri/fc_pearson"  
fmri_files = set(os.listdir(fmri_dir))

def pick_subject_id(row):
    run1 = row["subject_id_run1"] + ".npy"
    run2 = row["subject_id_run2"] + ".npy"
    if run1 in fmri_files:
        return row["subject_id_run1"]
    elif run2 in fmri_files:
        return row["subject_id_run2"]
    else:
        return np.nan

df["subject_id"] = df.apply(pick_subject_id, axis=1)


df = df[df["subject_id"].notna()]

# 6. 提取需要的列
df_final = df[["subject_id", "DX", "Gender"]]

print(df_final.head())
df_final.to_csv("/home/u500764/data/all_fmri/merged_labels.csv", index=False)
