import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


PHENOTYPICS_TSV = "/home/u500764/data/all_fmri/adhd200_preprocessed_phenotypics.tsv"
FMRI_DIR = "/home/u500764/data/all_fmri/fc_mi" 


OUTPUT_DIR = "/home/u500764/projects/svm_mi/data_splits" 
TRAIN_SUBJECTS_FILE = os.path.join(OUTPUT_DIR, "train_subjects.txt")
TEST_SUBJECTS_FILE = os.path.join(OUTPUT_DIR, "test_subjects.txt")
VALIDATION_SUBJECTS_FILE = os.path.join(OUTPUT_DIR, "validation_subjects.txt")



TEST_SET_SIZE = 0.2
VALIDATION_SET_SIZE_FROM_TRAIN = 0.15 
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


df = pd.read_csv(PHENOTYPICS_TSV, sep='\t')
df = df[(df["DX"] != "pending") & (df["QC_NIAK"] == 1.0) & (df["Gender"].notna())]
df["subject_id_run1"] = df["ScanDir ID"].apply(lambda x: f"fmri_X_{x}_session_1_run1")

fmri_files = set(os.listdir(FMRI_DIR))
def pick_subject_id(row):
    run1 = row["subject_id_run1"] + ".npy"
   
    if run1 in fmri_files:
        return row["subject_id_run1"]
    
df["subject_id"] = df.apply(pick_subject_id, axis=1)
df = df[df["subject_id"].notna()]
df_final = df[["subject_id", "DX"]] 


subjects = df_final["subject_id"].values
labels = df_final["DX"].values



train_val_subjects, test_subjects, train_val_labels, _ = train_test_split(
    subjects,
    labels,
    test_size=TEST_SET_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)

train_subjects, validation_subjects, _, _ = train_test_split(
    train_val_subjects,
    train_val_labels,
    test_size=VALIDATION_SET_SIZE_FROM_TRAIN,
    random_state=RANDOM_STATE, # 保持随机状态一致性
    stratify=train_val_labels
)



# 4. 将划分好的 subject ID 列表保存到文件
def save_list_to_file(filepath, data_list):
    with open(filepath, 'w') as f:
        for item in data_list:
            f.write(f"{item}\n")
    print(f"已保存列表到: {filepath}")

save_list_to_file(TRAIN_SUBJECTS_FILE, train_subjects)
save_list_to_file(VALIDATION_SUBJECTS_FILE, validation_subjects)
save_list_to_file(TEST_SUBJECTS_FILE, test_subjects)



print(pd.Series(labels).value_counts())