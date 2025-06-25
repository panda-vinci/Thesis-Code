import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from nilearn.datasets import fetch_atlas_aal


feature_dir = "/home/u500764/data/all_fmri/fc_mi"
label_path = "/home/u500764/data/all_fmri/merged_labels.csv"
base_outdir = "/home/u500764/projects/svm_mi/results" 
SPLIT_DIR = "/home/u500764/projects/svm_pearson/data_splits"


df_label = pd.read_csv(label_path)
label_dict = dict(zip(df_label["subject_id"], df_label["DX"]))
gender_dict = dict(zip(df_label["subject_id"], df_label["Gender"]))


with open(os.path.join(SPLIT_DIR, 'train_subjects.txt'), 'r') as f:
    train_subjects_base = [line.strip() for line in f.readlines()]
with open(os.path.join(SPLIT_DIR, 'validation_subjects.txt'), 'r') as f:
    validation_subjects = [line.strip() for line in f.readlines()]
with open(os.path.join(SPLIT_DIR, 'test_subjects.txt'), 'r') as f:
    test_subjects = [line.strip() for line in f.readlines()]


train_subjects_full = train_subjects_base + validation_subjects
print("   - Subject ID 加载完成。")

def build_dataset_from_ids(subject_ids, feature_dir, label_dict, dataset_name=""):
    X_data, y_data = [], []
    VALID_LABELS = {0, 1, 3}
    
    subjects_processed = 0
    subjects_skipped = 0
    for sid in tqdm(subject_ids, desc=f"加载 {dataset_name} 数据"):
        original_label = label_dict.get(sid)
        
        if original_label in VALID_LABELS:
            try:
                matrix = np.load(os.path.join(feature_dir, sid + ".npy"))
                X_data.append(matrix.flatten())
                y_data.append(original_label)
                subjects_processed += 1
            except FileNotFoundError:
                subjects_skipped += 1
                continue
        else:
            subjects_skipped += 1
            continue

def run_analysis_for_gender(gender_str, gender_code, train_ids_all, test_ids_all, gender_dict, feature_dir, label_dict, base_outdir):
    """
    为指定性别的数据子集运行完整的分析流程。
    """
    print(f"\n{'='*20} 开始为 {gender_str} 数据集进行分析 {'='*20}")


    gender_outdir = os.path.join(base_outdir, gender_str.lower())
    os.makedirs(gender_outdir, exist_ok=True)
   
    train_ids_gender = [sid for sid in train_ids_all if gender_dict.get(sid) == gender_code]
    test_ids_gender = [sid for sid in test_ids_all if gender_dict.get(sid) == gender_code]

    X_train, y_train = build_dataset_from_ids(train_ids_gender, feature_dir, label_dict, f"{gender_str} 训练集")
    X_test, y_test = build_dataset_from_ids(test_ids_gender, feature_dir, label_dict, f"{gender_str} 测试集")

    # d. 检查数据集是否为空
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"{gender_str} 的训练集或测试集为空，跳过此分析。")
        return


    pipeline = Pipeline([
        ('scaling', StandardScaler()),
        ('svm', SVC(class_weight='balanced', probability=True))
    ])
    param_grid = [
        {'svm__kernel': ['rbf'], 'svm__C': [0.1, 1, 5, 10], 'svm__gamma': ['scale', 0.001, 0.0001, 0.00001]},
        {'svm__kernel': ['linear'], 'svm__C': [1, 10, 100]},
        {'svm__kernel': ['poly'], 'svm__C': [1, 10, 100], 'svm__degree': [2, 3, 4]}
    ]
    CV_SPLITS = 5
    grid = GridSearchCV(pipeline, param_grid, cv=CV_SPLITS, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(f"{gender_str} 最优参数: {grid.best_params_}")
    print(f"{gender_str} 最优交叉验证得分: {grid.best_score_:.4f}")

    print(f"{gender_str} 测试集上评估最优模型...")
    y_pred = grid.predict(X_test)
    y_score = grid.predict_proba(X_test)

    # --- 结果保存 (路径已修改为 gender_outdir) ---
    report = classification_report(y_test, y_pred, target_names=["TD", "ADHD-C", "ADHD-I"], digits=4)
    print(f"\n   {gender_str} 测试集分类报告：\n" + report)
    with open(os.path.join(gender_outdir, "classification_report.txt"), "w") as f:
        f.write(f"Results for Gender: {gender_str}\n\n")
        f.write("Best Parameters:\n" + str(grid.best_params_) + "\n")
        f.write(f"\nCV Folds: {CV_SPLITS}\n\nClassification Report:\n" + report)

    f1_macro = f1_score(y_test, y_pred, average='macro')
    y_test_bin = label_binarize(y_test, classes=[0, 1, 3])
    auc_macro = roc_auc_score(y_test_bin, y_score, average='macro', multi_class='ovr')
    with open(os.path.join(gender_outdir, "metrics.txt"), "w") as f:
        f.write(f"F1 score (macro avg): {f1_macro:.4f}\n")
        f.write(f"AUC (macro avg, OvR): {auc_macro:.4f}\n")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 3])
    disp = ConfusionMatrixDisplay(cm, display_labels=["TD", "ADHD-C", "ADHD-I"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"SVM + MI Confusion Matrix ({gender_str})")
    plt.tight_layout()
    plt.savefig(os.path.join(gender_outdir, "confusion_matrix.png"))
    plt.close() 


    result = permutation_importance(grid.best_estimator_, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring='roc_auc_ovr')
    feature_importance = result.importances_mean
    
    try:
        atlas = fetch_atlas_aal(version='SPM12')
        aal_labels = atlas['labels'][1:] 
        N_REGIONS = 116
        if len(aal_labels) != N_REGIONS: aal_labels = [f"Region_{i+1}" for i in range(N_REGIONS)]
    except Exception:
        aal_labels = [f"Region_{i+1}" for i in range(N_REGIONS)]
        N_REGIONS = 116

    sorted_indices = np.argsort(feature_importance)[::-1]
    top_features_list = []
    seen_pairs = set()
    for flat_index in sorted_indices:
        row_i, col_j = divmod(flat_index, N_REGIONS)
        if row_i == col_j: continue
        region1_idx, region2_idx = min(row_i, col_j), max(row_i, col_j)
        if (region1_idx, region2_idx) in seen_pairs: continue
        seen_pairs.add((region1_idx, region2_idx))
        top_features_list.append({
            "Region 1": aal_labels[region1_idx], "Region 2": aal_labels[region2_idx],
            "Importance": feature_importance[flat_index], "Original Flat Index": flat_index
        })
        if len(top_features_list) >= 100: break
    
    df_top_features = pd.DataFrame(top_features_list)
    top_features_path = os.path.join(gender_outdir, "top_100_feature_pairs.csv")
    df_top_features.to_csv(top_features_path, index=False)
 

GENDERS_TO_RUN = [('Male', 0.0), ('Female', 1.0)]

for gender_name, gender_code_val in GENDERS_TO_RUN:
    run_analysis_for_gender(
        gender_str=gender_name,
        gender_code=gender_code_val,
        train_ids_all=train_subjects_full,
        test_ids_all=test_subjects,
        gender_dict=gender_dict,
        feature_dir=feature_dir,
        label_dict=label_dict,
        base_outdir=base_outdir
    )

print("done")