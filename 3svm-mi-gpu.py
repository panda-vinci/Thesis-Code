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
outdir = "/home/u500764/projects/svm_mi/results"
os.makedirs(outdir, exist_ok=True)
SPLIT_DIR = "/home/u500764/projects/svm_pearson/data_splits"


df_label = pd.read_csv(label_path)
label_dict = dict(zip(df_label["subject_id"], df_label["DX"]))

print("1. 加载预划分的 subject ID 列表...")
with open(os.path.join(SPLIT_DIR, 'train_subjects.txt'), 'r') as f:
    train_subjects_base = [line.strip() for line in f.readlines()]
with open(os.path.join(SPLIT_DIR, 'validation_subjects.txt'), 'r') as f:
    validation_subjects = [line.strip() for line in f.readlines()]
with open(os.path.join(SPLIT_DIR, 'test_subjects.txt'), 'r') as f:
    test_subjects = [line.strip() for line in f.readlines()]

train_subjects_full = train_subjects_base + validation_subjects


def build_dataset_from_ids(subject_ids, feature_dir, label_dict, dataset_name=""):
    X_data, y_data = [], []
    

    VALID_LABELS = {0, 1, 3}
    
    print(f"\n2. 正在为 {dataset_name} 数据集加载特征...")
    subjects_processed = 0
    subjects_skipped = 0
    for sid in tqdm(subject_ids, desc=f"加载 {dataset_name} 数据"):
        original_label = label_dict.get(sid)
        
        if original_label in VALID_LABELS:
            try:
                matrix = np.load(os.path.join(feature_dir, sid + ".npy"))
                feature_vector = matrix.flatten()
                X_data.append(feature_vector)
                y_data.append(original_label)
                subjects_processed += 1
            except FileNotFoundError:
                subjects_skipped += 1
                continue
        else:
            subjects_skipped += 1
            continue

    print(f"  - {dataset_name} 数据集处理完成：加载了 {subjects_processed} 个样本，跳过了 {subjects_skipped} 个样本。")
    return np.array(X_data), np.array(y_data)

X_train, y_train = build_dataset_from_ids(train_subjects_full, feature_dir, label_dict, "训练(Train+Val)")
X_test, y_test = build_dataset_from_ids(test_subjects, feature_dir, label_dict, "测试")


pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('svm', SVC(class_weight='balanced', probability=True)) # kernel, C, gamma 在网格中定义
])

param_grid = [
    {
        'svm__kernel': ['rbf'],
        'svm__C': [0.1 ,1, 5, 10,],
        'svm__gamma': ['scale', 0.001, 0.0001,0.00001]
    },
    {
        'svm__kernel': ['linear'],
        'svm__C': [1, 10, 100]
    },
    {
        'svm__kernel': ['poly'],
        'svm__C': [1, 10, 100],
        'svm__degree': [2, 3, 4]  
    }
]

CV_SPLITS = 5
grid = GridSearchCV(pipeline, param_grid, cv=CV_SPLITS, scoring='f1_macro', n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

print("\n 最优参数：", grid.best_params_)
print(f" 最优交叉验证得分：{grid.best_score_:.4f}")

print("\n4. 在测试集上评估最优模型...")
y_pred = grid.predict(X_test)
y_score = grid.predict_proba(X_test)

report = classification_report(
    y_test, 
    y_pred, 
    target_names=[
        "0 - Typically Developing",
        "1 - ADHD-Combined",
        "3 - ADHD-Inattentive"
    ],
    digits=4
)
print("\n 测试集分类报告：\n" + report)
with open(os.path.join(outdir, "classification_report.txt"), "w") as f:
    f.write("Best Parameters:\n")
    f.write(str(grid.best_params_))
    f.write(f"\n\nCV Folds: {CV_SPLITS}\n")
    f.write("\nClassification Report:\n")
    f.write(report)

f1_macro = f1_score(y_test, y_pred, average='macro')
y_test_bin = label_binarize(y_test, classes=[0, 1, 3]) # 明确指定类别
auc_macro = roc_auc_score(y_test_bin, y_score, average='macro', multi_class='ovr')

print(f"\n F1 score (macro avg): {f1_macro:.4f}")
print(f" AUC (macro avg, OvR): {auc_macro:.4f}")

with open(os.path.join(outdir, "metrics.txt"), "w") as f:
    f.write(f"F1 score (macro avg): {f1_macro:.4f}\n")
    f.write(f"AUC (macro avg, OvR): {auc_macro:.4f}\n")


cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm, display_labels=["TDC", "ADHD-C", "ADHD-I"])
disp.plot(cmap=plt.cm.Blues)
plt.title("SVM + MI Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "confusion_matrix.png"))

print(f"\n result saved to: {outdir}")



result = permutation_importance(
    grid.best_estimator_, 
    X_test, 
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
    scoring='roc_auc_ovr'
)
feature_importance = result.importances_mean



try:
    atlas = fetch_atlas_aal(version='SPM12')

    aal_labels = atlas['labels']
    if aal_labels[0].lower() == 'background':
        aal_labels = aal_labels[1:] # 去掉背景
        
    N_REGIONS = 116 
    if len(aal_labels) != N_REGIONS:
        print(f"警告：从 nilearn 加载的脑区标签数量 ({len(aal_labels)})与期望的 ({N_REGIONS}) 不符。请检查 atlas 版本。")
        aal_labels = [f"Region_{i+1}" for i in range(N_REGIONS)]
    else:
        print("  - AAL 标签加载成功。")
        
except Exception as e:
    print(f"警告：从 nilearn 加载 AAL atlas 失败: {e}。将使用通用名称。")
    aal_labels = [f"Region_{i+1}" for i in range(N_REGIONS)]
    N_REGIONS = 116


sorted_indices = np.argsort(feature_importance)[::-1] 

top_features_list = []
seen_pairs = set() #

for flat_index in sorted_indices:

    row_i = flat_index // N_REGIONS
    col_j = flat_index % N_REGIONS
    

    if row_i == col_j:
        continue

    region1_idx = min(row_i, col_j)
    region2_idx = max(row_i, col_j)
    
    if (region1_idx, region2_idx) in seen_pairs:
        continue
        
    seen_pairs.add((region1_idx, region2_idx))
    
    importance_value = feature_importance[flat_index]
    region1_name = aal_labels[region1_idx]
    region2_name = aal_labels[region2_idx]
    
    top_features_list.append({
        "Region 1": region1_name,
        "Region 2": region2_name,
        "Importance": importance_value,
        "Original Flat Index": flat_index
    })
    
    if len(top_features_list) >= 100:
        break

df_top_features = pd.DataFrame(top_features_list)
top_features_path = os.path.join(outdir, "top_100_feature_pairs.csv")
print(f"  - 保存最重要的100个脑区对到: {top_features_path}")
df_top_features.to_csv(top_features_path, index=False)


disp = ConfusionMatrixDisplay(cm, display_labels=["TDC", "ADHD-C", "ADHD-I"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Best Model")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "confusion_matrix.png"))
plt.close()