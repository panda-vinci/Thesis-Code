import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

feature_dir = "/home/u500764/data/all_fmri/fc_pearson"
label_path = "/home/u500764/data/all_fmri/merged_labels.csv"
outdir = "/home/u500764/projects/svm_pearson/results"
os.makedirs(outdir, exist_ok=True)
SPLIT_DIR = "/home/u500764/projects/svm_pearson/data_splits"

# === 读取标签表 ===
df_label = pd.read_csv(label_path)
label_dict = dict(zip(df_label["subject_id"], df_label["DX"]))


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


X_train, y_train = build_dataset_from_ids(train_subjects_full, feature_dir, label_dict, "训练(Train+Val)")
X_test, y_test = build_dataset_from_ids(test_subjects, feature_dir, label_dict, "测试")



# === Pipeline: 特征选择 + 标准化 + SVM ===
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
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "confusion_matrix.png"))

print(f"\ result saved to: {outdir}")