import os
import numpy as np
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_aal
from sklearn.metrics import mutual_info_score
from tqdm import tqdm

# === 路径配置 ===
fmri_dir = '/home/u500764/data/all_fmri/fmri'  
output_dir = '/home/u500764/data/all_fmri/fc_mi' 
os.makedirs(output_dir, exist_ok=True)

# === 获取 AAL atlas ===
atlas = fetch_atlas_aal(version='SPM12')
atlas_path = atlas['maps']  # AAL atlas nii 文件路径

# === 构建 masker：用来提取每个 ROI 的时间序列 ===
masker = NiftiLabelsMasker(labels_img=atlas_path, standardize=True)

def compute_mutual_information_matrix(time_series):
    n_rois = time_series.shape[1]
    mi_matrix = np.zeros((n_rois, n_rois))

  
    binned = np.zeros_like(time_series)
    for i in range(n_rois):
        binned[:, i] = np.digitize(time_series[:, i], bins=np.histogram_bin_edges(time_series[:, i], bins='fd'))

    # 计算互信息
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            mi = mutual_info_score(binned[:, i], binned[:, j])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
    return mi_matrix

# === 遍历所有 .nii 文件 ===
for filename in tqdm(os.listdir(fmri_dir)):
    if filename.endswith('.nii') or filename.endswith('.nii.gz'):
        subject_id = filename.replace('.nii', '').replace('.gz', '')
        fmri_path = os.path.join(fmri_dir, filename)

        try:
            # 1. 提取时间序列
            time_series = masker.fit_transform(fmri_path)

            # 2. 计算互信息矩阵
            mi_matrix = compute_mutual_information_matrix(time_series)

            # 3. 提取上三角特征
            feature_vector = mi_matrix[np.triu_indices_from(mi_matrix, k=1)]

            # 4. 保存为 .npy
            out_path = os.path.join(output_dir, f"{subject_id}.npy")
            np.save(out_path, feature_vector)

        except Exception as e:
            print(f"处理 {filename} 时出错：{e}")
