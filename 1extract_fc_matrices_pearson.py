import os
import numpy as np
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_aal
from tqdm import tqdm

# === 路径配置 ===
fmri_dir = '/home/u500764/data/all_fmri/fmri' 
output_dir = '/home/u500764/data/all_fmri/fc_pearson' 
os.makedirs(output_dir, exist_ok=True)

# === 获取 AAL atlas ===
atlas = fetch_atlas_aal(version='SPM12')
atlas_path = atlas['maps']  # AAL atlas nii 文件路径

# === 构建 masker：用来提取每个 ROI 的时间序列 ===
masker = NiftiLabelsMasker(labels_img=atlas_path, standardize=True)

# === 遍历所有 .nii 文件，提取 FC 矩阵 ===
for filename in tqdm(os.listdir(fmri_dir)):
    if filename.endswith('.nii') or filename.endswith('.nii.gz'):
        subject_id = filename.replace('.nii', '').replace('.gz', '')
        fmri_path = os.path.join(fmri_dir, filename)

        try:
            # 1. 提取时间序列：shape = (timepoints, ROIs)
            time_series = masker.fit_transform(fmri_path)

            # 2. 计算 Pearson 相关矩阵
            corr_matrix = np.corrcoef(time_series.T)  # shape = (ROIs, ROIs)
            
            # 3. 提取上三角为特征向量（去掉重复）
            feature_vector = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

            # 4. 保存 FC 矩阵到本地
            out_path = os.path.join(output_dir, f"{subject_id}.npy")
            np.save(out_path, corr_matrix)

        except Exception as e:
            print(f"处理 {filename} 时出错：{e}")
