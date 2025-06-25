# --- START OF FILE 4prepare_bnt_mi_data.py (添加了 StandardScaler 的版本) ---

import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def reconstruct_matrix_from_vector(vector, num_nodes):
    expected_len = num_nodes * (num_nodes - 1) // 2
    if len(vector) != expected_len:
        raise ValueError(f"输入向量长度 {len(vector)} 与期望长度 {expected_len} 不匹配 (节点数: {num_nodes})")
    
    matrix = np.zeros((num_nodes, num_nodes))
    indices = np.triu_indices(num_nodes, k=1)
    matrix[indices] = vector
    matrix = matrix + matrix.T
    return matrix

def load_mi_data_from_ids(subject_id_file, label_dict, label_mapping, fc_dir, num_nodes, dummy_timesteps):
    """
    根据给定的 subject ID 文件，加载 MI 数据并恢复成矩阵。
    
    返回:
        一个包含恢复后特征和标签的字典，如果找不到任何有效数据则返回 None。
    """
    
    with open(subject_id_file, 'r') as f:
        subject_list = [line.strip() for line in f.readlines()]
    print(f"  - 读取到 {len(subject_list)} 个 subject ID。")

    if not subject_list:
        print("  - 警告: subject ID 列表为空，跳过。")
        return None

    valid_timeseries, valid_mi, valid_node_features, valid_labels = [], [], [], []

    for subject_id in tqdm(subject_list, desc=f"加载 {os.path.basename(subject_id_file)}"):
        original_label = label_dict.get(subject_id)
        if original_label is None or original_label not in label_mapping:
            continue

        data_path = os.path.join(fc_dir, subject_id + ".npy")
        if not os.path.exists(data_path):
            continue

        try:
            mi_vector = np.load(data_path)
            mi_matrix = reconstruct_matrix_from_vector(mi_vector, num_nodes)
            
            valid_mi.append(mi_matrix)
            valid_node_features.append(mi_matrix)
            valid_labels.append(label_mapping[original_label])
            
        except (ValueError, FileNotFoundError) as e:
            print(f"\n  - 警告: 处理 {subject_id} 时出错: {e}. 跳过此样本。")
            continue
    

    final_mi_arr = np.stack(valid_mi)
    final_node_feature_arr = np.stack(valid_node_features)
    final_labels_arr = np.array(valid_labels)

    num_samples = len(final_labels_arr)
    final_timeseries_arr = np.zeros((num_samples, dummy_timesteps, num_nodes))


    data_dict = {
        'timeseires': final_timeseries_arr,
        'corr': final_mi_arr,
        'node_feature': final_node_feature_arr,
        'label': final_labels_arr
    }
    print(f"  - 加载完成，共 {num_samples} 个有效样本。")
    return data_dict



def main():

    FC_DIR = r"/home/u500764/data/all_fmri/fc_mi"
    LABEL_CSV = r"/home/u500764/data/all_fmri/merged_labels.csv"
    SPLIT_DIR = r"/home/u500764/projects/svm_mi/data_splits"
    OUTPUT_DIR = r"/home/u500764/projects/svm_mi/bnt_preprocessed_data"

    NUM_NODES = 116
    DUMMY_TIMESTEPS = 150

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = pd.read_csv(LABEL_CSV)
    label_dict = dict(zip(df["subject_id"], df["DX"]))
    label_mapping = {0: 0, 1: 1, 3: 2}


    load_args = (label_dict, label_mapping, FC_DIR, NUM_NODES, DUMMY_TIMESTEPS)
    train_data = load_mi_data_from_ids(os.path.join(SPLIT_DIR, "train_subjects.txt"), *load_args)
    validation_data = load_mi_data_from_ids(os.path.join(SPLIT_DIR, "validation_subjects.txt"), *load_args)
    test_data = load_mi_data_from_ids(os.path.join(SPLIT_DIR, "test_subjects.txt"), *load_args)

    if train_data is None:
        print("致命错误：训练数据加载失败，无法继续。请检查路径和文件。")
        return

    # --- 步骤 5: [核心] 使用 StandardScaler 标准化 node_feature ---
    print("\n--- 步骤 5: 开始标准化 node_feature (基于 MI 矩阵) ---")
    
    scaler = StandardScaler()
    train_features = train_data['node_feature']
    
    # 模型的特征是3D的 (样本数, 节点数, 特征数/节点数)，即 (N, 116, 116)
    # StandardScaler 需要2D输入，我们将其 reshape
    n_samples, n_nodes, n_features = train_features.shape
    train_features_reshaped = train_features.reshape(-1, n_features)
    print(f"  - 将训练特征从 {train_features.shape} reshape 为 {train_features_reshaped.shape} 以进行拟合。")

    # (Fit) 仅使用训练数据来“学习”标准化的参数
    print("  - 正在使用训练数据拟合 (fit) StandardScaler...")
    scaler.fit(train_features_reshaped)
    print("  - Scaler 拟合完成。")

    # (Transform) 使用学习到的参数来“应用”标准化到所有数据集
    print("  - 正在应用 (transform) 标准化到训练、验证和测试集...")

    # 转换训练集
    train_data['node_feature'] = scaler.transform(train_features_reshaped).reshape(n_samples, n_nodes, n_features)
    # MI矩阵也需要被标准化，因为它是模型的一部分输入'corr'
    train_data['corr'] = train_data['node_feature'] 
    
    # 转换验证集
    if validation_data:
        val_features = validation_data['node_feature']
        n, h, w = val_features.shape
        scaled_val_features = scaler.transform(val_features.reshape(-1, w)).reshape(n, h, w)
        validation_data['node_feature'] = scaled_val_features
        validation_data['corr'] = scaled_val_features
    
    # 转换测试集
    if test_data:
        test_features = test_data['node_feature']
        n, h, w = test_features.shape
        scaled_test_features = scaler.transform(test_features.reshape(-1, w)).reshape(n, h, w)
        test_data['node_feature'] = scaled_test_features
        test_data['corr'] = scaled_test_features

    print("  - 所有数据集的 node_feature 和 corr 已标准化。")

    # --- 步骤 6: [重构] 将处理好的数据保存到文件 ---
    print("\n--- 步骤 6: 正在保存标准化后的数据 ---")

    np.save(os.path.join(OUTPUT_DIR, "bnt_train_data.npy"), train_data, allow_pickle=True)
    print(f"  - 标准化后的训练数据已保存到: {os.path.join(OUTPUT_DIR, 'bnt_train_data.npy')}")
    
    if validation_data:
        np.save(os.path.join(OUTPUT_DIR, "bnt_validation_data.npy"), validation_data, allow_pickle=True)
        print(f"  - 标准化后的验证数据已保存到: {os.path.join(OUTPUT_DIR, 'bnt_validation_data.npy')}")

    if test_data:
        np.save(os.path.join(OUTPUT_DIR, "bnt_test_data.npy"), test_data, allow_pickle=True)
        print(f"  - 标准化后的测试数据已保存到: {os.path.join(OUTPUT_DIR, 'bnt_test_data.npy')}")




if __name__ == "__main__":
    try:
        main()
        print("done")
    except Exception as e:
        print("error")
        import traceback
        traceback.print_exc()
        print("error!")