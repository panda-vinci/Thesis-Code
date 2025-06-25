import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
print("--- 步骤 0: 库导入完成 ---")



def load_data_from_ids(subject_id_file, label_dict, label_mapping, fc_dir):
    """
    根据给定的 subject ID 文件，加载对应的特征矩阵和标签。
    
    返回:
        一个包含特征和标签的字典，如果找不到任何有效数据则返回 None。
    """
    print(f"\n--- 正在从 {os.path.basename(subject_id_file)} 加载数据 ---")
    
    with open(subject_id_file, 'r') as f:
        subject_list = [line.strip() for line in f.readlines()]
    print(f"  - 读取到 {len(subject_list)} 个 subject ID。")

    if not subject_list:
        print("  - 警告: subject ID 列表为空，跳过。")
        return None

    valid_timeseries, valid_pearson, valid_node_features, valid_labels = [], [], [], []

    for subject_id in tqdm(subject_list, desc=f"加载 {os.path.basename(subject_id_file)}"):
        original_label = label_dict.get(subject_id)

        if original_label is None or original_label not in label_mapping:
            continue

        data_path = os.path.join(fc_dir, subject_id + ".npy")
        if not os.path.exists(data_path):
            continue

        try:
            fc_matrix = np.load(data_path)
            if np.all(fc_matrix == 0):
                print(f"  - {data_path} is all zero, skip")
                continue

            dummy_timeseries = np.zeros_like(fc_matrix) 
            mapped_label = label_mapping[original_label]
            
            valid_timeseries.append(dummy_timeseries)
            valid_pearson.append(fc_matrix)
            valid_node_features.append(fc_matrix)
            valid_labels.append(mapped_label)

        except Exception as e:
            print(f"  - 错误: 加载或处理文件 {data_path} 时出错: {e}. 将跳过。")
            continue
    
    if not valid_labels:
        print(f"  - 错误：在处理 {subject_id_file} 后，没有找到任何有效的样本！")
        return None

    # 将数据堆叠成 Numpy 数组
    final_timeseries_arr = np.stack(valid_timeseries)
    final_pearson_arr = np.stack(valid_pearson)
    final_node_feature_arr = np.stack(valid_node_features)
    final_labels_arr = np.array(valid_labels)

    # 封装成字典返回
    data_dict = {
        'timeseires': final_timeseries_arr,
        'corr': final_pearson_arr,
        'node_feature': final_node_feature_arr,
        'label': final_labels_arr
    }

    return data_dict


def main():
    print("\n--- 步骤 1: 进入 main 函数，设置路径 ---")
    
    # --- 路径设置 ---
    FC_DIR = r"/home/u500764/data/all_fmri/fc_pearson"
    LABEL_CSV = r"/home/u500764/data/all_fmri/merged_labels.csv"
    SPLIT_DIR = r"/home/u500764/projects/svm_pearson/data_splits"
    OUTPUT_DIR = r"/home/u500764/projects/svm_pearson/bnt_preprocessed_data"


    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(LABEL_CSV)
    label_dict = dict(zip(df["subject_id"], df["DX"]))

    label_mapping = {0: 0, 1: 1, 3: 2}

    train_data = load_data_from_ids(os.path.join(SPLIT_DIR, "train_subjects.txt"), label_dict, label_mapping, FC_DIR)
    validation_data = load_data_from_ids(os.path.join(SPLIT_DIR, "validation_subjects.txt"), label_dict, label_mapping, FC_DIR)
    test_data = load_data_from_ids(os.path.join(SPLIT_DIR, "test_subjects.txt"), label_dict, label_mapping, FC_DIR)

    if train_data is None:
        print("error")
        return

    scaler = StandardScaler()


    train_features = train_data['node_feature']
    
    n_samples, n_nodes, n_features = train_features.shape
    train_features_reshaped = train_features.reshape(-1, n_features)
    print(f"  - 将训练特征从 {train_features.shape} reshape 为 {train_features_reshaped.shape} 以进行拟合。")

    print("  - 正在使用训练数据拟合 (fit) StandardScaler...")
    scaler.fit(train_features_reshaped)
    print("Scaler done")

    
    train_data['node_feature'] = scaler.transform(train_features_reshaped).reshape(n_samples, n_nodes, n_features)
    
 
    if validation_data:
        val_features = validation_data['node_feature']
        n, h, w = val_features.shape
        validation_data['node_feature'] = scaler.transform(val_features.reshape(-1, w)).reshape(n, h, w)
    
    if test_data:
        test_features = test_data['node_feature']
        n, h, w = test_features.shape
        test_data['node_feature'] = scaler.transform(test_features.reshape(-1, w)).reshape(n, h, w)


    np.save(os.path.join(OUTPUT_DIR, "bnt_train_data.npy"), train_data, allow_pickle=True)
    print(f"标准化后的训练数据已保存到: {os.path.join(OUTPUT_DIR, 'bnt_train_data.npy')}")
    
    if validation_data:
        np.save(os.path.join(OUTPUT_DIR, "bnt_validation_data.npy"), validation_data, allow_pickle=True)
        print(f"标准化后的验证数据已保存到: {os.path.join(OUTPUT_DIR, 'bnt_validation_data.npy')}")

    if test_data:
        np.save(os.path.join(OUTPUT_DIR, "bnt_test_data.npy"), test_data, allow_pickle=True)
        print(f"标准化后的测试数据已保存到: {os.path.join(OUTPUT_DIR, 'bnt_test_data.npy')}")





if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print("eroor")
        import traceback
        traceback.print_exc()
        print("error")
