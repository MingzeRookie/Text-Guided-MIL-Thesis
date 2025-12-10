import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_train_val_split_csv(
    original_csv_path,
    wsi_id_col,
    inflammation_label_col,
    output_dir,
    train_csv_name="train_inflammation_labels.csv",
    val_csv_name="val_inflammation_labels.csv",
    val_size=0.2,
    random_state=42
):
    """
    读取原始的标签CSV文件，按80:20的比例进行分层抽样，
    并为inflammation任务生成训练集和验证集的CSV文件。

    参数:
        original_csv_path (str): 原始 labels.csv 文件的路径。
        wsi_id_col (str): 原始CSV中包含WSI ID的列名。
        inflammation_label_col (str): 原始CSV中包含inflammation标签 (0-3) 的列名。
        output_dir (str): 保存新生成的CSV文件的目录。
        train_csv_name (str): 输出的训练集CSV文件名。
        val_csv_name (str): 输出的验证集CSV文件名。
        val_size (float): 验证集所占的比例。
        random_state (int): 随机种子，确保划分可复现。
    """
    try:
        df = pd.read_csv(original_csv_path)
        print(f"成功读取原始CSV文件: {original_csv_path}")
        print(f"原始数据共有 {len(df)} 条记录。")
        print(f"数据列名: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"错误: 原始CSV文件未找到: {original_csv_path}")
        return
    except Exception as e:
        print(f"读取原始CSV文件时发生错误: {e}")
        return

    if wsi_id_col not in df.columns:
        print(f"错误: WSI ID列 '{wsi_id_col}' 在CSV文件中未找到。")
        return
    if inflammation_label_col not in df.columns:
        print(f"错误: Inflammation标签列 '{inflammation_label_col}' 在CSV文件中未找到。")
        return

    # 提取 WSI ID 和 inflammation 标签
    ids = df[wsi_id_col]
    inflammation_labels = df[inflammation_label_col]

    print(f"\nInflammation 标签的分布情况:\n{inflammation_labels.value_counts(normalize=True).sort_index()}")

    # 进行分层抽样划分训练集和验证集
    # 我们基于 WSI ID 进行划分，使用 inflammation 标签进行分层
    try:
        train_ids, val_ids, train_labels_stratify, val_labels_stratify = train_test_split(
            ids,
            inflammation_labels, # 用于分层的标签
            test_size=val_size,
            random_state=random_state,
            stratify=inflammation_labels # 确保分层
        )
    except ValueError as e:
        print(f"错误: 进行分层抽样时发生错误: {e}")
        print("这通常发生在某个类别的样本数过少，无法满足分层要求时。")
        print("可以尝试不使用分层 (移除 stratify 参数) 或检查数据分布。")
        return


    # 根据划分出的 ID 筛选原始 DataFrame 中的数据
    train_df = df[df[wsi_id_col].isin(train_ids)].copy() # 使用 .copy() 避免 SettingWithCopyWarning
    val_df = df[df[wsi_id_col].isin(val_ids)].copy()

    # 为了与 YAML 配置中的 `label_column_name: "inflammation_label"` 匹配，
    # 我们将选定的 inflammation 标签列重命名为 "inflammation_label"
    # 并只保留 WSI ID 列和这个新的标签列
    
    # 为训练集准备输出 DataFrame
    train_output_df = pd.DataFrame({
        'slide_id': train_df[wsi_id_col], # YAML 中 wsi_id_column_name 期望是 'slide_id'
        'inflammation_label': train_df[inflammation_label_col] # YAML 中 label_column_name 期望是 'inflammation_label'
    })

    # 为验证集准备输出 DataFrame
    val_output_df = pd.DataFrame({
        'slide_id': val_df[wsi_id_col],
        'inflammation_label': val_df[inflammation_label_col]
    })
    
    # 如果原始 wsi_id_col 不是 'slide_id'，则上面的代码已经将其重命名为 'slide_id'
    # 如果原始 inflammation_label_col 不是 'inflammation_label'，上面的代码也已重命名

    # 创建输出目录 (如果不存在)
    os.makedirs(output_dir, exist_ok=True)

    # 保存新的 CSV 文件
    train_output_path = os.path.join(output_dir, train_csv_name)
    val_output_path = os.path.join(output_dir, val_csv_name)

    train_output_df.to_csv(train_output_path, index=False)
    val_output_df.to_csv(val_output_path, index=False)

    print(f"\n训练集数据共有 {len(train_output_df)} 条记录。")
    print(f"训练集 Inflammation 标签的分布情况:\n{train_output_df['inflammation_label'].value_counts(normalize=True).sort_index()}")
    print(f"训练集CSV已保存到: {train_output_path}")

    print(f"\n验证集数据共有 {len(val_output_df)} 条记录。")
    print(f"验证集 Inflammation 标签的分布情况:\n{val_output_df['inflammation_label'].value_counts(normalize=True).sort_index()}")
    print(f"验证集CSV已保存到: {val_output_path}")

if __name__ == '__main__':
    # --- 请根据您的实际情况修改以下参数 ---
    
    # 1. 原始 labels.csv 文件的完整路径
    #    您之前提到的是 /remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/labels.csv
    original_csv_file = "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/labels.csv"

    # 2. 原始CSV文件中包含WSI ID的列名
    #    根据您上传的 labels.csv 文件，这一列是 'slide_id'
    wsi_id_column = "ID" 

    # 3. 原始CSV文件中包含inflammation标签 (0-3) 的列名
    #    根据您上传的 labels.csv 文件，这一列是 'inflammation'
    inflammation_column = "inflammation"

    # 4. 新生成的训练集和验证集CSV文件的保存目录
    #    建议保存在与原始CSV相同的目录下，或者一个专门的split目录下
    output_directory = "/remote-home/share/lisj/Workspace/SOTA_NAS/BMW/datasets" 
    # 或者 output_directory = "/remote-home/share/lisj/Workspace/SOTA_NAS/datasets/core/inflammation_split/"

    # 5. (可选) 输出文件名
    train_file_name = "train_inflammation_labels.csv"
    val_file_name = "val_inflammation_labels.csv"

    # 6. (可选) 验证集比例
    validation_set_size = 0.2 # 20% 用于验证

    # 7. (可选) 随机种子
    seed = 123 # 使用固定的随机种子确保每次划分结果一致

    print(f"开始处理CSV文件: {original_csv_file}")
    create_train_val_split_csv(
        original_csv_path=original_csv_file,
        wsi_id_col=wsi_id_column,
        inflammation_label_col=inflammation_column,
        output_dir=output_directory,
        train_csv_name=train_file_name,
        val_csv_name=val_file_name,
        val_size=validation_set_size,
        random_state=seed
    )
    print("\n脚本执行完毕。")
