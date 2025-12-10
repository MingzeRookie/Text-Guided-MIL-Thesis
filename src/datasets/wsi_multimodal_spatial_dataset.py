import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class WsiMultimodalSpatialDataset(Dataset):
    """
    用于加载 WSI 多模态空间数据的 Dataset 类。
    每个样本包含:
    - WSI 的所有 patch 的图像特征 (MUSK)
    - 每个 patch 的原始物理坐标
    - 每个 patch 在 WSI 网格中的行列索引 (根据物理坐标和 patch_size 计算)
    - WSI 的整体网格形状 (最大行列索引 + 1)
    - 全局文本特征 (例如“小叶炎症”的 MUSK 文本特征)
    - WSI 级别的标签
    """
    def __init__(self,
                 data_manifest_csv_path,
                 image_feature_dir,
                 text_feature_path,
                 patch_size_pixels, # 例如 [256, 256] (height, width)
                 label_column_name='label',
                 wsi_id_column_name='wsi_id'):
        """
        初始化 Dataset.

        参数:
            data_manifest_csv_path (str): CSV文件的路径, 包含 wsi_id 和标签。
                                         wsi_id 应与图像特征 .pt 文件名（不含扩展名）对应。
            image_feature_dir (str): 存储图像特征 .pt 文件的目录。
            text_feature_path (str): 全局文本特征 .pt 文件的完整路径。
            patch_size_pixels (list or tuple): patch 的像素尺寸 [height, width]。
                                              用于将物理坐标转换为网格索引。
            label_column_name (str): CSV 文件中标签列的名称。
            wsi_id_column_name (str): CSV 文件中 WSI ID 列的名称。
        """
        super().__init__()

        self.image_feature_dir = image_feature_dir
        self.patch_size_h = patch_size_pixels[0]
        self.patch_size_w = patch_size_pixels[1]

        # 1. 加载数据清单 (包含 WSI ID 和标签)
        try:
            self.manifest = pd.read_csv(data_manifest_csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"错误: 数据清单 CSV 文件未找到: {data_manifest_csv_path}")
        
        self.wsi_ids = self.manifest[wsi_id_column_name].tolist()
        self.labels = self.manifest[label_column_name].tolist()

        if len(self.wsi_ids) != len(self.labels):
            raise ValueError("WSI ID 列表和标签列表的长度不匹配。请检查 CSV 文件。")

        # 2. 加载全局文本特征
        try:
            self.text_feature = torch.load(text_feature_path).squeeze().float() # 形状变为 [text_feature_dim]
            if self.text_feature.ndim == 0: # 如果squeeze后变成标量 (单个值)
                 self.text_feature = self.text_feature.unsqueeze(0) # 保持至少1维
            elif self.text_feature.ndim > 1 and self.text_feature.shape[0] == 1: # 如果是 [1, D]
                 self.text_feature = self.text_feature.squeeze(0)


            print(f"成功加载文本特征，形状: {self.text_feature.shape}")
        except FileNotFoundError:
            raise FileNotFoundError(f"错误: 文本特征文件未找到: {text_feature_path}")
        except Exception as e:
            raise IOError(f"加载文本特征时出错 {text_feature_path}: {e}")

        print(f"Dataset 初始化完成。共找到 {len(self.wsi_ids)} 个样本。")

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.wsi_ids)

    def __getitem__(self, idx):
        """
        获取指定索引的单个样本。

        参数:
            idx (int): 样本的索引。

        返回:
            tuple: 包含 (
                image_patch_features,      # Tensor [N_patches, image_feature_dim], float32
                patch_grid_indices,        # Tensor [N_patches, 2], int64 (row, col)
                text_feature,              # Tensor [text_feature_dim], float32
                label,                     # int or float
                grid_shape,                # Tensor [2], int64 (max_rows, max_cols)
                original_patch_coordinates # Tensor [N_patches, 2], int32 (原始物理坐标)
            )
        """
        wsi_id = self.wsi_ids[idx]
        label = self.labels[idx]

        # 1. 加载图像特征和原始物理坐标
        image_pt_path = os.path.join(self.image_feature_dir, f"{wsi_id}.pt")
        try:
            data = torch.load(image_pt_path, map_location='cpu') # 推荐加载到CPU，后续由DataLoader处理设备
        except FileNotFoundError:
            print(f"错误: 图像特征文件 {image_pt_path} 未找到。跳过样本 {wsi_id}。")
            # 返回一个占位符或引发更具体的错误，或在 __init__ 中预先检查所有文件
            # 为简单起见，这里可以返回 None 或引发错误，让 collate_fn 处理
            # 或者，更好的做法是在 __init__ 中过滤掉缺失文件的 wsi_id
            raise FileNotFoundError(f"图像特征文件 {image_pt_path} 未找到。")
        except Exception as e:
            raise IOError(f"加载图像特征时出错 {image_pt_path}: {e}")


        image_patch_features = data['bag_feats'].float()  # 从 float16 转换为 float32
        original_patch_coordinates = data['coords']     # dtype=torch.int32

        # 2. 根据物理坐标和 patch_size 计算网格索引和网格形状
        # 假设 original_patch_coordinates 是 [N_patches, 2]，其中每行是 (y_coord_top_left, x_coord_top_left)
        # 如果您的坐标是 (x,y)，请相应调整索引
        
        # 确保坐标是 LongTensor 以便进行整数除法和后续索引
        physical_coords_y = original_patch_coordinates[:, 0].long()
        physical_coords_x = original_patch_coordinates[:, 1].long()

        # 计算网格索引 (向下取整)
        patch_grid_indices_row = torch.div(physical_coords_y, self.patch_size_h, rounding_mode='floor')
        patch_grid_indices_col = torch.div(physical_coords_x, self.patch_size_w, rounding_mode='floor')
        
        patch_grid_indices = torch.stack([patch_grid_indices_row, patch_grid_indices_col], dim=1) # [N_patches, 2]

        # 计算网格形状 (基于0索引的最大行列号 + 1)
        if patch_grid_indices.numel() > 0: # 确保至少有一个 patch
            max_row_idx = torch.max(patch_grid_indices_row)
            max_col_idx = torch.max(patch_grid_indices_col)
            grid_shape = torch.tensor([max_row_idx.item() + 1, max_col_idx.item() + 1], dtype=torch.long)
        else:
            # 如果没有 patch (例如空的 .pt 文件或过滤后)，返回一个默认的 grid_shape
            grid_shape = torch.tensor([0, 0], dtype=torch.long)
            # 同时也应该没有 image_patch_features 和 patch_grid_indices
            # 这种情况最好在数据预处理或 __init__ 中避免

        # 3. 获取文本特征 (已在 __init__ 中加载并处理)
        # text_feature 已是 [text_feature_dim]

        # 4. 确保标签是适当的类型 (例如，long 类型用于 CrossEntropyLoss)
        if isinstance(label, (int, float)):
            label_tensor = torch.tensor(label, dtype=torch.long if isinstance(label, int) else torch.float)
        else:
            # 根据您的标签类型进行调整
            label_tensor = torch.tensor(label)


        return (
            image_patch_features,
            patch_grid_indices,
            self.text_feature.clone(), # 返回克隆以避免多进程问题
            label_tensor,
            grid_shape,
            original_patch_coordinates
        )

# --- 使用示例 (假设您已准备好相应文件) ---
if __name__ == '__main__':
    # 1. 准备一个虚拟的 labels.csv 文件
    dummy_wsi_ids = [f'wsi_{i:03d}' for i in range(5)]
    dummy_labels = [0, 1, 0, 1, 0]
    dummy_manifest_df = pd.DataFrame({
        'wsi_id': dummy_wsi_ids,
        'label': dummy_labels
    })
    dummy_csv_path = 'dummy_manifest.csv'
    dummy_manifest_df.to_csv(dummy_csv_path, index=False)

    # 2. 准备虚拟的图像特征 .pt 文件
    dummy_image_feature_dir = 'dummy_image_features_spatial'
    os.makedirs(dummy_image_feature_dir, exist_ok=True)
    
    num_patches_example = [100, 150, 80, 200, 120] # 每个WSI的patch数量不同
    image_feature_dim = 1024 # 与您的 MUSK 输出一致
    
    for i, wsi_id in enumerate(dummy_wsi_ids):
        num_p = num_patches_example[i]
        # 模拟 float16 特征
        dummy_bag_feats = torch.randn(num_p, image_feature_dim, dtype=torch.float16)
        # 模拟物理坐标 (y, x)
        dummy_coords_y = torch.randint(0, 180000, (num_p,), dtype=torch.int32)
        dummy_coords_x = torch.randint(0, 70000, (num_p,), dtype=torch.int32)
        dummy_coords = torch.stack([dummy_coords_y, dummy_coords_x], dim=1)
        
        torch.save({'bag_feats': dummy_bag_feats, 'coords': dummy_coords},
                   os.path.join(dummy_image_feature_dir, f"{wsi_id}.pt"))

    # 3. 准备虚拟的文本特征 .pt 文件
    text_feature_dim = 1024 # 与您的 MUSK 输出一致
    dummy_text_feat = torch.randn(1, text_feature_dim) # 假设原始保存为 [1, dim]
    dummy_text_feature_path = 'dummy_text_feature.pt'
    torch.save(dummy_text_feat, dummy_text_feature_path)

    # 4. 定义 patch 大小 (像素)
    patch_h, patch_w = 224, 224 # 假设您的 patch 是 224x224

    # 5. 初始化 Dataset
    print("正在初始化 WsiMultimodalSpatialDataset...")
    try:
        dataset = WsiMultimodalSpatialDataset(
            data_manifest_csv_path=dummy_csv_path,
            image_feature_dir=dummy_image_feature_dir,
            text_feature_path=dummy_text_feature_path,
            patch_size_pixels=[patch_h, patch_w],
            label_column_name='label',
            wsi_id_column_name='wsi_id'
        )
        print(f"Dataset 长度: {len(dataset)}")

        # 6. 获取并打印一个样本
        if len(dataset) > 0:
            print("\n获取第一个样本:")
            sample = dataset[0]
            img_feats, grid_indices, txt_feat, lbl, gr_shape, orig_coords = sample
            
            print(f"  图像 Patch 特征形状: {img_feats.shape}, dtype: {img_feats.dtype}")
            print(f"  Patch 网格索引形状: {grid_indices.shape}, dtype: {grid_indices.dtype}")
            print(f"  Patch 网格索引示例 (前5个): \n{grid_indices[:5]}")
            print(f"  文本特征形状: {txt_feat.shape}, dtype: {txt_feat.dtype}")
            print(f"  标签: {lbl}, dtype: {lbl.dtype}")
            print(f"  网格形状 (max_rows, max_cols): {gr_shape}, dtype: {gr_shape.dtype}")
            print(f"  原始物理坐标形状: {orig_coords.shape}, dtype: {orig_coords.dtype}")
            print(f"  原始物理坐标示例 (前5个): \n{orig_coords[:5]}")

            # 验证 grid_shape 是否合理
            if grid_indices.numel() > 0:
                assert gr_shape[0] >= torch.max(grid_indices[:, 0]) + 1
                assert gr_shape[1] >= torch.max(grid_indices[:, 1]) + 1
                print("  grid_shape 验证通过。")
            else:
                 print("  没有 patch，无法验证 grid_shape。")


    except Exception as e:
        print(f"创建或使用 Dataset 时发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理虚拟文件和目录
        print("\n清理虚拟文件和目录...")
        if os.path.exists(dummy_csv_path):
            os.remove(dummy_csv_path)
        if os.path.exists(dummy_text_feature_path):
            os.remove(dummy_text_feature_path)
        if os.path.exists(dummy_image_feature_dir):
            for f_name in os.listdir(dummy_image_feature_dir):
                os.remove(os.path.join(dummy_image_feature_dir, f_name))
            os.rmdir(dummy_image_feature_dir)
        print("清理完成。")

