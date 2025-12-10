import torch
from torch.nn.utils.rnn import pad_sequence

def multimodal_spatial_collate_fn(batch):
    """
    自定义的 collate_fn 用于 WsiMultimodalSpatialDataset。
    处理批次中每个 WSI 可能有不同数量 patch 的情况。

    参数:
        batch (list): 一个列表，其中每个元素是 WsiMultimodalSpatialDataset.__getitem__ 返回的元组:
                      (image_patch_features, patch_grid_indices, text_feature, 
                       label, grid_shape, original_patch_coordinates)

    返回:
        tuple: 包含批处理后的张量:
               (image_patch_features_batch, patch_grid_indices_batch, 
                text_feat_batch, labels_batch, grid_shapes_batch,
                original_patch_coordinates_batch, patch_mask_batch)
    """
    # 按类型分离数据
    image_patch_features_list = []
    patch_grid_indices_list = []
    text_features_list = []
    labels_list = []
    grid_shapes_list = []
    original_patch_coordinates_list = []
    
    num_patches_list = []

    for sample in batch:
        img_feats, grid_indices, txt_feat, lbl, gr_shape, orig_coords = sample
        
        image_patch_features_list.append(img_feats)
        patch_grid_indices_list.append(grid_indices)
        text_features_list.append(txt_feat)
        labels_list.append(lbl)
        grid_shapes_list.append(gr_shape)
        original_patch_coordinates_list.append(orig_coords)
        
        num_patches_list.append(img_feats.shape[0] if img_feats.numel() > 0 else 0)

    # 1. 处理 image_patch_features (变长)
    #    使用 pad_sequence 进行填充，batch_first=True 表示输出形状为 (B, max_len, D)
    #    pad_sequence 默认使用 0 进行填充
    if any(n > 0 for n in num_patches_list): # 只有当至少有一个样本有patch时才padding
        image_patch_features_batch = pad_sequence(image_patch_features_list, batch_first=True, padding_value=0.0)
    else: # 如果所有样本都没有patch
        # 创建一个形状合理的空tensor或特定占位符，取决于模型如何处理这种情况
        # 这里假设如果所有样本都为空，我们仍然创建一个形状为 (B, 0, D) 的张量
        # 或者，如果知道特征维度，可以创建 (B, 0, feature_dim)
        # 为简单起见，如果第一个样本的特征维度已知，用它
        feature_dim = batch[0][0].shape[1] if batch[0][0].numel() > 0 else 0
        image_patch_features_batch = torch.empty(len(batch), 0, feature_dim, dtype=torch.float32)


    # 2. 处理 patch_grid_indices (变长)
    if any(n > 0 for n in num_patches_list):
        patch_grid_indices_batch = pad_sequence(patch_grid_indices_list, batch_first=True, padding_value=0) # 用0填充坐标可能需要mask小心处理
    else:
        patch_grid_indices_batch = torch.empty(len(batch), 0, 2, dtype=torch.long)


    # 3. 处理 original_patch_coordinates (变长)
    if any(n > 0 for n in num_patches_list):
        original_patch_coordinates_batch = pad_sequence(original_patch_coordinates_list, batch_first=True, padding_value=0)
    else:
        original_patch_coordinates_batch = torch.empty(len(batch), 0, 2, dtype=torch.int32)


    # 4. 创建 patch_mask_batch
    max_n_patches = image_patch_features_batch.shape[1] # 从已padding的张量获取最大长度
    patch_mask_batch = torch.zeros(len(batch), max_n_patches, dtype=torch.bool)
    for i, n_patches in enumerate(num_patches_list):
        if n_patches > 0:
            patch_mask_batch[i, :n_patches] = True

    # 5. 处理固定大小的张量 (text_feature, label, grid_shape)
    text_feat_batch = torch.stack(text_features_list, dim=0)
    labels_batch = torch.stack(labels_list, dim=0) # 假设标签已经是tensor
    grid_shapes_batch = torch.stack(grid_shapes_list, dim=0)


    return (
        image_patch_features_batch,
        patch_grid_indices_batch,
        text_feat_batch,
        labels_batch,
        grid_shapes_batch,
        original_patch_coordinates_batch,
        patch_mask_batch
    )

if __name__ == '__main__':
    # --- 准备虚拟数据进行测试 ---
    # 假设我们有一个 WsiMultimodalSpatialDataset 的实例
    # 这里我们直接构造一些符合 Dataset 输出格式的虚拟样本
    
    print("开始测试 multimodal_spatial_collate_fn...")

    # 样本1
    img_feats1 = torch.randn(50, 1024) # 50 patches
    grid_idx1 = torch.randint(0, 20, (50, 2), dtype=torch.long)
    txt_feat1 = torch.randn(1024)
    lbl1 = torch.tensor(0, dtype=torch.long)
    gs1 = torch.tensor([20, 20], dtype=torch.long)
    orig_coords1 = torch.randint(0, 50000, (50,2), dtype=torch.int32)
    sample1 = (img_feats1, grid_idx1, txt_feat1, lbl1, gs1, orig_coords1)

    # 样本2
    img_feats2 = torch.randn(75, 1024) # 75 patches
    grid_idx2 = torch.randint(0, 25, (75, 2), dtype=torch.long)
    txt_feat2 = torch.randn(1024)
    lbl2 = torch.tensor(1, dtype=torch.long)
    gs2 = torch.tensor([25, 22], dtype=torch.long)
    orig_coords2 = torch.randint(0, 60000, (75,2), dtype=torch.int32)
    sample2 = (img_feats2, grid_idx2, txt_feat2, lbl2, gs2, orig_coords2)
    
    # 样本3 (没有 patch)
    img_feats3 = torch.empty(0, 1024) 
    grid_idx3 = torch.empty(0, 2, dtype=torch.long)
    txt_feat3 = torch.randn(1024)
    lbl3 = torch.tensor(0, dtype=torch.long)
    gs3 = torch.tensor([0,0], dtype=torch.long) # 网格形状为0
    orig_coords3 = torch.empty(0,2, dtype=torch.int32)
    sample3 = (img_feats3, grid_idx3, txt_feat3, lbl3, gs3, orig_coords3)


    batch = [sample1, sample2, sample3]

    (image_patch_features_b, patch_grid_indices_b, 
     text_feat_b, labels_b, grid_shapes_b,
     original_patch_coordinates_b, patch_mask_b) = multimodal_spatial_collate_fn(batch)

    print("\n--- 批处理后输出形状 ---")
    print(f"Image Patch Features Batch: {image_patch_features_b.shape}")
    print(f"Patch Grid Indices Batch: {patch_grid_indices_b.shape}")
    print(f"Text Features Batch: {text_feat_b.shape}")
    print(f"Labels Batch: {labels_b.shape}")
    print(f"Grid Shapes Batch: {grid_shapes_b.shape}")
    print(f"Original Patch Coordinates Batch: {original_patch_coordinates_b.shape}")
    print(f"Patch Mask Batch: {patch_mask_b.shape}")

    print("\n--- 批处理后输出数据类型 ---")
    print(f"Image Patch Features Batch dtype: {image_patch_features_b.dtype}")
    print(f"Patch Grid Indices Batch dtype: {patch_grid_indices_b.dtype}")
    print(f"Text Features Batch dtype: {text_feat_b.dtype}")
    print(f"Labels Batch dtype: {labels_b.dtype}")
    print(f"Grid Shapes Batch dtype: {grid_shapes_b.dtype}")
    print(f"Original Patch Coordinates Batch dtype: {original_patch_coordinates_b.dtype}")
    print(f"Patch Mask Batch dtype: {patch_mask_b.dtype}")

    # 验证形状
    batch_size = len(batch)
    max_len = 75 # 在这个例子中，样本2有最多的patch (75)
    
    assert image_patch_features_b.shape == (batch_size, max_len, 1024)
    assert patch_grid_indices_b.shape == (batch_size, max_len, 2)
    assert text_feat_b.shape == (batch_size, 1024)
    assert labels_b.shape == (batch_size,)
    assert grid_shapes_b.shape == (batch_size, 2)
    assert original_patch_coordinates_b.shape == (batch_size, max_len, 2)
    assert patch_mask_b.shape == (batch_size, max_len)

    # 验证掩码
    print("\n--- 掩码验证 ---")
    print(f"Patch Mask for sample 1 (first 10): {patch_mask_b[0, :10]}")
    print(f"Sum of mask for sample 1: {patch_mask_b[0].sum()} (Expected: 50)")
    assert patch_mask_b[0].sum() == 50
    
    print(f"Patch Mask for sample 2 (first 10): {patch_mask_b[1, :10]}")
    print(f"Sum of mask for sample 2: {patch_mask_b[1].sum()} (Expected: 75)")
    assert patch_mask_b[1].sum() == 75

    print(f"Patch Mask for sample 3 (first 10): {patch_mask_b[2, :10]}")
    print(f"Sum of mask for sample 3: {patch_mask_b[2].sum()} (Expected: 0)")
    assert patch_mask_b[2].sum() == 0


    print("\nmultimodal_spatial_collate_fn 测试成功!")
