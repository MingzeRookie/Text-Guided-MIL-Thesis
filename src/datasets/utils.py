import torch
import torch.nn.functional as F # 如果需要在这里用，但主要是在Dataset里用

def get_patch_grid_and_maps(coords_cpu: torch.Tensor):
    # coords_cpu: (N, 2) tensor
    if coords_cpu.shape[0] == 0:
        # 返回空的或默认的grid结构
        return torch.empty(0, 0, dtype=torch.long), {}, {}, 0, 0
        
    unique_r_coords = torch.unique(coords_cpu[:, 0])
    unique_c_coords = torch.unique(coords_cpu[:, 1])
    
    r_map = {r.item(): i for i, r in enumerate(unique_r_coords)}
    c_map = {c.item(): i for i, c in enumerate(unique_c_coords)}
    
    grid_rows = len(unique_r_coords)
    grid_cols = len(unique_c_coords)
    
    patch_grid = torch.full((grid_rows, grid_cols), -1, dtype=torch.long, device='cpu')
    for i in range(coords_cpu.shape[0]):
        r_val = coords_cpu[i, 0].item()
        c_val = coords_cpu[i, 1].item()
        # 确保坐标值在映射中存在 (通常unique会保证)
        if r_val in r_map and c_val in c_map:
            r_idx = r_map[r_val]
            c_idx = c_map[c_val]
            if patch_grid[r_idx, c_idx] == -1: # 只记录第一个出现的patch索引
                patch_grid[r_idx, c_idx] = i
        else:
            print(f"Warning: Coordinate value {(r_val, c_val)} not in unique maps. This shouldn't happen.")
            
    return patch_grid, r_map, c_map, grid_rows, grid_cols


def select_windows_from_grid_and_feats(
    coords_all_patches: torch.Tensor,      # (N_total, 2)
    features_all_patches: torch.Tensor,   # (N_total, D)
    cos_similarities_all_patches: torch.Tensor, # (N_total,)
    window_size: int,
    stride: int,
    top_k_percent: float
):
    # 确保所有输入都在CPU上
    coords_cpu = coords_all_patches.cpu()
    features_cpu = features_all_patches.cpu()
    cos_sim_cpu = cos_similarities_all_patches.cpu()

    if coords_cpu.shape[0] == 0:
        return []

    patch_grid, _, _, grid_rows, grid_cols = get_patch_grid_and_maps(coords_cpu)
    if grid_rows == 0 or grid_cols == 0 : # 如果无法构建grid
        return []

    window_scores_data = [] # Store {'score': float, 'patch_features': tensor_K_D}

    for r_start in range(0, grid_rows - window_size + 1, stride):
        for c_start in range(0, grid_cols - window_size + 1, stride):
            window_patch_indices_on_grid = patch_grid[r_start:r_start+window_size, c_start:c_start+window_size]
            # 获取窗口内有效的 *原始 patch 索引* (相对于 features_all_patches)
            valid_original_patch_indices = window_patch_indices_on_grid[window_patch_indices_on_grid != -1].tolist()
            
            if not valid_original_patch_indices:
                continue
            
            current_window_total_sim = cos_sim_cpu[valid_original_patch_indices].sum().item()
            # 提取这些patch的实际特征
            current_window_patch_features = features_cpu[valid_original_patch_indices] # (K, D)
            
            window_scores_data.append({
                'score': current_window_total_sim,
                'patch_features': current_window_patch_features 
            })

    if not window_scores_data:
        return []

    window_scores_data.sort(key=lambda x: x['score'], reverse=True)
    
    num_windows_to_keep = 0
    if len(window_scores_data) > 0:
        num_windows_to_keep = max(1, int(len(window_scores_data) * top_k_percent))
    
    selected_data = window_scores_data[:num_windows_to_keep]
    
    # 返回的是一个列表，每个元素是一个字典，包含被选窗口的 patch 特征
    return selected_data