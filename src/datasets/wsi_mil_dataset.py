# src/datasets/wsi_mil_dataset.py
import torch
from torch.utils.data import Dataset
import os
import torch.nn.functional as F
from pathlib import Path
from .utils import get_patch_grid_and_maps, select_windows_from_grid_and_feats # 从同级utils导入

class WsiMilDataset(Dataset):
    def __init__(self, patch_feature_dir: str,global_text_feature_path: str, case_id_list: list, labels_dict : dict, config: object):
        super().__init__()

        self.patch_feature_dir = Path(patch_feature_dir)
        self.global_text_feature = torch.load(global_text_feature_path, map_location='cpu').float()
        self.case_id_list = case_id_list
        self.labels_dict = labels_dict

        self.window_size = config.get('window_size', 3)
        self.stride = config.get('stride', 3)
        self.top_k_percent = config.get('top_k_percent', 0.5)
        self.feature_dim = config.get('feature_dim', 1024)

        print(f"Dataset initialized with {len(self.case_id_list)} cases.")
        print(f"Global text feature loaded, shape: {self.global_text_feature.shape}")

    def __len__(self):
        return len(self.case_id_list)

    def _load_and_format_patch_data(self, case_id: str):
        case_pt_path = self.patch_feature_dir / f"{case_id}.pt"
        if not os.path.exists(case_pt_path):
            return torch.empty(0, self.feature_dim, dtype=torch.float32), \
                     torch.empty(0, 2, dtype=torch.int32) 
        feature_dict = torch.load(case_pt_path, map_location='cpu')
        if not feature_dict:
            print(f"Warning: No features found for case {case_id}.")
            return torch.empty(0, self.feature_dim, dtype=torch.float32), \
                        torch.empty(0, 2, dtype=torch.int32)
        patch_filenames = list(feature_dict.keys())
        try:
            bag_feats_list = [feature_dict[fname].float() for fname in patch_filenames]
            if not bag_feats_list: # 如果所有文件名都无法提取特征（不太可能，但作为检查）
                 raise ValueError("No features could be extracted from feature_dict")
            bag_feats = torch.stack(bag_feats_list, dim=0) # (N, D)
        except Exception as e:
            print(f"Error stacking features for case {case_id}: {e}. Filenames: {patch_filenames[:5]}")
            # 如果堆叠失败，可能因为某些特征为空或维度不一致
            return torch.empty(0, self.feature_dim, dtype=torch.float32), \
                   torch.empty(0, 2, dtype=torch.int32)

        coords_list = []
        valid_indices_for_feats = [] # 用于处理无法解析坐标的情况
        for i, fname in enumerate(patch_filenames):
            try:
                # 假设文件名为 "col_row.png"
                name_without_ext = Path(fname).stem # "col_row"
                col_str, row_str = name_without_ext.split('_')
                coords_list.append([int(col_str), int(row_str)])
                valid_indices_for_feats.append(i)
            except ValueError:
                print(f"Warning: Could not parse coordinates from filename '{fname}' for case {case_id}. Skipping this patch.")
        
        if not coords_list: # 如果所有坐标都无法解析
            print(f"Warning: No valid coordinates found for case {case_id}")
            return torch.empty(0, self.feature_dim, dtype=torch.float32), \
                   torch.empty(0, 2, dtype=torch.int32)

        # 确保bag_feats只包含那些坐标成功解析的patch
        bag_feats = bag_feats[valid_indices_for_feats]
        coords = torch.tensor(coords_list, dtype=torch.int32) # (N, 2)

        if bag_feats.shape[0] != coords.shape[0]:
            # This should not happen if logic is correct, but as a safeguard
            print(f"Critical Error: Mismatch in feature ({bag_feats.shape[0]}) and coord ({coords.shape[0]}) counts for case {case_id} after parsing.")
            # Fallback to empty to prevent further issues
            return torch.empty(0, self.feature_dim, dtype=torch.float32), \
                   torch.empty(0, 2, dtype=torch.int32)
                   
        return bag_feats, coords            
    
    def __getitem__(self, idx: int):
        case_id = self.case_id_list[idx]
        label = torch.tensor(self.labels_dict.get(case_id, -1), dtype=torch.long)
        bag_feats, coords = self._load_and_format_patch_data(case_id)
        selected_window_patches_features_list = []
        if bag_feats.shape[0] > 0:
            cos_similarities = F.cosine_similarity(bag_feats, self.global_text_feature, dim=1)
            selected_window_data = select_windows_from_grid_and_feats(
                coords, bag_feats, cos_similarities,
                self.window_size, self.stride, self.top_k_percent
            )
            if selected_window_data:
                for window_data in selected_window_data:
                    selected_window_patches_features_list.append(window_data['patch_feats'])
        return{
            "case_id": case_id,
            "selected_window_patches_list": selected_window_patches_features_list,
            "global_text_feature": self.global_text_feature.clone(),
            "label": label
        }