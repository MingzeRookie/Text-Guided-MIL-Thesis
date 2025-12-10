# src/models/multimodal_text_guided_mil.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unittest 
import random # Ensure random is imported
import logging # Recommended for better logging

logger = logging.getLogger(__name__)

# Try to import your real modules from the current directory or defined paths
REAL_MODULES_LOADED = False
try:
    from .attention_layers import SelfAttentionLayer, CrossAttentionLayer
    from .mil_aggregators import AttentionMIL, MILAttentionAggregator 
    # print("Step 1/2 Success: Attempted relative import of real modules successfully.") # Less verbose
    REAL_MODULES_LOADED = True
except ImportError as e_rel:
    # print(f"Step 1/2 Fail: Relative import failed: {e_rel}. Attempting absolute import...") # Less verbose
    try:
        from BMW.src.models.attention_layers import SelfAttentionLayer, CrossAttentionLayer
        from BMW.src.models.mil_aggregators import AttentionMIL, MILAttentionAggregator
        # print("Step 2/2 Success: Attempted absolute import of real modules successfully.") # Less verbose
        REAL_MODULES_LOADED = True
    except ImportError as e_abs:
        # print(f"Step 2/2 Fail: Absolute import also failed: {e_abs}.") # Less verbose
        raise ImportError(
            "Critical custom modules failed to load. Check PYTHONPATH, __init__.py files, and how you are running the command.\n"
            f"Relative import error: {e_rel}\nAbsolute import error: {e_abs}"
        )

if not REAL_MODULES_LOADED:
    raise RuntimeError("Real module loading status inconsistent, failed to load real modules.")

# print("Confirmation: Real modules will be used. No 'Placeholder...initializing' messages should follow.") # Less verbose

# NaN/Inf check helper function
def check_tensor_nan_inf(tensor, name="Tensor", batch_item_idx=None, critical=True, print_content=False):
    prefix = "CRITICAL_NaN_CHECK" if critical else "DEBUG_NaN_CHECK"
    loc_info = f" (Batch Item {batch_item_idx})" if batch_item_idx is not None else ""
    
    if tensor is None:
        return False 

    try:
        tensor_cpu = tensor.detach().cpu()
        has_nan = torch.isnan(tensor_cpu).any()
        has_inf = torch.isinf(tensor_cpu).any()
    except TypeError: 
        return False
    except RuntimeError as e:
        logger.error(f"ERROR_NaN_CHECK: Could not move tensor {name} to CPU for NaN/Inf check: {e}")
        return False

    if has_nan or has_inf:
        issues = []
        if has_nan: issues.append("NaN")
        if has_inf: issues.append("Inf")
        logger.warning(f"{prefix}: {name}{loc_info} contains {' and '.join(issues)}! Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}")
        if print_content:
            logger.warning(f"  {name} values (on CPU): {tensor_cpu}") 
        return True
    return False

class MultimodalTextGuidedMIL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_feature_dim = config.patch_feature_dim
        self.num_classes = config.num_classes

        # Ablation flags from config
        self.ablation_image_only = config.model_params.get('ablation_image_only', False)
        self.ablation_no_window = config.model_params.get('ablation_no_window', False)

        if self.ablation_image_only and self.ablation_no_window:
            raise ValueError("ablation_image_only and ablation_no_window cannot both be True.")

        if not self.ablation_image_only:
            self.text_feature_dim = config.text_feature_dim # Required for multimodal and no_window ablation

        if not self.ablation_no_window: # Window-based architecture (original or image-only)
            self.window_patch_rows = config.model_params.window_params.patch_rows
            self.window_patch_cols = config.model_params.window_params.patch_cols
            self.patches_per_window = self.window_patch_rows * self.window_patch_cols
            self.stride_rows = config.model_params.window_params.stride_rows
            self.stride_cols = config.model_params.window_params.stride_cols
            self.num_selected_windows = config.model_params.window_params.num_selected_windows
            self.pre_similarity_window_agg_type = config.model_params.window_params.pre_similarity_window_agg_type
            
            if not self.ablation_image_only:
                self.similarity_projection_dim = config.model_params.similarity_projection_dim
                self.text_proj_sim = nn.Linear(self.text_feature_dim, self.similarity_projection_dim)
                self.patch_proj_sim = nn.Linear(self.patch_feature_dim, self.similarity_projection_dim)

            if self.pre_similarity_window_agg_type == 'attention_light':
                self.light_window_aggregator = AttentionMIL( 
                    input_dim=self.patch_feature_dim,
                    hidden_dim=config.model_params.window_params.light_agg_D,
                    dropout_rate=config.model_params.window_params.light_agg_dropout,
                    output_dim=self.patch_feature_dim 
                )
            elif self.pre_similarity_window_agg_type == 'mean':
                def robust_mean_agg(x, mask):
                    if mask is None:
                        check_tensor_nan_inf(x, "x in robust_mean_agg (mask is None)")
                        return x.mean(dim=1)
                    float_mask = mask.unsqueeze(-1).float()
                    num_valid_patches = float_mask.sum(dim=1) 
                    masked_x = x * float_mask 
                    sum_feats = masked_x.sum(dim=1) 
                    safe_num_valid_patches = num_valid_patches + 1e-8 
                    aggregated_repr = sum_feats / safe_num_valid_patches
                    if torch.isnan(aggregated_repr).any():
                        aggregated_repr = torch.nan_to_num(aggregated_repr, nan=0.0, posinf=0.0, neginf=0.0)
                    return aggregated_repr
                self.light_window_aggregator = robust_mean_agg
            elif self.pre_similarity_window_agg_type == 'max':
                 self.light_window_aggregator = lambda x, mask: x.masked_fill(~mask.unsqueeze(-1).bool(), -1e9).max(dim=1)[0] if mask is not None and mask.any() else x.max(dim=1)[0]
            elif self.ablation_image_only: # If image_only and no specific agg_type for it, it might not be used if selection is random
                pass
            else:
                raise ValueError(f"Unsupported pre_similarity_window_agg_type: {self.pre_similarity_window_agg_type}")

            self.window_self_attention = SelfAttentionLayer(
                embed_dim=self.patch_feature_dim, 
                num_heads=config.model_params.self_attn_heads,
                dropout=config.model_params.self_attn_dropout
            )
            self.window_mil_output_dim = config.model_params.window_mil_output_dim
            self.window_mil_aggregator = AttentionMIL(
                input_dim=self.patch_feature_dim, 
                hidden_dim=config.model_params.window_mil_D, 
                dropout_rate=config.model_params.window_mil_dropout, 
                output_dim=self.window_mil_output_dim
            )
            self.final_image_feature_dim = config.model_params.final_image_feature_dim
            self.inter_window_aggregator = AttentionMIL(
                input_dim=self.window_mil_output_dim,
                hidden_dim=config.model_params.inter_window_mil_D, 
                dropout_rate=config.model_params.inter_window_mil_dropout, 
                output_dim=self.final_image_feature_dim
            )

        # Cross attention and classifier setup common or modified for ablations
        if not self.ablation_image_only: # Multimodal (original or no_window)
            cross_attn_query_dim = self.final_image_feature_dim if not self.ablation_no_window else self.patch_feature_dim
            # For no_window, embed_dim might be different, e.g. a new config param
            cross_attention_embed_dim = config.model_params.get('direct_cross_attention_embed_dim', self.patch_feature_dim if self.ablation_no_window else self.final_image_feature_dim)

            self.cross_attention = CrossAttentionLayer( 
                query_dim=cross_attn_query_dim,
                key_dim=self.text_feature_dim,
                embed_dim=cross_attention_embed_dim, 
                num_heads=config.model_params.cross_attn_heads,
                dropout=config.model_params.cross_attn_dropout
            )
            if self.ablation_no_window:
                final_mil_output_dim = config.model_params.get('direct_final_mil_output_dim', 512)
                self.final_mil_aggregator = AttentionMIL(
                    input_dim=cross_attention_embed_dim, # Output of direct cross-attention
                    hidden_dim=config.model_params.get('direct_final_mil_hidden_dim', 128),
                    dropout_rate=config.model_params.get('direct_final_mil_dropout', 0.25),
                    output_dim=final_mil_output_dim
                )
                self.fused_feature_dim = final_mil_output_dim 
                # Optionally, could concat text_feature_dim again: + self.text_feature_dim
            else: # Original multimodal with windows
                self.fused_feature_dim = self.final_image_feature_dim + cross_attention_embed_dim # Original cross_attention_output_dim was self.final_image_feature_dim
        else: # Ablation: Image-only
            self.cross_attention = None
            self.fused_feature_dim = self.final_image_feature_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_feature_dim, config.model_params.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.model_params.classifier_dropout),
            nn.Linear(config.model_params.classifier_hidden_dim, self.num_classes)
        )

    def _generate_candidate_spatial_windows(self, all_patch_features_wsi, all_patch_grid_indices_wsi, wsi_grid_shape, batch_item_idx_for_debug=None):
        # This function is only used if not self.ablation_no_window
        if self.ablation_no_window:
            return [], [] # Should not be called in this ablation

        if all_patch_features_wsi is None or all_patch_features_wsi.numel() == 0: return [], []
        if check_tensor_nan_inf(all_patch_features_wsi, "all_patch_features_wsi (in _generate_candidate_spatial_windows)", batch_item_idx_for_debug): return [],[]
        
        max_r, max_c = wsi_grid_shape[0].item(), wsi_grid_shape[1].item()
        candidate_windows_feats_list = []
        candidate_windows_masks_list = []
        
        if all_patch_grid_indices_wsi.numel() == 0: return [], []
        try:
            coord_to_idx_map = {tuple(coord.tolist()): i for i, coord in enumerate(all_patch_grid_indices_wsi)}
        except Exception as e:
            logger.error(f"ERROR_NaN_CHECK: Failed to create coord_to_idx_map for batch item {batch_item_idx_for_debug}: {e}")
            return [], []

        eff_window_rows = min(self.window_patch_rows, max_r)
        eff_window_cols = min(self.window_patch_cols, max_c)

        if eff_window_rows <= 0 or eff_window_cols <= 0: return [], []

        for r_start in range(0, max_r - eff_window_rows + 1, self.stride_rows):
            for c_start in range(0, max_c - eff_window_cols + 1, self.stride_cols):
                current_window_patch_indices = []
                for r_offset in range(eff_window_rows): 
                    for c_offset in range(eff_window_cols):
                        abs_r, abs_c = r_start + r_offset, c_start + c_offset
                        if (abs_r, abs_c) in coord_to_idx_map:
                            current_window_patch_indices.append(coord_to_idx_map[(abs_r, abs_c)])
                
                if len(current_window_patch_indices) > 0:
                    window_feats = all_patch_features_wsi[current_window_patch_indices]
                    if check_tensor_nan_inf(window_feats, f"window_feats (r{r_start}c{c_start})", batch_item_idx_for_debug): continue 
                    num_actual_patches = window_feats.shape[0]
                    padded_window_feats = torch.zeros(self.patches_per_window, self.patch_feature_dim,
                                                      device=all_patch_features_wsi.device, dtype=all_patch_features_wsi.dtype)
                    window_mask = torch.zeros(self.patches_per_window, device=all_patch_features_wsi.device, dtype=torch.bool)
                    num_to_fill = min(num_actual_patches, self.patches_per_window)
                    if num_to_fill > 0: 
                        padded_window_feats[:num_to_fill] = window_feats[:num_to_fill]
                        window_mask[:num_to_fill] = True
                    candidate_windows_feats_list.append(padded_window_feats)
                    candidate_windows_masks_list.append(window_mask)
        return candidate_windows_feats_list, candidate_windows_masks_list

    def _select_windows(self, all_patch_features_wsi, text_feat_wsi, all_patch_grid_indices_wsi, wsi_grid_shape, batch_item_idx_for_debug=None):
        # This function is only used if not self.ablation_no_window
        if self.ablation_no_window:
            # This logic will be bypassed
            return torch.zeros(self.num_selected_windows, self.patches_per_window, self.patch_feature_dim, device=all_patch_features_wsi.device, dtype=all_patch_features_wsi.dtype), \
                   torch.zeros(self.num_selected_windows, self.patches_per_window, device=all_patch_features_wsi.device, dtype=torch.bool)

        def return_zeros_for_select(dev_ref, dtype_ref): # Pass reference for device/dtype
            dev = dev_ref.device if dev_ref is not None else torch.device('cpu')
            dtype = dtype_ref.dtype if dtype_ref is not None else torch.float32
            feats = torch.zeros(self.num_selected_windows, self.patches_per_window, self.patch_feature_dim, device=dev, dtype=dtype)
            masks = torch.zeros(self.num_selected_windows, self.patches_per_window, device=dev, dtype=torch.bool)
            return feats, masks

        if all_patch_features_wsi is None or all_patch_features_wsi.numel() == 0:
            return return_zeros_for_select(all_patch_features_wsi, all_patch_features_wsi) 
        if check_tensor_nan_inf(all_patch_features_wsi, "all_patch_features_wsi (in _select_windows)", batch_item_idx_for_debug): 
            return return_zeros_for_select(all_patch_features_wsi, all_patch_features_wsi)
        
        candidate_windows_feats_list, candidate_windows_masks_list = \
            self._generate_candidate_spatial_windows(all_patch_features_wsi, all_patch_grid_indices_wsi, wsi_grid_shape, batch_item_idx_for_debug)

        if not candidate_windows_feats_list:
            return return_zeros_for_select(all_patch_features_wsi, all_patch_features_wsi)

        final_selected_feats_list = []
        final_selected_masks_list = []
        num_candidates = len(candidate_windows_feats_list)
        num_to_select = min(self.num_selected_windows, num_candidates)

        if self.ablation_image_only or text_feat_wsi is None: # Image-only mode uses random selection
            if num_to_select > 0:
                if num_candidates <= num_to_select:
                    selected_indices = list(range(num_candidates))
                else:
                    selected_indices = random.sample(range(num_candidates), num_to_select)
                for idx in selected_indices:
                    final_selected_feats_list.append(candidate_windows_feats_list[idx])
                    final_selected_masks_list.append(candidate_windows_masks_list[idx])
        else: # Original text-guided logic
            if check_tensor_nan_inf(text_feat_wsi, "text_feat_wsi (in _select_windows)", batch_item_idx_for_debug): 
                return return_zeros_for_select(all_patch_features_wsi, all_patch_features_wsi)
            
            aggregated_candidate_reprs = []
            valid_candidate_indices = [] 
            for i, window_feats in enumerate(candidate_windows_feats_list):
                window_mask = candidate_windows_masks_list[i]
                if not window_mask.any(): continue
                if check_tensor_nan_inf(window_feats, f"window_feats to light_agg (candidate {i})", batch_item_idx_for_debug): continue
                current_window_feats_unsqueezed = window_feats.unsqueeze(0)
                current_mask_unsqueezed = window_mask.unsqueeze(0)

                if self.pre_similarity_window_agg_type == 'attention_light':
                    if not hasattr(self, 'light_window_aggregator'): # Should not happen if config is correct
                         raise AttributeError("light_window_aggregator not initialized for attention_light type")
                    agg_repr_tuple = self.light_window_aggregator(current_window_feats_unsqueezed, instance_mask=current_mask_unsqueezed)
                    agg_repr = agg_repr_tuple[0] 
                else: 
                    agg_repr = self.light_window_aggregator(current_window_feats_unsqueezed, current_mask_unsqueezed)
                
                if check_tensor_nan_inf(agg_repr, f"agg_repr from light_agg (candidate {i})", batch_item_idx_for_debug): continue
                aggregated_candidate_reprs.append(agg_repr.squeeze(0))
                valid_candidate_indices.append(i)

            if not aggregated_candidate_reprs:
                 return return_zeros_for_select(all_patch_features_wsi, all_patch_features_wsi)

            stacked_candidate_reprs = torch.stack(aggregated_candidate_reprs)
            if check_tensor_nan_inf(stacked_candidate_reprs, "stacked_candidate_reprs", batch_item_idx_for_debug): 
                return return_zeros_for_select(all_patch_features_wsi, all_patch_features_wsi)

            proj_text_feat = self.text_proj_sim(text_feat_wsi)
            if check_tensor_nan_inf(proj_text_feat, "proj_text_feat", batch_item_idx_for_debug): 
                return return_zeros_for_select(all_patch_features_wsi, all_patch_features_wsi)
            
            proj_candidate_reprs = self.patch_proj_sim(stacked_candidate_reprs)
            if check_tensor_nan_inf(proj_candidate_reprs, "proj_candidate_reprs", batch_item_idx_for_debug): 
                return return_zeros_for_select(all_patch_features_wsi, all_patch_features_wsi)

            similarity_scores = F.cosine_similarity(proj_text_feat.unsqueeze(0), proj_candidate_reprs, dim=1)
            if check_tensor_nan_inf(similarity_scores, "similarity_scores", batch_item_idx_for_debug): 
                return return_zeros_for_select(all_patch_features_wsi, all_patch_features_wsi)
            
            num_valid_candidates_for_topk = similarity_scores.shape[0]
            num_to_select_topk = min(self.num_selected_windows, num_valid_candidates_for_topk)

            if num_to_select_topk > 0:
                if not (torch.isnan(similarity_scores).any() or torch.isinf(similarity_scores).any()):
                    _, top_k_relative_indices = torch.topk(similarity_scores, k=num_to_select_topk, dim=0)
                    top_k_absolute_indices = [valid_candidate_indices[i] for i in top_k_relative_indices.tolist()]
                    for idx in top_k_absolute_indices:
                        final_selected_feats_list.append(candidate_windows_feats_list[idx])
                        final_selected_masks_list.append(candidate_windows_masks_list[idx])
        
        # Padding logic (same as your provided version)
        if final_selected_feats_list:
            padded_selected_feats = torch.stack(final_selected_feats_list, dim=0)
            padded_selected_masks = torch.stack(final_selected_masks_list, dim=0)
        else: 
            dev = all_patch_features_wsi.device if all_patch_features_wsi is not None else torch.device('cpu')
            dtype = all_patch_features_wsi.dtype if all_patch_features_wsi is not None else torch.float32
            padded_selected_feats = torch.zeros(0, self.patches_per_window, self.patch_feature_dim, device=dev, dtype=dtype)
            padded_selected_masks = torch.zeros(0, self.patches_per_window, device=dev, dtype=torch.bool)
        
        num_padding_windows = self.num_selected_windows - padded_selected_feats.shape[0]
        if num_padding_windows > 0:
            dev = padded_selected_feats.device if padded_selected_feats.numel() > 0 else (all_patch_features_wsi.device if all_patch_features_wsi is not None else torch.device('cpu'))
            dtype = padded_selected_feats.dtype if padded_selected_feats.numel() > 0 else (all_patch_features_wsi.dtype if all_patch_features_wsi is not None else torch.float32)
            padding_f = torch.zeros(num_padding_windows, self.patches_per_window, self.patch_feature_dim, device=dev, dtype=dtype)
            padding_m = torch.zeros(num_padding_windows, self.patches_per_window, device=dev, dtype=torch.bool)
            if padded_selected_feats.numel() > 0 : 
                padded_selected_feats = torch.cat([padded_selected_feats, padding_f], dim=0)
                padded_selected_masks = torch.cat([padded_selected_masks, padding_m], dim=0)
            else: 
                padded_selected_feats = padding_f
                padded_selected_masks = padding_m
        return padded_selected_feats, padded_selected_masks


    def forward(self, image_patch_features_batch, patch_grid_indices_batch, grid_shapes_batch, text_feat_batch=None, original_patch_coordinates_batch=None, patch_mask_batch=None):
        # Initial NaN/Inf checks (slightly adjusted for clarity)
        if image_patch_features_batch is None: return torch.full((1, self.num_classes), float('nan'), device=torch.device('cpu'), dtype=torch.float32) # Default if no batch size known
        batch_size = image_patch_features_batch.shape[0] # Define batch_size early
        def nan_return_val():
            return torch.full((batch_size, self.num_classes), float('nan'), device=image_patch_features_batch.device, dtype=image_patch_features_batch.dtype)

        if check_tensor_nan_inf(image_patch_features_batch, "image_patch_features_batch @ FORWARD_ENTRY"): return nan_return_val()
        
        if not self.ablation_image_only: # Text features are needed for original multimodal and no-window ablation
            if text_feat_batch is None:
                logger.error("MultimodalTextGuidedMIL: text_feat_batch is None but required for current mode.")
                return nan_return_val()
            if check_tensor_nan_inf(text_feat_batch, "text_feat_batch @ FORWARD_ENTRY"): return nan_return_val()
        
        # --- Ablation: No Windowing, Direct Cross-Attention ---
        if self.ablation_no_window:
            final_representations_for_classifier = []
            for i in range(batch_size):
                current_patch_feats_padded = image_patch_features_batch[i]
                current_text_feat = text_feat_batch[i] # text_feat_batch is guaranteed to be not None here
                
                active_patch_feats = current_patch_feats_padded
                current_valid_patches_mask_for_mil = None

                if patch_mask_batch is not None and patch_mask_batch[i] is not None:
                    valid_indices = patch_mask_batch[i].bool()
                    active_patch_feats = current_patch_feats_padded[valid_indices]
                    if active_patch_feats.shape[0] > 0: # Only create mask if there are active patches
                       current_valid_patches_mask_for_mil = torch.ones(active_patch_feats.shape[0], dtype=torch.bool, device=active_patch_feats.device)
                else: # No explicit mask, try to infer from non-zero patches if padding is zeros
                    non_zero_mask = active_patch_feats.abs().sum(dim=1) > 1e-6 # Assuming zero padding
                    active_patch_feats = active_patch_feats[non_zero_mask]
                    if active_patch_feats.shape[0] > 0:
                        current_valid_patches_mask_for_mil = torch.ones(active_patch_feats.shape[0], dtype=torch.bool, device=active_patch_feats.device)
                
                if active_patch_feats.shape[0] == 0:
                    cls_input_dim = self.classifier[0].in_features
                    zero_repr = torch.zeros(cls_input_dim, device=image_patch_features_batch.device, dtype=image_patch_features_batch.dtype)
                    final_representations_for_classifier.append(zero_repr)
                    continue

                query_img_feats = active_patch_feats.unsqueeze(0) # (1, N_active, D_patch)
                kv_text_feat = current_text_feat.unsqueeze(0).unsqueeze(0) # (1, 1, D_text)

                if check_tensor_nan_inf(query_img_feats, f"query_img_feats (no_window ablation, item {i})"): return nan_return_val()
                
                cross_attended_feats = self.cross_attention(query=query_img_feats, key_value=kv_text_feat)
                if check_tensor_nan_inf(cross_attended_feats, f"cross_attended_feats (no_window ablation, item {i})"): return nan_return_val()
                
                mil_input_mask = current_valid_patches_mask_for_mil.unsqueeze(0) if current_valid_patches_mask_for_mil is not None else None # (1, N_active)
                aggregated_repr, _ = self.final_mil_aggregator(cross_attended_feats, instance_mask=mil_input_mask)
                if check_tensor_nan_inf(aggregated_repr, f"aggregated_repr from final_mil (no_window ablation, item {i})"): return nan_return_val()
                
                final_representations_for_classifier.append(aggregated_repr.squeeze(0))

            if not final_representations_for_classifier: return nan_return_val()
            final_batch_representation = torch.stack(final_representations_for_classifier, dim=0)

        # --- Original Window-based logic (Multimodal or Image-Only) ---
        else: # Not ablation_no_window (i.e., original or image-only ablation)
            all_selected_windows_feats_b = []
            all_selected_windows_masks_b = []
            for i in range(batch_size):
                current_patch_feats = image_patch_features_batch[i]
                current_grid_indices = patch_grid_indices_batch[i]
                current_text_feat_sample = text_feat_batch[i] if text_feat_batch is not None and not self.ablation_image_only else None
                current_grid_shape = grid_shapes_batch[i]
                
                active_patch_feats = current_patch_feats
                active_grid_indices = current_grid_indices
                if patch_mask_batch is not None and patch_mask_batch[i] is not None:
                    # ... (patch masking logic as in your provided code) ...
                    current_patch_mask = patch_mask_batch[i]
                    num_original_patches = current_patch_feats.shape[0]
                    if current_patch_mask.shape[0] != num_original_patches:
                        len_to_use = min(current_patch_mask.shape[0], num_original_patches)
                        valid_patches_mask_i = current_patch_mask[:len_to_use].bool()
                        active_patch_feats = current_patch_feats[:len_to_use][valid_patches_mask_i]
                        active_grid_indices = current_grid_indices[:len_to_use][valid_patches_mask_i]
                    else:
                        valid_patches_mask_i = current_patch_mask.bool()
                        active_patch_feats = current_patch_feats[valid_patches_mask_i]
                        active_grid_indices = current_grid_indices[valid_patches_mask_i]

                if check_tensor_nan_inf(active_patch_feats, f"active_patch_feats (window mode, item {i})"): return nan_return_val()

                if active_patch_feats.shape[0] == 0:
                    dev = current_patch_feats.device if current_patch_feats is not None else image_patch_features_batch.device
                    dtype = current_patch_feats.dtype if current_patch_feats is not None else image_patch_features_batch.dtype
                    s_feats = torch.zeros(self.num_selected_windows, self.patches_per_window, self.patch_feature_dim, device=dev, dtype=dtype)
                    s_mask = torch.zeros(self.num_selected_windows, self.patches_per_window, device=dev, dtype=torch.bool)
                else:
                    s_feats, s_mask = self._select_windows( # Renamed from _select_text_guided_windows for clarity
                        active_patch_feats, current_text_feat_sample, active_grid_indices, current_grid_shape, batch_item_idx_for_debug=i
                    )
                if check_tensor_nan_inf(s_feats, f"s_feats from _select_windows (item {i})"): return nan_return_val()
                all_selected_windows_feats_b.append(s_feats)
                all_selected_windows_masks_b.append(s_mask)

            selected_windows_feats = torch.stack(all_selected_windows_feats_b, dim=0)
            if check_tensor_nan_inf(selected_windows_feats, "selected_windows_feats (after stack)"): return nan_return_val()
            selected_windows_mask = torch.stack(all_selected_windows_masks_b, dim=0)
            
            k_w = self.num_selected_windows
            proc_windows_feats = selected_windows_feats.reshape(batch_size * k_w, self.patches_per_window, self.patch_feature_dim)
            proc_windows_mask = selected_windows_mask.reshape(batch_size * k_w, self.patches_per_window)
            if check_tensor_nan_inf(proc_windows_feats, "proc_windows_feats (INPUT to self_attn)"): return nan_return_val()
            
            current_key_padding_mask = ~proc_windows_mask.bool() if proc_windows_mask is not None else None
            attended_patch_feats = torch.zeros_like(proc_windows_feats)
            if current_key_padding_mask is not None:
                not_fully_masked_window_indices = (~current_key_padding_mask.all(dim=1)).nonzero(as_tuple=True)[0]
            else:
                not_fully_masked_window_indices = torch.arange(proc_windows_feats.shape[0], device=proc_windows_feats.device)

            if not_fully_masked_window_indices.numel() > 0:
                feats_to_attend = proc_windows_feats[not_fully_masked_window_indices]
                mask_for_attention = current_key_padding_mask[not_fully_masked_window_indices] if current_key_padding_mask is not None else None
                if check_tensor_nan_inf(feats_to_attend, "feats_to_attend (subset for self_attn)"): return nan_return_val()
                output_from_attention = self.window_self_attention(feats_to_attend, key_padding_mask=mask_for_attention)
                if check_tensor_nan_inf(output_from_attention, "output_from_attention (from non-fully-masked self_attn)"): return nan_return_val()
                attended_patch_feats.index_copy_(0, not_fully_masked_window_indices, output_from_attention)
            if check_tensor_nan_inf(attended_patch_feats, "attended_patch_feats (AFTER selective self_attn)"): return nan_return_val()
            
            aggregated_window_reprs, _ = self.window_mil_aggregator(attended_patch_feats, instance_mask=proc_windows_mask) 
            if check_tensor_nan_inf(aggregated_window_reprs, "aggregated_window_reprs (from window_mil_aggregator)"): return nan_return_val()
            aggregated_window_reprs = aggregated_window_reprs.view(batch_size, k_w, self.window_mil_output_dim)
            if check_tensor_nan_inf(aggregated_window_reprs, "aggregated_window_reprs (reshaped for inter_window_agg)"): return nan_return_val()
            
            inter_window_mask = selected_windows_mask.any(dim=2) 
            if check_tensor_nan_inf(inter_window_mask.float(), "inter_window_mask", critical=False): return nan_return_val()
            final_image_repr, _ = self.inter_window_aggregator(aggregated_window_reprs, instance_mask=inter_window_mask) 
            if check_tensor_nan_inf(final_image_repr, "final_image_repr (from inter_window_aggregator)"): return nan_return_val()
            
            if not self.ablation_image_only and self.cross_attention is not None:
                final_image_repr_seq = final_image_repr.unsqueeze(1)
                # text_feat_batch should be valid here due to earlier checks
                # Need to ensure each item in text_feat_batch is also valid for cross attention
                cross_attn_text_inputs = []
                valid_cross_attn_indices = []
                for item_idx in range(batch_size):
                    if text_feat_batch[item_idx] is not None:
                        cross_attn_text_inputs.append(text_feat_batch[item_idx].unsqueeze(0).unsqueeze(0)) # (1,1,D_text)
                        valid_cross_attn_indices.append(item_idx)
                
                if not valid_cross_attn_indices: # No valid text features for cross attention in the batch
                    logger.warning("No valid text features for cross-attention in the batch.")
                    # Fallback: treat as image-only for this batch or handle error
                    final_batch_representation = final_image_repr
                else:
                    # Process only valid items for cross-attention
                    valid_final_image_repr_seq = final_image_repr_seq[valid_cross_attn_indices]
                    valid_text_feat_batch_seq = torch.cat(cross_attn_text_inputs, dim=0)

                    fused_representation_valid = self.cross_attention(
                        query=valid_final_image_repr_seq,
                        key_value=valid_text_feat_batch_seq
                    )
                    if check_tensor_nan_inf(fused_representation_valid, "fused_representation_valid (from cross_attention)"): return nan_return_val()
                    fused_representation_valid = fused_representation_valid.squeeze(1)

                    # Initialize final_batch_representation with image features and update with fused ones
                    final_batch_representation = final_image_repr.clone() # Start with image features
                    # Concatenate for the valid indices
                    for i, original_idx in enumerate(valid_cross_attn_indices):
                         final_batch_representation[original_idx] = torch.cat([final_image_repr[original_idx], fused_representation_valid[i]], dim=-1)

            else: # Image-only ablation for window-based model
                final_batch_representation = final_image_repr
        
        # --- End of conditional logic for ablations ---

        if check_tensor_nan_inf(final_batch_representation, "final_batch_representation (input to classifier)"): return nan_return_val()
        logits = self.classifier(final_batch_representation)
        if check_tensor_nan_inf(logits, "logits (FINAL OUTPUT from classifier)"):
            if not (torch.isnan(final_batch_representation).any() or torch.isinf(final_batch_representation).any()):
                 logger.warning(f"  final_batch_representation (input to classifier) min: {final_batch_representation.min().item()}, max: {final_batch_representation.max().item()}, mean: {final_batch_representation.mean().item()}")
        return logitsni

# --- SimpleConfig (保持不变, 但确保配置能驱动ablation_image_only, ablation_no_window) ---
class SimpleConfig:
    def __init__(self, **kwargs):
        self.patch_feature_dim = kwargs.get('patch_feature_dim', 1024)
        self.text_feature_dim = kwargs.get('text_feature_dim', 1024) # Keep for non-image-only
        self.num_classes = kwargs.get('num_classes', 2)
        
        model_params_dict = kwargs.get('model_params', {})
        if not isinstance(model_params_dict, dict): # Ensure it's a dict before passing to SimpleConfigModelParams
            try:
                model_params_dict = model_params_dict._asdict() # If it's a NamedTuple from OmegaConf
            except AttributeError:
                model_params_dict = vars(model_params_dict) # If it's a simple class instance

        # Ensure model_params_dict has a 'get' method or convert it
        # A simple way to ensure it behaves like a dict for 'get'
        class ConfigWrapper:
            def __init__(self, d):
                self._d = d
            def get(self, key, default=None):
                return self._d.get(key, default)
            def __getattr__(self, key): # Allow attribute access as well
                return self._d.get(key)


        if not isinstance(model_params_dict, SimpleConfigModelParams):
             self.model_params = SimpleConfigModelParams(**model_params_dict)
        else:
            self.model_params = model_params_dict
            
    def get(self, key, default=None): return getattr(self, key, default)

class SimpleConfigModelParams: # Ensure this can handle all new ablation params
    def __init__(self, **kwargs):
        self.ablation_image_only = kwargs.get('ablation_image_only', False)
        self.ablation_no_window = kwargs.get('ablation_no_window', False)

        self.similarity_projection_dim = kwargs.get('similarity_projection_dim', 256)
        
        window_params_config = kwargs.get('window_params', {})
        # if not isinstance(window_params_config, SimpleConfigWindowParams): # Already a dict by now
        self.window_params = SimpleConfigWindowParams(**window_params_config)
        
        self.self_attn_heads = kwargs.get('self_attn_heads', 4)
        self.self_attn_dropout = kwargs.get('self_attn_dropout', 0.1)
        self.window_mil_output_dim = kwargs.get('window_mil_output_dim', 512)
        self.window_mil_D = kwargs.get('window_mil_D', 128) 
        self.window_mil_dropout = kwargs.get('window_mil_dropout', 0.1) 
        self.final_image_feature_dim = kwargs.get('final_image_feature_dim', 512)
        self.inter_window_mil_D = kwargs.get('inter_window_mil_D', 128) # Corrected from 64 to match recent logs if needed
        self.inter_window_mil_dropout = kwargs.get('inter_window_mil_dropout', 0.1) 
        
        self.cross_attn_heads = kwargs.get('cross_attn_heads', 4)
        self.cross_attn_dropout = kwargs.get('cross_attn_dropout', 0.1)
        
        # Params for no_window ablation
        self.direct_cross_attention_embed_dim = kwargs.get('direct_cross_attention_embed_dim', kwargs.get('patch_feature_dim', 1024))
        self.direct_final_mil_output_dim = kwargs.get('direct_final_mil_output_dim', 512)
        self.direct_final_mil_hidden_dim = kwargs.get('direct_final_mil_hidden_dim', 128)
        self.direct_final_mil_dropout = kwargs.get('direct_final_mil_dropout', 0.25)

        self.classifier_hidden_dim = kwargs.get('classifier_hidden_dim', 256)
        self.classifier_dropout = kwargs.get('classifier_dropout', 0.25)

    def get(self, key, default=None): return getattr(self, key, default)


class SimpleConfigWindowParams: # Ensure this is robust to missing keys if ablation_no_window=True
    def __init__(self, **kwargs):
        self.patch_rows = kwargs.get('patch_rows', 3)
        self.patch_cols = kwargs.get('patch_cols', 3)
        self.stride_rows = kwargs.get('stride_rows', 3)
        self.stride_cols = kwargs.get('stride_cols', 3)
        self.num_selected_windows = kwargs.get('num_selected_windows', 5)
        self.pre_similarity_window_agg_type = kwargs.get('pre_similarity_window_agg_type', 'mean')
        self.light_agg_D = kwargs.get('light_agg_D', 64) 
        self.light_agg_dropout = kwargs.get('light_agg_dropout', 0.1)


# --- 单元测试部分 (需要大幅修改以适应新的模型逻辑和ablation flags) ---
class TestMultimodalTextGuidedMIL(unittest.TestCase):
    def setUp(self):
        # Base config, will be overridden for specific tests
        self.base_config_dict = {
            'patch_feature_dim': 64, 
            'text_feature_dim': 32, # Still provide for structure, even if not used in image_only
            'num_classes': 2,
            'model_params': {
                'ablation_image_only': False, # Default
                'ablation_no_window': False,  # Default

                'similarity_projection_dim': 16,
                'window_params': {
                    'patch_rows': 2, 'patch_cols': 2, 
                    'stride_rows': 1, 'stride_cols': 1,
                    'num_selected_windows': 3,        
                    'pre_similarity_window_agg_type': 'mean', 
                    'light_agg_D': 8, 
                    'light_agg_dropout': 0.0, 
                },
                'self_attn_heads': 2, 'self_attn_dropout': 0.0,
                'window_mil_output_dim': 24, 
                'window_mil_D': 12, 
                'window_mil_dropout': 0.0, 
                'final_image_feature_dim': 20, 
                'inter_window_mil_D': 10, 
                'inter_window_mil_dropout': 0.0, 
                
                'cross_attn_heads': 2, 'cross_attn_dropout': 0.0,
                'direct_cross_attention_embed_dim': 64, # For no_window test
                'direct_final_mil_output_dim': 22,    # For no_window test
                'direct_final_mil_hidden_dim': 11,    # For no_window test
                'direct_final_mil_dropout': 0.0,      # For no_window test

                'classifier_hidden_dim': 16, 'classifier_dropout': 0.0,
            }
        }
        # Common data for tests
        self.patch_dim = self.base_config_dict['patch_feature_dim']
        self.text_dim = self.base_config_dict['text_feature_dim']
        self.N_total_patches = 25 
        self.all_patch_features_wsi = torch.randn(self.N_total_patches, self.patch_dim)
        grid_r, grid_c = torch.meshgrid(torch.arange(5), torch.arange(5), indexing='ij')
        self.all_patch_grid_indices_wsi = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=1)
        self.text_feat_wsi = torch.randn(self.text_dim)
        self.wsi_grid_shape = torch.tensor([5, 5], dtype=torch.long)

    def test_original_model_forward(self):
        print("\nTesting: Original Model Forward Pass")
        cfg = SimpleConfig(**self.base_config_dict)
        model = MultimodalTextGuidedMIL(cfg)
        model.eval()
        
        bs = 1
        img_feats_b = self.all_patch_features_wsi.unsqueeze(0)
        grid_idx_b = self.all_patch_grid_indices_wsi.unsqueeze(0)
        txt_feat_b = self.text_feat_wsi.unsqueeze(0)
        grid_shape_b = self.wsi_grid_shape.unsqueeze(0)
        
        logits = model(img_feats_b, grid_idx_b, grid_shape_b, text_feat_batch=txt_feat_b)
        self.assertEqual(logits.shape, (bs, cfg.num_classes))
        self.assertFalse(torch.isnan(logits).any(), "Logits should not be NaN in original model test")
        print(f"  Original model forward pass successful. Logits shape: {logits.shape}")

    def test_ablation_image_only_forward(self):
        print("\nTesting: Ablation Image-Only Forward Pass")
        image_only_config_dict = self.base_config_dict.copy()
        image_only_config_dict['model_params']['ablation_image_only'] = True
        cfg_img_only = SimpleConfig(**image_only_config_dict)
        
        model_img_only = MultimodalTextGuidedMIL(cfg_img_only)
        model_img_only.eval()

        bs = 1
        img_feats_b = self.all_patch_features_wsi.unsqueeze(0)
        grid_idx_b = self.all_patch_grid_indices_wsi.unsqueeze(0)
        grid_shape_b = self.wsi_grid_shape.unsqueeze(0)
        
        # Pass text_feat_batch=None for image-only
        logits = model_img_only(img_feats_b, grid_idx_b, grid_shape_b, text_feat_batch=None) 
        self.assertEqual(logits.shape, (bs, cfg_img_only.num_classes))
        self.assertFalse(torch.isnan(logits).any(), "Logits should not be NaN in image-only ablation")
        print(f"  Image-only ablation forward pass successful. Logits shape: {logits.shape}")

    def test_ablation_no_window_forward(self):
        print("\nTesting: Ablation No-Window Forward Pass")
        no_window_config_dict = self.base_config_dict.copy()
        no_window_config_dict['model_params']['ablation_no_window'] = True
        cfg_no_window = SimpleConfig(**no_window_config_dict)

        model_no_window = MultimodalTextGuidedMIL(cfg_no_window)
        model_no_window.eval()

        bs = 1
        # For no-window, image_patch_features_batch is (B, Max_Patches, D_patch)
        # And patch_mask_batch is (B, Max_Patches)
        max_patches_in_batch = 30
        img_feats_b_padded = torch.zeros(bs, max_patches_in_batch, self.patch_dim)
        patch_mask_b = torch.zeros(bs, max_patches_in_batch, dtype=torch.bool)

        img_feats_b_padded[0, :self.N_total_patches] = self.all_patch_features_wsi
        patch_mask_b[0, :self.N_total_patches] = True
        
        txt_feat_b = self.text_feat_wsi.unsqueeze(0) # (B, D_text)
        
        # grid_indices and grid_shapes are not strictly used by this ablation's forward but passed by dataloader
        grid_idx_b_dummy = torch.zeros(bs, max_patches_in_batch, 2, dtype=torch.long)
        grid_shape_b_dummy = torch.zeros(bs, 2, dtype=torch.long)


        logits = model_no_window(
            image_patch_features_batch=img_feats_b_padded, 
            patch_grid_indices_batch=grid_idx_b_dummy, # Dummy
            grid_shapes_batch=grid_shape_b_dummy,    # Dummy
            text_feat_batch=txt_feat_b,
            patch_mask_batch=patch_mask_b
        )
        self.assertEqual(logits.shape, (bs, cfg_no_window.num_classes))
        self.assertFalse(torch.isnan(logits).any(), "Logits should not be NaN in no-window ablation")
        print(f"  No-window ablation forward pass successful. Logits shape: {logits.shape}")


if __name__ == '__main__':
    # For basic logging during tests if logger isn't configured elsewhere
    logging.basicConfig(level=logging.INFO)
    print("Starting MultimodalTextGuidedMIL model component and forward pass unit tests...")
    suite = unittest.TestSuite()
    # Add tests selectively or all
    suite.addTest(unittest.makeSuite(TestMultimodalTextGuidedMIL))
    runner = unittest.TextTestRunner()
    runner.run(suite)
    print("\nAll unit tests executed.")