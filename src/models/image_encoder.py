# src/models/image_encoder.py
import torch
import torch.nn as nn
from .attention_layers import SelfAttentionLayer # Assuming it's in the same package
from .mil_aggregators import AttentionMIL

class WSIImageEncoder(nn.Module):
    def __init__(self, feature_dim, self_attn_heads, mil_hidden_dim, mil_output_dim, dropout_sa=0.1, dropout_mil=0.25):
        super().__init__()
        self.self_attention_module = SelfAttentionLayer(embed_dim=feature_dim, num_heads=self_attn_heads, dropout=dropout_sa)
        self.mil_aggregator = AttentionMIL(input_dim=feature_dim, hidden_dim=mil_hidden_dim, output_dim=mil_output_dim, dropout_rate=dropout_mil)

    def forward(self, selected_window_patches_list, device):
        # selected_window_patches_list: list of tensors, each tensor is (num_patches_in_window, feature_dim)
        # All tensors in the list should be moved to the target device

        all_attended_patch_features_for_mil = []
        if not selected_window_patches_list: # Handle empty case
            # Return a zero tensor of appropriate shape for MIL aggregator if it expects input
            # Or handle this case further up in the main model
            # For now, let mil_aggregator handle empty input if it can, or return None/zeros
            # Assuming mil_output_dim is the expected final feature dim
            return torch.zeros(1, self.mil_aggregator.output_dim, device=device) # Or handle as error/None

        for window_feats in selected_window_patches_list:
            window_feats = window_feats.to(device) # Move individual window data to device
            if window_feats.nelement() == 0: continue # Skip empty windows

            # SelfAttentionLayer expects (batch_size, seq_len, embed_dim)
            window_feats_batched = window_feats.unsqueeze(0)
            attended_feats_in_window = self.self_attention_module(window_feats_batched).squeeze(0)
            all_attended_patch_features_for_mil.append(attended_feats_in_window)

        if not all_attended_patch_features_for_mil: # If all windows were empty
             return torch.zeros(1, self.mil_aggregator.output_dim, device=device)

        # Concatenate all attended patch features for MIL
        aggregated_attended_features = torch.cat(all_attended_patch_features_for_mil, dim=0) # (Total_Attended_Patches, feature_dim)

        # MIL aggregation
        image_level_feature, _ = self.mil_aggregator(aggregated_attended_features) # (1, mil_output_dim)
        return image_level_feature