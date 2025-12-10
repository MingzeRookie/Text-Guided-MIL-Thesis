# src/models/mil_aggregators.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# 用户提供的 AttentionMIL 类 (已做内部变量定义鲁棒性微调)
class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1024, dropout_rate=0.25, output_att_scores=1):
        super(AttentionMIL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim # 这是通过 bottleneck 层后的最终输出特征维度
        self.output_att_scores = output_att_scores # K, 通常为1

        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, self.output_att_scores)

        bottleneck_input_dim = input_dim * self.output_att_scores
        self.bottleneck = nn.Sequential(
            nn.Linear(bottleneck_input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, bag_feats: torch.Tensor, instance_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # bag_feats: (B, N, D_in)
        # instance_mask: (B, N) bool, True for valid instances
        
        if bag_feats.dim() == 2: # Handle single bag case
            bag_feats = bag_feats.unsqueeze(0) # (1, N, D_in)
        
        batch_size, num_instances, _ = bag_feats.shape

        A_V = self.attention_V(bag_feats)  # (B, N, H)
        A_U = self.attention_U(bag_feats)  # (B, N, H)
        att_raw = self.attention_weights(A_V * A_U) # (B, N, K)
        
        if instance_mask is not None:
            if not (instance_mask.shape[0] == batch_size and instance_mask.shape[1] == num_instances):
                raise ValueError(
                    f"AttentionMIL: instance_mask shape {instance_mask.shape} is not compatible with bag_feats shape {bag_feats.shape}"
                )
            mask_expanded = instance_mask.bool().unsqueeze(-1).expand_as(att_raw) # (B, N, K)
            att_raw = att_raw.masked_fill_(~mask_expanded, float('-inf'))
        
        A_softmax = F.softmax(att_raw, dim=1)  # (B, N, K)
        
        # Variable to track if any bag in the batch had all its instances masked.
        # This helps in zeroing out 'M' for such bags later.
        any_bag_is_fully_masked = torch.zeros(batch_size, dtype=torch.bool, device=bag_feats.device)
        if instance_mask is not None:
            # A bag is fully masked if all its instances have False in instance_mask
            # instance_mask.sum(dim=1) == 0 means all instances are masked (False)
            any_bag_is_fully_masked = (instance_mask.sum(dim=1) == 0)

        # Handle NaNs in A_softmax, which can occur if a K-slice of att_raw was all -inf
        if torch.isnan(A_softmax).any():
            # Replace NaNs with 0.0. This means fully masked instances (or attention heads for them)
            # will have zero attention weight.
            A_softmax = torch.nan_to_num(A_softmax, nan=0.0)

        M = torch.bmm(A_softmax.transpose(1, 2), bag_feats) # (B, K, D_in)
        
        # If any bag was fully masked (all its instances were invalid),
        # its aggregated representation M should be zero.
        if any_bag_is_fully_masked.any():
            M[any_bag_is_fully_masked] = 0.0

        if self.output_att_scores == 1:
            M_reshaped = M.squeeze(1) # (B, D_in)
        else:
            M_reshaped = M.view(batch_size, -1) # (B, K*D_in)
            if self.bottleneck[0].in_features != self.input_dim * self.output_att_scores:
                 print(f"警告: AttentionMIL bottleneck 输入维度 ({self.bottleneck[0].in_features}) " \
                       f"与期望的 K*D_in ({self.input_dim * self.output_att_scores}) 不匹配。K={self.output_att_scores}")

        final_feature = self.bottleneck(M_reshaped) # (B, output_dim)
        
        # Return attention scores consistent with K
        att_return = A_softmax.squeeze(-1) if self.output_att_scores == 1 else A_softmax
        return final_feature, att_return

# MILAttentionAggregator 类 (修复了原始 UnboundLocalError 并增强鲁棒性)
# 这个类是针对您traceback中 self.inter_window_aggregator 引发的错误设计的。
class MILAttentionAggregator(nn.Module):
    def __init__(self, feature_dim: int, use_gate: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_gate = use_gate

        self.attention_fc = nn.Linear(feature_dim, 1)
        if self.use_gate:
            # Assuming gate_fc also operates on feature_dim and outputs feature_dim for element-wise product
            self.gate_fc = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim)
            )

    def forward(
        self,
        bag_features: torch.Tensor, # (B, N, D_in)
        attention_mask: Optional[torch.Tensor] = None, # (B, N), True for valid, False for masked
        return_attention_weights: bool = False, # Parameter to control returning attention weights
    ) -> Tuple[torch.Tensor, torch.Tensor]: # Returns (aggregated_features, attention_scores)

        B, N, D = bag_features.shape

        # 1. 计算原始注意力分数
        A_raw = self.attention_fc(bag_features)  # Shape: (B, N, 1)

        # 2. 初始化掩码相关变量
        #    all_masked_in_batch: True if all bags in the entire batch are fully masked.
        #    This is the variable that caused UnboundLocalError in your original code.
        all_masked_in_batch = torch.tensor(False, device=bag_features.device)
        
        #    individual_bags_fully_masked: True for each bag if it's fully masked.
        individual_bags_fully_masked = torch.zeros(B, dtype=torch.bool, device=bag_features.device)

        # 3. 应用注意力掩码 (如果提供)
        if attention_mask is not None:
            if not (attention_mask.shape[0] == B and attention_mask.shape[1] == N):
                raise ValueError(
                    f"MILAttentionAggregator: attention_mask shape {attention_mask.shape} is not compatible with bag_features shape {bag_features.shape}"
                )
            
            # 将掩码位置的注意力分数设为负无穷
            # A_raw is (B,N,1), attention_mask is (B,N) -> unsqueeze for broadcasting
            A_raw[~attention_mask.unsqueeze(-1)] = -float('inf')

            # 确定哪些包是完全被掩码的
            individual_bags_fully_masked = (~attention_mask).all(dim=1)  # Shape: (B,)

            # 确定整个批次是否完全被掩码 (即所有包都完全被掩码)
            if individual_bags_fully_masked.all(): # .all() on a (B,) tensor gives a scalar tensor
                all_masked_in_batch = torch.tensor(True, device=bag_features.device)
        
        # 4. 处理整个批次完全被掩码的极端情况 (这是对原始代码中报错行的修正)
        if all_masked_in_batch: # This was `if all_masked_in_batch.any():`
            # 如果整个批次都被掩码，softmax(-inf) 会导致 NaN。
            # 返回零聚合特征和零注意力权重以防止 NaN。
            # logging.warning("All instances in the batch are masked for MILAttentionAggregator. Returning zero aggregation.") # 可选日志
            aggregated_features = torch.zeros(B, D, device=bag_features.device, dtype=bag_features.dtype)
            attention_scores = torch.zeros(B, N, device=bag_features.device, dtype=A_raw.dtype) # 保持与 A_raw 类型一致
            
            # 调用者 multimodal_text_guided_mil.py 使用: final_image_repr, _ = self.inter_window_aggregator(...)
            # 因此总是返回两个值
            return aggregated_features, attention_scores

        # 5. 计算注意力权重 (Softmax)
        A_softmax_input = A_raw.squeeze(-1) # Shape (B, N)
        A = torch.softmax(A_softmax_input, dim=1)  # Shape: (B, N)

        # 6. 处理单个包完全被掩码导致 A 中出现 NaN 的情况
        if attention_mask is not None and individual_bags_fully_masked.any():
            # 如果任何一个包被完全掩码 (individual_bags_fully_masked[i] is True),
            # 它的 A[i] 行在 softmax 后会是 NaN。将这些 NaN 行替换为零。
            A[individual_bags_fully_masked] = 0.0
            # 检查是否还有NaN（调试用）
            # if torch.isnan(A).any():
            #     print("Warning: NaNs still present in MILAttentionAggregator attention weights A after handling.")

        # 7. 应用门控机制 (如果启用)
        gated_bag_features = bag_features
        if self.use_gate:
            gate_values = self.gate_fc(bag_features)  # Shape: (B, N, D)
            gated_bag_features = bag_features * torch.sigmoid(gate_values)

        # 8. 计算加权聚合特征
        H = torch.sum(A.unsqueeze(-1) * gated_bag_features, dim=1)  # Shape: (B, D)

        # 根据参数决定是否返回注意力权重，但为了兼容通常返回两个值
        # 如果调用者不需要，可以忽略第二个返回值。
        # if return_attention_weights:
        #     return H, A
        return H, A # 总是返回两个值