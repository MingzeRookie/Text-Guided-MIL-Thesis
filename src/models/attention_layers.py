import torch
import torch.nn as nn
import torch.nn.functional as F 

# NaN/Inf 检查辅助函数
def check_tensor_nan_inf_attn(tensor, name="Tensor", layer_name="Layer", critical=True, print_content=False, check_extreme=False, check_variance=False):
    prefix = "CRITICAL_NaN_CHECK" if critical else "DEBUG_NaN_CHECK"
    
    if tensor is None:
        return False 

    try:
        tensor_cpu = tensor.detach().cpu() # Check on CPU
        has_nan = torch.isnan(tensor_cpu).any()
        has_inf = torch.isinf(tensor_cpu).any()
    except TypeError: 
        return False
    except RuntimeError as e: 
        print(f"ERROR_NaN_CHECK ({layer_name}): Could not move tensor {name} to CPU for NaN/Inf check: {e}")
        return False 

    if has_nan or has_inf:
        issues = []
        if has_nan: issues.append("NaN")
        if has_inf: issues.append("Inf")
        print(f"{prefix} ({layer_name}): {name} contains {' and '.join(issues)}! Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}")
        if print_content:
            slice_print = tensor_cpu.flatten()[:min(20, tensor_cpu.numel())] # Print even smaller slice
            print(f"  {name} values (on CPU, first 20 flat): {slice_print}") 
        return True # Indicates NaN/Inf found
    
    if check_extreme and tensor.is_floating_point(): # Only check extreme for float tensors
        abs_tensor = torch.abs(tensor_cpu)
        abs_max = abs_tensor.max().item()
        # Check for very small non-zero values only if there are non-zero elements
        # abs_min_non_zero = torch.min(abs_tensor[abs_tensor > 1e-20]).item() if (abs_tensor > 1e-20).any() else 0.0
        mean_abs = abs_tensor.mean().item()
        if abs_max > 1e6: 
            print(f"WARNING_EXTREME_VALUE ({layer_name}): {name} has large absolute value: max_abs={abs_max:.2e}, mean_abs={mean_abs:.2e}. Shape: {tensor.shape}")
            # if print_content:
            #     slice_print = tensor_cpu.flatten()[:min(20, tensor_cpu.numel())]
            #     print(f"  {name} values (on CPU, first 20 flat): {slice_print}")
    
    if check_variance and tensor.is_floating_point() and tensor.ndim > 1:
        var_per_sample = torch.var(tensor_cpu, dim=list(range(1, tensor_cpu.ndim)), unbiased=False) # Variance over seq/feature dims for each batch item
        if (var_per_sample < 1e-8).any(): # If any sample has very low variance
            low_var_indices = (var_per_sample < 1e-8).nonzero(as_tuple=True)[0]
            print(f"WARNING_LOW_VARIANCE ({layer_name}): {name} has very low variance (<1e-8) for samples at batch indices: {low_var_indices.tolist()[:5]}. Min var: {var_per_sample.min().item():.2e}")
            # if print_content and low_var_indices.numel() > 0:
            #     print(f"  Example low variance sample data (idx {low_var_indices[0]}): {tensor_cpu[low_var_indices[0]].flatten()[:20]}")
    return False # No NaN/Inf found by this function call

class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"SelfAttentionLayer: embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
            
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout) # User's code uses self.dropout

    def forward(self, x, key_padding_mask=None):
        layer_name = "SelfAttentionLayer"
        
        if check_tensor_nan_inf_attn(x, "Input x", layer_name, check_extreme=True, check_variance=True, print_content=False):
            return torch.full_like(x, float('nan'))

        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                # print(f"WARNING ({layer_name}): key_padding_mask is not bool (dtype: {key_padding_mask.dtype}). Casting to bool.")
                key_padding_mask = key_padding_mask.bool()
            if key_padding_mask.all(dim=1).any(): # Should have been handled by caller
                print(f"ERROR ({layer_name}): Received key_padding_mask where some sequences are FULLY MASKED. This should be handled by the caller.")
                # return torch.full_like(x, float('nan')) # Or handle by zeroing out corresponding x rows before MHA

        attn_output = None # Initialize to None
        attn_weights = None # Initialize to None
        try:
            # Request attention weights for debugging
            attn_output, attn_weights = self.multihead_attn(x, x, x, 
                                                            key_padding_mask=key_padding_mask,
                                                            need_weights=True, average_attn_weights=False) 
                                                            # Get per-head weights: (N, H, L, S)
        except RuntimeError as e:
            print(f"CRITICAL_ERROR ({layer_name}): RuntimeError during multihead_attn: {e}")
            check_tensor_nan_inf_attn(x, "  Input x to MHA (RuntimeError context)", layer_name, print_content=True)
            if key_padding_mask is not None: print(f"  key_padding_mask any True: {key_padding_mask.any()}, all True on any row: {key_padding_mask.all(dim=1).any()}")
            return torch.full_like(x, float('nan'))

        if check_tensor_nan_inf_attn(attn_weights, "attn_weights (from MHA, per head)", layer_name, print_content=False): # Check weights first
             print(f"  Context for NaN attn_weights in {layer_name}:")
             check_tensor_nan_inf_attn(x, "  Input x to MHA for NaN weights", layer_name, print_content=True)
             if key_padding_mask is not None:
                 check_tensor_nan_inf_attn(key_padding_mask.float(), "  Input key_padding_mask for NaN weights", layer_name, print_content=True)
        
        if check_tensor_nan_inf_attn(attn_output, "attn_output (from MHA)", layer_name, check_extreme=True, print_content=True):
            # This is where NaN was previously detected
            print(f"  Context for NaN attn_output in {layer_name} (already printed if weights were also NaN):")
            if not (attn_weights is not None and torch.isnan(attn_weights).any()): # If weights were not NaN, print context again
                check_tensor_nan_inf_attn(x, "  Input x to MHA for NaN output", layer_name, print_content=True)
                if key_padding_mask is not None:
                    check_tensor_nan_inf_attn(key_padding_mask.float(), "  Input key_padding_mask for NaN output", layer_name, print_content=True)
            # return torch.full_like(x, float('nan')) # Propagate NaN if MHA output is NaN

        # Residual connection
        x_res = x + self.dropout(attn_output) # User's code uses self.dropout
        if check_tensor_nan_inf_attn(x_res, "x_res (after residual+dropout)", layer_name, check_extreme=True, check_variance=True):
            pass 

        # LayerNorm
        x_norm = self.norm(x_res)
        if check_tensor_nan_inf_attn(x_norm, "x_norm (final output)", layer_name, check_extreme=True, check_variance=True, print_content=True):
            pass
            
        return x_norm

class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads, dropout=0.1, embed_dim=None):
        super(CrossAttentionLayer, self).__init__()
        self.mha_embed_dim = embed_dim if embed_dim is not None else query_dim
        if self.mha_embed_dim % num_heads != 0:
            raise ValueError(f"CrossAttentionLayer: mha_embed_dim ({self.mha_embed_dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(query_dim, self.mha_embed_dim)
        self.k_proj = nn.Linear(key_dim, self.mha_embed_dim) 
        self.v_proj = nn.Linear(key_dim, self.mha_embed_dim) 

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.mha_embed_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(self.mha_embed_dim) 
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, kv_padding_mask=None):
        layer_name = "CrossAttentionLayer"
        if check_tensor_nan_inf_attn(query, "Input query", layer_name, check_extreme=True, check_variance=True): return torch.full((query.shape[0], query.shape[1], self.mha_embed_dim), float('nan'), device=query.device, dtype=query.dtype)
        if check_tensor_nan_inf_attn(key_value, "Input key_value", layer_name, check_extreme=True, check_variance=True): return torch.full((query.shape[0], query.shape[1], self.mha_embed_dim), float('nan'), device=query.device, dtype=query.dtype)

        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)

        if check_tensor_nan_inf_attn(q, "Projected q", layer_name, check_extreme=True): return torch.full_like(q, float('nan'))
        if check_tensor_nan_inf_attn(k, "Projected k", layer_name, check_extreme=True): return torch.full_like(q, float('nan'))
        if check_tensor_nan_inf_attn(v, "Projected v", layer_name, check_extreme=True): return torch.full_like(q, float('nan'))
        
        attn_output = None
        attn_weights = None
        try:
            attn_output, attn_weights = self.multihead_attn(q, k, v, 
                                                            key_padding_mask=kv_padding_mask,
                                                            need_weights=True, average_attn_weights=False)
        except RuntimeError as e:
            print(f"CRITICAL_ERROR ({layer_name}): RuntimeError during multihead_attn: {e}")
            return torch.full_like(q, float('nan'))

        if check_tensor_nan_inf_attn(attn_weights, "attn_weights (from MHA, per head)", layer_name, print_content=False):
             print(f"  Context for NaN attn_weights in {layer_name}:")
             check_tensor_nan_inf_attn(q, "  Input q to MHA for NaN weights", layer_name, print_content=True)
             check_tensor_nan_inf_attn(k, "  Input k to MHA for NaN weights", layer_name, print_content=True)
             check_tensor_nan_inf_attn(v, "  Input v to MHA for NaN weights", layer_name, print_content=True)
             if kv_padding_mask is not None:
                 check_tensor_nan_inf_attn(kv_padding_mask.float(), "  Input kv_padding_mask for NaN weights", layer_name, print_content=True)

        if check_tensor_nan_inf_attn(attn_output, "attn_output (from MHA)", layer_name, check_extreme=True, print_content=True):
            pass 

        # Residual with projected query 'q'
        output_after_residual = q + self.output_dropout(attn_output)
        if check_tensor_nan_inf_attn(output_after_residual, "output_after_residual", layer_name, check_extreme=True):
            pass
        
        output_norm = self.norm(output_after_residual)
        if check_tensor_nan_inf_attn(output_norm, "output_norm (final)", layer_name, check_extreme=True, print_content=True):
            pass
            
        return output_norm
