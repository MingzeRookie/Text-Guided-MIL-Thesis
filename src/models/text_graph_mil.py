import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data, Batch

# 导入刚才修改好的 GCN 模块
from .graph_module.model_graph_mil import DeepGraphConv_Surv

class TextGuidedGCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_feature_dim = config.patch_feature_dim
        self.text_feature_dim = config.text_feature_dim
        self.num_classes = config.num_classes
        
        # === 创新点 1: 文本引导筛选 ===
        # 将图像和文本投影到同一空间计算相似度
        self.img_proj = nn.Linear(self.patch_feature_dim, 256)
        self.text_proj = nn.Linear(self.text_feature_dim, 256)
        
        # 筛选数量 (Top-K)
        # 尝试从配置读取，如果没有则默认 512
        self.k_sample = getattr(config.model_params, 'k_sample', 512)

        # === 创新点 2: 空间图卷积 ===
        # 使用修改后的 Patch-GCN 骨干
        self.gcn_backbone = DeepGraphConv_Surv(
            num_features=self.patch_feature_dim, 
            hidden_dim=256, 
            dropout=0.25,
            n_classes=self.num_classes
        )

    def forward(self, image_patch_features_batch, original_patch_coordinates_batch, text_feat_batch, **kwargs):
        """
        参数:
        - image_patch_features_batch: [B, N, 1024] 图像特征
        - original_patch_coordinates_batch: [B, N, 2] 真实坐标 (x, y)
        - text_feat_batch: [B, 1024] 文本特征
        """
        batch_size = image_patch_features_batch.shape[0]
        data_list = [] # 用于存放 PyG 图对象

        # 逐个样本处理 (WSI 任务 Batch Size 通常很小)
        for i in range(batch_size):
            img_feat = image_patch_features_batch[i]
            coords = original_patch_coordinates_batch[i]
            text_feat = text_feat_batch[i]
            
            # --- Step 1: 文本引导筛选 (Top-K) ---
            img_emb = self.img_proj(img_feat)       # [N, 256]
            txt_emb = self.text_proj(text_feat)     # [256]
            
            # 归一化并计算 Cosine 相似度
            img_emb = F.normalize(img_emb, p=2, dim=1)
            txt_emb = F.normalize(txt_emb, p=2, dim=0)
            scores = torch.mv(img_emb, txt_emb)     # [N]
            
            # 选出分数最高的 K 个 Patch
            # 确保 k 不超过当前 Patch 总数
            k = min(self.k_sample, img_feat.shape[0])
            _, topk_indices = torch.topk(scores, k)
            
            # 提取关键数据
            selected_x = img_feat[topk_indices]   # [K, 1024]
            selected_pos = coords[topk_indices]   # [K, 2]
            
            # --- Step 2: 动态构图 (k-NN) ---
            # 基于物理坐标构建图结构 (k=8)
            edge_index = knn_graph(selected_pos.float(), k=8, loop=False)
            
            # 创建 PyG Data 对象并放到 GPU
            data = Data(x=selected_x, edge_index=edge_index)
            data_list.append(data.to(img_feat.device))

        # --- Step 3: 图卷积推理 ---
        # 将 List 拼成大 Batch
        batch_data = Batch.from_data_list(data_list)
        
        # 输入 GCN
        logits = self.gcn_backbone(x_path=batch_data)
        
        return logits