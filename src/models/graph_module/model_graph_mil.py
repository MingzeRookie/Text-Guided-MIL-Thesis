import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GENConv, DeepGCNLayer

# 注意：这里改成了相对引用，适配你的目录结构
from .model_utils import *

class DeepGraphConv_Surv(torch.nn.Module):
    def __init__(self, edge_agg='spatial', resample=0, num_features=1024, hidden_dim=256, 
        linear_dim=256, use_edges=False, dropout=0.25, n_classes=2):
        super(DeepGraphConv_Surv, self).__init__()
        self.use_edges = use_edges
        self.resample = resample
        self.edge_agg = edge_agg

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample)])

        self.conv1 = GINConv(nn.Sequential(nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        
        # Attention Pooling 层
        self.path_attention_head = Attn_Net_Gated(L=hidden_dim, D=hidden_dim, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        
        # 分类器
        self.classifier = torch.nn.Linear(hidden_dim, n_classes)

    def forward(self, **kwargs):
        # 1. 解包 PyG 数据
        data = kwargs['x_path']
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # 2. 图卷积特征提取
        if self.resample:
            x = self.fc(x)

        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2, edge_index))
        
        # 3. Attention Pooling (聚合)
        h_path = x3
        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h = self.path_rho(h_path).squeeze()
        
        # 4. 分类输出 (直接输出 Logits)
        logits = self.classifier(h)
        
        # 确保 Batch=1 时维度正确 [1, n_classes]
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
            
        return logits