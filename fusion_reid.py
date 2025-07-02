import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedReIDFusion(nn.Module):
    def __init__(self, feature_dim=712, hidden_dim=256, max_features=16):
        """
        feature_dim: ReID特征维度 (默认712)
        hidden_dim: 内部隐藏层维度
        max_features: 支持的最大特征数
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.max_features = max_features
        
        # 特征有效性门控
        self.validity_gate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 特征重要性权重生成
        self.importance_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # 特征交互模块
        self.feature_interaction = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True,
            kdim=feature_dim,
            vdim=feature_dim
        )
        
        # 特征增强变换
        self.feature_enhancer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # 全局融合节点 (用于无有效特征情况)
        self.global_node = nn.Parameter(torch.randn(1, feature_dim))
        nn.init.xavier_uniform_(self.global_node)

    def forward(self, features):
        """
        融合多个ReID特征
        features: 特征列表，每个元素为 [batch_size, feature_dim] 张量
        返回: 融合后的特征 [batch_size, feature_dim]
        """
        batch_size = features[0].size(0) if features else 1
        num_features = len(features)
        
        # 1. 特征有效性评估
        validity_scores = []
        valid_features = []
        
        for feat in features:
            # 计算特征有效性分数 (0-1)
            validity = self.validity_gate(feat)  # [batch_size, 1]
            validity_scores.append(validity)
            
            # 只保留有效特征 (有效性>0.3)
            mask = (validity > 0.3).float()
            valid_features.append(feat * mask)
        
        # 2. 构建特征矩阵 [batch_size, num_features, feature_dim]
        if num_features > 0:
            feature_matrix = torch.stack(valid_features, dim=1)
        else:
            # 无有效特征时返回全局节点
            return self.global_node.expand(batch_size, -1)
        
        # 3. 特征重要性加权
        importance_weights = self.importance_net(feature_matrix)  # [batch_size, num_features, 1]
        weighted_features = feature_matrix * importance_weights
        
        # 4. 特征交互融合
        # 使用自注意力增强特征间交互
        fused, _ = self.feature_interaction(
            weighted_features, weighted_features, weighted_features
        )
        
        # 5. 特征聚合 (加权平均)
        fused_feature = torch.sum(fused * importance_weights, dim=1)
        
        # 6. 特征增强
        enhanced_feature = self.feature_enhancer(fused_feature)
        
        # 残差连接保留原始信息
        final_feature = fused_feature + enhanced_feature
        
        return final_feature

    def adaptive_fusion(self, feature_sets):
        """
        批量处理多组特征
        feature_sets: 特征集列表，每个元素是特征列表
        返回: 融合特征列表
        """
        return [self.forward(features) for features in feature_sets]