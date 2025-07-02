import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock

class SwinFusion(nn.Module):
    def __init__(self, in_channels=4, embed_dim=256, depths=[2,2], num_heads=[4,8]):
        super().__init__()
        # 初始投影层 (4C -> D)
        self.proj = nn.Conv2d(in_channels*4, embed_dim, 3, padding=1)
        
        # Swin Transformer 块
        self.blocks = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = nn.ModuleList()
            for i_block in range(depths[i_layer]):
                block = SwinTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads[i_layer],
                    window_size=8,
                    shift_size=0 if (i_block % 2 == 0) else 4
                )
                layer.append(block)
            self.blocks.append(layer)
        
        # 输出归一化
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Conv2d(embed_dim, in_channels, 1)

    def forward(self, feats):
        """
        feats: list of 4 [B, C, H, W] tensors
        """
        # 拼接特征 [B, 4C, H, W]
        x = torch.cat(feats, dim=1)
        
        # 初始投影 [B, D, H, W]
        x = self.proj(x)
        B, C, H, W = x.shape
        
        # 转换为序列格式 [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)
        
        # Swin Transformer处理
        for layer in self.blocks:
            for blk in layer:
                x = blk(x, (H, W))
        
        # 序列转回图像格式
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        
        return self.output_proj(x)


class ChannelAttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//8, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 输入特征图 [B, C, H, W]
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_weights = self.sigmoid(avg_out + max_out)
        return x * channel_weights.expand_as(x)


class MultiViewFuser(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.swin_fusion = SwinFusion(in_channels=latent_dim)
        self.channel_attention = ChannelAttentionFusion(latent_dim)
        
    def forward(self, latent_list):
        """
        latent_list: list of 4 VAE latent features [B, C, H, W]
        """
        fused = self.swin_fusion(latent_list)
        return self.channel_attention(fused)