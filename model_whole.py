# models.py
import torch
import torch.nn as nn
import math

# ==============================================================================
# === 强大的扩散模型 (AdvancedUNet)                                          ===
# ==============================================================================

# --- 辅助模块 ---

class SinusoidalPositionEmbeddings(nn.Module):
    """ 将时间步 t 转换为位置嵌入向量 """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """ 带有时间嵌入的残差块 """
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(), # SiLU (Swish) 通常比 ReLU 效果更好
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1) # 扩展维度以匹配 h
        h = self.conv2(h)
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    """ 自注意力块 """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).view(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(dim=1)

        attn = torch.einsum('b h c i, b h c j -> b h i j', q, k) * ((C // self.num_heads) ** -0.5)
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h c j -> b h c i', attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj_out(out)

class DownBlock(nn.Module):
    """ 下采样块 """
    def __init__(self, in_channels, out_channels, time_emb_dim, has_attn=False):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

    def forward(self, x, time_emb):
        x = self.res(x, time_emb)
        x = self.attn(x)
        return self.downsample(x), x # 返回下采样后的结果和用于 skip-connection 的结果

class UpBlock(nn.Module):
    """ 上采样块 """
    def __init__(self, in_channels, out_channels, time_emb_dim, has_attn=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # 输入通道是 in_channels (来自上一层) + out_channels (来自skip-connection)
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_emb_dim)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x, skip_x, time_emb):
        x = self.upsample(x)
        x = torch.cat([skip_x, x], dim=1) # 拼接 skip-connection
        x = self.res(x, time_emb)
        x = self.attn(x)
        return x

# --- 主模型: AdvancedUNet ---
class AdvancedUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        time_emb_dim=128,
        base_dim=64, # 基础通道数
        dim_mults=(1, 2, 4) # 通道数乘数
    ):
        super().__init__()

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        dims = [base_dim] + [base_dim * m for m in dim_mults]
        
        # 下采样路径
        self.down_blocks = nn.ModuleList([])
        for i in range(len(dim_mults)):
            self.down_blocks.append(DownBlock(dims[i], dims[i+1], time_emb_dim, has_attn=(i >= 1)))
        
        # 中间瓶颈层
        self.mid_block1 = ResidualBlock(dims[-1], dims[-1], time_emb_dim)
        self.mid_attn = AttentionBlock(dims[-1])
        self.mid_block2 = ResidualBlock(dims[-1], dims[-1], time_emb_dim)

        # 上采样路径
        self.up_blocks = nn.ModuleList([])
        for i in reversed(range(len(dim_mults))):
            self.up_blocks.append(UpBlock(dims[i+1], dims[i], time_emb_dim, has_attn=(i >= 1)))

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, out_channels, 1)
        )

    def forward(self, x, time):
        # 1. 时间嵌入
        t = self.time_mlp(time)
        
        # 2. 初始卷积
        x = self.init_conv(x)
        
        # 3. 下采样
        skip_connections = []
        for down_block in self.down_blocks:
            x, skip_x = down_block(x, t)
            skip_connections.append(skip_x)
            
        # 4. 中间层
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        # 5. 上采样
        for up_block in self.up_blocks:
            skip_x = skip_connections.pop()
            x = up_block(x, skip_x, t)
            
        # 6. 输出
        return self.final_conv(x)

# ==============================================================================
# === 异构分类器模型 (保持不变)                                               ===
# ==============================================================================

# 基础模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # ... (代码不变) ...
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        # ... (代码不变) ...
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 模型1: 更宽的网络
class SimpleCNN_Wide(nn.Module):
    def __init__(self):
        super(SimpleCNN_Wide, self).__init__()
        # ... (代码不变) ...
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(96 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        # ... (代码不变) ...
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 96 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 模型2: 更深的网络
class SimpleCNN_Deep(nn.Module):
    def __init__(self):
        super(SimpleCNN_Deep, self).__init__()
        # ... (代码不变) ...
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        # ... (代码不变) ...
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x