import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# === EDM U-Net 模型及其辅助模块 (改编自 NVIDIA 官方实现)                     ===
# ==============================================================================

# --- 内部辅助模块 (为 EDM U-Net 服务) ---
class _EDM_Normalize(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
    def forward(self, x):
        return self.norm(x)

class _EDM_ResBlock(nn.Module):
    def __init__(self, in_channels, emb_channels, out_channels, dropout=0.1):
        super().__init__()
        self.in_layers = nn.Sequential(
            _EDM_Normalize(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels),
        )
        self.out_layers = nn.Sequential(
            _EDM_Normalize(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        h = h + emb_out.unsqueeze(-1).unsqueeze(-1) # 扩展维度
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class _EDM_AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = _EDM_Normalize(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        qkv = qkv.view(b, self.num_heads, c // self.num_heads * 3, h * w).permute(0, 3, 1, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        
        scale = 1. / math.sqrt(float(c // self.num_heads))
        weight = torch.einsum('b n i d, b n j d -> b n i j', q, k) * scale
        weight = torch.softmax(weight, dim=-1)
        
        a = torch.einsum('b n i j, b n j d -> b n i d', weight, v)
        a = a.permute(0, 2, 3, 1).reshape(b, c, h, w)
        return x + self.proj_out(a)


# --- 主模型: EDMUNet ---
class EDMUNet(nn.Module):
    def __init__(
        self,
        img_resolution=32,      # 图像分辨率
        in_channels=3,          # 输入通道
        out_channels=3,         # 输出通道
        label_dim=10,           # 类别条件维度 (CIFAR-10 为 10)
        model_channels=128,     # 基础通道数
        channel_mult=[1,2,2,2], # 通道数乘数
        num_blocks=2,           # 每个分辨率的残差块数
        attn_resolutions=[16],  # 在 16x16 分辨率时使用注意力
        dropout=0.1,
    ):
        super().__init__()
        self.label_dim = label_dim
        
        # EDM 的时间步嵌入 (傅里叶特征)
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # 类别条件嵌入 (将 one-hot 标签映射到嵌入维度)
        if self.label_dim > 0:
            self.label_embed = nn.Linear(label_dim, time_embed_dim)

        # 下采样部分
        self.input_blocks = nn.ModuleList([nn.Conv2d(in_channels, model_channels, 3, padding=1)])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_blocks):
                layers = [_EDM_ResBlock(ch, time_embed_dim, mult * model_channels, dropout)]
                ch = mult * model_channels
                if ds * (img_resolution // 2**level) in attn_resolutions:
                    layers.append(_EDM_AttentionBlock(ch))
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.AvgPool2d(kernel_size=2, stride=2))
                input_block_chans.append(ch)
                ds *= 2

        # 中间部分
        self.middle_block = nn.Sequential(
            _EDM_ResBlock(ch, time_embed_dim, ch, dropout),
            _EDM_AttentionBlock(ch),
            _EDM_ResBlock(ch, time_embed_dim, ch, dropout),
        )

        # 上采样部分
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_blocks + 1):
                ich = input_block_chans.pop()
                layers = [_EDM_ResBlock(ch + ich, time_embed_dim, model_channels * mult, dropout)]
                ch = model_channels * mult
                if ds * (img_resolution // 2**level) in attn_resolutions:
                    layers.append(_EDM_AttentionBlock(ch))
                if level and i == num_blocks:
                    layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
        
        self.out = nn.Sequential(
            _EDM_Normalize(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, sigma, class_labels):
        # 1. EDM 预处理与时间步嵌入 (sigma 是噪声水平)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        c_noise = sigma.log() / 4 
        model_channels = self.time_embed[0].in_features 
        half_dim = model_channels // 2
        # 计算不同频率
        frequencies = torch.exp(torch.arange(half_dim, device=x.device) * -math.log(10000) / (half_dim - 1))
        
        # 将 c_noise 投影到不同频率上
        fourier_features = c_noise * frequencies[None, :]
        
        # 拼接 sin 和 cos
        timestep_embedding = torch.cat([fourier_features.sin(), fourier_features.cos()], dim=1)
        # 确保在奇数维度时也能工作
        if model_channels % 2 == 1:
            timestep_embedding = F.pad(timestep_embedding, (0, 1), "constant", 0)
        # 现在的 timestep_embedding 形状是 [batch_size, model_channels], e.g., [128, 128]
        
        emb = self.time_embed(timestep_embedding)

        # 2. 类别条件嵌入
        if self.label_dim > 0:
            if class_labels is None:
                class_labels = torch.zeros(x.shape[0], self.label_dim, device=x.device)
            # class_labels 必须是 [batch_size, label_dim] 的 one-hot 张量
            emb = emb + self.label_embed(class_labels.to(torch.float32))

        # 3. EDM U-Net 主体
        skips = []
        h = self.input_blocks[0](x) # 第一个是单独的 Conv2d
        skips.append(h)

        # 遍历 input_blocks (从索引1开始)
        for module in self.input_blocks[1:]:
            if isinstance(module, nn.AvgPool2d):
                h = module(h)
            elif isinstance(module, nn.Sequential):
                for layer in module: # 遍历 nn.Sequential 内部的层
                    if isinstance(layer, _EDM_ResBlock):
                        h = layer(h, emb)
                    elif isinstance(layer, _EDM_AttentionBlock):
                        h = layer(h)
                    else:
                        h = layer(h)
            skips.append(h)
        
        # 遍历 middle_block
        for layer in self.middle_block:
            if isinstance(layer, _EDM_ResBlock):
                h = layer(h, emb)
            elif isinstance(layer, _EDM_AttentionBlock):
                h = layer(h)
            else:
                h = layer(h)

        # 遍历 output_blocks
        for module in self.output_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            for layer in module:
                if isinstance(layer, _EDM_ResBlock):
                    h = layer(h, emb)
                elif isinstance(layer, _EDM_AttentionBlock):
                    h = layer(h)
                elif isinstance(layer, nn.Upsample):
                    h = layer(h)
                else:
                    h = layer(h)

        h = self.out(h)
        return h

# ==============================================================================
# === 异构分类器模型                                                          ===
# ==============================================================================

# # 基础模型
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         # ... (代码不变) ...
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 8 * 8, 512)
#         self.fc2 = nn.Linear(512, 10)
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         # ... (代码不变) ...
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 8 * 8)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # 模型1: 更宽的网络
# class SimpleCNN_Wide(nn.Module):
#     def __init__(self):
#         super(SimpleCNN_Wide, self).__init__()
#         # ... (代码不变) ...
#         self.conv1 = nn.Conv2d(3, 48, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(96 * 8 * 8, 512)
#         self.fc2 = nn.Linear(512, 10)
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         # ... (代码不变) ...
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(-1, 96 * 8 * 8)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # 模型2: 更深的网络
# class SimpleCNN_Deep(nn.Module):
#     def __init__(self):
#         super(SimpleCNN_Deep, self).__init__()
#         # ... (代码不变) ...
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 8 * 8, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 10)
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         # ... (代码不变) ...
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 8 * 8)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# --- 通用的 ResNet 基类 ---
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # --- VVV 骨干网络 (Backbone) VVV ---
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_layer(block, 64, num_blocks[0], stride=1),
            self._make_layer(block, 128, num_blocks[1], stride=2),
            self._make_layer(block, 256, num_blocks[2], stride=2),
            self._make_layer(block, 512, num_blocks[3], stride=2),
            nn.AdaptiveAvgPool2d((1, 1)) # 平均池化层
        )
        
        feature_dim = 512 * block.expansion

        # --- VVV 分类器 (Classifier) VVV ---
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # --- VVV 投影层 (Projector), 用于 LTE 和 FedD3A VVV ---
        self.projector = nn.Linear(feature_dim, 512) # 将特征映射到512维公共空间

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def extract_features(self, x):
        """仅执行骨干网络以提取特征"""
        features = self.backbone(x)
        # 将特征展平为向量
        return features.view(features.size(0), -1)

    def forward(self, x):
        """执行完整的正向传播"""
        features = self.extract_features(x)
        output = self.classifier(features)
        return output

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()

        # --- VVV 骨干网络 (Backbone) VVV ---
        # 标准 LeNet-5 结构，适配 3通道 32x32 输入
        self.backbone = nn.Sequential(
            # C1: 卷积层 (3 -> 6), kernel=5
            # Input: 3x32x32 -> Output: 6x28x28
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            
            # S2: 池化层
            # Input: 6x28x28 -> Output: 6x14x14
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # C3: 卷积层 (6 -> 16), kernel=5
            # Input: 6x14x14 -> Output: 16x10x10
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            
            # S4: 池化层
            # Input: 16x10x10 -> Output: 16x5x5
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 展平
            nn.Flatten(),
            
            # C5: 全连接层 (16*5*5 -> 120)
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            
            # F6: 全连接层 (120 -> 84)
            nn.Linear(120, 84),
            nn.ReLU()
        )
        
        # LeNet 倒数第二层的输出维度是 84
        feature_dim = 84

        # --- VVV 分类器 (Classifier) VVV ---
        # 对应 LeNet 的输出层
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # --- VVV 投影层 (Projector) VVV ---
        # 保持与 ResNet 相同的接口，将 84 维特征映射到 512 维公共空间
        # 这对于你后续的特征对齐或对比学习至关重要
        self.projector = nn.Linear(feature_dim, 512)

    def extract_features(self, x):
        """仅执行骨干网络以提取特征"""
        # 输出维度: [batch_size, 84]
        return self.backbone(x)

    def forward(self, x):
        """执行完整的正向传播"""
        features = self.extract_features(x)
        output = self.classifier(features)
        return output

# --- 便捷调用函数 ---
def LeNet5_CIFAR():
    """LeNet-5 adapted for CIFAR-10"""
    return LeNet(num_classes=10)

# --- 创建不同规模的 ResNet 变体 (用于异构场景) ---

def ResNet18_CIFAR():
    """ResNet-18 for CIFAR-10"""
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34_CIFAR():
    """ResNet-34 for CIFAR-10"""
    return ResNet(BasicBlock, [3, 4, 6, 3])

# 这是一个自定义的、更小的 ResNet 变体，用于增加异构性
def ResNet10_CIFAR():
    """A smaller ResNet-10 for CIFAR-10"""
    return ResNet(BasicBlock, [1, 1, 1, 1])
