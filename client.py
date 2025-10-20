# client.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import torch.nn.functional as F
from dkd_loss import dkd_loss

# 从我们自己的模块中导入
# 确保这些模块路径正确且包含所需的类和变量
from models import AdvancedUNet
from diffusion_utils import (
    TIMESTEPS, q_sample, alphas, alphas_cumprod, betas, recip_sqrt_alphas
)

class Client:
    def __init__(self, client_id, local_data, device, model_instance, lte_all_classes, lambda_ltc, bounding_radius):
        self.id = client_id
        self.device = device
        self.data_loader = DataLoader(local_data, batch_size=32, shuffle=True)
        self.projection_matrix = None
        
        self.model = model_instance.to(device)
        self.optimizer_model = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        self.diffusion_model = AdvancedUNet().to(device) 
        self.optimizer_diffusion = optim.Adam(self.diffusion_model.parameters(), lr=1e-4)
        
        self.neighbors = []
        
        # --- VVV 新增：存储 LTE 和超参数 VVV ---
        self.lte_all_classes = lte_all_classes
        self.lambda_ltc = lambda_ltc
        self.bounding_radius_sq = bounding_radius ** 2 # 存储半径的平方以避免开方运算

    def add_neighbor(self, neighbor_client):
        self.neighbors.append(neighbor_client)

    # ==============================================================================
    # === 1. 基础训练方法 ===
    # ==============================================================================

    def train_local_model_step(self, epochs=1):
        """本地训练，结合了交叉熵损失和 Bounding Loss"""
        self.model.train()
        total_loss, batch_count = 0, 0
        for _ in range(epochs):
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer_model.zero_grad()
                
                # 1. 正常的前向传播和分类损失
                output = self.model(data)
                loss_ce = F.cross_entropy(output, target)
                
                # 2. 计算 Bounding Loss (参考论文 Eq. 3)
                # 提取特征并投影到与LTE相同的维度
                features = self.model.extract_features(data)
                projected_features = self.model.projector(features)
                
                # 获取这批数据对应的LTE锚点
                batch_ltes = self.lte_all_classes[target]
                
                # 计算特征与LTE锚点之间的距离的平方
                dist_sq = ((projected_features - batch_ltes) ** 2).sum(dim=1)
                
                # Bounding Loss = max(0, distance^2 - radius^2)
                loss_bounding = F.relu(dist_sq - self.bounding_radius_sq).mean()
                
                # 3. 组合损失
                total_loss_batch = loss_ce + self.lambda_ltc * loss_bounding
                
                total_loss_batch.backward()
                self.optimizer_model.step()
                
                total_loss += total_loss_batch.item()
                batch_count += 1
                
        return total_loss / batch_count if batch_count > 0 else 0.0

    # ==============================================================================
    # === 2. 扩散模型相关方法 (用于 'generated' 数据源) ===
    # ==============================================================================

    def train_local_diffusion_step(self, epochs=1):
        self.diffusion_model.train()
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            for real_imgs, _ in self.data_loader:
                self.optimizer_diffusion.zero_grad()
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)
                t = torch.randint(0, TIMESTEPS, (batch_size,), device=self.device).long()
                noise = torch.randn_like(real_imgs)
                noisy_imgs = q_sample(x_start=real_imgs, t=t, noise=noise)
                predicted_noise = self.diffusion_model(noisy_imgs, t)
                loss = loss_fn(noise, predicted_noise)
                loss.backward()
                self.optimizer_diffusion.step()

    def perform_generator_consensus(self):
        if not self.neighbors:
            return
        diffusion_state_dict = self.diffusion_model.state_dict()
        for neighbor in self.neighbors:
            for key in diffusion_state_dict:
                diffusion_state_dict[key] += neighbor.diffusion_model.state_dict()[key]
        for key in diffusion_state_dict:
            diffusion_state_dict[key] /= (len(self.neighbors) + 1)
        self.diffusion_model.load_state_dict(diffusion_state_dict)

    @torch.no_grad()
    def generate_normal_samples_diffusion(self, num_samples=10):
        self.diffusion_model.eval()
        img = torch.randn((num_samples, 3, 32, 32), device=self.device)
        for i in reversed(range(0, TIMESTEPS)):
            t = torch.full((num_samples,), i, device=self.device, dtype=torch.long)
            predicted_noise = self.diffusion_model(img, t)
            alpha_t = alphas[i].to(self.device); alpha_t_cumprod = alphas_cumprod[i].to(self.device)
            beta_t = betas[i].to(self.device); recip_sqrt_alpha_t = recip_sqrt_alphas[i].to(self.device)
            term1 = recip_sqrt_alpha_t * (img - (beta_t / torch.sqrt(1. - alpha_t_cumprod)) * predicted_noise)
            noise = torch.randn_like(img) if i > 0 else torch.zeros_like(img)
            img = term1 + torch.sqrt(beta_t) * noise
        img = (img.clamp(-1, 1) + 1) / 2
        return img.detach()

    def generate_scarcity_hard_samples_diffusion(self, num_samples):
        """
        生成“困难样本”，优先生成该客户端本地数据量稀缺的类别。
        """
        self.diffusion_model.eval()

        # 1. 根据数据稀缺性定义采样概率
        # 为避免除以零，给类别计数加上一个很小的数
        epsilon = 1e-6
        # 使用计数的倒数作为权重。计数越低，权重越高。
        weights = 1.0 / (self.class_counts + epsilon)
        
        # 将权重归一化，得到一个概率分布
        probabilities = weights / weights.sum()

        # 2. 根据这个新的概率分布，采样 num_samples 个类别标签
        # 这会导致数据量少的类别被更频繁地选中
        class_labels = torch.multinomial(probabilities, num_samples, replacement=True).to(self.device)
        
        # 3. 使用条件扩散模型，为这些被优先选择的类别生成图像
        # (这部分代码与 generate_normal_samples_diffusion 中的生成逻辑完全相同)
        class_emb = self.lte_all_classes[class_labels]
        
        img = torch.randn((num_samples, 3, 32, 32), device=self.device)
        for i in reversed(range(0, TIMESTEPS)):
            t = torch.full((num_samples,), i, device=self.device, dtype=torch.long)
            
            # 将 LTE 嵌入作为条件传入模型
            predicted_noise = self.diffusion_model(img, t, class_emb)
            
            # DDPM 采样步骤
            alpha_t = alphas[i].to(self.device)
            alpha_t_cumprod = alphas_cumprod[i].to(self.device)
            beta_t = betas[i].to(self.device)
            recip_sqrt_alpha_t = recip_sqrt_alphas[i].to(self.device)
            
            term1 = recip_sqrt_alpha_t * (img - (beta_t / torch.sqrt(1. - alpha_t_cumprod)) * predicted_noise)
            noise = torch.randn_like(img) if i > 0 else torch.zeros_like(img)
            img = term1 + torch.sqrt(beta_t) * noise
        
        img = (img.clamp(-1, 1) + 1) / 2
        return img.detach()

    # ==============================================================================
    # === 3. 知识蒸馏方法 ===
    # ==============================================================================
    def distill(self, distillation_data, teacher_selector, sample_filter, hard_sample_ratio, distill_mode, dkd_alpha, dkd_beta):
        """
        统一的、经过重构的知识蒸馏函数。
        """
        if not self.neighbors: return 0.0
        self.model.train()

        # ==================== 阶段一：样本筛选 (可选) ====================
        
        filtered_data = distillation_data
        
        if sample_filter == 'confidence':
            with torch.no_grad():
                all_teacher_models = [n.model for n in self.neighbors]
                if not all_teacher_models: return 0.0
                for teacher in all_teacher_models: teacher.eval()

                all_confidences = [torch.max(F.softmax(teacher(distillation_data), dim=1), dim=1)[0] for teacher in all_teacher_models]
                avg_confidence = torch.stack(all_confidences).mean(dim=0)
                
                num_hard_samples = int(distillation_data.shape[0] * hard_sample_ratio)
                if num_hard_samples == 0: return 0.0
                
                _, hard_indices = torch.topk(avg_confidence, k=num_hard_samples, largest=False)
                filtered_data = distillation_data[hard_indices]

        if filtered_data.shape[0] == 0: return 0.0

        # ==================== 阶段二：教师选择与权重计算 ====================

        # 默认使用所有邻居作为教师
        active_teachers = [n.model for n in self.neighbors]
        alpha = None
        
        # --- 根据选择器策略，决定 active_teachers 和 alpha ---
        if teacher_selector == 'all':
            num_teachers = len(active_teachers)
            if num_teachers == 0: return 0.0
            alpha = torch.ones(filtered_data.shape[0], num_teachers, device=self.device) / num_teachers
        
        elif teacher_selector == 'random':
            if not active_teachers: return 0.0
            num_teachers = len(active_teachers)
            alpha = torch.zeros(filtered_data.shape[0], num_teachers, device=self.device)
            chosen_idx = random.randint(0, num_teachers - 1)
            alpha[:, chosen_idx] = 1.0

        else: # 处理所有自适应模式 ('fedd3a', 'top1', 'top2', 'top3')
            teacher_models = [n.model for n in self.neighbors]
            teacher_matrices = [n.projection_matrix for n in self.neighbors]
            
            # 筛选出同时拥有模型和投影矩阵的有效教师
            valid_teachers = [(m, p) for m, p in zip(teacher_models, teacher_matrices) if p is not None]
            if not valid_teachers:
                print(f"客户端 {self.id}: 在自适应模式下没有可用的教师投影矩阵。")
                return 0.0
            
            # **关键修正**: 将 active_teachers 更新为筛选后的有效教师列表
            active_teachers, teacher_matrices = zip(*valid_teachers)
            
            # 1. 计算原始相似度 r
            raw_student_features = self.model.extract_features(filtered_data)
            projected_student_features = self.model.projector(raw_student_features)
            cos = nn.CosineSimilarity(dim=1)
            r = torch.stack([cos(projected_student_features, projected_student_features @ P_k) for P_k in teacher_matrices], dim=1)
            
            # 2. 根据选择器计算 alpha
            if teacher_selector == 'fedd3a':
                if r.shape[1] > 1:
                    r_normalized = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-8)
                else:
                    r_normalized = r
                alpha = F.softmax(r_normalized, dim=1)
            
            elif 'top' in teacher_selector:
                k = int(teacher_selector[-1])
                num_valid_teachers = len(active_teachers)
                alpha = torch.zeros_like(r)
                
                # 使用 min(k, num_valid_teachers) 确保 k 不会超过可用教师数量
                actual_k = min(k, num_valid_teachers)
                if actual_k > 0:
                    topk_scores, topk_indices = torch.topk(r, k=actual_k, dim=1)
                    topk_weights = F.softmax(topk_scores, dim=1)
                    alpha.scatter_(1, topk_indices, topk_weights)

        # 确保 alpha 被成功赋值
        if alpha is None:
            print(f"警告: 客户端 {self.id} 的 alpha 权重未能计算，跳过蒸馏。")
            return 0.0

        # ==================== 阶段三：执行知识蒸馏 ====================
        
        temperature = 2.0
        with torch.no_grad():
            for teacher in active_teachers: teacher.eval()
            # **关键修正**: 这里的 teacher_logits 是基于 active_teachers 计算的
            teacher_logits = torch.stack([teacher(filtered_data) for teacher in active_teachers], dim=0)

        # 现在 alpha 和 teacher_logits 的维度一定匹配
        y_hat = torch.einsum('bt,tbc->bc', alpha, teacher_logits)
        student_logits = self.model(filtered_data)
        if distill_mode == 'dkd':
            # 使用 DKD 损失
            # 我们需要一个目标标签。对于 data-free 场景，最好的选择是使用教师集体的预测作为伪标签。
            with torch.no_grad():
                pseudo_labels = torch.argmax(y_hat, dim=1)
            
            loss = dkd_loss(
                student_logits=student_logits,
                teacher_logits=y_hat,
                target_labels=pseudo_labels,
                alpha=dkd_alpha,
                beta=dkd_beta,
                temperature=temperature
            )
        else: # 默认为 'kd' (经典KL散度)
            loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(y_hat / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)

        # loss = F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(y_hat, dim=1), reduction='batchmean')
        
        self.optimizer_model.zero_grad()
        loss.backward()
        self.optimizer_model.step()

        return loss.item()

    # --- VVV FedD3A 核心方法 VVV ---
    @torch.no_grad()
    def calculate_projection_matrix(self):
        """(客户端功能) 使用SVD为本地数据计算子空间投影矩阵。"""
        self.model.eval()
        all_projected_features = []
        for data, _ in self.data_loader:
            data = data.to(self.device)
            # 1. 提取原始特征
            raw_features = self.model.extract_features(data)
            # 2. VVV 新增：将特征投影到公共空间 VVV
            projected_features = self.model.projector(raw_features)
            all_projected_features.append(projected_features)
        
        if not all_projected_features:
            return

        # 3. 在公共空间中（例如512维）构建特征矩阵Z
        Z_projected = torch.cat(all_projected_features, dim=0)
        try:
            # 4. 在公共空间中计算投影矩阵
            U, _, _ = torch.linalg.svd(Z_projected.T, full_matrices=False)
            self.projection_matrix = U @ U.T
        except torch.linalg.LinAlgError:
            print(f"客户端 {self.id} 的 SVD 计算失败，跳过投影矩阵更新。")

    # ==============================================================================
    # === 4. 评估方法 ===
    # ==============================================================================
    def evaluate(self, test_loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100.0 * correct / total