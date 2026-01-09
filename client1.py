# client.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import torch.nn.functional as F
from dkd_loss1 import dkd_loss
import numpy as np
import sys
import pickle
import os  
from datetime import datetime  
from torchvision.utils import save_image
import math
from torch.optim.lr_scheduler import LambdaLR

# 从 models.py 导入新模型
from models import EDMUNet, ResNet18_CIFAR, ResNet34_CIFAR
from diffusion_utils import (
    TIMESTEPS, q_sample, alphas, alphas_cumprod, betas, recip_sqrt_alphas
)

class Client:
    def __init__(self, client_id, local_data, device, model_instance, rounds):
        self.id = client_id
        self.device = device
        self.data_loader = DataLoader(local_data, batch_size=128, shuffle=True)
        # --- VVV 新增：计算并存储本地数据的类别计数 VVV ---
        if hasattr(local_data.dataset, 'targets'):
            all_targets = np.array(local_data.dataset.targets)
            client_labels = all_targets[local_data.indices]
            self.real_class_counts = torch.tensor(np.bincount(client_labels, minlength=10), dtype=torch.float32, device=device)
        else:
            self.real_class_counts = torch.ones(10, device=device) # Fallback
        
        self.dynamic_class_counts = self.real_class_counts
        
        self.cached_complementary_probs = None

        self.projection_matrix = None
        
        self.model = model_instance.to(device)
        self.optimizer_model = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        self.scheduler_model = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_model, T_max=rounds, eta_min=1e-4)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # self.base_lr = 0.1
        # self.optimizer_model = optim.SGD(self.model.parameters(), lr=self.base_lr, momentum=0.9, weight_decay=1e-4)

        # # ------------------------------------------------------------------
        # # 4. [修改核心] 自定义学习率调度器 (LambdaLR)
        # # ------------------------------------------------------------------
        # # 保存调度所需的参数
        # self.total_rounds = rounds        # 总轮次 (例如 500)
        # self.split_round = 300            # 切换点 (例如 300)
        # self.min_lr = 1e-4                # 最低下降值
        # self.restart_lr = 0.01            # 第301轮跳回的"灵活值"

        # # 生成 lambda 函数并传入调度器
        # lr_lambda_func = self._get_lr_lambda_func()
        # self.scheduler_model = LambdaLR(self.optimizer_model, lr_lambda=lr_lambda_func)
        
        self.diffusion_model = EDMUNet(
            img_resolution=32, 
            in_channels=3, 
            out_channels=3,
            label_dim=10 # CIFAR-10 有10个类别
        ).to(device)
        self.optimizer_diffusion = optim.Adam(self.diffusion_model.parameters(), lr=1e-4)

        self.neighbors = []

    def _get_lr_lambda_func(self):
        """
        构造并返回用于 LambdaLR 的计算函数。
        """
        # 为了闭包能访问到 self 中的属性，我们在内部定义 logic
        def lr_lambda(epoch):
            # --- 辅助函数：计算带下限的 Cosine ---
            def get_cosine_factor(cur_step, total_steps, peak_val, floor_val):
                cosine_decay = 0.5 * (1 + math.cos(math.pi * cur_step / total_steps))
                target_lr = floor_val + (peak_val - floor_val) * cosine_decay
                return target_lr / self.base_lr  # 返回相对于 base_lr (0.1) 的比例

            # 阶段 1: 0 ~ 299 轮 (下降到 min_lr)
            if epoch < self.split_round:
                return get_cosine_factor(
                    cur_step=epoch, 
                    total_steps=self.split_round, 
                    peak_val=self.base_lr, 
                    floor_val=self.min_lr
                )
            
            # 阶段 2: 300 ~ End (从 restart_lr 再次下降)
            else:
                return get_cosine_factor(
                    cur_step=epoch - self.split_round, 
                    total_steps=self.total_rounds - self.split_round, 
                    peak_val=self.restart_lr,  # 这里设置你想跳回的那个灵活值
                    floor_val=self.min_lr
                )
        
        return lr_lambda

    def UpdateClassCounts(self, labels, device):
        # 直接在 GPU 上计算，保持全 PyTorch 流程
        new_counts = torch.bincount(labels, minlength=10).to(dtype=torch.float32, device=device)
        self.dynamic_class_counts = self.real_class_counts + new_counts

    def get_real_sample_batch(self, batch_size):
        """从本地数据加载器中随机抽取一批真实数据"""
        try:
            # 尝试从现有的迭代器获取
            if not hasattr(self, '_train_loader_iter'):
                self._train_loader_iter = iter(self.data_loader)
            images, labels = next(self._train_loader_iter)
        except StopIteration:
            # 如果迭代器用尽，重新创建一个
            self._train_loader_iter = iter(self.data_loader)
            images, labels = next(self._train_loader_iter)
        
        # 如果获取的数据不足 batch_size (比如最后一个batch)，则可能需要补齐，这里简化处理直接返回
        return images.to(self.device), labels.to(self.device)
    
    def add_neighbor(self, neighbor_client):
        self.neighbors.append(neighbor_client)

    # ==============================================================================
    # === 1. 基础训练方法 ===
    # ==============================================================================

    def train_local_model_step(self, epochs=1, aux_data=None, num_aux_data=0):
        self.model.train()
        total_loss, batch_count = 0, 0
        
        # --- VVV 核心修改：数据预处理与合并 VVV ---
        
        # 1. 收集本地数据 (从 DataLoader 中提取并合并)
        local_imgs_list = []
        local_lbls_list = []
        
        for batch_data, batch_labels in self.data_loader:
            local_imgs_list.append(batch_data)
            local_lbls_list.append(batch_labels)
        
        # 将列表拼接成单个大张量，并移动到 GPU
        if len(local_imgs_list) > 0:
            train_images = torch.cat(local_imgs_list, dim=0).to(self.device)
            train_targets = torch.cat(local_lbls_list, dim=0).to(self.device)
        else:
            # 防止本地数据为空的极端情况
            train_images = torch.tensor([], device=self.device)
            train_targets = torch.tensor([], device=self.device)

        # 2. 如果有辅助数据，进行拼接 (Mix)
        if aux_data is not None:
            aux_imgs, aux_lbls = aux_data
            aux_imgs = aux_imgs.to(self.device)
            aux_lbls = aux_lbls.to(self.device)
            
            # 拼接本地数据和辅助数据
            train_images = torch.cat([train_images, aux_imgs], dim=0)
            train_targets = torch.cat([train_targets, aux_lbls], dim=0)
            
        # 获取当前混合后的总样本数 (修正：不能只用 num_aux_data)
        current_num_samples = train_images.size(0)
        
        if current_num_samples == 0:
            return 0.0
        # --- ^^^ 预处理结束 ^^^ ---

        for _ in range(epochs):
            # 3. 生成随机索引 (Shuffle)
            indices = torch.randperm(current_num_samples, device=self.device)
            
            # 4. 标准 Batch 循环
            # 建议使用 self.local_batch_size 或默认 128
            batch_size = 128 
            for start_idx in range(0, current_num_samples, batch_size):
                end_idx = min(start_idx + batch_size, current_num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # 现在 train_images 是 Tensor，可以正确支持 Tensor索引
                data = train_images[batch_indices]
                target = train_targets[batch_indices]
                
                self.optimizer_model.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer_model.step()
                
                total_loss += loss.item()
                batch_count += 1
                
        return total_loss / batch_count if batch_count > 0 else 0.0

    # ==============================================================================
    # === 2. 扩散模型相关方法 (用于 'generated' 数据源) ===
    # ==============================================================================

    def train_local_diffusion_step(self, epochs=1):
        self.diffusion_model.train()
        for _ in range(epochs):
            for real_imgs, labels in self.data_loader:
                real_imgs, labels = real_imgs.to(self.device), labels.to(self.device)
                
                # 将标签转换为 one-hot 编码
                class_labels = F.one_hot(labels, num_classes=10).to(torch.float32)
                
                self.optimizer_diffusion.zero_grad()
                
                # 1. EDM 的噪声采样方式 (P_mean=-1.2, P_std=1.2)
                sigma = (torch.randn(real_imgs.shape[0], device=self.device) * 1.2 - 1.2).exp()
                noise = torch.randn_like(real_imgs) * sigma.reshape(-1, 1, 1, 1)
                
                # 2. 加噪
                noisy_imgs = real_imgs + noise
                
                # 3. EDM 的输入缩放
                c_in = 1 / (sigma ** 2 + 1).sqrt()
                model_input = noisy_imgs * c_in.reshape(-1, 1, 1, 1)
                
                # 4. 模型预测
                predicted_output = self.diffusion_model(model_input, sigma, class_labels)
                
                # 5. EDM 的目标和损失计算 (与预训练方案不同，这里我们使用简化的 l2 损失)
                # EDM 论文中的标准 Karras 损失函数
                c_skip = 1 / (sigma**2 + 1)
                c_out = sigma / (sigma**2 + 1).sqrt()
                
                target = (real_imgs - c_skip.reshape(-1, 1, 1, 1) * noisy_imgs) / c_out.reshape(-1, 1, 1, 1)
                
                loss = (predicted_output - target).pow(2).mean()

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
    def generate_edm_samples(self, num_samples, class_labels, num_steps=18):
        self.diffusion_model.eval()
        
        # 将整数标签转换为 one-hot 编码
        class_labels_one_hot = F.one_hot(class_labels, num_classes=10).to(torch.float32)

        # 1. 定义采样步长 (sigma schedule)
        sigma_min = 0.002
        sigma_max = 80
        rho = 7
        step_indices = torch.arange(num_steps, device=self.device, dtype=torch.float32)
        sigmas = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])]) # 添加 sigma=0

        # 2. 从纯噪声开始
        x = torch.randn((num_samples, 3, 32, 32), device=self.device) * sigmas[0]

        # 3. 执行 Heun's 2nd order ODE 求解器
        for i in range(num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i+1]
            
            # 第一次模型评估 (Euler step)
            c_in = 1 / (sigma ** 2 + 1).sqrt()
            c_skip = 1 / (sigma**2 + 1)
            c_out = sigma / (sigma**2 + 1).sqrt()

            pred = self.diffusion_model(x * c_in.reshape(-1,1,1,1), sigma.repeat(num_samples), class_labels_one_hot)
            denoised = pred * c_out.reshape(-1,1,1,1) + x * c_skip.reshape(-1,1,1,1)
            d = (x - denoised) / sigma
            
            x_next = x + d * (sigma_next - sigma)
            
            # 第二次模型评估 (Corrector step)
            if i < num_steps - 1:
                c_in_next = 1 / (sigma_next ** 2 + 1).sqrt()
                c_skip_next = 1 / (sigma_next**2 + 1)
                c_out_next = sigma_next / (sigma_next**2 + 1).sqrt()

                pred_next = self.diffusion_model(x_next * c_in_next.reshape(-1,1,1,1), sigma_next.repeat(num_samples), class_labels_one_hot)
                denoised_next = pred_next * c_out_next.reshape(-1,1,1,1) + x_next * c_skip_next.reshape(-1,1,1,1)
                d_next = (x_next - denoised_next) / sigma_next
                
                # Heun's method
                x = x + (d + d_next) * ((sigma_next - sigma) / 2)
            else:
                x = x_next
        
        # 4. 后处理
        x = (x.clamp(-1, 1) + 1) / 2
        return x.detach()

    # ==============================================================================
    # === 3. 知识蒸馏方法 ===
    # ==============================================================================

    def select_complementary_teachers(self, num_select, round):
        """
        [最终修正版] 
        选择逻辑：Softmax (强者优先)
        权重逻辑：Inverse Score (反比加权，拒绝0权重)
        """
        if len(self.neighbors) <= num_select:
            # 邻居很少时，基于分数的反比计算权重
            scores = []
            for n in self.neighbors:
                s = torch.std(self.dynamic_class_counts + n.dynamic_class_counts).item()
                scores.append(s)
            scores_tensor = torch.tensor(scores, device=self.device)
            # 反比权重: 1 / score
            inv_scores = 1.0 / (scores_tensor + 1e-6)
            weights = inv_scores / inv_scores.sum()
            return [n.model for n in self.neighbors], [n.id for n in self.neighbors], weights

        # ============================================================
        # 1. 缓存原始分数 (只计算一次)
        # ============================================================
        if self.cached_complementary_probs is None or round % 10 == 1: # 这里借用这个变量名存分数
            imbalance_scores = []
            for neighbor in self.neighbors:
                # 计算标准差
                score = torch.std(self.dynamic_class_counts + neighbor.dynamic_class_counts).item()
                imbalance_scores.append(score)
            
            # 存为 Tensor
            self.cached_complementary_probs = torch.tensor(imbalance_scores, device=self.device)
        
        # 获取缓存的分数
        raw_scores = self.cached_complementary_probs

        # ============================================================
        # 2. 计算选择概率 (用于决定"选谁")
        # ============================================================
        # 使用 Z-Score + Softmax，这是为了"大概率选到好老师"
        # 如果你觉得还是太极端，可以把这里的 T 从 2.0 降到 1.0 或 0.5
        norm_scores = (raw_scores - raw_scores.mean()) / (raw_scores.std() + 1e-6)
        selection_probs = F.softmax(-norm_scores * 1.0, dim=0) # T=1.0 温和一点
        
        # 采样索引
        selected_indices = torch.multinomial(selection_probs, num_select, replacement=False)
        
        # ============================================================
        # 3. 计算聚合权重 (用于决定"听多少") - 线性反比
        # ============================================================
        # 提取被选中老师的原始分数 (标准差)
        selected_raw_scores = raw_scores[selected_indices]
        
        # 逻辑：分数(std)越小越好 -> 权重应越大 -> 取倒数 1/score
        # 加上 1e-6 防止除以 0
        inverse_scores = 1.0 / (selected_raw_scores + 1e-6)
        
        # 还可以加一个幂次来控制平滑度 (可选)
        # power = 1.0 (线性反比) -> 结果可能是 0.6/0.4
        # power = 2.0 (平方反比) -> 结果可能是 0.8/0.2 (更偏向好的)
        # power = 0.5 (根号反比) -> 结果可能是 0.55/0.45 (更平均)
        power = 2.0 
        inverse_scores = inverse_scores ** power
        
        # 归一化
        selected_weights = inverse_scores / inverse_scores.sum()
        
        # 4. 提取结果
        selected_teachers = [self.neighbors[i].model for i in selected_indices]
        selected_ids = [self.neighbors[i].id for i in selected_indices]
        
        return selected_teachers, selected_ids, selected_weights
    
    def distill(self, distillation_data, distill_labels, teacher_selector, sample_filter, hard_sample_ratio, distill_mode, dkd_alpha, dkd_beta, round_idx, total_rounds):
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
        num_teachers = len(active_teachers)
        alpha = None
        teacher_ids = []
        
        # --- 根据选择器策略，决定 active_teachers 和 alpha ---
        if teacher_selector == 'all' or teacher_selector == 'random':
            active_teachers = [n.model for n in self.neighbors]
            teacher_ids = [n.id for n in self.neighbors] # 记录所有邻居的ID
            num_teachers = len(active_teachers)

            if num_teachers == 0: return 0.0

            if teacher_selector == 'all':
                alpha = torch.ones(filtered_data.shape[0], num_teachers, device=self.device) / num_teachers
            
            elif teacher_selector == 'random':
                alpha = torch.zeros(filtered_data.shape[0], num_teachers, device=self.device)
                chosen_idx = random.randint(0, num_teachers - 1)
                alpha[:, chosen_idx] = 1.0
        
        elif teacher_selector == 'complementary':
            # 每一轮随机取 2 个 (您可以根据需要修改 k)
            k = 1
            active_teachers, teacher_ids, comp_weights= self.select_complementary_teachers(num_select=k, round=round_idx)
            num_teachers = len(active_teachers)

            if isinstance(comp_weights, torch.Tensor):
                weights_list = comp_weights.tolist()
            else:
                # 假如它已经是 list，但里面是 tensor，则需要逐个转 item
                weights_list = [w.item() if isinstance(w, torch.Tensor) else w for w in comp_weights]
            # 2. 现在 weights_list 里全是 float，可以安全使用 round_idx
            formatted_weights = [round(w, 4) for w in weights_list]
            # print(f"Client {self.id} [Complementary] -> 选中教师: {teacher_ids}, 权重: {formatted_weights}")
            
            if num_teachers == 0: return 0.0

            alpha = comp_weights.view(1, -1).expand(filtered_data.shape[0], -1)

        elif 'expert' in teacher_selector:
            active_teachers = [n.model for n in self.neighbors]
            teacher_ids = [n.id for n in self.neighbors] # 记录所有邻居的ID
            if not active_teachers: return 0.0
            with torch.no_grad():
                teacher_class_counts = torch.stack([n.dynamic_class_counts for n in self.neighbors])
                target_labels = distill_labels
                relevant_counts_t = teacher_class_counts[:, target_labels].T # 形状: [batch_size, num_teachers]
            
            alpha = torch.zeros_like(relevant_counts_t)
            num_teachers = len(active_teachers)
            k = 0
            
            if teacher_selector == 'expert_top1': k = 1
            elif teacher_selector == 'expert_top2': k = 2
            elif teacher_selector == 'expert_top3': k = 3
            
            if k > 0:
                actual_k = min(k, num_teachers)
                if actual_k > 0:
                    topk_scores, topk_indices = torch.topk(relevant_counts_t, k=actual_k, dim=1)
                    if k == 1:
                         alpha.scatter_(1, topk_indices, 1.0)
                    else:
                        topk_weights = F.softmax(topk_scores.float(), dim=1)
                        alpha.scatter_(1, topk_indices, topk_weights)

            elif teacher_selector == 'expert_all':
                expert_mask = (relevant_counts_t > 0)
                num_experts = expert_mask.sum(dim=1, keepdim=True)
                alpha = torch.where(num_experts > 0, expert_mask.float() / num_experts, 0.0)
                no_expert_mask = (num_experts == 0).squeeze()
                if no_expert_mask.any():
                    alpha[no_expert_mask] = 1.0 / num_teachers


        else: # 处理所有自适应模式 ('fedd3a', 'top1', 'top2', 'top3')
            valid_teachers_data = [(n.model, n.projection_matrix, n.id) for n in self.neighbors if n.projection_matrix is not None]
            if not valid_teachers_data:
                print(f"客户端 {self.id}: 在 {teacher_selector} 模式下没有可用的教师投影矩阵。")
                return 0.0
            
            active_teachers, teacher_matrices, teacher_ids = zip(*valid_teachers_data) # 解包，获取ID
            
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
        
        # if round_idx is not None and round_idx % 10 == 0:
        #     # 计算批次中每个教师获得的平均权重
        #     avg_weights_per_teacher = alpha.mean(dim=0).cpu().numpy()
            
        #     # 格式化输出
        #     log_str = f"  [调试 日志 | 轮次 {round_idx} | 学生 {self.id} | 模式: {teacher_selector}]\n"
        #     log_str += "    > 教师ID:    " + " | ".join([f"{tid:^6}" for tid in teacher_ids]) + "\n"
        #     log_str += "    > 平均权重: " + " | ".join([f"{w:^6.3f}" for w in avg_weights_per_teacher])
        #     print(log_str)

        # ==================== 阶段三：执行知识蒸馏 ====================
        temperature = 2.0
        student_logits = self.model(filtered_data)
        ce_loss = F.cross_entropy(student_logits, distill_labels)
        total_loss = 0.0
        
        # 获取 alpha 的具体数值 (如果它是 Tensor)
        # alpha 形状: [Batch, Num_Teachers]
        
        if distill_mode == 'dkd':
            # 遍历每一个活跃的教师
            for i, teacher in enumerate(active_teachers):
                teacher.eval()
                with torch.no_grad():
                    # 1. 获取 Logits
                    t_logits = teacher(filtered_data) # [B, C]
                    
                    # 2. 计算 Softmax 概率
                    t_probs = F.softmax(t_logits, dim=1)
                    
                    # 3. [关键修改] 获取真实标签对应的置信度 (作为 NCKD 系数)
                    # distill_labels 需要是 LongTensor [B]
                    # target_probs: [B] (范围 0.0 ~ 1.0)
                    target_probs = t_probs.gather(1, distill_labels.view(-1, 1)).squeeze()
                    scale_factor = 1.0 
                    
                    beta_i = target_probs * scale_factor      # [B]
                    alpha_i = (1.0 - target_probs) * scale_factor  # [B]

                # 5. 获取该教师的专家权重 (基于数据量)
                # weight_i: [B]
                weight_i = alpha[:, i] 
                
                if weight_i.sum() == 0:
                    continue

                # 6. 计算 Loss (传入 alpha 和 beta 向量)
                loss_i_vector = self.compute_sample_wise_dkd(
                    student_logits, t_logits, distill_labels, 
                    alpha_i, beta_i, temperature
                )
                
                # 7. 加权累加 (Expert Weight)
                weighted_loss_i = (loss_i_vector * weight_i).sum() / student_logits.size(0)
                total_loss += weighted_loss_i

        else: # 默认为 'kd' (经典KL散度)
            with torch.no_grad():
                for teacher in active_teachers: teacher.eval()
                # **关键修正**: 这里的 teacher_logits 是基于 active_teachers 计算的
                teacher_logits = torch.stack([teacher(filtered_data) for teacher in active_teachers], dim=0)

            y_hat = torch.einsum('bt,tbc->bc', alpha, teacher_logits)
            total_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(y_hat / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)
        # 反向传播
        total_loss += ce_loss
        self.optimizer_model.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer_model.step()

        return total_loss.item()

    # --- 辅助函数：计算样本级 DKD Loss (不求平均) ---
    def compute_sample_wise_dkd(self, student_logits, teacher_logits, target_labels, alpha_vector, beta_vector, temperature):
        """
        计算样本级 DKD Loss。
        参数:
            alpha_vector: [Batch_Size] 动态 TCKD 系数
            beta_vector:  [Batch_Size] 动态 NCKD 系数
        """
        gt_mask = F.one_hot(target_labels, num_classes=teacher_logits.shape[1]).bool()
        other_mask = ~gt_mask

        # --- 1. 计算 TCKD 项 (Target Class Knowledge Distillation) ---
        s_t = F.log_softmax(student_logits / temperature, dim=1)
        t_t = F.softmax(teacher_logits / temperature, dim=1)
        # 结果形状 [B, 1] -> squeeze -> [B]
        tckd_loss = (-(t_t * s_t).sum(dim=1, keepdim=True)) * gt_mask
        tckd_loss = tckd_loss.sum(dim=1) 

        # --- 2. 计算 NCKD 项 (Non-Target Class Knowledge Distillation) ---
        s_oth = F.log_softmax(student_logits / temperature - 1000.0 * gt_mask, dim=1)
        t_oth = F.softmax(teacher_logits / temperature - 1000.0 * gt_mask, dim=1)
        # 结果形状 [B]
        nckd_loss = (-(t_oth * s_oth).sum(dim=1, keepdim=True)) * other_mask
        nckd_loss = nckd_loss.sum(dim=1)

        # --- 3. 动态加权组合 ---
        # 注意：这里直接进行向量乘法 [B] * [B]
        total_loss_vector = alpha_vector * tckd_loss + beta_vector * nckd_loss
        
        return total_loss_vector
        # temperature = 2.0
        # with torch.no_grad():
        #     for teacher in active_teachers: teacher.eval()
        #     # **关键修正**: 这里的 teacher_logits 是基于 active_teachers 计算的
        #     teacher_logits = torch.stack([teacher(filtered_data) for teacher in active_teachers], dim=0)

        # # 现在 alpha 和 teacher_logits 的维度一定匹配
        # y_hat = torch.einsum('bt,tbc->bc', alpha, teacher_logits)

        # with torch.no_grad():
        #     teacher_probs = F.softmax(y_hat, dim=1) # [B, C]
        #     # max_probs: [B]，存储每个样本的最高预测概率
        #     max_probs, _ = torch.max(teacher_probs, dim=1)
            
        #     # 2. 设计自适应函数：使用置信度来缩放基础 Beta 值
        #     # 逻辑: adaptive_beta = base_beta * scaling_factor
            
        #     # 基础 Beta
        #     base_beta_tensor = torch.tensor(dkd_beta, device=self.device)
            
        #     # 缩放因子 (Scaling Factor)：
        #     # 我们直接使用教师的置信度 (max_probs) 作为缩放因子。
        #     # 
        #     # 策略：线性缩放
        #     # 如果置信度为 1.0 (极度自信) -> scaling_factor=1.0 -> adaptive_beta=dkd_beta
        #     # 如果置信度为 0.5 (中等自信) -> scaling_factor=0.5 -> adaptive_beta=0.5*dkd_beta
            
        #     # 设定最低激活阈值 (可选)：防止教师在瞎猜时提供 NCKD (例如，阈值设为 1/C)
        #     num_classes = teacher_probs.shape[1] # 10
        #     min_conf_threshold = 1.0 / num_classes # 0.1
            
        #     # 归一化和非线性（可选，这里使用最简单的线性/ReLU）
        #     # 我们使用 ReLU(max_probs - threshold) 来抑制低于阈值的置信度
        #     scaling_factor = torch.relu(max_probs - min_conf_threshold) 
            
        #     # 重新归一化 scaling_factor 的范围，防止高置信度时 Beta 值过小
        #     # 将 (0, 0.9) 映射到 (0, 1) 的近似范围
        #     max_possible_conf = 1.0 - min_conf_threshold # 0.9
        #     adaptive_beta_scale = scaling_factor / max_possible_conf
        #     adaptive_beta_scale = torch.clamp(adaptive_beta_scale, min=0.0, max=1.0) # 限制在 [0, 1]
            
        #     # 最终的样本级 Beta 张量
        #     adaptive_betas = base_beta_tensor * adaptive_beta_scale

        # # if round_idx % 5 == 0:
        # #     with torch.no_grad():
        # #         # 1. 计算标准 Softmax (T=1) 来查看 "硬" 概率
        # #         teacher_hard_probs = F.softmax(y_hat, dim=1)
                
        # #         # 2. 确定要记录的样本数量 (最多10个)
        # #         num_to_log = min(10, filtered_data.shape[0])
                
        # #         if num_to_log > 0:
        # #             try:
        # #                 # 3. 准备目录和批次时间戳 (用于关联文件名)
        # #                 os.makedirs(self.distill_image_dir, exist_ok=True)
        # #                 batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # #                 all_top_probs = []
                        
        # #                 # 4. 使用 'a' (append) 模式打开日志文件
        # #                 with open(self.distill_log_file, 'a', encoding='utf-8') as f:
        # #                     # 写入一个标题，包含唯一的批次ID (时间戳)
        # #                     f.write(f"====== [客户端 {self.id} 蒸馏日志] - 批次ID: {batch_timestamp} - 记录 {num_to_log} / {filtered_data.shape[0]} 个样本 ======\n")
                            
        # #                     # 5. 循环写入日志并保存图像
        # #                     for i in range(num_to_log):
        # #                         # a. 获取概率信息
        # #                         prob_sample = teacher_hard_probs[i].cpu().numpy()
        # #                         prob_str = "[" + ", ".join(f"{p:.3f}" for p in prob_sample) + "]"
        # #                         top_prob = np.max(prob_sample)
        # #                         top_class = np.argmax(prob_sample)

        # #                         all_top_probs.append(top_prob)
                                
        # #                         # b. 定义图像文件名和路径
        # #                         # 文件名格式: batch_时间戳_sample_序号.png
        # #                         image_filename = f"batch_{batch_timestamp}_sample_{i+1:02d}.png"
        # #                         image_save_path = os.path.join(self.distill_image_dir, image_filename)
                                
        # #                         # c. 保存图像 (save_image 期望 [C, H, W] 格式, 且数据在 [0, 1] 范围)
        # #                         # 您的生成函数似乎已将图像缩放到 [0, 1]，所以这里直接保存
        # #                         image_tensor = filtered_data[i]
        # #                         save_image(image_tensor, image_save_path)
                                
        # #                         f.write(f"  样本 {i+1:02d}: {prob_str} (最高: C{top_class} @ {top_prob:.3f}) -> 保存为: {image_filename}\n")
                            
        # #                     if all_top_probs: # 确保列表不为空
        # #                         avg_top_prob = np.mean(all_top_probs)
        # #                         f.write("  " + "-" * 76 + "\n") # 写入一个小的分隔符
        # #                         f.write(f"  [本批次平均最高置信度 (Avg Top Prob)]: {avg_top_prob:.4f}\n")
        # #                         # 写入一个分隔符
        # #                     f.write("-" * 80 + "\n\n")

        # #             except Exception as e:
        # #                 # 如果文件写入或图像保存失败，则在控制台打印错误
        # #                 print(f"警告: 客户端 {self.id} 写入蒸馏日志或保存图像失败。错误: {e}")

        # student_logits = self.model(filtered_data)
        # if distill_mode == 'dkd':
        #     # 使用 DKD 损失
        #     # 我们需要一个目标标签。对于 data-free 场景，最好的选择是使用教师集体的预测作为伪标签。
        #     with torch.no_grad():
        #         pseudo_labels = torch.argmax(y_hat, dim=1)
            
        #     loss = dkd_loss(
        #         student_logits=student_logits,
        #         teacher_logits=y_hat,
        #         target_labels=pseudo_labels,
        #         dkd_alpha=dkd_alpha,
        #         beta=adaptive_betas,
        #         temperature=temperature
        #     )
        # else: # 默认为 'kd' (经典KL散度)
        #     loss = F.kl_div(
        #         F.log_softmax(student_logits / temperature, dim=1),
        #         F.softmax(y_hat / temperature, dim=1),
        #         reduction='batchmean'
        #     ) * (temperature ** 2)

        # # loss = F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(y_hat, dim=1), reduction='batchmean')
        # # loss *= 10
        # self.optimizer_model.zero_grad()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        # self.optimizer_model.step()

        # return loss.item()
        

    # --- VVV FedD3A 核心方法 VVV ---
    @torch.no_grad()
    def calculate_projection_matrix(self):
        """(客户端功能) 使用SVD为本地数据计算子空间投影矩阵。"""
        self.model.eval()
        all_projected_features = []
        for data, _ in self.data_loader:
            data = data.to(self.device)
            with torch.no_grad(): # 在推理时使用 no_grad() 是个好习惯
                raw_features = self.model.extract_features(data)
                projected_features = self.model.projector(raw_features)
            all_projected_features.append(projected_features.cpu()) # 移动到CPU，避免GPU内存累积

        if not all_projected_features:
            print(f"客户端 {self.id} 没有数据，跳过投影矩阵更新。")
            return

        Z_projected = torch.cat(all_projected_features, dim=0).to(self.device) # 计算时再移回设备

        # === 新增：健壮性检查 ===
        # 1. 检查是否存在无效值
        if torch.isnan(Z_projected).any() or torch.isinf(Z_projected).any():
            print(f"客户端 {self.id} 的特征矩阵 Z_projected 包含 NaN 或 Inf，跳过更新。")
            return

        # 2. 检查数据量是否过少 (可选，但推荐)
        # 假设特征维度是 D，至少需要 D 个样本才能有较好的覆盖
        feature_dim = Z_projected.shape[1]
        if Z_projected.shape[0] < feature_dim:
            print(f"客户端 {self.id} 的样本数 ({Z_projected.shape[0]}) 小于特征维度 ({feature_dim})，SVD 可能不稳定，跳过更新。")
            return
        # ========================

        try:
            # 使用双精度以提高数值稳定性，但会消耗更多资源
            # Z_projected_double = Z_projected.to(torch.float64)
            # U, _, _ = torch.linalg.svd(Z_projected_double.T, full_matrices=False)
            # U = U.to(torch.float32) # 计算完转回来
            
            U, _, _ = torch.linalg.svd(Z_projected.T, full_matrices=False)
            
            # 增加一个对U的检查
            if torch.isnan(U).any():
                print(f"客户端 {self.id} 的 SVD 计算结果 U 包含 NaN，跳过投影矩阵更新。")
                return

            self.projection_matrix = U @ U.T
        except torch.linalg.LinAlgError:
            print(f"客户端 {self.id} 的 SVD 计算失败 (LinAlgError)，跳过投影矩阵更新。")
            # 增加更多诊断信息
            print(f"    - Z_projected 形状: {Z_projected.shape}")
            print(f"    - Z_projected 均值: {Z_projected.mean().item():.4f}, 标准差: {Z_projected.std().item():.4f}")
            print(f"    - Z_projected 最大值: {Z_projected.max().item():.4f}, 最小值: {Z_projected.min().item():.4f}")
    # def calculate_projection_matrix(self):
    #     """(客户端功能) 使用SVD为本地数据计算子空间投影矩阵。"""
    #     self.model.eval()
    #     all_projected_features = []
    #     for data, _ in self.data_loader:
    #         data = data.to(self.device)
    #         # 1. 提取原始特征
    #         raw_features = self.model.extract_features(data)
    #         # 2. VVV 新增：将特征投影到公共空间 VVV
    #         projected_features = self.model.projector(raw_features)
    #         all_projected_features.append(projected_features)
        
    #     if not all_projected_features:
    #         return

    #     # 3. 在公共空间中（例如512维）构建特征矩阵Z
    #     Z_projected = torch.cat(all_projected_features, dim=0)
    #     try:
    #         # 4. 在公共空间中计算投影矩阵
    #         U, _, _ = torch.linalg.svd(Z_projected.T, full_matrices=False)
    #         self.projection_matrix = U @ U.T
    #     except torch.linalg.LinAlgError:
    #         print(f"客户端 {self.id} 的 SVD 计算失败，跳过投影矩阵更新。")

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