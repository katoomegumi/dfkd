# dkd_loss.py
import torch
import torch.nn.functional as F

def dkd_loss(student_logits, teacher_logits, target_labels, alpha, beta, temperature):
    """
    计算解耦知识蒸馏 (Decoupled Knowledge Distillation) 损失。
    参考论文: Decoupled Knowledge Distillation (CVPR 2022)
    在 FedDKDGen 论文中对应公式 (8), (9), (10)。

    参数:
    - student_logits: 学生模型的 logits 输出
    - teacher_logits: 教师模型的 logits 输出 (或聚合后的 y_hat)
    - target_labels: 目标类别标签 (可以是真实标签或伪标签)
    - alpha: TCKD 的权重
    - beta: NCKD 的权重
    - temperature: 蒸馏温度
    """
    # 对 logits 应用温度
    student_logits_T = student_logits / temperature
    teacher_logits_T = teacher_logits / temperature

    # 计算 softmax 概率
    student_probs = F.softmax(student_logits_T, dim=1)
    teacher_probs = F.softmax(teacher_logits_T, dim=1)

    # --- 1. 计算 TCKD (Target Class Knowledge Distillation) ---
    
    # 获取学生和教师在目标类别上的概率
    student_target_prob = student_probs.gather(1, target_labels.unsqueeze(1)).squeeze()
    teacher_target_prob = teacher_probs.gather(1, target_labels.unsqueeze(1)).squeeze()

    # TCKD 衡量的是目标类概率分布的差异
    # 可以看作是对 [p_target, 1-p_target] 这个二分类分布进行 KL 散度计算
    tckd_loss = F.kl_div(
        torch.log(torch.stack([student_target_prob, 1. - student_target_prob], dim=1)),
        torch.stack([teacher_target_prob, 1. - teacher_target_prob], dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)

    # --- 2. 计算 NCKD (Non-Target Class Knowledge Distillation) ---

    # 创建一个 mask，将目标类别的 logits 设为负无穷，以便在 softmax 中忽略它们
    mask = torch.ones_like(student_logits_T).scatter_(1, target_labels.unsqueeze(1), 0)
    
    # 学生模型的非目标 logits
    student_non_target_logits = student_logits_T - (1 - mask) * 1e8 # 使用一个较大的负数
    # 教师模型的非目标 logits
    teacher_non_target_logits = teacher_logits_T - (1 - mask) * 1e8

    # 计算非目标类别的概率分布
    student_non_target_probs = F.log_softmax(student_non_target_logits, dim=1)
    teacher_non_target_probs = F.softmax(teacher_non_target_logits, dim=1)

    # NCKD 衡量的是所有非目标类概率分布的差异
    nckd_loss = F.kl_div(
        student_non_target_probs,
        teacher_non_target_probs,
        reduction='batchmean'
    ) * (temperature ** 2)

    # --- 3. 组合损失 ---
    return alpha * tckd_loss + beta * nckd_loss