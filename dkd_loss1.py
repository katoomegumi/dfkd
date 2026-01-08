import torch
import torch.nn.functional as F

# 确保 target_labels 是长整型 (LongTensor)，dkd_alpha 和 temperature 是浮点数
def dkd_loss(student_logits, teacher_logits, target_labels, dkd_alpha, beta, temperature):
    """
    DKD (Decoupled Knowledge Distillation) 损失函数。
    
    参数:
        student_logits (Tensor): 学生模型的 Logits [B, C]
        teacher_logits (Tensor): 教师模型的 Logits [B, C]
        target_labels (Tensor): 真实/伪标签 (argmax) [B]
        dkd_alpha (float): TCKD (目标类) 损失的权重
        beta (float 或 Tensor[B]): NCKD (非目标类) 损失的权重。
                                   如果传入 Tensor，则实现样本级加权。
        temperature (float): 蒸馏温度 T
    """
    
    gt_mask = F.one_hot(target_labels, num_classes=teacher_logits.shape[1]).bool()
    other_mask = ~gt_mask

    # 1. TCKD Loss (目标类知识蒸馏)
    # 仅关注目标类别，相当于一个交叉熵损失
    s_t = F.log_softmax(student_logits / temperature, dim=1)
    t_t = F.softmax(teacher_logits / temperature, dim=1)
    
    tckd_loss = (-(t_t * s_t).sum(dim=1, keepdim=True)) * gt_mask
    tckd_loss = tckd_loss.sum() / tckd_loss.size(0)

    # 2. NCKD Loss (非目标类知识蒸馏)
    # 对非目标类别进行 L2-norm 归一化，再计算 KL 散度
    
    # 归一化非目标类 Logits (学生)
    s_oth = F.log_softmax(student_logits / temperature - 1000.0 * gt_mask, dim=1)
    # 归一化非目标类 Logits (教师)
    t_oth = F.softmax(teacher_logits / temperature - 1000.0 * gt_mask, dim=1)
    
    nckd_loss = (-(t_oth * s_oth).sum(dim=1, keepdim=True)) * other_mask
    nckd_loss = nckd_loss.sum(dim=1) # 形状 [B]

    # 3. 组合损失 (核心修改部分)
    if isinstance(beta, torch.Tensor):
        # 样本级加权: beta [B] 乘以 nckd_loss [B]，再求 Batch 平均
        weighted_nckd = (nckd_loss * beta).mean()
    else:
        # 标量 beta 保持不变
        weighted_nckd = beta * nckd_loss.mean()
        
    final_loss = dkd_alpha * tckd_loss + weighted_nckd
    
    return final_loss