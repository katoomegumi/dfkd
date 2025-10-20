# data_utils.py
import numpy as np
from torch.utils.data import Subset

def get_non_iid_data(num_clients, dataset, seed=42):
    """
    将数据集划分为 Non-IID 的客户端子集。
    """
    rng = np.random.default_rng(seed)
    num_classes = 10
    alpha = 0.5
    min_size = 0
    num_samples = len(dataset)
    
    # 确保每个客户端至少有10个样本
    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            # 获取属于类别k的样本索引
            idx_k = np.where(np.array(dataset.targets) == k)[0]
            rng.shuffle(idx_k)
            
            # 使用狄利克雷分布生成不均衡的比例
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < num_samples / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            # 分配样本
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            
    # 创建Subset对象
    client_subsets = [Subset(dataset, idx) for idx in idx_batch]
    return client_subsets