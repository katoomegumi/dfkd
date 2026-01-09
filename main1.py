# main.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
import random
import time
import csv
from datetime import datetime
import os
import math
import torch.nn.functional as F
from torchvision.utils import save_image

# 从我们自己的模块中导入所有需要的工具
from data_utils import get_non_iid_data
from topology import setup_topology
from factory import create_clients

MAX_GEN_SAMPLES = 5000
SATURATION_ROUND = 400

# torch.autograd.set_detect_anomaly(True)

# def nan_hook(module, input, output):
#     """
#     一个 PyTorch hook, 用于检查模块输出中是否存在 NaN 或 Inf。
#     """
#     if isinstance(output, torch.Tensor):
#         if torch.isnan(output).any() or torch.isinf(output).any():
#             print(f"\n!!!!!! [调试信息] 在模块 {module.__class__.__name__} 的输出中检测到 NaN/Inf !!!!!!\n")
#     elif isinstance(output, (tuple, list)):
#         for i, out in enumerate(output):
#             if isinstance(out, torch.Tensor):
#                 if torch.isnan(out).any() or torch.isinf(out).any():
#                     print(f"\n!!!!!! [调试信息] 在模块 {module.__class__.__name__} 的第 {i} 个输出中检测到 NaN/Inf !!!!!!\n")

def set_seed(seed):
    """
    设置所有相关的随机种子，以确保实验的可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 适用于多GPU
    
    # 确保cuDNN的确定性，这可能会牺牲一点速度，但对于复现性至关重要
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def evaluate_and_log_teacher_probs(clients, test_loader, output_base_dir, timestamp, round_num, device):
    """
    在一次评估中，计算所有客户端在测试集上的表现，
    并为每个客户端保存其邻居教师的详细预测概率。
    """
    print(f"--- 第 {round_num} 轮详细评估：正在缓存所有教师的预测... ---")

    # --- 阶段一：计算并缓存所有客户端对测试集的预测 ---
    
    all_predictions = {}
    true_labels = []

    # 1. 遍历测试集一次，获取所有真实标签
    for _, targets in test_loader:
        true_labels.append(targets.numpy())
    true_labels = np.concatenate(true_labels) # 形状: [10000]

    # 2. 遍历所有客户端，获取它们对测试集的预测
    for client in clients:
        client.model.eval()
        client_probs = []
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            logits = client.model(inputs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            client_probs.append(probs)
        
        # 将该客户端的所有预测拼接成一个大数组
        all_predictions[client.id] = np.concatenate(client_probs) # 形状: [10000, 10]

    print("所有教师的预测已缓存。")
    return all_predictions, true_labels

def get_probabilistic_scarcity_batch(client, dataloader_iterator, full_dataloader, target_batch_size, device, current_round, total_rounds, annealing_enabled):
    """
    为单个客户端，根据其本地数据稀缺性，从公共数据集中概率性地构建一个批次。
    """
    # 1. 根据数据稀缺性，计算每个类别的采样概率
    class_counts = client.real_class_counts
    if annealing_enabled:
        # --- 退火机制逻辑 ---
        # a. 计算退火因子 gamma，它会从 1.0 线性下降到 0.0
        gamma = 1.0 - (current_round / total_rounds)
        
        # b. 计算稀缺性权重 (偏向 non-IID)
        epsilon = 1e-6
        scarcity_weights = 1.0 / (class_counts + epsilon)
        
        # c. 定义均匀权重 (代表 IID)
        uniform_weights = torch.ones_like(class_counts)
        
        # d. 线性插值：早期 gamma 接近1，侧重稀缺性；后期 gamma 接近0，侧重均匀性
        final_weights = gamma * scarcity_weights + (1.0 - gamma) * uniform_weights
        
        # e. 归一化得到最终概率
        probabilities = final_weights / final_weights.sum()
    else:
        # --- 原始的、无退火的逻辑 ---
        epsilon = 1e-6
        weights = 1.0 / (class_counts + epsilon)
        probabilities = weights / weights.sum()
    # 2. 根据这个概率分布，生成一个包含 target_batch_size 个的目标类别“购物清单”
    target_labels = torch.multinomial(probabilities, target_batch_size, replacement=True)
    
    # 3. 持续从公共数据集中“淘宝”，直到凑齐“购物清单”上的所有样本
    collected_samples = []
    # 使用 bincount 快速统计每类需要多少个样本
    needed_counts = torch.bincount(target_labels, minlength=10).cpu()
    
    while sum(needed_counts) > 0:
        try:
            batch_data, batch_labels = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(full_dataloader)
            batch_data, batch_labels = next(dataloader_iterator)
            
        # 遍历当前批次中的每个样本
        for i in range(len(batch_labels)):
            label = batch_labels[i].item()
            # 如果这个类别是我们需要的
            if needed_counts[label] > 0:
                collected_samples.append(batch_data[i].unsqueeze(0))
                needed_counts[label] -= 1
                # 如果购物清单已满，提前退出
                if sum(needed_counts) == 0:
                    break
    
    # 4. 拼接成最终的批次
    final_data_batch = torch.cat(collected_samples, dim=0)
    
    return final_data_batch.to(device), dataloader_iterator, target_labels

def get_random_batch(client, dataloader_iterator, full_dataloader, target_batch_size, device, current_round, total_rounds, annealing_enabled):
    """
    为单个客户端，根据其本地数据稀缺性，从公共数据集中概率性地构建一个批次。
    """
    # 1. 根据数据稀缺性，计算每个类别的采样概率
    class_counts = client.real_class_counts
        
        # e. 归一化得到最终概率
    uniform_weights = torch.ones_like(class_counts)
    probabilities = uniform_weights / uniform_weights.sum()

    target_labels = torch.multinomial(probabilities, target_batch_size, replacement=True)
    
    # 3. 持续从公共数据集中“淘宝”，直到凑齐“购物清单”上的所有样本
    collected_samples = []
    # 使用 bincount 快速统计每类需要多少个样本
    needed_counts = torch.bincount(target_labels, minlength=10).cpu()
    
    while sum(needed_counts) > 0:
        try:
            batch_data, batch_labels = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(full_dataloader)
            batch_data, batch_labels = next(dataloader_iterator)
            
        # 遍历当前批次中的每个样本
        for i in range(len(batch_labels)):
            label = batch_labels[i].item()
            # 如果这个类别是我们需要的
            if needed_counts[label] > 0:
                collected_samples.append(batch_data[i].unsqueeze(0))
                needed_counts[label] -= 1
                # 如果购物清单已满，提前退出
                if sum(needed_counts) == 0:
                    break
    
    # 4. 拼接成最终的批次
    final_data_batch = torch.cat(collected_samples, dim=0)
    
    return final_data_batch.to(device), dataloader_iterator, target_labels

def main(args):
    seed = 42
    random.seed(seed)
    set_seed(args.seed)
    print(f"--- 已设置全局随机种子: {args.seed} ---")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 数据准备
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    public_dataloader = DataLoader(train_dataset, batch_size=args.distill_batch_size, shuffle=True)
    public_data_iterator = iter(public_dataloader)

    # 2. 创建客户端 (使用 factory)
    client_datasets = get_non_iid_data(args.num_clients, train_dataset, seed=seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    clients = create_clients(args.num_clients, client_datasets, device, args.rounds)

    clients = create_clients(args.num_clients, client_datasets, device, args.rounds)

    print(f"\n--- 正在统计并保存客户端数据分布 (CSV) ---")
    dist_save_filename = f"client_distributions_{timestamp}.csv"
    dist_folder = "distribution"
    dist_save_path = os.path.join(dist_folder, dist_save_filename)
    os.makedirs(os.path.dirname(dist_save_path), exist_ok=True)
    
    # CIFAR-10 有 10 个类别
    num_classes = 10
    
    with open(dist_save_path, 'w', newline='') as csvfile:
        # 定义表头: Client_ID, Class_0, Class_1, ..., Class_9, Total
        fieldnames = ['Client_ID'] + [f'Class_{i}' for i in range(num_classes)] + ['Total']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        total_samples_all = 0
        
        for i, subset in enumerate(client_datasets):
            # 利用 Subset 特性快速获取标签，无需遍历 DataLoader
            client_indices = subset.indices
            # 注意: subset.dataset 是原始完整数据集
            client_targets = np.array(subset.dataset.targets)[client_indices]
            
            # 统计各类别数量
            unique, counts = np.unique(client_targets, return_counts=True)
            counts_dict = {int(k): int(v) for k, v in zip(unique, counts)}
            
            # 准备写入的一行数据
            row_data = {'Client_ID': i}
            client_total = 0
            
            for c in range(num_classes):
                count = counts_dict.get(c, 0)
                row_data[f'Class_{c}'] = count
                client_total += count
            
            row_data['Total'] = client_total
            total_samples_all += client_total
            
            writer.writerow(row_data)

    print(f"✅ 客户端分布 CSV 已保存至: {dist_save_path}")

    # ... (继续执行 "3. 设置网络拓扑")
    setup_topology(clients, args)
    
    if args.load_diffusion_path:
        print(f"\n--- 正在从 '{args.load_diffusion_path}' 加载预训练的扩散模型 ---")
        all_models_loaded = True
        for client in clients:
            # 构造每个客户端的模型文件路径
            model_filename = f"diffusion_model_client_{client.id}.pt"
            model_load_path = os.path.join(args.load_diffusion_path, model_filename)
            
            if os.path.exists(model_load_path):
                try:
                    # 加载状态字典 (state_dict)
                    state_dict = torch.load(model_load_path, map_location=device)
                    # 将加载的权重应用到客户端的 diffusion_model 上
                    client.diffusion_model.load_state_dict(state_dict)
                    print(f"成功加载客户端 {client.id} 的模型。")
                except Exception as e:
                    print(f"错误：加载客户端 {client.id} 的模型失败: {e}")
                    all_models_loaded = False
            else:
                print(f"警告：找不到客户端 {client.id} 的模型文件: {model_load_path}")
                all_models_loaded = False
        
        if all_models_loaded:
            print("所有客户端的扩散模型均已成功加载。")
        else:
            print("警告：部分或全部模型加载失败，将从随机初始化开始训练。")
    else:
        print("\n--- 未指定预训练模型路径，所有模型将从随机初始化开始训练 ---")
        
    # 3. 设置网络拓扑 (使用 topology 模块)
    setup_topology(clients, args)

    # 预热
    if args.distill_data_source == 'generated' and args.diffusion_warmup_rounds > 0:
        print(f"\n--- 正在开始扩散模型预热 {args.diffusion_warmup_rounds} 轮 ---")
        
        
        for warmup_round in range(1, args.diffusion_warmup_rounds + 1):
            print(f"  预热轮次 {warmup_round}/{args.diffusion_warmup_rounds}...")
            
            for c in clients:
                c.train_local_diffusion_step(epochs=1)
            
            # 2. 执行生成器共识（邻居间聚合）
            for client in clients:
                client.perform_generator_consensus()
                
        print("--- 扩散模型预热完成 ---")
    else:
        print("\n--- 跳过扩散模型预热 ---")

    # 4. 准备日志和结果文件
    output_base_dir = "alpha0.1_result"
    image_output_dir = "generated_images"
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(output_base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result_filename = f"{args.distill_data_source}_weights-{args.teacher_selector}_{args.sample_filter}_{args.distill_mode}_late-{args.late_mix}_early-{args.early_mix}_{timestamp}.csv"
    result_file_path =os.path.join(output_base_dir, result_filename)
    print(f"结果将保存至: {result_file_path}")
    with open(result_file_path, "w") as f:
        f.write("Round,Avg_Train_Loss,Avg_Distillation_Loss,Avg_Test_Accuracy,std acc,Generation_Time_s\n")
    print(f"\n--- 开始训练 (蒸馏数据源: {args.distill_data_source}) ---")
    print(f"\n--- 开始训练 (蒸馏模式: {args.sample_filter}, 教师模式: {args.teacher_selector}) ---")
    if args.teacher_weights:
        print(f"使用自定义教师权重: {args.teacher_weights}")
    else:
        print("所有教师使用相等权重。")

    # 5. 主训练循环
    aux_data_for_clients = [None] * args.num_clients
    num_to_generate = 0
    for round_num in range(1, args.rounds + 1):
        print(f"\n--- 第 {round_num}/{args.rounds} 轮 ---")
        # # 当第二个 epoch (索引通常为 1) 即将开始时, 注册 hook
        # if round_num == 2:
        #     print("\n--- [调试模式]：为所有客户端模型注册 NaN/Inf 检测 Hook ---")
            
        #     # 假设你的客户端列表是 `clients`
        #     # 并且每个客户端有一个 `.model` 属性
        #     for client in clients:
        #         for name, module in client.model.named_modules():
        #             module.register_forward_hook(nan_hook)
        #     print("--- Hook 注册完毕，即将开始第二个 Epoch 的计算 ---\n")

        # --- VVV FedD3A 新增步骤: 在本地训练前，计算投影矩阵 VVV ---
        if args.distill_data_source == 'generated' and args.late_mix:

            
            # 判断是否是“生成轮”（起始轮，或之后每隔10轮）
            if round_num % args.generate_interval == 1:
                print(f"  [策略B] 晚期增强 (生成更新): 正在生成新数据 (本数据将使用10轮)...")

                k = 3.0 
                current_x = min(1.0, round_num / SATURATION_ROUND)

                # 分子: 1 - exp(-k * current_progress)
                # 分母: 1 - exp(-k * 1.0) -> 用于归一化，确保最后一轮一定是 1.0
                ratio = (1 - math.exp(-k * current_x)) / (1 - math.exp(-k))

                num_to_generate = int(MAX_GEN_SAMPLES * ratio)
                for i, client in enumerate(clients):
                    # 1. 计算需要生成的数量 (mix_ratio逻辑)
                    # ratio = args.mix_ratio
                    # if ratio >= 1.0: ratio = 0.99 
                    # num_to_generate = int(client.num_samples * (ratio / (1 - ratio)))
                    if num_to_generate < 1: num_to_generate = 0

                    if num_to_generate > 0:
                        gamma = max(0.0, 1.0 - (round_num / args.rounds))

                        # [修改] 2. 混合两种采样权重
                        # 确保 class_counts 是 tensor 格式以便计算
                        counts_tensor = torch.as_tensor(client.real_class_counts, device=device, dtype=torch.float32)

                        # A. 稀缺性权重 (原始逻辑: 样本越少权重越大)
                        scarcity_weights = 1.0 / (counts_tensor + 1e-6)
                        scarcity_probs = scarcity_weights / scarcity_weights.sum()
                        uniform_probs = torch.ones_like(counts_tensor) / len(counts_tensor)
                        final_probs = gamma * scarcity_probs + (1 - gamma) * uniform_probs
                        aux_labels = torch.multinomial(final_probs, num_to_generate, replacement=True).to(device)
                        
                        # 3. 生成图像
                        aux_imgs = client.generate_edm_samples(num_to_generate, aux_labels)
                        
                        # 更新缓存
                        aux_data_for_clients[i] = (aux_imgs, aux_labels)
                    client.UpdateClassCounts(aux_labels, device)
            else:
                print(f"  [策略B] 晚期增强 (复用数据): 使用上一轮生成的辅助数据")
        local_losses = []
        for i, c in enumerate(clients):
            loss = c.train_local_model_step(aux_data = aux_data_for_clients[i], num_aux_data = num_to_generate)
            local_losses.append(loss)
        if round_num % args.eval_interval == 0:
            
            # --- 1. 执行您要求的新的、详细的概率记录 ---
            # (这个新函数会处理所有教师的评估和保存)
            all_predictions, true_labels = evaluate_and_log_teacher_probs(clients=clients, test_loader=test_loader, output_base_dir=output_base_dir, timestamp=timestamp, round_num=round_num, device=device)

            # --- 2. 计算并打印常规的平均日志 ---
            # (我们仍然执行此操作，以便您可以在控制台快速查看)
            avg_local_loss = np.mean(local_losses) if local_losses else 0.0
            avg_distill_loss = np.mean(distillation_losses) if distillation_losses else 0.0
            
            # 从刚才缓存的结果中计算平均准确率 (避免重复计算)
            all_accuracies = []
            for client_id, probs in all_predictions.items():
                predicted_labels = np.argmax(probs, axis=1)
                acc = np.mean(predicted_labels == true_labels) * 100
                all_accuracies.append(acc)
                # (我们已经在新函数中打印了每个教师的个体准确率)

        if args.distill_data_source == 'generated':
            print("正在训练本地 EDM 扩散模型...")
            for c in clients: c.train_local_diffusion_step()
            print("正在执行 EDM 生成器共识...")
            for client in clients: client.perform_generator_consensus()
        
        distillation_losses = []
        total_generation_time = 0

        for student_client in clients:
            if not student_client.neighbors: continue

            local_distillation_data = None
            distill_labels = None

            if args.distill_data_source == 'original':
            # 'original' 模式下，从公共数据加载器中取一个批次的数据
                local_distillation_data, public_data_iterator, distill_labels = get_random_batch(
                    client=student_client,
                    dataloader_iterator=public_data_iterator,
                    full_dataloader=public_dataloader,
                    target_batch_size=args.distill_batch_size,
                    device=device,
                    current_round=round_num,
                    total_rounds=args.rounds,
                    annealing_enabled=args.scarcity_annealing
                )
                
            if args.distill_data_source == 'original_scarcity':
                local_distillation_data, public_data_iterator, distill_labels = get_probabilistic_scarcity_batch(
                    client=student_client,
                    dataloader_iterator=public_data_iterator,
                    full_dataloader=public_dataloader,
                    target_batch_size=args.distill_batch_size,
                    device=device,
                    current_round=round_num,
                    total_rounds=args.rounds,
                    annealing_enabled=args.scarcity_annealing
                )
            
            # 如果是 'generated' 模式，现在为每个客户端独立生成数据
            if args.distill_data_source == 'generated':
                start_time = time.time()
                class_labels = None
                if args.sample_filter == 'hard_scarcity':
                    # 使用稀缺性概率生成类别标签
                    # weights = 1.0 / (student_client.class_counts + 1e-6)
                    # probabilities = weights / weights.sum()
                    
                    gamma = 1.0 - ( round_num / args.rounds)
        
                    # b. 计算稀缺性权重 (偏向 non-IID)
                    epsilon = 1e-6
                    scarcity_weights = 1.0 / (student_client.dynamic_class_counts + epsilon)
                    
                    # c. 定义均匀权重 (代表 IID)
                    uniform_weights = torch.ones_like(student_client.dynamic_class_counts)
                    
                    # d. 线性插值：早期 gamma 接近1，侧重稀缺性；后期 gamma 接近0，侧重均匀性
                    final_weights = gamma * scarcity_weights + (1.0 - gamma) * uniform_weights
                    
                    # e. 归一化得到最终概率
                    probabilities = final_weights / final_weights.sum()
                    class_labels = torch.multinomial(probabilities, args.distill_batch_size, replacement=True).to(device)
                else: # 'normal'
                    # 随机生成类别标签
                    class_labels = torch.randint(0, 10, (args.distill_batch_size,), device=device)
                
                distill_labels = class_labels
                local_distillation_data = student_client.generate_edm_samples(
                    num_samples=args.distill_batch_size,
                    class_labels=class_labels
                )
                if args.early_mix:
                    # 从本地取一批真实数据
                    real_imgs, real_labels = student_client.get_real_sample_batch(batch_size=args.distill_batch_size)
                    
                    # 混合：将真实数据拼接到生成数据后面
                    # 这样蒸馏数据就变成了 [生成(32) + 真实(32)]，共64个
                    if local_distillation_data is not None:
                        local_distillation_data = torch.cat([local_distillation_data, real_imgs], dim=0)
                        local_distill_labels = torch.cat([class_labels, real_labels], dim=0)
                    distill_labels = local_distill_labels
                total_generation_time += (time.time() - start_time)

                if local_distillation_data is None: continue
                
                # 保存生成的图像
                if args.distill_data_source == 'generated' and round_num % args.eval_interval == 0:
                    filename = os.path.join(image_output_dir, f"round_{round_num:03d}_client_{student_client.id:02d}_{args.sample_filter}.png")
                    save_image(local_distillation_data, filename, nrow=8)
                

            # --- VVV 根据 teacher_selector 执行不同的蒸馏策略 VVV ---
            loss = student_client.distill(
                distillation_data=local_distillation_data,
                distill_labels = distill_labels,
                teacher_selector=args.teacher_selector,
                sample_filter=args.sample_filter,
                hard_sample_ratio=args.hard_sample_ratio,
                distill_mode=args.distill_mode,
                dkd_alpha=args.dkd_alpha,
                dkd_beta=args.dkd_beta,
                round_idx = round_num,
                total_rounds = args.rounds
            )
            
            distillation_losses.append(loss)
        
        # 定期评估并记录结果
        # if round_num % args.eval_interval == 0:
            
        #     avg_local_loss = np.mean(local_losses) if local_losses else 0.0
        #     avg_distill_loss = np.mean(distillation_losses) if distillation_losses else 0.0
        #     avg_acc = np.mean([c.evaluate(test_loader) for c in clients])
            
        #     log_message = (f"第 {round_num}/{args.rounds} 轮评估 | 平均本地损失: {avg_local_loss:.4f} | "
        #                    f"平均蒸馏损失: {avg_distill_loss:.4f} | 平均测试准确率: {avg_acc:.2f}% | "
        #                    f"样本生成耗时: {total_generation_time:.2f}s")
        #     print(log_message)
            
        #     with open(result_file_path, "a") as f:
        #         f.write(f"{round_num},{avg_local_loss:.4f},{avg_distill_loss:.4f},{avg_acc:.2f},{total_generation_time:.2f}\n")
        if round_num % args.eval_interval == 0:
            
            # --- 1. 执行您要求的新的、详细的概率记录 ---
            # (这个新函数会处理所有教师的评估和保存)
            all_predictions, true_labels = evaluate_and_log_teacher_probs(clients=clients, test_loader=test_loader, output_base_dir=output_base_dir, timestamp=timestamp, round_num=round_num, device=device)

            # --- 2. 计算并打印常规的平均日志 ---
            # (我们仍然执行此操作，以便您可以在控制台快速查看)
            avg_local_loss = np.mean(local_losses) if local_losses else 0.0
            avg_distill_loss = np.mean(distillation_losses) if distillation_losses else 0.0
            
            # 从刚才缓存的结果中计算平均准确率 (避免重复计算)
            all_distill_accuracies = []
            for client_id, probs in all_predictions.items():
                predicted_labels = np.argmax(probs, axis=1)
                acc = np.mean(predicted_labels == true_labels) * 100
                all_distill_accuracies.append(acc)
                # (我们已经在新函数中打印了每个教师的个体准确率)
            
            avg_acc = np.mean(all_distill_accuracies)
            std_acc = np.std(all_distill_accuracies)
            
            log_message = (f"第 {round_num}/{args.rounds} 轮评估 | 平均本地损失: {avg_local_loss:.4f} | "
                           f"平均蒸馏损失: {avg_distill_loss:.4f} | "
                           f"平均准确率: {avg_acc:.2f} | "
                           f"样本生成耗时: {total_generation_time:.2f}s")
            print(log_message)
            
            # --- 3. 写入您的主 CSV 日志文件 (不变) ---
            teacher_acc_str = ",".join([f"{acc:.2f}" for acc in all_accuracies])
            with open(result_file_path, "a") as f:
                f.write(f"{round_num},{avg_local_loss:.4f},{avg_distill_loss:.4f},{avg_acc:.2f},{std_acc:.2f},{total_generation_time:.2f},{teacher_acc_str}\n")

        for client in clients:
            client.scheduler_model.step()
    
    print("\n--- 训练结束，正在保存最后一个轮次的扩散模型 ---")
    models_save_dir = "saved_model"
    os.makedirs(models_save_dir, exist_ok=True)
    models_save_path = os.path.join(models_save_dir, f"saved_models_{timestamp}")
    os.makedirs(models_save_path, exist_ok=True)
    print(f"模型将被保存至: {models_save_path}")
    for client in clients:
        # 为每个客户端的模型创建一个包含其ID的唯一文件名
        model_filename = f"diffusion_model_client_{client.id}.pt"
        model_save_path = os.path.join(models_save_path, model_filename)
        
        # 使用 torch.save 保存模型的状态字典 (state_dict)
        # 这是推荐的、最灵活的保存方式
        torch.save(client.diffusion_model.state_dict(), model_save_path)

    print("所有客户端的扩散模型均已成功保存。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="异构模型下的多教师加权蒸馏去中心化联邦学习实验")
    # ... 参数解析部分与您提供的代码完全相同 ...
    parser.add_argument('--num_clients',            type=int, default=10, help='客户端总数 (semidfl 拓扑需要10个)')
    parser.add_argument('--rounds',                 type=int, default=50, help='总通信轮数')
    parser.add_argument('--seed', type=int, default=42, help='全局随机种子，用于保证模型初始化的可复现性')
    parser.add_argument('--topology',               type=str, default='semidfl', choices=['ring', 'fully_connected', 'semidfl'], help='客户端连接拓扑')
    parser.add_argument('--eval_interval',          type=int, default=10, help='评估与图像保存的间隔轮数')
    parser.add_argument('--distill_batch_size',     type=int, default=128, help='每轮蒸馏用的样本数量')
    parser.add_argument('--distillation_data_mode', type=str, default='normal', choices=['hard', 'normal'], help='用于蒸馏的数据类型: hard (困难样本), normal (普通样本)')
    parser.add_argument('--guidance_scale',         type=float, default=0.01, help='困难样本生成的引导强度')
    parser.add_argument('--gpu_id',                 type=int, default=0, help='指定使用的GPU索引')
    parser.add_argument('--scarcity_annealing', type=bool, default=1, 
                        help='(仅在original_scarcity模式下生效) 启用稀缺性退火，使采样概率随轮次增加逐渐趋向于IID')
    parser.add_argument('--teacher_weights',        type=float, nargs='*', default=None, help='用空格分隔的教师模型权重列表。若不提供，则默认为所有教师权重相等。')
    parser.add_argument('--teacher_selector',       type=str, default='fedd3a', choices=['all', 'random', 'fedd3a', 'top1', 'top2', 'top3', 'expert_top1','expert_top2', 'expert_top3', 'expert_all','complementary'], 
                        help='''选择教师的策略: 
                                all(所有邻居平均), 
                                random(随机单个邻居), 
                                fedd3a(FedD3A软权重), 
                                top1/top2/top3(选择最相似的k个教师)
                                expert_top1/expert_top2/expert_top3/expert_all(样本class与邻居在该class上数据最多的)''')
    parser.add_argument('--sample_filter', type=str, default='none', choices=['none', 'confidence', 'hard_scarcity'], 
                        help='''选择蒸馏样本的策略: 
                                none(使用所有样本), 
                                confidence(只用教师置信度最低的困难样本)
                                hard_scarcity(优先生成数据稀缺的类别)''')
    parser.add_argument('--distill_mode', type=str, default='kd', 
                        choices=['kd', 'dkd'], 
                        help='选择知识蒸馏的模式: kd(经典KL散度), dkd(解耦知识蒸馏)')

    # 2. 定义 DKD 的超参数 alpha 和 beta
    parser.add_argument('--dkd_alpha', type=float, default=1.0, 
                        help='DKD中TCKD的权重')
    parser.add_argument('--dkd_beta', type=float, default=8.0, 
                        help='DKD中NCKD的权重 (通常设置得比alpha大)')
    parser.add_argument('--hard_sample_ratio', type=float, default=0.5, help='在 confidence 模式下，选择的困难样本比例')
    parser.add_argument('--distill_data_source', type=str, default='generated', choices=['generated', 'original', 'original_scarcity'], help='选择用于蒸馏的数据来源: generated (由扩散模型生成), original (从原始训练集中采样)')
    parser.add_argument('--load_diffusion_path', type=str, default=None, 
                        help='指定包含预训练扩散模型(.pt文件)的文件夹路径')
    parser.add_argument('--diffusion_warmup_rounds', type=int, default=20, help='在主训练循环开始前，专门用于预热扩散模型的轮数 (例如: 20)')
    

    parser.add_argument('--late_mix', type=bool, default=0, 
                        help='是否混合用于本地训练')
    parser.add_argument('--early_mix', type=bool, default=0, 
                        help='是否将本地真实数据混入蒸馏数据中，以提升质量')
    # 混合比例 (可选，这里简单处理为 1:1 混合或固定数量)
    parser.add_argument('--mix_ratio', type=float, default=0.5, help='数据混合时的比例 (0.0-1.0)')
    parser.add_argument('--generate_interval', type=int, default=10, help='多少轮生成一次用于本地训练的混合数据')
    args = parser.parse_args()
    main(args)