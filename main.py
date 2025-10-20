# main.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
import random
import time
from datetime import datetime
import os
from torchvision.utils import save_image

# 从我们自己的模块中导入所有需要的工具
from data_utils import get_non_iid_data
from topology import setup_topology
from factory import create_clients
from text_utils import get_label_text_embeddings

def main(args):
    seed = 42
    random.seed(seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 数据准备
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # --- VVV 新增：生成所有类别的 LTE VVV ---
    cifar10_classes = train_dataset.classes
    lte_all_classes = get_label_text_embeddings(cifar10_classes, device)
    if lte_all_classes is None:
        return # 如果CLIP加载失败，则退出
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    public_dataloader = DataLoader(train_dataset, batch_size=args.distill_batch_size, shuffle=True)
    public_data_iterator = iter(public_dataloader)

    # 2. 创建客户端 (使用 factory)
    client_datasets = get_non_iid_data(args.num_clients, train_dataset, seed=seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dist_filename = f"client_data_distribution_{timestamp}.csv"
    data_distribution_file = "distribution"
    os.makedirs(data_distribution_file,exist_ok=True)
    dist_filepath = os.path.join(data_distribution_file, dist_filename)
    with open(dist_filepath, "w") as f:
        # 写入CSV文件的表头 (header)
        header = "client_id," + ",".join([f"class_{i}" for i in range(10)]) + "\n"
        f.write(header)
        
        targets = np.array(train_dataset.targets)
        for i, client_subset in enumerate(client_datasets):
            client_labels = targets[client_subset.indices]
            class_counts = np.bincount(client_labels, minlength=10)
            
            # 准备要写入文件的一行数据
            row_data = f"{i}," + ",".join(map(str, class_counts.tolist())) + "\n"
            f.write(row_data)
            
            # 同时也在控制台打印出来，方便实时查看
            print(f"客户端 {i}: {class_counts.tolist()}")
    
    print(f"数据分布详情已保存至: {dist_filepath}")
    print("------------------------------------\n")

    clients = create_clients(args.num_clients, client_datasets, device, lte_all_classes, args.lambda_ltc, args.bounding_radius)
    
    # 3. 设置网络拓扑 (使用 topology 模块)
    setup_topology(clients, args)

    # 4. 准备日志和结果文件
    output_base_dir = "result"
    image_output_dir = "generated_images"
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(output_base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result_filename = f"{args.distill_data_source}_weights-{args.teacher_selector}_{args.sample_filter}_{args.distill_mode}_{timestamp}.txt"
    result_file_path =os.path.join(output_base_dir, result_filename)
    print(f"结果将保存至: {result_file_path}")
    with open(result_file_path, "w") as f:
        f.write("Round,Avg_Train_Loss,Avg_Distillation_Loss,Avg_Test_Accuracy,Generation_Time_s\n")
    print(f"\n--- 开始训练 (蒸馏数据源: {args.distill_data_source}) ---")
    print(f"\n--- 开始训练 (蒸馏模式: {args.distillation_data_mode}, 教师模式: {args.teacher_selector}) ---")
    if args.teacher_weights:
        print(f"使用自定义教师权重: {args.teacher_weights}")
    else:
        print("所有教师使用相等权重。")

    # 5. 主训练循环
    for round_num in range(1, args.rounds + 1):
        print(f"\n--- 第 {round_num}/{args.rounds} 轮 ---")

        # --- VVV FedD3A 新增步骤: 在本地训练前，计算投影矩阵 VVV ---
        if args.teacher_selector == 'fedd3a' or args.teacher_selector == 'top1' or args.teacher_selector == 'top2' or args.teacher_selector == 'top3':
            print("各客户端根据本地数据计算投影矩阵...")
            for client in clients:
                # 调用 Client 类中的新方法
                client.calculate_projection_matrix()

        local_losses = [c.train_local_model_step() for c in clients]
        # 只有在 'generated' 模式下才需要训练和同步扩散模型
        if args.distill_data_source == 'generated':
            for c in clients: c.train_local_diffusion_step()
            for client in clients: client.perform_generator_consensus()
        
        distillation_losses = []
        total_generation_time = 0

        # 准备蒸馏数据
        distillation_data = None
        if args.distill_data_source == 'original':
            # 'original' 模式下，从公共数据加载器中取一个批次的数据
            try:
                distillation_data, _ = next(public_data_iterator)
            except StopIteration:
                # 如果数据用完了，就重置加载器
                public_data_iterator = iter(public_dataloader)
                distillation_data, _ = next(public_data_iterator)
            distillation_data = distillation_data.to(device)

        for student_client in clients:
            if not student_client.neighbors: continue
            
            local_distillation_data = distillation_data
            # 如果是 'generated' 模式，现在为每个客户端独立生成数据
            if args.distill_data_source == 'generated':
                start_time = time.time()
                if args.distillation_data_mode == 'hard':
                    local_distillation_data = student_client.generate_consensus_hard_samples_diffusion(num_samples=args.distill_batch_size, guidance_scale=args.guidance_scale)
                else:
                    local_distillation_data = student_client.generate_normal_samples_diffusion(num_samples=args.distill_batch_size)
                total_generation_time += (time.time() - start_time)

                if local_distillation_data is None: continue
                
                # 保存生成的图像
                if args.distill_data_source == 'generated' and round_num % args.eval_interval == 0:
                    filename = os.path.join(image_output_dir, f"round_{round_num:03d}_client_{student_client.id:02d}_{args.distillation_data_mode}.png")
                    save_image(local_distillation_data, filename, nrow=8)
                
            # --- VVV 根据 teacher_selector 执行不同的蒸馏策略 VVV ---
            loss = student_client.distill(
                distillation_data=local_distillation_data,
                teacher_selector=args.teacher_selector,
                sample_filter=args.sample_filter,
                hard_sample_ratio=args.hard_sample_ratio,
                distill_mode=args.distill_mode,
                dkd_alpha=args.dkd_alpha,
                dkd_beta=args.dkd_beta
            )
            
            distillation_losses.append(loss)
        
        # 定期评估并记录结果
        if round_num % args.eval_interval == 0:
            avg_local_loss = np.mean(local_losses) if local_losses else 0.0
            avg_distill_loss = np.mean(distillation_losses) if distillation_losses else 0.0
            avg_acc = np.mean([c.evaluate(test_loader) for c in clients])
            
            log_message = (f"第 {round_num}/{args.rounds} 轮评估 | 平均本地损失: {avg_local_loss:.4f} | "
                           f"平均蒸馏损失: {avg_distill_loss:.4f} | 平均测试准确率: {avg_acc:.2f}% | "
                           f"样本生成耗时: {total_generation_time:.2f}s")
            print(log_message)
            
            with open(result_file_path, "a") as f:
                f.write(f"{round_num},{avg_local_loss:.4f},{avg_distill_loss:.4f},{avg_acc:.2f},{total_generation_time:.2f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="异构模型下的多教师加权蒸馏去中心化联邦学习实验")
    # ... 参数解析部分与您提供的代码完全相同 ...
    parser.add_argument('--num_clients', type=int, default=10, help='客户端总数 (semidfl 拓扑需要10个)')
    parser.add_argument('--rounds', type=int, default=50, help='总通信轮数')
    parser.add_argument('--topology', type=str, default='semidfl', choices=['ring', 'fully_connected', 'semidfl'], help='客户端连接拓扑')
    parser.add_argument('--eval_interval', type=int, default=5, help='评估与图像保存的间隔轮数')
    parser.add_argument('--distill_batch_size', type=int, default=32, help='每轮蒸馏用的样本数量')
    parser.add_argument('--distillation_data_mode', type=str, default='normal', choices=['hard', 'normal'], help='用于蒸馏的数据类型: hard (困难样本), normal (普通样本)')
    parser.add_argument('--guidance_scale', type=float, default=0.01, help='困难样本生成的引导强度')
    parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的GPU索引')
    parser.add_argument('--teacher_weights', type=float, nargs='*', default=None, help='用空格分隔的教师模型权重列表。若不提供，则默认为所有教师权重相等。')
    parser.add_argument('--teacher_selector', type=str, default='fedd3a', choices=['all', 'random', 'fedd3a', 'top1', 'top2', 'top3'], 
                        help='''选择教师的策略: 
                                all(所有邻居平均), 
                                random(随机单个邻居), 
                                fedd3a(FedD3A软权重), 
                                top1/top2/top3(选择最相似的k个教师)''')
    parser.add_argument('--sample_filter', type=str, default='none', choices=['none', 'confidence', 'hard_scarcity'], 
                        help='''选择蒸馏样本的策略: 
                                none(使用所有样本), 
                                confidence(只用教师置信度最低的困难样本)
                                hard_scarcity(优先生成数据稀缺的类别)''')
    parser.add_argument('--distill_mode', type=str, default='dkd', 
                        choices=['kd', 'dkd'], 
                        help='选择知识蒸馏的模式: kd(经典KL散度), dkd(解耦知识蒸馏)')

    # 2. 定义 DKD 的超参数 alpha 和 beta
    parser.add_argument('--dkd_alpha', type=float, default=1.0, 
                        help='DKD中TCKD的权重')
    parser.add_argument('--dkd_beta', type=float, default=8.0, 
                        help='DKD中NCKD的权重 (通常设置得比alpha大)')
    parser.add_argument('--hard_sample_ratio', type=float, default=0.5, help='在 confidence 模式下，选择的困难样本比例')
    parser.add_argument('--distill_data_source', type=str, default='generated', choices=['generated', 'original'], help='选择用于蒸馏的数据来源: generated (由扩散模型生成), original (从原始训练集中采样)')
    parser.add_argument('--lambda_ltc', type=float, default=0, help='Bounding Loss 的权重超参数')
    parser.add_argument('--bounding_radius', type=float, default=0.1, help='Bounding Loss 的半径超参数 r')
    args = parser.parse_args()
    main(args)