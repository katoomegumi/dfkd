# factory.py
# VVV 确保从 models 导入了新的 ResNet 函数 VVV
from models import ResNet10_CIFAR, ResNet18_CIFAR, ResNet34_CIFAR, LeNet5_CIFAR
from client1 import Client

def create_clients(num_clients, client_datasets, device, rounds):
    """
    客户端创建工厂函数，现在使用 ResNet 模型。
    """
    # --- VVV 定义一个包含所有异构 ResNet 架构的列表 VVV ---
    model_architectures = [ResNet18_CIFAR, ResNet34_CIFAR, ResNet10_CIFAR]
    model_architectures = [ResNet18_CIFAR, ResNet34_CIFAR, LeNet5_CIFAR]
    clients = []
    
    for i in range(num_clients):
        # 使用取模运算 (%) 循环地从模型列表中选择一个架构
        model_class_to_use = model_architectures[i % len(model_architectures)]
        model_instance = model_class_to_use() # 实例化选中的模型
            
        client = Client(
            client_id=i, 
            local_data=client_datasets[i], 
            device=device, 
            model_instance=model_instance,
            rounds=rounds
        )
        
        clients.append(client)
        print(f"客户端 {i} 已创建，分配的模型为: {model_class_to_use.__name__}")
        
    return clients