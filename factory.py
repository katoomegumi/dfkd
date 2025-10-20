# factory.py
from client import Client
from models import SimpleCNN, SimpleCNN_Wide, SimpleCNN_Deep

# VVV 修正：更新函数签名以接收新参数 VVV
def create_clients(num_clients, client_datasets, device, lte_all_classes, lambda_ltc, bounding_radius):
    """
    客户端创建工厂函数。此函数现在会接收并传递
    LTE相关的参数给每一个客户端。
    """
    model_types = ['base', 'wide', 'deep']
    clients = []
    
    for i in range(num_clients):
        model_type = model_types[i % len(model_types)]
        
        # 根据类型选择相应的模型类
        if model_type == 'base':
            model_instance = SimpleCNN()
        elif model_type == 'wide':
            model_instance = SimpleCNN_Wide()
        else: # deep
            model_instance = SimpleCNN_Deep()
            
        # VVV 修正：调用Client构造函数时传入所有参数 VVV
        client = Client(
            client_id=i, 
            local_data=client_datasets[i], 
            device=device, 
            model_instance=model_instance,
            lte_all_classes=lte_all_classes,
            lambda_ltc=lambda_ltc,
            bounding_radius=bounding_radius
        )
        
        clients.append(client)
        print(f"客户端 {i} 已创建，分配的模型为: {model_instance.__class__.__name__}")
        
    return clients