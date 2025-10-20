def setup_topology(clients, args):
    """
    根据指定的拓扑结构为客户端设置邻居关系。

    参数:
        clients (list): 客户端对象的列表。
        args (argparse.Namespace): 包含 num_clients 和 topology 等配置的命名空间。
    """
    num_clients = args.num_clients
    topology = args.topology
    
    print(f"设置网络拓扑: {topology}")
    
    if topology == 'ring':
        for i in range(num_clients):
            clients[i].add_neighbor(clients[(i - 1 + num_clients) % num_clients])
            clients[i].add_neighbor(clients[(i + 1) % num_clients])
            
    elif topology == 'fully_connected':
        for i in range(num_clients):
            for j in range(num_clients):
                if i != j:
                    clients[i].add_neighbor(clients[j])
                    
    elif topology == 'semidfl':
        if num_clients != 10:
            print("警告: 'semidfl' 拓扑是为10个客户端专门设计的。客户端数量不为10，将回退到 'ring' 拓扑。")
            # 回退到ring拓扑
            for i in range(num_clients):
                clients[i].add_neighbor(clients[(i - 1 + num_clients) % num_clients])
                clients[i].add_neighbor(clients[(i + 1) % num_clients])
        else:
            # 为10个客户端专门设计的邻接列表 (0-indexed)
            adjacency_list = {
                0: [1, 2, 6, 7], 1: [0, 3, 4, 5], 2: [0, 3, 4, 7], 3: [1, 2, 4, 9], 4: [1, 2, 3, 8],
                5: [1, 6, 8, 9], 6: [0, 5, 7, 8], 7: [0, 2, 6, 9], 8: [4, 5, 6, 9], 9: [3, 5, 7, 8]
            }
            for client_id, neighbors in adjacency_list.items():
                for neighbor_id in neighbors:
                    clients[client_id].add_neighbor(clients[neighbor_id])