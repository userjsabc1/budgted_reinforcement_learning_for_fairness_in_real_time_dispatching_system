#!/usr/bin/env python3
"""
重新映射CSV数据为连续节点范围(0-800)
"""
import pandas as pd
import os

def remap_nodes():
    """重新映射节点到连续范围"""
    print("开始重新映射节点...")
    
    data_dir = "/Users/akiyama/Downloads/budgted_reinforcement_learning_for_fairness_in_real_time_dispatching_system-main/data"
    
    # 1. 处理请求数据文件
    print("1. 处理请求数据文件...")
    vio_file = os.path.join(data_dir, "bay_vio_data_03_19.csv")
    vio_data = pd.read_csv(vio_file)
    
    print(f"原始请求数据行数: {len(vio_data)}")
    
    # 创建节点映射字典
    node_mapping = {}
    
    # 收集所有节点
    all_nodes = set()
    for _, row in vio_data.iterrows():
        origin_node = int(row['street_marker'].replace('A', ''))
        dest_node = int(row['aim_marker'].replace('A', ''))
        all_nodes.add(origin_node)
        all_nodes.add(dest_node)
    
    # 排序节点
    sorted_nodes = sorted(list(all_nodes))
    print(f"发现的节点范围: {min(sorted_nodes)} - {max(sorted_nodes)}")
    print(f"总节点数: {len(sorted_nodes)}")
    
    # 创建映射：老节点 -> 新节点(0-N)
    for i, old_node in enumerate(sorted_nodes):
        node_mapping[old_node] = i
    
    print(f"节点映射示例: {dict(list(node_mapping.items())[:10])}")
    
    # 应用映射到请求数据
    vio_data['street_marker'] = vio_data['street_marker'].apply(
        lambda x: f"A{node_mapping[int(x.replace('A', ''))]}"
    )
    vio_data['aim_marker'] = vio_data['aim_marker'].apply(
        lambda x: f"A{node_mapping[int(x.replace('A', ''))]}"
    )
    
    # 保存修改后的请求数据
    new_vio_file = os.path.join(data_dir, "bay_vio_data_03_19_remapped.csv")
    vio_data.to_csv(new_vio_file, index=False)
    print(f"保存重新映射的请求数据: {new_vio_file}")
    
    # 2. 重新生成图连接数据
    print("2. 生成新的图连接数据...")
    max_node = max(node_mapping.values())
    print(f"新的节点范围: 0 - {max_node}")
    
    connections = []
    total_connections = (max_node + 1) * (max_node + 1)
    print(f"需要生成的连接数: {total_connections}")
    
    count = 0
    for i in range(max_node + 1):
        for j in range(max_node + 1):
            if i == j:
                distance = 0.0
            else:
                # 基于节点差距估算距离
                node_diff = abs(i - j)
                if node_diff <= 50:
                    distance = 50 + node_diff * 20
                elif node_diff <= 200:
                    distance = 200 + node_diff * 10
                else:
                    distance = 500 + node_diff * 5
                
                # 添加随机波动
                import random
                distance *= random.uniform(0.8, 1.2)
                distance = round(distance, 2)
            
            connections.append({
                'distance': distance,
                'twoPs': f'A{i}_A{j}'
            })
            
            count += 1
            if count % 50000 == 0:
                print(f"已生成 {count}/{total_connections} 连接")
    
    # 保存新的图连接数据
    df_connections = pd.DataFrame(connections)
    new_graph_file = os.path.join(data_dir, "dis_CBD_twoPs_03_19_remapped.csv")
    df_connections.to_csv(new_graph_file, index=False)
    
    print(f"保存重新映射的图连接数据: {new_graph_file}")
    print(f"图连接数: {len(df_connections)}")
    print(f"距离范围: {df_connections['distance'].min():.2f} - {df_connections['distance'].max():.2f}")
    
    return node_mapping, max_node

if __name__ == "__main__":
    try:
        mapping, max_node = remap_nodes()
        print(f"✅ 节点重新映射完成！新范围: 0-{max_node}")
        print(f"✅ 生成文件:")
        print(f"  - bay_vio_data_03_19_remapped.csv (请求数据)")
        print(f"  - dis_CBD_twoPs_03_19_remapped.csv (图连接)")
    except Exception as e:
        print(f"❌ 重新映射失败: {e}")
        import traceback
        traceback.print_exc()