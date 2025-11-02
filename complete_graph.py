#!/usr/bin/env python3
"""
脚本：补全图数据，确保所有节点都有连接
目标：让 bay_vio_data_03_19.csv 中的所有节点都在图中有边连接
"""

import pandas as pd
import numpy as np
import os
from itertools import combinations
import random

def analyze_existing_data():
    """分析现有数据的节点范围"""
    print("=== 分析现有数据 ===")
    
    # 分析请求数据中的节点
    vio_data = pd.read_csv('/Users/akiyama/Downloads/budgted_reinforcement_learning_for_fairness_in_real_time_dispatching_system-main/data/bay_vio_data_03_19.csv')
    
    origins = set()
    destinations = set()
    
    for _, row in vio_data.iterrows():
        if isinstance(row['street_marker'], str) and row['street_marker'].startswith('A'):
            origins.add(int(row['street_marker'][1:]))
        if isinstance(row['aim_marker'], str) and row['aim_marker'].startswith('A'):
            destinations.add(int(row['aim_marker'][1:]))
    
    print(f"请求数据中的起始节点范围: {min(origins)} - {max(origins)} (共{len(origins)}个)")
    print(f"请求数据中的终点节点范围: {min(destinations)} - {max(destinations)} (共{len(destinations)}个)")
    
    # 分析现有图数据
    graph_data = pd.read_csv('/Users/akiyama/Downloads/budgted_reinforcement_learning_for_fairness_in_real_time_dispatching_system-main/data/dis_CBD_twoPs_03_19.csv')
    
    graph_nodes = set()
    for _, row in graph_data.iterrows():
        nodes = row['twoPs'].split('_')
        for node in nodes:
            if node.startswith('A'):
                graph_nodes.add(int(node[1:]))
    
    print(f"现有图数据中的节点范围: {min(graph_nodes)} - {max(graph_nodes)} (共{len(graph_nodes)}个)")
    
    all_needed_nodes = origins.union(destinations)
    missing_nodes = all_needed_nodes - graph_nodes
    
    print(f"缺失的节点: {len(missing_nodes)}个")
    if len(missing_nodes) < 50:
        print(f"缺失节点列表: {sorted(missing_nodes)}")
    else:
        print(f"部分缺失节点: {sorted(list(missing_nodes))[:20]}...")
    
    return all_needed_nodes, graph_nodes, missing_nodes

def generate_distances_from_existing():
    """从现有数据中分析距离分布，用于生成新的距离"""
    graph_data = pd.read_csv('/Users/akiyama/Downloads/budgted_reinforcement_learning_for_fairness_in_real_time_dispatching_system-main/data/dis_CBD_twoPs_03_19.csv')
    
    # 排除自环（A0_A0这种）
    non_zero_distances = graph_data[graph_data['distance'] > 0]['distance'].values
    
    stats = {
        'mean': np.mean(non_zero_distances),
        'std': np.std(non_zero_distances),
        'min': np.min(non_zero_distances),
        'max': np.max(non_zero_distances),
        'median': np.median(non_zero_distances)
    }
    
    print(f"\n=== 现有距离统计 ===")
    print(f"平均距离: {stats['mean']:.2f}")
    print(f"标准差: {stats['std']:.2f}")
    print(f"最小距离: {stats['min']:.2f}")
    print(f"最大距离: {stats['max']:.2f}")
    print(f"中位数: {stats['median']:.2f}")
    
    return stats

def generate_realistic_distance(node1, node2, distance_stats):
    """生成较为现实的距离值"""
    if node1 == node2:
        return 0
    
    # 根据节点差值生成基础距离
    node_diff = abs(node1 - node2)
    
    if node_diff == 1:
        # 相邻节点，距离较小
        base_distance = np.random.normal(500, 200)
    elif node_diff <= 10:
        # 近邻节点
        base_distance = np.random.normal(1500, 500)
    elif node_diff <= 50:
        # 中等距离
        base_distance = np.random.normal(3000, 1000)
    else:
        # 远距离节点
        base_distance = np.random.normal(distance_stats['mean'], distance_stats['std'])
    
    # 确保距离在合理范围内
    distance = max(100, min(15000, abs(base_distance)))
    
    return round(distance, 2)

def create_complete_graph():
    """创建完整的图数据"""
    print("\n=== 开始生成完整图数据 ===")
    
    all_needed_nodes, existing_nodes, missing_nodes = analyze_existing_data()
    distance_stats = generate_distances_from_existing()
    
    # 读取现有图数据
    existing_graph = pd.read_csv('/Users/akiyama/Downloads/budgted_reinforcement_learning_for_fairness_in_real_time_dispatching_system-main/data/dis_CBD_twoPs_03_19.csv')
    
    # 存储所有边
    edges = {}
    
    # 添加现有边
    for _, row in existing_graph.iterrows():
        edges[row['twoPs']] = row['distance']
    
    print(f"现有边数量: {len(edges)}")
    
    # 为所有需要的节点生成连接
    all_nodes = sorted(list(all_needed_nodes))
    
    print(f"需要确保连通的节点总数: {len(all_nodes)}")
    
    # 1. 确保每个节点至少与几个其他节点相连（保证连通性）
    print("生成基本连通性...")
    for i, node in enumerate(all_nodes):
        # 每个节点至少连接到前后几个节点
        connect_to = []
        
        # 连接到前后2个节点
        if i > 0:
            connect_to.append(all_nodes[i-1])
        if i < len(all_nodes) - 1:
            connect_to.append(all_nodes[i+1])
        
        # 随机连接到几个其他节点
        other_nodes = [n for n in all_nodes if n != node and n not in connect_to]
        if other_nodes:
            random.shuffle(other_nodes)
            connect_to.extend(other_nodes[:min(3, len(other_nodes))])
        
        # 生成边
        for target in connect_to:
            edge1 = f"A{node}_A{target}"
            edge2 = f"A{target}_A{node}"
            
            if edge1 not in edges and edge2 not in edges:
                distance = generate_realistic_distance(node, target, distance_stats)
                edges[edge1] = distance
                edges[edge2] = distance
    
    # 2. 添加自环（A0_A0 = 0）
    print("添加自环...")
    for node in all_nodes:
        self_edge = f"A{node}_A{node}"
        if self_edge not in edges:
            edges[self_edge] = 0
    
    # 3. 为高频节点对添加直接连接
    print("添加高频节点对连接...")
    vio_data = pd.read_csv('/Users/akiyama/Downloads/budgted_reinforcement_learning_for_fairness_in_real_time_dispatching_system-main/data/bay_vio_data_03_19.csv')
    
    # 确保所有请求中的节点对都有边
    print("确保所有请求的节点对都有连接...")
    for _, row in vio_data.iterrows():
        if (isinstance(row['street_marker'], str) and row['street_marker'].startswith('A') and
            isinstance(row['aim_marker'], str) and row['aim_marker'].startswith('A')):
            
            origin = int(row['street_marker'][1:])
            dest = int(row['aim_marker'][1:])
            
            edge1 = f"A{origin}_A{dest}"
            edge2 = f"A{dest}_A{origin}"
            
            # 如果这个节点对没有连接，就添加
            if edge1 not in edges and edge2 not in edges:
                distance = generate_realistic_distance(origin, dest, distance_stats)
                edges[edge1] = distance
                # 注意：这里不添加反向边，因为图可能是有向的
    
    print(f"最终图中的边数量: {len(edges)}")
    
    # 创建新的DataFrame
    new_edges_data = []
    for edge_name, distance in edges.items():
        new_edges_data.append({'distance': distance, 'twoPs': edge_name})
    
    new_graph_df = pd.DataFrame(new_edges_data)
    new_graph_df = new_graph_df.sort_values('twoPs').reset_index(drop=True)
    
    # 保存到新文件
    output_file = '/Users/akiyama/Downloads/budgted_reinforcement_learning_for_fairness_in_real_time_dispatching_system-main/data/dis_CBD_twoPs_03_19_complete.csv'
    new_graph_df.to_csv(output_file, index=False)
    
    print(f"完整图数据已保存到: {output_file}")
    
    # 验证覆盖性
    verify_coverage(output_file)
    
    return output_file

def verify_coverage(graph_file):
    """验证图是否覆盖所有需要的节点对"""
    print("\n=== 验证图覆盖性 ===")
    
    # 读取请求数据
    vio_data = pd.read_csv('/Users/akiyama/Downloads/budgted_reinforcement_learning_for_fairness_in_real_time_dispatching_system-main/data/bay_vio_data_03_19.csv')
    
    # 读取图数据
    graph_data = pd.read_csv(graph_file)
    
    # 提取图中的所有边
    graph_edges = set()
    for _, row in graph_data.iterrows():
        graph_edges.add(row['twoPs'])
    
    # 检查请求数据中的所有节点对是否都在图中
    missing_edges = []
    total_requests = 0
    
    for _, row in vio_data.iterrows():
        if (isinstance(row['street_marker'], str) and row['street_marker'].startswith('A') and
            isinstance(row['aim_marker'], str) and row['aim_marker'].startswith('A')):
            
            origin = row['street_marker']
            dest = row['aim_marker']
            edge1 = f"{origin}_{dest}"
            edge2 = f"{dest}_{origin}"
            
            total_requests += 1
            
            if edge1 not in graph_edges and edge2 not in graph_edges:
                missing_edges.append((origin, dest))
    
    print(f"总请求数: {total_requests}")
    print(f"缺失的边: {len(missing_edges)}")
    
    if missing_edges:
        print("部分缺失的边:")
        for i, (o, d) in enumerate(missing_edges[:10]):
            print(f"  {o} -> {d}")
        if len(missing_edges) > 10:
            print(f"  ... 还有 {len(missing_edges) - 10} 个缺失边")
        return False
    else:
        print("✅ 所有请求的节点对都在图中有对应的边！")
        return True

def update_utils_py(new_graph_file):
    """更新utils.py以使用新的完整图文件"""
    print(f"\n=== 更新utils.py以使用新图文件 ===")
    
    # 备份原文件
    utils_file = '/Users/akiyama/Downloads/budgted_reinforcement_learning_for_fairness_in_real_time_dispatching_system-main/data/utils.py'
    backup_file = utils_file + '.backup'
    
    with open(utils_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已备份原文件到: {backup_file}")
    
    # 更新文件路径
    new_filename = os.path.basename(new_graph_file)
    updated_content = content.replace(
        "file_path = os.path.join(current_dir, 'dis_CBD_twoPs_03_19.csv')",
        f"file_path = os.path.join(current_dir, '{new_filename}')"
    )
    
    with open(utils_file, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"已更新utils.py使用新图文件: {new_filename}")

if __name__ == "__main__":
    print("🚀 开始补全图数据...")
    
    # 设置随机种子以获得可重复的结果
    random.seed(42)
    np.random.seed(42)
    
    try:
        # 创建完整图
        new_graph_file = create_complete_graph()
        
        # 更新utils.py
        update_utils_py(new_graph_file)
        
        print("\n🎉 图数据补全完成！")
        print("现在可以运行实验了:")
        print("PYTHONPATH=. python train/train.py --select 0")
        print("PYTHONPATH=. python train/train.py --select 1")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()