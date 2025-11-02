#!/usr/bin/env python3
"""
生成完整图连接脚本
基于现有数据生成全连接图，确保所有节点间都有距离信息
"""
import pandas as pd
import numpy as np
import os
from itertools import combinations
import random

def analyze_existing_data():
    """分析现有数据的距离范围"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 读取现有的图数据
    existing_files = [
        'dis_CBD_twoPs_03_19.csv',
        'dis_CBD_twoPs_03_19(1).csv'
    ]
    
    all_distances = []
    existing_pairs = set()
    
    for file in existing_files:
        file_path = os.path.join(current_dir, file)
        if os.path.exists(file_path):
            print(f"分析文件: {file}")
            data = pd.read_csv(file_path)
            all_distances.extend(data['distance'].tolist())
            
            # 收集已存在的节点对
            for _, row in data.iterrows():
                if pd.notna(row['twoPs']):
                    existing_pairs.add(row['twoPs'])
            
            print(f"  - 行数: {len(data)}")
            print(f"  - 距离范围: {data['distance'].min():.2f} - {data['distance'].max():.2f}")
    
    return all_distances, existing_pairs

def get_node_ranges():
    """获取节点范围"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 从bay_vio_data_03_19.csv获取实际使用的节点范围
    vio_data_path = os.path.join(current_dir, 'bay_vio_data_03_19.csv')
    vio_data = pd.read_csv(vio_data_path)
    
    # 提取起点和终点节点ID
    origin_nodes = set()
    dest_nodes = set()
    
    for _, row in vio_data.iterrows():
        if pd.notna(row['street_marker']):
            origin_id = int(row['street_marker'].replace('A', ''))
            origin_nodes.add(origin_id)
        
        if pd.notna(row['aim_marker']):
            dest_id = int(row['aim_marker'].replace('A', ''))
            dest_nodes.add(dest_id)
    
    print(f"起点节点范围: {min(origin_nodes)} - {max(origin_nodes)} (共{len(origin_nodes)}个)")
    print(f"终点节点范围: {min(dest_nodes)} - {max(dest_nodes)} (共{len(dest_nodes)}个)")
    
    return origin_nodes, dest_nodes

def estimate_distance(node1, node2, all_distances, base_distance=1000):
    """
    基于现有数据估算两点间距离
    使用启发式方法：根据节点ID差距和现有距离分布进行估算
    """
    if node1 == node2:
        return 0.0
    
    # 计算节点ID差距
    id_diff = abs(node1 - node2)
    
    # 基于现有距离分布计算统计信息
    min_dist = min(all_distances)
    max_dist = max(all_distances)
    avg_dist = np.mean(all_distances)
    std_dist = np.std(all_distances)
    
    # 启发式距离估算
    if id_diff == 1:
        # 相邻节点，使用较小距离
        estimated = random.uniform(min_dist * 0.8, avg_dist * 0.6)
    elif id_diff <= 10:
        # 近距离节点
        estimated = random.uniform(avg_dist * 0.5, avg_dist * 1.2)
    elif id_diff <= 50:
        # 中距离节点
        estimated = random.uniform(avg_dist * 0.8, avg_dist * 1.8)
    else:
        # 远距离节点
        estimated = random.uniform(avg_dist * 1.2, max_dist * 0.8)
    
    # 添加随机性，但保持合理性
    noise = random.uniform(0.9, 1.1)
    estimated *= noise
    
    # 确保在合理范围内
    estimated = max(min_dist * 0.5, min(estimated, max_dist * 1.2))
    
    return round(estimated, 2)

def generate_complete_graph():
    """生成完整的图连接"""
    print("开始生成完整图连接...")
    
    # 分析现有数据
    all_distances, existing_pairs = analyze_existing_data()
    origin_nodes, dest_nodes = get_node_ranges()
    
    # 合并所有节点
    all_nodes = sorted(list(origin_nodes.union(dest_nodes)))
    print(f"总节点数: {len(all_nodes)}")
    print(f"现有连接数: {len(existing_pairs)}")
    
    # 读取现有数据作为基础
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_file = os.path.join(current_dir, 'dis_CBD_twoPs_03_19.csv')
    
    if os.path.exists(base_file):
        existing_data = pd.read_csv(base_file)
        result_data = existing_data.copy()
    else:
        result_data = pd.DataFrame(columns=['distance', 'twoPs'])
    
    # 生成所有可能的节点对
    total_pairs = len(all_nodes) * (len(all_nodes) - 1) // 2  # 无向图的边数
    print(f"需要生成的总连接数: {total_pairs}")
    
    new_connections = []
    generated_count = 0
    
    # 为所有节点对生成连接
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i <= j:  # 避免重复（无向图）
                pair_key1 = f"A{node1}_A{node2}"
                pair_key2 = f"A{node2}_A{node1}"
                
                # 检查是否已存在
                if pair_key1 not in existing_pairs and pair_key2 not in existing_pairs:
                    # 估算距离
                    distance = estimate_distance(node1, node2, all_distances)
                    
                    # 添加双向连接
                    new_connections.append({
                        'distance': distance,
                        'twoPs': pair_key1
                    })
                    
                    if node1 != node2:  # 避免自环重复
                        new_connections.append({
                            'distance': distance,
                            'twoPs': pair_key2
                        })
                    
                    generated_count += 1
        
        # 进度显示
        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1}/{len(all_nodes)} 个节点...")
    
    print(f"新生成连接数: {generated_count}")
    print(f"新连接边数: {len(new_connections)}")
    
    # 合并数据
    if new_connections:
        new_df = pd.DataFrame(new_connections)
        result_data = pd.concat([result_data, new_df], ignore_index=True)
    
    # 去重并排序
    result_data = result_data.drop_duplicates(subset=['twoPs'])
    result_data = result_data.sort_values('twoPs')
    
    # 保存结果
    output_file = os.path.join(current_dir, 'dis_CBD_twoPs_03_19_complete_full.csv')
    result_data.to_csv(output_file, index=False)
    
    print(f"完整图数据已保存到: {output_file}")
    print(f"总连接数: {len(result_data)}")
    print(f"距离范围: {result_data['distance'].min():.2f} - {result_data['distance'].max():.2f}")
    
    return output_file

if __name__ == "__main__":
    print("="*50)
    print("生成完整图连接脚本")
    print("="*50)
    
    try:
        output_file = generate_complete_graph()
        print(f"✅ 成功生成完整图文件: {output_file}")
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()