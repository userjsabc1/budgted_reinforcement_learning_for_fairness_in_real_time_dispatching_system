#!/usr/bin/env python3
"""
快速生成完整图连接CSV
直接生成800个节点(0-399 + 5000-5399)的完全连接图
"""
import pandas as pd
import numpy as np
import random
import os

def generate_full_graph_csv():
    """生成完整的图CSV文件"""
    print("开始生成完整图连接CSV...")
    
    # 定义节点范围
    origin_nodes = list(range(0, 401))  # 0-400
    dest_nodes = list(range(5000, 5401))  # 5000-5400 (401个节点)
    all_nodes = origin_nodes + dest_nodes
    
    print(f"起点节点: 0-400 ({len(origin_nodes)}个)")
    print(f"终点节点: 5000-5400 ({len(dest_nodes)}个)")
    print(f"总节点数: {len(all_nodes)}")
    
    # 基于现有数据设定距离范围
    min_distance = 50    # 最小距离
    max_distance = 20000 # 最大距离
    avg_distance = 5000  # 平均距离
    
    connections = []
    total_connections = len(all_nodes) * len(all_nodes)
    
    print(f"需要生成连接数: {total_connections}")
    
    count = 0
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            # 计算距离
            if node1 == node2:
                distance = 0.0
            else:
                # 基于节点差距估算距离
                if abs(node1 - node2) <= 50:
                    # 近距离节点
                    distance = random.uniform(min_distance, avg_distance * 0.8)
                elif abs(node1 - node2) <= 200:
                    # 中距离节点
                    distance = random.uniform(avg_distance * 0.5, avg_distance * 1.5)
                else:
                    # 远距离节点
                    distance = random.uniform(avg_distance * 0.8, max_distance * 0.8)
                
                # 添加随机噪声
                distance *= random.uniform(0.9, 1.1)
                distance = round(distance, 2)
            
            # 添加连接
            connections.append({
                'distance': distance,
                'twoPs': f'A{node1}_A{node2}'
            })
            
            count += 1
            
            # 进度显示
            if count % 50000 == 0:
                print(f"已生成 {count}/{total_connections} 连接 ({count/total_connections*100:.1f}%)")
    
    # 创建DataFrame并保存
    print("创建DataFrame...")
    df = pd.DataFrame(connections)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, 'dis_CBD_twoPs_03_19_full.csv')
    
    print("保存CSV文件...")
    df.to_csv(output_file, index=False)
    
    print(f"✅ 成功生成完整图文件: {output_file}")
    print(f"总连接数: {len(df)}")
    print(f"距离范围: {df['distance'].min():.2f} - {df['distance'].max():.2f}")
    print(f"文件大小: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    return output_file

if __name__ == "__main__":
    print("="*60)
    print("快速生成完整图连接CSV (800×800)")
    print("="*60)
    
    try:
        output_file = generate_full_graph_csv()
        print("生成完成！")
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()