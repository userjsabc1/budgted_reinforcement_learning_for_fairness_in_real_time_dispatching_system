#!/usr/bin/env python3
"""
调试脚本：检查图连接和节点问题
"""
import sys
import os
sys.path.append(os.getcwd())

from data.utils import create_graph, import_requests_from_csv
import pandas as pd

def debug_graph_and_data():
    """调试图连接和数据问题"""
    print("="*60)
    print("调试图连接和数据")
    print("="*60)
    
    # 1. 检查图
    print("1. 创建图...")
    graph = create_graph()
    print(f"图节点数: {graph.number_of_nodes()}")
    print(f"图边数: {graph.number_of_edges()}")
    
    # 检查节点范围
    nodes = list(graph.nodes())
    print(f"节点范围: {min(nodes)} - {max(nodes)}")
    print(f"前10个节点: {sorted(nodes)[:10]}")
    print(f"后10个节点: {sorted(nodes)[-10:]}")
    
    # 2. 检查司机初始位置
    from data.utils import random_list
    print(f"\n2. 司机初始位置: {random_list[:10]}")
    
    # 检查这些位置是否在图中
    missing_driver_positions = []
    for pos in random_list[:5]:  # 只检查前5个司机
        if pos not in graph.nodes():
            missing_driver_positions.append(pos)
    
    if missing_driver_positions:
        print(f"❌ 司机位置不在图中: {missing_driver_positions}")
    else:
        print("✅ 所有司机位置都在图中")
    
    # 3. 检查订单数据
    print(f"\n3. 检查订单数据...")
    requests = import_requests_from_csv()
    print(f"订单批次数: {len(requests)}")
    if requests:
        first_batch = requests[0]
        print(f"第一批订单数: {len(first_batch)}")
        
        if first_batch:
            sample_request = first_batch[0]
            print(f"示例订单 - 起点: {sample_request.origin}, 终点: {sample_request.destination}")
            
            # 检查订单节点是否在图中
            missing_origins = []
            missing_destinations = []
            
            for req in first_batch[:10]:  # 检查前10个订单
                if req.origin not in graph.nodes():
                    missing_origins.append(req.origin)
                if req.destination not in graph.nodes():
                    missing_destinations.append(req.destination)
            
            if missing_origins:
                print(f"❌ 订单起点不在图中: {set(missing_origins)}")
            else:
                print("✅ 订单起点都在图中")
                
            if missing_destinations:
                print(f"❌ 订单终点不在图中: {set(missing_destinations)}")
            else:
                print("✅ 订单终点都在图中")
    
    # 4. 测试具体的边连接
    print(f"\n4. 测试边连接...")
    test_pairs = [
        (0, 1),      # 起点范围内
        (0, 5000),   # 起点到终点
        (100, 5100), # 中间节点
        (400, 5400)  # 边界节点
    ]
    
    for node1, node2 in test_pairs:
        if node1 in graph.nodes() and node2 in graph.nodes():
            edge_data = graph.get_edge_data(node1, node2)
            if edge_data:
                print(f"✅ {node1} -> {node2}: 距离 {edge_data['distance']}")
            else:
                print(f"❌ {node1} -> {node2}: 没有边连接")
        else:
            missing = []
            if node1 not in graph.nodes(): missing.append(node1)
            if node2 not in graph.nodes(): missing.append(node2)
            print(f"❌ {node1} -> {node2}: 节点不存在 {missing}")

if __name__ == "__main__":
    try:
        debug_graph_and_data()
    except Exception as e:
        print(f"调试过程出错: {e}")
        import traceback
        traceback.print_exc()