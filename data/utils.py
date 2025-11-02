import os
import random

import networkx as nx
import numpy as np
import pandas as pd
import csv
import tempfile
import shutil

import requests
from tqdm import tqdm


def create_graph():
    """使用重新映射的本地数据文件创建图结构"""
    # 使用新的重新映射文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'dis_CBD_twoPs_03_19_remapped.csv')
    
    try:
        print(f"加载重新映射的图数据文件: {file_path}")
        data = pd.read_csv(file_path)
        print(f"数据行数: {len(data)}")
    except FileNotFoundError:
        print(f"重新映射文件未找到，使用旧文件")
        file_path = os.path.join(current_dir, 'dis_CBD_twoPs_03_19_full.csv')
        try:
            data = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"所有图文件都未找到，创建默认图")
            # 创建一个简单的默认图
            graph = nx.Graph()
            for i in range(800):  # 0-799
                graph.add_node(i)
            # 添加一些基本连接
            for i in range(799):
                graph.add_edge(i, i+1, distance=100)
            return graph
    
    graph = nx.Graph()
    processed_count = 0

    for row in data.itertuples(index=False):
        dis = row.distance
        nodes = row.twoPs.split('_')
        node1 = nodes[0]
        node2 = nodes[1]
        try:
            node1_id = int(node1.replace("A", ""))
            node2_id = int(node2.replace("A", ""))
        except ValueError:
            continue
            
        # 添加节点（如果不存在）
        if node1_id not in graph.nodes:
            graph.add_node(node1_id)
        if node2_id not in graph.nodes:
            graph.add_node(node2_id)
            
        # 添加边（无向图，只需要添加一次）
        if not graph.has_edge(node1_id, node2_id):
            graph.add_edge(node1_id, node2_id, distance=dis)
            
        processed_count += 1
        if processed_count % 100000 == 0:
            print(f"已处理 {processed_count} 条边...")
    
    print(f"✅ 图创建完成：{graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")
    return graph

# create_graph()  # 注释掉自动执行，避免导入时创建图
def choose_random_node(graph):
    nodes = list(graph.nodes)
    random_node = random.choice(nodes)
    return random_node


import pandas as pd
def import_requests_from_csv():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "bay_vio_data_03_19_remapped.csv")
    if not os.path.exists(file_path):
        # 回退到原始文件
        file_path = os.path.join(os.path.dirname(current_dir), "data", "bay_vio_data_03_19.csv")
    
    requests = [[]]
    data = pd.read_csv(file_path)
    for row in data.itertuples(index=False):
        timestamp = int(row.RequestTime)
        destination = change_node_to_int(row.aim_marker)
        origin = change_node_to_int(row.street_marker)
        request = Request(timestamp, destination, origin)
        if request.destination != origin:
            requests[0].append(request)

    return requests

# 定义请求结构
class Request:
    def __init__(self, timestamp, destination, origin):
        self.timestamp = timestamp
        self.destination = destination
        self.origin = origin
        self.state = 0


class Driver:
    def __init__(self, speed):
        self.on_road = None
        self.start_time = 0
        self.Request = None
        self.idx = None
        self.money = None
        self.speed = speed
        self.pos = None

    def __str__(self):
        if self.Request is not None:
            return f"Driver(speed={self.speed}, idx={self.idx}, money={self.money},on_road={self.on_road}, start_time={self.start_time},Request={self.Request.origin,self.Request.destination,self.Request.state,self.Request.timestamp},pos={self.pos})"

        return f"Driver(speed={self.speed}, idx={self.idx}, money={self.money},on_road={self.on_road}, start_time={self.start_time},pos={self.pos})"

    # 从CSV文件中导入请求





def change_node_to_int(node):
    try:
        return int(node.replace("A", ""))
    except ValueError:
        return 0





def load_budget():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'fairness.npy')
    array = np.load(file_path)
    return array

print(load_budget())

def load_location():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'init_location.npy')
    array = np.load(file_path)
    return array

def load_minuium_budget():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'fairness_min_max.npy')
    array = np.load(file_path)
    return array

def generate_npy():
    random_ints = np.random.randint(0,5000,size = 25)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'init_location.npy')
    np.save(file_path,random_ints)

def get_random():
    """生成司机初始位置，确保在图的节点范围内"""
    # 从0-793范围选择初始位置（重新映射后的节点范围）
    return [random.randint(0, 793) for _ in range(15)]

random_list = get_random()
