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
    # too large around 2.7gb, use 1drive link to dl
    response = requests.get("https://onedrive.live.com/download?resid=AlDeipOkaENXazCoXrgWvD0uaX0", stream=True,timeout=10,proxies={"http": None, "https": None}, verify=False)
    response.raise_for_status()
    project_dir = os.path.dirname(os.getcwd())
    file_path = project_dir + 'data/dis_CBD_twoPs_03_19.csv'
    total_size = int(response.headers.get('content-length', 0))

    with open(file_path, "wb") as file, tqdm(
            desc=file_path,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
            progress_bar.update(len(chunk))

    data = pd.read_csv(file_path)
    graph = nx.Graph()

    for row in data.itertuples(index=False):
        dis = row.distance
        nodes = row.twoPs.split('_')
        node1 = nodes[0]
        node2 = nodes[1]
        try:
            int(node1.replace("A", ""))
            int(node2.replace("A", ""))
        except ValueError:
            continue
        # 检查节点是否已经存在于图中
        if int(node1.replace("A", "")) not in graph.nodes:
            graph.add_node(int(node1.replace("A", "")))
        if int(node2.replace("A", "")) not in graph.nodes:
            graph.add_node(int(node2.replace("A", "")))
        # 检查边是否已经存在于图中
        if not graph.has_edge(int(node1.replace("A", "")), int(node2.replace("A", ""))):
            graph.add_edge(int(node1.replace("A", "")), int(node2.replace("A", "")), distance=dis)
        if not graph.has_edge(int(node2.replace("A", "")), int(node1.replace("A", ""))):
            graph.add_edge(int(node2.replace("A", "")), int(node1.replace("A", "")), distance=dis)
    import sys
    return graph

create_graph()
def choose_random_node(graph):
    nodes = list(graph.nodes)
    random_node = random.choice(nodes)
    return random_node


import pandas as pd
def import_requests_from_csv():
    project_dir = os.path.dirname(os.getcwd())
    file_path = project_dir + "/data/bay_vio_data_03_19.csv"
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
    project_dir = os.path.dirname(os.getcwd())
    file_path = project_dir+'/data/fairness.npy'
    array = np.load(file_path)
    return array

print(load_budget())

def load_location():
    project_dir = os.path.dirname(os.getcwd())
    file_path = project_dir +'/data/init_location.npy'
    array = np.load(file_path)
    return array

def load_minuium_budget():
    project_dir = os.path.dirname(os.getcwd())
    file_path = project_dir +'/data/fairness_min_max.npy'
    array = np.load(file_path)
    return array

def generate_npy():
    random_ints = np.random.randint(0,5000,size = 25)
    project_dir = os.path.dirname(os.getcwd())
    file_path = project_dir + '/data/init_location.npy'
    np.save(file_path,random_ints)

def get_random():
    return [random.randint(0, 999) for _ in range(15)]
random_list=get_random()
