import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

#####################################
#UpperBodyNodes 25, 24, 12, 11, 10, 9, 21, 3, 4, 5, 6, 7, 8,  22, 23, 2
#reindex         0, 1,  2,  3,  4,  5, 6,  7, 8, 9, 10,11,12, 13, 14, 15
##########################################

num_node_velocity = 16
self_link_velocity = [(i, i) for i in range(num_node_velocity)]
inward_ori_index_velocity = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (6,9), (9,10), (10,11), (11, 12), (12, 13), (13, 14), (14,15)]
inward_velocity = [(i, j) for (i, j) in inward_ori_index_velocity]
outward_velocity = [(j, i) for (i, j) in inward_velocity]
neighbor_velocity = inward_velocity + outward_velocity

#####################################
#UpperBodyNodes  2, 1, 17, 18, 19, 20, 13, 14, 15, 16
#reindex         0, 1, 2,  3,  4,  5,  6,  7,  8,  9
##########################################

num_node_bone = 10
self_link_bone = [(i, i) for i in range(num_node_bone)]
inward_ori_index_bone = [(0,1), (1,2), (2,3), (3,4), (4,5), (1,6), (6,7), (7,8), (8,9)]
inward_bone = [(i, j) for (i, j) in inward_ori_index_bone]
outward_bone = [(j, i) for (i, j) in inward_bone]
neighbor_bone = inward_bone + outward_bone


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A

class Graph_velocity:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node_velocity
        self.self_link = self_link_velocity
        self.inward = inward_velocity
        self.outward = outward_velocity
        self.neighbor = neighbor_velocity
        self.A = self.get_adjacency_matrix(labeling_mode)
    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A

class Graph_bone:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node_bone
        self.self_link = self_link_bone
        self.inward = inward_bone
        self.outward = outward_bone
        self.neighbor = neighbor_bone
        self.A = self.get_adjacency_matrix(labeling_mode)
    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A
