import sys

sys.path.extend(['../'])
from graph import tools

num_node = 15
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [ (14,1), (15, 14), (9, 15), (8, 15), (10, 8), (12, 10), (11, 9), (13, 11), (3, 14), (5, 3), (7, 5), (2, 14),
        (4, 2), (6, 4)]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


tsh_pairs = (
    (14,1), (15, 14), (9, 15), (8, 15), (10, 8), (12, 10), 
    (11, 9), (13, 11), (3, 14), (5, 3), (7, 5), (2, 14),
    (4, 2), (6, 4)
)


#####################################
#UpperBodyNodes  7, 5, 3, 14, 1, 2, 4, 6
#reindex         0, 1, 2,  3, 4, 5, 6, 7
##########################################

num_node_velocity = 8
self_link_velocity = [(i, i) for i in range(num_node_velocity)]
inward_ori_index_velocity = [(0,1), (1,2), (2,3), (3,4), (3, 5), (5, 6), (6,7)]
inward_velocity = [(i, j) for (i, j) in inward_ori_index_velocity]
outward_velocity = [(j, i) for (i, j) in inward_velocity]
neighbor_velocity = inward_velocity + outward_velocity



class Graph_velocity:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node_velocity
        self.self_link = self_link_velocity
        self.inward = inward_velocity
        self.outward = outward_velocity
        self.neighbor = neighbor_velocity

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


#####################################
#UpperBodyNodes  15, 9, 11, 13, 8, 10, 12
#reindex         0, 1, 2,  3,  4,  5,  6,
##########################################

num_node_bone = 7
self_link_bone = [(i, i) for i in range(num_node_bone)]
inward_ori_index_bone = [(0,1), (1,2), (2,3), (0,4), (4,5), (5,6)]
inward_bone = [(i, j) for (i, j) in inward_ori_index_bone]
outward_bone = [(j, i) for (i, j) in inward_bone]
neighbor_bone = inward_bone + outward_bone



class Graph_bone:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node_bone
        self.self_link = self_link_bone
        self.inward = inward_bone
        self.outward = outward_bone
        self.neighbor = neighbor_bone

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
