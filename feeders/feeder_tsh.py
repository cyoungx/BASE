import numpy as np
import pickle
from torch.utils.data import Dataset
import h5py
import torch
from feeders import tools
import feeders.augmentations as augmentations

unseen_list_tyt = {
    '1':   [11, 29, 12,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 15, 16, 24, 26, 27, 14, 17, 21],
    '2':   [12, 13, 14, 15, 23, 24, 25, 26, 27, 28, 30, 16, 17, 18, 19, 21, 22, 10,  8,  9,  7]
}

def get_mapping(run):
    labels = [j for j in range(31)]
    label_mapping = []
    for i in labels:
        if i not in unseen_list_tyt[str(run)]:
            label_mapping.append(i)
    return label_mapping

class Feeder(Dataset):
    def __init__(self, data_path, run=1, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, test_data_path=None, test_label_path=None, arg=None):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """
        self.arg = arg
        if self.arg.run == None:
            self.run = run
        else:
            self.run = self.arg.run
        
        self.label_mapping = get_mapping(self.run)

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.load_data()
        if normalization:
            self.get_mean_map()
    def load_data(self):

        if self.split == 'train' or self.split == 'pretrain_train':
            # data: N C V T M
            npz_train_data = np.load(self.data_path)
            with open(self.label_path, 'rb') as f:
                pkl_label = pickle.load(f)
            N, C, T, V, M = npz_train_data.shape
        elif self.split == 'test_seen' or self.split == 'test_unseen' or self.split == 'pretrain_test':
            npz_test_data = np.load(self.data_path)
            with open(self.label_path, 'rb') as f:
                pkl_test_label = pickle.load(f)
            N, C, T, V, M = npz_test_data.shape

        if self.split == 'train':
            self.data = []
            self.label = []
            self.data_raw = npz_train_data
            self.label_raw = pkl_label[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
            for ind, (data, label, name) in enumerate(zip(self.data_raw, self.label_raw, self.data_raw)):
                if label not in unseen_list_tyt[str(self.run)]:
                    self.data.append(data)
                    label = self.label_mapping.index(label)
                    self.label.append(label)
                    self.sample_name.append(name)
        elif self.split == 'test_seen' or self.split == 'test_unseen':
            self.data_raw = npz_test_data
            self.data = []
            self.label_raw = pkl_test_label[1]
            self.label = []
            self.sample_name_raw = ['test_' + str(i) for i in range(len(self.data_raw))]
            self.sample_name = []
            if self.split == 'test_seen':
                for ind, (data, label, name) in enumerate(zip(self.data_raw, self.label_raw, self.data_raw)):
                    if label not in unseen_list_tyt[str(self.run)]:
                        self.data.append(data)
                        label = self.label_mapping.index(label)
                        self.label.append(label)
                        self.sample_name.append(name)
            elif self.split == 'test_unseen':
                for ind, (data, label, name) in enumerate(zip(self.data_raw, self.label_raw, self.data_raw)):
                    if label in unseen_list_tyt[str(self.run)]:
                        self.data.append(data)
                        label = 40
                        self.label.append(label)
                        self.sample_name.append(name)
    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
    def __len__(self):
        return len(self.label)
    def __iter__(self):
        return self
    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        target_len = 300
        C, T, V, M = data_numpy.shape
        if T > target_len:
            idx = np.linspace(0, T - 1, target_len).astype(int)
            data_numpy = data_numpy[:, idx, :, :]
        elif T < target_len:
            data_numpy = tools.auto_pading(data_numpy, target_len)
        
        target_len = 300
        C, T, V, M = data_numpy.shape
        if T > target_len:
            idx = np.linspace(0, T - 1, target_len).astype(int)
            data_numpy = data_numpy[:, idx, :, :]
        elif T < target_len:
            data_numpy = tools.auto_pading(data_numpy, target_len)

        data_numpy_v1 = data_numpy

        if self.normalization:
            data_numpy_v1 = (data_numpy_v1 - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy_v1 = tools.random_shift(data_numpy_v1)
        if self.random_choose:
            data_numpy_v1 = tools.random_choose(data_numpy_v1, self.window_size)
        elif self.window_size > 0:
            data_numpy_v1 = tools.auto_pading(data_numpy_v1, self.window_size)
        if self.random_move:
            data_numpy_v1 = tools.random_move(data_numpy_v1)

        from .bone_pairs import tsh_pairs
        bone_data_numpy = np.zeros_like(data_numpy)
        for v1, v2 in tsh_pairs:
            bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
        vel_data_numpy = np.zeros_like(data_numpy)
        vel_data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
        vel_data_numpy[:, -1] = 0
        
        bone_data_numpy_v1 = np.zeros_like(data_numpy_v1)
        for v1, v2 in tsh_pairs:
            bone_data_numpy_v1[:, :, v1 - 1] = data_numpy_v1[:, :, v1 - 1] - data_numpy_v1[:, :, v2 - 1]
        vel_data_numpy_v1 = np.zeros_like(data_numpy_v1)
        vel_data_numpy_v1[:, :-1] = data_numpy_v1[:, 1:] - data_numpy_v1[:, :-1]
        vel_data_numpy_v1[:, -1] = 0

        if self.split == 'train':
            return [data_numpy, bone_data_numpy, vel_data_numpy], label, index
        else:
            return [data_numpy, bone_data_numpy, vel_data_numpy], label, index
        
    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
