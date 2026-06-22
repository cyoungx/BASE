import numpy as np
from torch.utils.data import Dataset
import h5py
import torch
from feeders import tools
import feeders.augmentations as augmentations

seen_list_ntu120 = {'1': [   0,  37,  52,  70,  96,  92,  91,   4,  39,  12,  46,  81,  87,  31,  72,  48,  16,  62,  42, 102, 112,  68,  56,  49,  22,  11,  88, 107,  93,  43],
                    '2': [  17,  90,  47,  80,  79,  48,  27,  82,  61,  53,  96, 117,  62,  35,  23,  85,   8,  98, 104,  77,  51,  75,  56, 105,  54,  25,  18,  44,  40, 109],
                    '3': [  76,   9,  57,  59,   5,  51,  83, 104,  73,  27,  92,  72,  42, 111, 100,  67, 105,   4, 101,  12,  84, 119,  15,  33,  78,  62,  82,  24,  65, 108],
                    '4': [  48,  12,  26,  63,  20, 109,  80,  33,  79,  67, 100,   6,  24,  11,  76,  61,  10,  59,   0,  99,  19,   4,  90,  58,  28,  88,  44,  95,  72,  18],
                    '5': [  45,   0,  44,  13, 100,  14,  32,  72, 101,  17,  39,  63,  20,  56, 105,  71,  78,  73,   8,  99,  19, 115,  23,  54,  12, 109,  15,  37,  88,  18]}

def get_mapping120(run):
    labels = [j for j in range(120)]
    label_mapping = []
    for i in labels:
        if i in seen_list_ntu120[str(run)]:
            label_mapping.append(i)
    return label_mapping

class Feeder(Dataset):
    def __init__(self, data_path, run=1, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, arg=None):
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
        
        self.label_mapping = get_mapping120(self.run)
        
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
        self.bone = bone
        self.vel = vel
        self.load_data()
        self.upper = [25, 24, 12, 11, 10, 9, 21, 3, 4, 5, 6, 7, 8,  22, 23, 2]
        self.lower = [2, 1, 17, 18, 19, 20, 13, 14, 15, 16]
        self.upper = [item-1 for item in self.upper]
        self.lower = [item-1 for item in self.lower]
        if normalization:
            self.get_mean_map()
    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = []
            self.label = []
            self.data_raw = npz_data['x_train']
            self.label_raw = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
            for ind, (data, label, name) in enumerate(zip(self.data_raw, self.label_raw, self.data_raw)):
                if label in seen_list_ntu120[str(self.run)]:
                    self.data.append(data)
                    label = self.label_mapping.index(label)
                    self.label.append(label)
                    self.sample_name.append(name)    

        elif self.split == 'test_seen' or self.split == 'test_unseen':
            self.data_raw = npz_data['x_test']
            self.data = []
            self.label_raw = np.where(npz_data['y_test'] > 0)[1]
            self.label = []
            self.sample_name_raw = ['test_' + str(i) for i in range(len(self.data_raw))]
            self.sample_name = []
            if self.split == 'test_seen':
                for ind, (data, label, name) in enumerate(zip(self.data_raw, self.label_raw, self.data_raw)):
                    if label in seen_list_ntu120[str(self.run)]:
                        self.data.append(data)
                        label = self.label_mapping.index(label)
                        self.label.append(label)
                        self.sample_name.append(name)
            elif self.split == 'test_unseen':
                for ind, (data, label, name) in enumerate(zip(self.data_raw, self.label_raw, self.data_raw)):
                    if label not in seen_list_ntu120[str(self.run)]:
                        self.data.append(data)
                        label = 40
                        self.label.append(label)
                        self.sample_name.append(name)  
        else:
            raise NotImplementedError('data split only supports train/test')
        self.data = np.stack(self.data, 0)
        self.label = np.stack(self.label, 0)
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)    # (26719, 3, 300, 25, 2)
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
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        # if self.random_rot:
        # data_numpy_v1_crop = augmentations.temporal_cropresize(data_numpy, number_of_frames, self.l_ratio, self.input_size)
        data_numpy_v1 = data_numpy
        # randomly select  one of the spatial augmentations 
        data_numpy_v1 = tools.random_rot(data_numpy_v1)
        import random
        flip_prob  = random.random()
        if flip_prob < 0.5:
            data_numpy_v1 = augmentations.pose_augmentation(data_numpy_v1)
        else:
            data_numpy_v1 = augmentations.joint_courruption(data_numpy_v1)

        data_numpy = tools.random_rot(data_numpy)
        from .bone_pairs import ntu_pairs
        bone_data_numpy = np.zeros_like(data_numpy)
        for v1, v2 in ntu_pairs:
            bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
        vel_data_numpy = np.zeros_like(data_numpy)
        vel_data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
        vel_data_numpy[:, -1] = 0

        
        bone_data_numpy_v1 = np.zeros_like(data_numpy_v1)
        for v1, v2 in ntu_pairs:
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
