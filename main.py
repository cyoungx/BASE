#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
import csv
import numpy as np
import glob
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, roc_curve, confusion_matrix
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import logging
import pprint
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlights.torchlight import DictAction

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

from sklearn.neighbors import NearestNeighbors

def init_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

init_seed(1)

from module import EDL_loss, Text_Projection, CrossModalInteraction

def create_dirs(args):

    def create_dir(path, dir_name):
        if not os.path.exists(path):
            os.makedirs(path)

    dir_path = args.file_path
    dir_path_name = 'ANA'
    create_dir(dir_path, dir_path_name)

    experiments_id_path = dir_path + '/' + args.experiment_id + '_' + args.timestamp
    experiments_id_path_name = 'ID'
    create_dir(experiments_id_path, experiments_id_path_name)

    experiments_logs_path = experiments_id_path + '/' + 'logs'
    experiments_logs_path_name = 'log'
    create_dir(experiments_logs_path, experiments_logs_path_name)

    experiments_npz_file_path = experiments_id_path + '/' + 'npz_files'
    experiments_npz_file_path_name = 'npz'
    create_dir(experiments_npz_file_path, experiments_npz_file_path_name)

    experiments_models_file_path = experiments_id_path + '/' + 'models'
    experiments_models_file_path_name = 'models'
    create_dir(experiments_models_file_path, experiments_models_file_path_name)

    experiments_pics_path = experiments_id_path + '/' + 'pics'
    experiments_pics_path_name = 'vis'
    create_dir(experiments_pics_path, experiments_pics_path_name)

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.bone() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.bone() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='Spatial Temporal Graph Convolution Network')
    parser.add_argument('--work-dir', default='./work_dir/temp', help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('--config', default='./config/nturgbd-cross-view/test_bone.yaml', help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score', type=str2bool, default=False, help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save_epoch', type=int, default=30, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=1, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=64, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', action=DictAction, default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-args-unseen', action=DictAction, default=dict(), help='the arguments of data loader for test')
    parser.add_argument('--test-feeder-args-seen', action=DictAction, default=dict(), help='the arguments of data loader for test')
    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args', action=DictAction, default=dict(), help='the arguments of model')
    parser.add_argument('--model-args-velocity', action=DictAction, default=dict(), help='the arguments of model')
    parser.add_argument('--model-args-bone', action=DictAction, default=dict(), help='the arguments of model')
    
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--weights_velocity', default=None, help='the weights for network initialization')
    parser.add_argument('--weights_bone', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--base-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--step', type=int, default=None, nargs='+', help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=64, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=64, help='test batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num_epoch', type=int, default=60, help='stop training in which epoch')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--run', type=int, default=None)

    # text
    parser.add_argument('--flag_text_proj', action='store_true', default=False)
    parser.add_argument('--text_gpt4_path', type=str, default='data/language/ntu120_des_gpt4_embeddings.npy')
    parser.add_argument('--text_class_atom_path', type=str, default='data/language/ntu120_des_class_atom_embeddings.npy')

    # loss
    parser.add_argument('--flag_loss_edl', action='store_true', default=False)
    parser.add_argument('--flag_loss_aes', action='store_true', default=False)

    parser.add_argument('--flag_semlp', action='store_true', default=False)
    parser.add_argument('--flag_module_cmi', action='store_true', default=False)

    parser.add_argument('--flag_cmi_lhsg', action='store_true', default=False)
    parser.add_argument('--num_proto', type=int, default=3)
    parser.add_argument('--flag_loss_enh', action='store_true', default=False)
    
    parser.add_argument('--flag_cmi_bpom', action='store_true', default=False)
    parser.add_argument('--flag_loss_bcl', action='store_true', default=False)
    parser.add_argument('--temp_id', type=float, default=0.05)
    parser.add_argument('--temp_eid', type=float, default=0.07)
    parser.add_argument('--temp_pood', type=float, default=0.10)
    
    parser.add_argument('--dim', type=int, default=None, help='GCN:256, Transformer:216')
    parser.add_argument('--lambda_a', type=float, default=0.1)
    parser.add_argument('--kn', type=int, default=3)
    parser.add_argument('--gamma_base', type=float, default=0.5)
    parser.add_argument('--eta_base', type=float, default=0.4)
    
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--eval_interval',type=int, default=10)

    parser.add_argument('--file_path', type=str, help='root path')
    parser.add_argument('--experiment_id', type=str)
    parser.add_argument('--timestamp', type=str)
    return parser

class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0

        self.load_data()
        self.load_model()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.model = self.model.cuda(self.output_device)
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=self.output_device)
        
        if self.arg.flag_loss_edl:
            self.loss_edl = EDL_loss(arg=arg).cuda(self.output_device)

        self.text_gpt4_embedding = torch.Tensor(np.load(self.arg.text_gpt4_path)).cuda()

        self.load_optimizer()

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.train_feeder_args, arg=self.arg),
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker,
            drop_last=True,
            worker_init_fn=init_seed,
            pin_memory=True)
        self.data_loader['test_seen'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args_seen, arg=self.arg),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed,
            pin_memory=True)
        self.data_loader['test_unseen'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args_unseen, arg=self.arg),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed,
            pin_memory=True)
        
        self.label_mapping_list = Feeder(**self.arg.train_feeder_args, arg=self.arg).label_mapping

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        
        logging.info('{}'.format(Model))
        self.model = Model(**self.arg.model_args, arg=self.arg).cuda()
        self.model_velocitybody = Model(**self.arg.model_args, arg=self.arg).cuda()
        self.model_bonebody = Model(**self.arg.model_args, arg=self.arg).cuda()
        
        logging.info('\n{}'.format(self.model))

        if self.arg.weights:
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            logging.info('Load weights from {}.'.format(self.arg.weights))
            weights = torch.load(self.arg.weights)
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)
            # ---------------------------------------------------------------
            weights_velocity = torch.load(self.arg.weights_velocity)
            weights_velocity = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights_velocity.items()])
            keys = list(weights_velocity.keys())
            self.model_velocitybody.load_state_dict(weights_velocity)
            # ---------------------------------------------------------------
            weights_bone = torch.load(self.arg.weights_bone)
            weights_bone = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights_bone.items()])
            self.model_bonebody.load_state_dict(weights_bone)
            # ---------------------------------------------------------------
        
        self.model = torch.nn.DataParallel(self.model)
        self.model_velocitybody = torch.nn.DataParallel(self.model_velocitybody)
        self.model_bonebody = torch.nn.DataParallel(self.model_bonebody)

        if self.arg.flag_text_proj:
            self.text_model = Text_Projection(arg=self.arg).cuda()
            self.text_model = torch.nn.DataParallel(self.text_model)
        else:
            self.text_model = None

        
        if self.arg.flag_module_cmi:
            self.module_cmi = CrossModalInteraction(arg=self.arg, text_model=self.text_model, label_mapping=self.label_mapping_list).cuda()
            self.module_cmi = torch.nn.DataParallel(self.module_cmi)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            params = ([*self.model.parameters()] +
                      [*self.model_bonebody.parameters()] +
                      [*self.model_velocitybody.parameters()])
            
            if self.arg.flag_text_proj:
                params += [*self.text_model.parameters()]
            
            if self.arg.flag_module_cmi:
                params += [*self.module_cmi.parameters()]
            
            self.optimizer = optim.SGD(
                params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError("Select Optimizer!")

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))
        logging.info('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        logging.info('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        for batch_idx, (datalist, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = datalist[0].float().cuda(self.output_device)
                data_bone = datalist[1].float().cuda(self.output_device)
                data_velocity = datalist[2].float().cuda(self.output_device)
                label = label.long().cuda(self.output_device) 
            
            self.optimizer.zero_grad()
            bs = data.shape[0]
            timer['dataloader'] += self.split_time()

            # forward
            with torch.cuda.amp.autocast():
                output_all, feature = self.model(data)
                output_velocity_all, feature_velocity = self.model_velocitybody(data_velocity)
                output_bone_all, feature_bone = self.model_bonebody(data_bone)

                # text features
                if self.arg.flag_text_proj:
                    text_gpt4_feature = self.text_model(self.text_gpt4_embedding)
                    idx = torch.tensor(self.label_mapping_list, dtype=torch.long)
                    new_text_gpt4_feature = text_gpt4_feature[idx]

                if self.arg.flag_loss_edl:
                    output_list = [output_all, output_bone_all, output_velocity_all]
                    edl_info_dict = self.loss_edl(output_list, label)
                    loss_edl = edl_info_dict['loss_edl']
                    loss_aes = edl_info_dict['loss_aes']
                    edl_info_dict_list = edl_info_dict['edl_info_dict_list']
                    edl_info_dict_j, edl_info_dict_b, edl_info_dict_v = edl_info_dict_list[0], edl_info_dict_list[1], edl_info_dict_list[2]
                else:
                    loss_edl = torch.tensor(0).cuda()
                    loss_aes = torch.tensor(0).cuda()
                    edl_info_dict_list = {}

                if self.arg.flag_module_cmi:
                    processed_text_dict = {'text_gpt4_feature':None}
                    processed_text_gpt4_feature = new_text_gpt4_feature.unsqueeze(0)
                    processed_text_dict['text_gpt4_feature'] = processed_text_gpt4_feature
                    
                    # only for multi GPUs，single GPU removes `module`
                    layer_list = [self.model.module.fc, self.model_bonebody.module.fc, self.model_velocitybody.module.fc]

                    ske_logits_list = [output_all, output_bone_all, output_velocity_all]
                    ske_feat_list = [feature, feature_bone, feature_velocity]
                    ske_edl_info_dict_list = [edl_info_dict_j, edl_info_dict_b, edl_info_dict_v]
                    cmi_info_dict = self.module_cmi(ske_logits_list, ske_feat_list, label, ske_edl_info_dict_list, processed_text_dict, layer_list)

                    loss_enh = cmi_info_dict['loss_enh']
                    loss_bcl = cmi_info_dict['loss_bcl']

                else:
                    loss_enh = torch.tensor(0).cuda()
                    loss_bcl = torch.tensor(0).cuda()

                loss = loss_edl + 0.1 * loss_aes + 0.1 * loss_enh + 0.1 * loss_bcl

            # backward
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output_all[:bs].data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()
            #break
        
        # statistics of time consumption and loss
        proportion = {k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values())))) for k, v in timer.items()}
        self.print_log('\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
        logging.info('Mean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))
        logging.info('Time consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model and (epoch + 1) % self.arg.eval_interval == 0:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            state_dict_velocity = self.model_velocitybody.state_dict()
            weights_velocity = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict_velocity.items()])
            state_dict_bone = self.model_bonebody.state_dict()
            weights_bone = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict_bone.items()])
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')
            torch.save(weights_velocity, self.arg.model_saved_name + '-velocity' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')
            torch.save(weights_bone, self.arg.model_saved_name + '-bone' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')
    
    def eval_osr(self, y_true, y_pred):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        auroc = roc_auc_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        aupr = auc(recall, precision)
        fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
        operation_idx = np.abs(tpr - 0.95).argmin()
        fpr95 = fpr[operation_idx]
        return auroc, aupr, fpr95
    def eval_uosr(self, y_true, y_pred):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        auroc = roc_auc_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        aupr = auc(recall, precision)
        fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
        operation_idx = np.abs(tpr - 0.95).argmin()
        fpr95 = fpr[operation_idx]
        return auroc, aupr, fpr95
    def eval(self, epoch, save_score=False, loader_name=['test_seen', 'test_unseen'], wrong_file=None, result_file=None, num_class=None):
        num_class = self.arg.num_class
        threshold_v = 0.9
        threshold_m = 0.9
        total = torch.zeros(num_class+1)
        correct_mean_seen = torch.zeros(num_class+1)
        correct_var_seen = torch.zeros(num_class+1)
        correct_mean_unseen = torch.zeros(num_class+1)
        correct_var_unseen = torch.zeros(num_class+1)
        all_prob_seen = []
        all_preds_seen = []
        all_labels_seen = []
        all_prob_unseen = []
        all_labels_unseen = []
        all_preds_unseen = []
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        logging.info('Eval epoch: {}'.format(epoch + 1))
        step = 0
        
        process = tqdm(self.data_loader['test_seen'], ncols=40)
        for batch_idx, (datalist, label,index) in enumerate(process):
            with torch.no_grad():
                data = datalist[0].float().cuda(self.output_device)
                data_bone = datalist[1].float().cuda(self.output_device)
                data_velocity = datalist[2].float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                output, feature = self.model(data)
                output_bone, feature_bone = self.model_bonebody(data_bone)
                output_velocity, feature_velocity = self.model_velocitybody(data_velocity)

                output = (output + output_bone + output_velocity)/3
                
                output = torch.nn.functional.softmax(output, dim = -1)
                probab, predicted = torch.max(output, 1)
                all_preds_seen.append(predicted)
                all_prob_seen.append(probab)
                all_labels_seen.append(label)
                for k in range(len(predicted)):
                    total[label[k]] += 1
                    if predicted[k] == label[k]:
                        correct_mean_seen[predicted[k]] += 1
                    if predicted[k] == label[k]:
                        correct_var_seen[predicted[k]] += 1
        all_prob_seen = torch.cat(all_prob_seen, 0)
        all_labels_seen = torch.cat(all_labels_seen,0)
        all_preds_seen = torch.cat(all_preds_seen)

        
        process = tqdm(self.data_loader['test_unseen'], ncols=40)
        for batch_idx, (datalist, label,index) in enumerate(process):
            with torch.no_grad():
                
                data = datalist[0].float().cuda(self.output_device)
                data_bone = datalist[1].float().cuda(self.output_device)
                data_velocity = datalist[2].float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                output, feature = self.model(data)
                output_bone, feature_bone = self.model_bonebody(data_bone)
                output_velocity, feature_velocity = self.model_velocitybody(data_velocity)
                
                output = (output + output_bone + output_velocity)/3
                
                output = torch.nn.functional.softmax(output, dim=-1)
                probab, predicted = torch.max(output, 1)
                all_preds_unseen.append(predicted)
                all_prob_unseen.append(probab)
                all_labels_unseen.append(label)
                for k in range(len(predicted)):
                    if predicted[k] == label[k]:
                        correct_mean_unseen[predicted[k]] += 1
        all_prob_unseen = torch.cat(all_prob_unseen, 0)
        all_labels_unseen = torch.cat(all_labels_unseen, 0)
        all_preds_unseen = torch.cat(all_preds_unseen, 0)              
        N = all_labels_seen.shape[0]
        correct_mean = correct_mean_seen
        mixed_acc = torch.sum(correct_mean)/N * 100
        ###############################calculate OS auc############################
        all_prob = torch.cat([all_prob_seen, all_prob_unseen])
        all_prob = 1 - all_prob
        binary_label_uncertainty = torch.cat([torch.zeros(all_labels_seen.shape[0]), torch.ones(all_labels_unseen.shape[0])], 0)
        auroc, aupr, fpr95 = self.eval_osr(y_true=binary_label_uncertainty, y_pred=all_prob)
        ###############################calculate UOS auc###########################
        N = all_labels_seen.shape[0]
        topK = N - int(N*0.85)
        uncertainty_seen = 1- all_prob_seen
        threshold = torch.sort(uncertainty_seen, 0)[0][N-topK+1]
        inc_labels = torch.zeros(all_preds_seen[uncertainty_seen<=threshold].shape[0])
        inw_labels = torch.ones(all_preds_seen[uncertainty_seen>threshold].shape[0])
        labels_seen = torch.cat([inc_labels, inw_labels], 0)
        preds_seen = torch.cat([uncertainty_seen[uncertainty_seen<=threshold], uncertainty_seen[uncertainty_seen>threshold]], 0)
        preds = torch.cat([preds_seen, 1-all_prob_unseen], 0)
        labels_uosr = torch.cat([labels_seen.cuda(),torch.ones(all_labels_unseen.shape[0]).cuda()], 0)
        auroc_uosr, aupr_uosr, fpr95_uosr = self.eval_uosr(y_true=labels_uosr, y_pred=preds)
        print('####Epoch: ', epoch+1, ' -----ACC: ', mixed_acc, ' ------osauc:', auroc, ' ------aupr:', aupr)
        logging.info('####Epoch: {}  -----ACC: {}  ------osauc: {}  ------aupr: {}'.format(epoch+1, mixed_acc, auroc, aupr))
        if (epoch+1) % 10 == 0:
            torch.save(self.model.state_dict(), self.arg.work_dir + '/checkpoints_epoch_{}.pt'.format(epoch+1))
        with open('{}/each_epoch_resuts.csv'.format(self.arg.work_dir), 'w') as f:
            writer = csv.writer(f)
            writer.writerow('Epoch_{}_MixedAcc_{}_OSAUC_{}_UOSAUC_{}'.format(epoch, mixed_acc, auroc, auroc_uosr))

    def eval_inference(self, epoch, save_score=False, loader_name=['test_seen', 'test_unseen'], wrong_file=None, result_file=None):
        from numpy import linalg as LA

        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        logging.info('Eval epoch: {}'.format(epoch + 1))
        step = 0

        joints_train_rep = []
        joints_train_out = []

        train_label = []

        bones_train_rep = []
        bones_train_out = []
        vels_train_rep = []
        vels_train_out = []
        
        joints_test_seen_rep = []
        joints_test_seen_out = []
        bones_test_seen_rep = []
        bones_test_seen_out = []
        vels_test_seen_rep = []
        vels_test_seen_out = []
        test_seen_label = []

        joints_test_unseen_rep = []
        joints_test_unseen_out = []
        bones_test_unseen_rep = []
        bones_test_unseen_out = []
        vels_test_unseen_rep = []
        vels_test_unseen_out = []
        test_unseen_label = []

        process = tqdm(self.data_loader['train'], ncols=40)
        for batch_idx, (datalist, label,index) in enumerate(process):
            with torch.no_grad():
                data = datalist[0].float().cuda(self.output_device)
                data_bone = datalist[1].float().cuda(self.output_device)
                data_velocity = datalist[2].float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                output, rep1 = self.model(data)
                output_bone, rep2 = self.model_bonebody(data_bone)
                output_velocity, rep3 = self.model_velocitybody(data_velocity)
                        
                for i in range(rep1.shape[0]):
                    joints_train_rep.append(rep1[i].cpu().numpy()/LA.norm(rep1[i].cpu().numpy()))
                    joints_train_out.append(output[i].cpu().numpy())
                    train_label.append(label[i].cpu().numpy())
                for i in range(rep2.shape[0]):
                    bones_train_rep.append(rep2[i].cpu().numpy()/LA.norm(rep2[i].cpu().numpy()))
                    bones_train_out.append(output_bone[i].cpu().numpy())
                for i in range(rep3.shape[0]):
                    vels_train_rep.append(rep3[i].cpu().numpy()/LA.norm(rep3[i].cpu().numpy()))
                    vels_train_out.append(output_velocity[i].cpu().numpy())
        
        process = tqdm(self.data_loader['test_seen'], ncols=40)
        for batch_idx, (datalist, label,index) in enumerate(process):
            with torch.no_grad():
                data = datalist[0].float().cuda(self.output_device)
                data_bone = datalist[1].float().cuda(self.output_device)
                data_velocity = datalist[2].float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                output, rep1 = self.model(data)
                output_bone, rep2 = self.model_bonebody(data_bone)
                output_velocity, rep3 = self.model_velocitybody(data_velocity)

                for i in range(rep1.shape[0]):
                    joints_test_seen_rep.append(rep1[i].cpu().numpy()/LA.norm(rep1[i].cpu().numpy()))
                    joints_test_seen_out.append(output[i].cpu().numpy())
                    test_seen_label.append(label[i].cpu().numpy())
                for i in range(rep2.shape[0]):
                    bones_test_seen_rep.append(rep2[i].cpu().numpy()/LA.norm(rep2[i].cpu().numpy()))
                    bones_test_seen_out.append(output_bone[i].cpu().numpy())
                for i in range(rep3.shape[0]):
                    vels_test_seen_rep.append(rep3[i].cpu().numpy()/LA.norm(rep3[i].cpu().numpy()))
                    vels_test_seen_out.append(output_velocity[i].cpu().numpy())
        
        process = tqdm(self.data_loader['test_unseen'], ncols=40)
        for batch_idx, (datalist, label,index) in enumerate(process):
            with torch.no_grad():
                data = datalist[0].float().cuda(self.output_device)
                data_bone = datalist[1].float().cuda(self.output_device)
                data_velocity = datalist[2].float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                output, rep1 = self.model(data)
                output_bone, rep2 = self.model_bonebody(data_bone)
                output_velocity, rep3 = self.model_velocitybody(data_velocity)

                for i in range(rep1.shape[0]):
                    joints_test_unseen_rep.append(rep1[i].cpu().numpy()/LA.norm(rep1[i].cpu().numpy()))
                    joints_test_unseen_out.append(output[i].cpu().numpy())
                    test_unseen_label.append(label[i].cpu().numpy())
                for i in range(rep2.shape[0]):
                    bones_test_unseen_rep.append(rep2[i].cpu().numpy()/LA.norm(rep2[i].cpu().numpy()))
                    bones_test_unseen_out.append(output_bone[i].cpu().numpy())
                for i in range(rep3.shape[0]):
                    vels_test_unseen_rep.append(rep3[i].cpu().numpy()/LA.norm(rep3[i].cpu().numpy()))
                    vels_test_unseen_out.append(output_velocity[i].cpu().numpy())
        
        dicty = {'joints_train_rep':joints_train_rep, 
                'joints_train_out':joints_train_out, 
                'bones_train_rep':bones_train_rep,
                'bones_train_out':bones_train_out, 
                'vels_train_rep':vels_train_rep, 
                'vels_train_out':vels_train_out, 
                'train_label':train_label, 

                'joints_test_seen_rep':joints_test_seen_rep,
                'joints_test_seen_out':joints_test_seen_out, 
                'bones_test_seen_rep':bones_test_seen_rep, 
                'bones_test_seen_out':bones_test_seen_out,
                'vels_test_seen_rep':vels_test_seen_rep, 
                'vels_test_seen_out':vels_test_seen_out, 
                'test_seen_label':test_seen_label, 
                
                'joints_test_unseen_rep':joints_test_unseen_rep, 
                'joints_test_unseen_out':joints_test_unseen_out, 
                'bones_test_unseen_rep':bones_test_unseen_rep,
                'bones_test_unseen_out':bones_test_unseen_out, 
                'vels_test_unseen_rep':vels_test_unseen_rep, 
                'vels_test_unseen_out':vels_test_unseen_out, 
                'test_unseen_label':test_unseen_label,
        }

        import pickle as pkl
        pkl_path = self.arg.file_path + '/' + self.arg.experiment_id + '_' + self.arg.timestamp + '/' + 'feature_test.pkl'
        f = open(pkl_path, 'wb')
        pkl.dump(obj=dicty, file=f)
        f.close()
        
        joints_train_rep = np.stack(joints_train_rep)
        joints_train_out = np.stack(joints_train_out)
        bones_train_rep = np.stack(bones_train_rep)
        bones_train_out = np.stack(bones_train_out)
        vels_train_rep =np.stack(vels_train_rep)
        vels_train_out = np.stack(vels_train_out)
        train_label = np.stack(train_label)

        joints_test_seen_rep = np.stack(joints_test_seen_rep)
        joints_test_seen_out = np.stack(joints_test_seen_out)
        bones_test_seen_rep = np.stack(bones_test_seen_rep)
        bones_test_seen_out = np.stack(bones_test_seen_out)
        vels_test_seen_rep = np.stack(vels_test_seen_rep)
        vels_test_seen_out = np.stack(vels_test_seen_out)
        test_seen_label = np.stack(test_seen_label)

        joints_test_unseen_rep = np.stack(joints_test_unseen_rep)
        joints_test_unseen_out = np.stack(joints_test_unseen_out)
        bones_test_unseen_rep = np.stack(bones_test_unseen_rep)
        bones_test_unseen_out = np.stack(bones_test_unseen_out)
        vels_test_unseen_rep = np.stack(vels_test_unseen_rep)
        vels_test_unseen_out = np.stack(vels_test_unseen_out)
        test_unseen_label = np.stack(test_unseen_label)

        ###########################################################
        # calculate samples
        ###########################################################
        print('KNN joints')
        logging.info('KNN joints')
        neigh_joints =  NearestNeighbors(n_neighbors=3).fit(joints_train_rep)
        print('KNN bones')
        logging.info('KNN bones')
        neigh_bones =  NearestNeighbors(n_neighbors=3).fit(bones_train_rep)
        print('KNN vels')
        logging.info('KNN vels')
        neigh_vels =  NearestNeighbors(n_neighbors=3).fit(vels_train_rep)

        dist_joints = neigh_joints.kneighbors(np.concatenate([joints_test_seen_rep, joints_test_unseen_rep],0))[0]
        pred_ind = neigh_joints.kneighbors(np.concatenate([joints_test_seen_rep, joints_test_unseen_rep],0))[1][:,0]
        dist_bones = neigh_bones.kneighbors(np.concatenate([bones_test_seen_rep, bones_test_unseen_rep],0))[0]
        pred_ind_bones = neigh_bones.kneighbors(np.concatenate([bones_test_seen_rep, bones_test_unseen_rep],0))[1][:,0]
        dist_vels = neigh_vels.kneighbors(np.concatenate([vels_test_seen_rep, vels_test_unseen_rep],0))[0]
        pred_ind_vels = neigh_vels.kneighbors(np.concatenate([vels_test_seen_rep, vels_test_unseen_rep],0))[1][:,0]
        dist_concat = np.stack([dist_joints[:,0], dist_bones[:,0], dist_vels[:,0]], 0)
        pred_concat = np.stack([pred_ind, pred_ind_bones, pred_ind_vels], 0)

        index = np.argmin(dist_concat, axis=0)
        pred_indl = []
        
        for ind in range(pred_concat.shape[1]):
            pred_indl.append(pred_concat[index[ind], ind])
        pred_indl = np.stack(pred_indl)
        pred_labels = train_label[pred_indl[:test_seen_label.shape[0]]]
        acc = 0.0
        for i, item in enumerate(pred_labels):
            if item == test_seen_label[i]:
                acc += 1

        probab_joints = np.max(dist_joints, -1)[np.newaxis, :][0]
        probab_bones = np.max(dist_bones, -1)[np.newaxis, :][0]
        probab_vels = np.max(dist_vels, -1)[np.newaxis, :][0]

        ###############################eval before partition of the inw and ino #####################################################
        probab_labels = np.concatenate([np.zeros(joints_test_seen_rep.shape[0]), np.ones(joints_test_unseen_rep.shape[0])])
        probab_labels = np.concatenate([np.zeros(bones_test_seen_rep.shape[0]), np.ones(bones_test_unseen_rep.shape[0])])

        ###############################eval after partition of the inw and ino #####################################################
        all_dist = (probab_bones + probab_joints + probab_vels)/3
        all_dist = np.clip(all_dist, 1e-6, 1 - 1e-6)

        joints_pred = torch.Tensor(np.concatenate([joints_test_seen_out, joints_test_unseen_out]))
        bones_pred = torch.Tensor(np.concatenate([bones_test_seen_out, bones_test_unseen_out]))
        vels_pred = torch.Tensor(np.concatenate([vels_test_seen_out, vels_test_unseen_out]))
        all_prob = (joints_pred + bones_pred + vels_pred)/3
        
        l_prob = torch.max(torch.nn.functional.softmax(all_prob, -1), dim=-1)[0].numpy()
        l_pred = torch.max(torch.nn.functional.softmax(all_prob, -1), dim=-1)[1].numpy()

        l_prob_joints_recal = []
        l_prob_bones_recal = []
        l_prob_vels_recal = []

        for ind in range(joints_pred.shape[0]):
            item_joints = joints_pred[ind]
            item_bones = bones_pred[ind]
            item_vels = vels_pred[ind]
            pos = l_pred[ind]
            dist = all_dist[ind]
            mask = torch.ones(self.arg.num_class)
            mask[pos] = 0
            mask = mask.bool()

            j_upper = torch.sum(torch.exp(item_joints[mask]* dist**2 )) * (1-dist)
            j_unter = dist
            item_joints[pos] = torch.log(j_upper / j_unter)
            
            item_joints[mask] = item_joints[mask] * dist**2
            
            l_prob_joints_recal.append(item_joints)
            b_upper = torch.sum(torch.exp(item_bones[mask]* dist**2))* (1-dist)
            b_unter = dist
            item_bones[pos] = torch.log(b_upper / b_unter)
            item_bones[mask] = item_bones[mask] * dist**2
            l_prob_bones_recal.append(item_bones)
            v_upper = torch.sum(torch.exp(item_vels[mask]* dist**2))* (1-dist)
            v_unter = dist
            item_vels[pos] = torch.log(v_upper / v_unter)
            item_vels[mask] = item_vels[mask] * dist**2
            l_prob_vels_recal.append(item_vels)

        l_prob_joints_recal = torch.stack(l_prob_joints_recal)
        l_prob_bones_recal = torch.stack(l_prob_bones_recal)
        l_prob_vels_recal = torch.stack(l_prob_vels_recal)
        prob_j, pred_j = torch.max(torch.nn.functional.softmax(l_prob_joints_recal, dim=-1), dim=-1)
        prob_b, pred_b = torch.max(torch.nn.functional.softmax(l_prob_bones_recal, dim=-1), dim=-1)
        prob_v, pred_v = torch.max(torch.nn.functional.softmax(l_prob_vels_recal, dim=-1), dim=-1)
        l_prob = (l_prob_bones_recal + l_prob_vels_recal + l_prob_joints_recal)/3
        prob, pred = torch.max(torch.nn.functional.softmax(l_prob, dim=-1), dim=-1)

        def eval_osr(y_true, y_pred):
            auroc = roc_auc_score(y_true, y_pred)
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            aupr = auc(recall, precision)
            fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
            operation_idx = np.abs(tpr - 0.95).argmin()
            fpr95 = fpr[operation_idx]
            return auroc, aupr, fpr95

        auroc, aupr, fpr95 = eval_osr(probab_labels, 1-prob)
        
        acc = 0.0
        for i, item in enumerate(pred[:test_seen_label.shape[0]]):
            if item == test_seen_label[i]:
                acc += 1
        c_acc = acc/pred_labels.shape[0]

        print('O-AUROC: {}'.format(auroc))
        print('O-AUPR:  {}'.format(aupr))
        print('C-ACC:   {}'.format(c_acc))
        print('FPR@95:  {}'.format(fpr95))
        
        logging.info('O-AUROC: {}'.format(auroc))
        logging.info('O-AUPR:  {}'.format(aupr))
        logging.info('C-ACC:   {}'.format(c_acc))
        logging.info('FPR@95:  {}'.format(fpr95))
        logging.info('\n')
        logging.info('auroc,aupr,c_acc,fpr95')
        logging.info('{},{},{},{}'.format(auroc, aupr, c_acc, fpr95))

    def start(self):
        if self.arg.phase == 'train':
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)
                if ((epoch+1) % self.arg.eval_interval == 0) and (epoch>self.arg.eval_epoch):
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['test_unseen', 'test_seen'])
        
            self.print_log('Training Done.\n')
            logging.info('Training Done.\n')
        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            logging.info('Model:   {}.'.format(self.arg.model))
            logging.info('Weights: {}.'.format(self.arg.weights))

            self.eval_inference(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Test Done.\n')
            logging.info('Test Done.\n')
        else:
            pass

scaler = torch.cuda.amp.GradScaler(init_scale=65536.0)
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    parser = get_parser()
    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    
    create_dirs(arg)
    experiments_id_path = arg.file_path + '/' + arg.experiment_id + '_' + arg.timestamp

    if arg.phase == 'train':
        log_filename = experiments_id_path + '/' + 'logs' + '/' + arg.experiment_id + '_' + arg.timestamp + '_' + 'training_log.log'
    elif arg.phase == 'test':
        log_filename = experiments_id_path + '/' + 'logs' + '/' + arg.experiment_id + '_' + arg.timestamp + '_' + 'test_log.log'
    else:
        raise ValueError("Select train or test!")
    
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(pprint.pformat(arg))

    processor = Processor(arg)
    processor.start()