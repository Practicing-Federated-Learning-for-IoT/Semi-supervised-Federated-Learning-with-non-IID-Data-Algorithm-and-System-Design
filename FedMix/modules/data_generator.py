import os
import pdb
import cv2
import time
import random
import argparse
import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0,'..')
from utils.misc import *
from third_party.mixture_loader.mixture import *

class DataGenerator:

    def __init__(self, opt):
        self.opt = opt
        self.base_dir = os.path.join(self.opt.task_path, self.opt.task) 
        self.did_to_dname = {
            0: 'cifar10',
            1: 'cifar100',
            2: 'mnist',
            3: 'svhn',
            4: 'fashion_mnist',
            5: 'traffic_sign',
            6: 'face_scrub',
            7: 'not_mnist',
        }
        self.load_third_party_data()

    def load_third_party_data(self):
        processed = os.path.join(self.opt.task_path, self.opt.mixture_fname)
        if os.path.exists(processed):
            print('loading mixture data: {}'.format(processed))
            self.mixture = np.load(processed, allow_pickle=True)
        else:
            print('downloading & processing mixture data')
            self.mixture = get(base_dir=self.opt.task_path, fixed_order=True)
            np.save(processed, self.mixture)
        return 

    def get_dataset(self, dataset_id):
        print('load {} from third party ...'.format(self.did_to_dname[dataset_id]))
        self.dataset_id = dataset_id
        data = self.mixture[0][dataset_id]
        # concat train & test from third party data
        x_train = data['train']['x']
        y_train = data['train']['y']
        x_test = data['test']['x']
        y_test = data['test']['y']
        x = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test])
        # shuffle dataset

        # x, y = self.shuffle(x, y)

        #print('{}: {}'.format(self.did_to_dname[self.dataset_id], np.shape(x)))

        return x, y
    
    def generate_data(self):
        print('generating {} ...'.format(self.opt.task))
        start_time = time.time()
        self.task_cnt = -1 #
        self.is_labels_at_server = True if 'labels-at-server' in self.opt.scenario else False#根据输入信息查看label的位置
        self.is_labels_at_all = True if 'labels-at-all' in self.opt.scenario else False
        self.is_imbalanced = True if 'imb' in self.opt.task else False
        self.is_streaming = True if 'simb' in self.opt.task else False
        for dataset_id in self.opt.datasets:
            x, y = self.get_dataset(dataset_id)
            self.generate_task(x, y)
        print('{} - done ({}s)'.format(self.opt.task, time.time()-start_time))

    def generate_task(self, x, y):
        #x_train, y_train = self.split_train_test_valid(x, y)
        x_train, y_train = self.split_train_test(x, y)
        s, u = self.split_train(x_train, y_train)
        self.split_s(s)
        self.split_u(u)

    def split_train_test(self, x, y):
        self.num_examples = len(x)
        self.num_train = self.num_examples - self.opt.num_test
        self.num_test = self.opt.num_test
        self.labels = np.unique(y)
        # train set
        x_train = x[:self.num_train]
        y_train = y[:self.num_train]# x =[0,1,2,3,4,5,6,7,8,9,10] num_train = 8 test = 2 x_train = [0,1,2,3,4,5,6,7]
        # test set
        x_test = x[self.num_train:self.num_train+self.num_test]#x_test=[8,9]
        y_test = y[self.num_train:self.num_train+self.num_test]
        l_test = np.unique(y_test)#l_test=[0,0,1,2,0,1,2,1,2,1,2]->[0,1,2]
        y_test = tf.keras.utils.to_categorical(y_test, len(self.labels))
        self.save_task({
            'x': x_test,
            'y': y_test,
            'labels': l_test,
            'name': 't_{}'.format(self.did_to_dname[self.dataset_id])
        })
        # valid set
        # x_valid = x[self.num_train+self.num_test:]#[train,test,vaild]
        # y_valid = y[self.num_train+self.num_test:]
        # l_valid = np.unique(y_valid)
        # y_valid = tf.keras.utils.to_categorical(y_valid, len(self.labels))
        # self.save_task({
        #     'x': x_valid,
        #     'y': y_valid,
        #     'labels': l_valid,
        #     'name': 'v_{}'.format(self.did_to_dname[self.dataset_id])
        # })
        return x_train, y_train

    def split_train(self, x, y):
        if self.is_labels_at_server:
            self.num_s = self.opt.num_labels_per_class
        elif self.is_labels_at_all:
            self.num_s_server = self.opt.num_labels_per_class_server
            self.num_s_client = self.opt.num_labels_per_class_client * self.opt.num_clients
        else:
            self.num_s = self.opt.num_labels_per_class * self.opt.num_clients
        # for class-wise extraction
        data_by_label = {}
        for label in self.labels:#labels=[0,1,2,3,4,5]
            idx = np.where(y[:]==label)[0]
            data_by_label[label] = {
                'x': x[idx],
                'y': y[idx]
            }

        self.num_u = 0
        s_by_label, u_by_label = {}, {}
        for label, data in data_by_label.items():#key,value
            if self.is_labels_at_all:
                s_by_label[label] = {
                    'x': data['x'][:self.num_s_server+self.num_s_client],
                    'y': data['y'][:self.num_s_server+self.num_s_client]
                }
                u_by_label[label] = {
                    'x': data['x'][self.num_s_server+self.num_s_client:],
                    'y': data['y'][self.num_s_server+self.num_s_client:]
                }
                self.num_u += len(u_by_label[label]['x'])
            else:
                s_by_label[label] = {
                    'x': data['x'][:self.num_s],
                    'y': data['y'][:self.num_s]
                }
                u_by_label[label] = {
                    'x': data['x'][self.num_s:],
                    'y': data['y'][self.num_s:]
                }
                self.num_u += len(u_by_label[label]['x'])
        # print('num_u', self.num_u)
        # print('num_s', self.num_s_server + self.num_s_client)
        return s_by_label, u_by_label
        
    def split_s(self, s):
        if self.is_labels_at_server:
            x_labeled = []
            y_labeled = []
            for label, data in s.items():#key,value
                x_labeled = data['x'] if len(x_labeled)==0 else np.concatenate([x_labeled, data['x']], axis=0)
                y_labeled = data['y'] if len(y_labeled)==0 else np.concatenate([y_labeled, data['y']], axis=0)
            x_labeled, y_labeled = self.shuffle(x_labeled, y_labeled)
            self.save_task({
                'x': x_labeled,
                'y': tf.keras.utils.to_categorical(y_labeled, len(self.labels)),
                'name': 's_{}'.format(self.did_to_dname[self.dataset_id]),
                'labels': np.unique(y_labeled)
            })
        elif self.is_labels_at_all:
            x_labeled = []
            y_labeled = []
            for label, data in s.items():  # key,value
                x_labeled = data['x'][:self.opt.num_labels_per_class_server] if len(x_labeled) == 0 else np.concatenate([x_labeled, data['x'][:self.opt.num_labels_per_class_server]], axis=0)
                y_labeled = data['y'][:self.opt.num_labels_per_class_server] if len(y_labeled) == 0 else np.concatenate([y_labeled, data['y'][:self.opt.num_labels_per_class_server]], axis=0)
            x_labeled, y_labeled = self.shuffle(x_labeled, y_labeled)
            self.save_task({
                'x': x_labeled,
                'y': tf.keras.utils.to_categorical(y_labeled, len(self.labels)),
                'name': 's_{}'.format(self.did_to_dname[self.dataset_id]),
                'labels': np.unique(y_labeled)
            })
            for cid in range(self.opt.num_clients):
                x_labeled = []
                y_labeled = []
                for label, data in s.items():
                    start = self.opt.num_labels_per_class_server + self.opt.num_labels_per_class_client * cid
                    end = self.opt.num_labels_per_class_server + self.opt.num_labels_per_class_client * (cid+1)
                    x_labeled = data['x'][start:end] if len(x_labeled)==0 else np.concatenate([x_labeled, data['x'][start:end]], axis=0)
                    y_labeled = data['y'][start:end] if len(y_labeled)==0 else np.concatenate([y_labeled, data['y'][start:end]], axis=0)
                x_labeled, y_labeled = self.shuffle(x_labeled, y_labeled)
                self.save_task({
                    'x': x_labeled,
                    'y': tf.keras.utils.to_categorical(y_labeled, len(self.labels)),
                    'name': 's_{}_{}'.format(self.did_to_dname[self.dataset_id], cid),
                    'labels': np.unique(y_labeled)
                })
        else:
            for cid in range(self.opt.num_clients):
                x_labeled = []
                y_labeled = []
                for label, data in s.items():
                    start = self.opt.num_labels_per_class * cid
                    end = self.opt.num_labels_per_class * (cid+1)
                    x_labeled = data['x'][start:end] if len(x_labeled)==0 else np.concatenate([x_labeled, data['x'][start:end]], axis=0)
                    y_labeled = data['y'][start:end] if len(y_labeled)==0 else np.concatenate([y_labeled, data['y'][start:end]], axis=0)
                x_labeled, y_labeled = self.shuffle(x_labeled, y_labeled)
                self.save_task({
                    'x': x_labeled,
                    'y': tf.keras.utils.to_categorical(y_labeled, len(self.labels)),
                    'name': 's_{}_{}'.format(self.did_to_dname[self.dataset_id], cid),
                    'labels': np.unique(y_labeled)
                })


    def split_u(self, u):
        if self.is_imbalanced:
            '''
            ten_types_of_class_imbalanced_dist = [
                [0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15], # type 0
                [0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03], # type 1 
                [0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03,0.03], # type 2 
                [0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02,0.03], # type 3 
                [0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03,0.02], # type 4 
                [0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03,0.03], # type 5 
                [0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03,0.03], # type 6 
                [0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15,0.03], # type 7 
                [0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50,0.15], # type 8 
                [0.15,0.03,0.03,0.03,0.02,0.03,0.03,0.03,0.15,0.50], # type 9
            ]
            '''
            z = np.random.dirichlet((0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), size=10)
            for i in range(len(z)):
                sum = np.sum(z[i])
                for k in range(len(z)):
                    z[i][k] = z[i][k] / sum
            labels = list(u.keys())
            num_u_per_client = int(self.num_u/self.opt.num_clients)
            offset_per_label = {label:0 for label in labels}
            for cid in range(self.opt.num_clients):
                if self.is_streaming:
                    # streaming-imbalanced
                    x_unlabeled = {tid:[] for tid in range(self.opt.num_tasks)}
                    y_unlabeled = {tid:[] for tid in range(self.opt.num_tasks)}
                    dist_type = cid%len(labels)
                    freqs = np.random.choice(labels, num_u_per_client, p=z[dist_type])
                    frq = []
                    for label, data in u.items():
                        num_instances = len(freqs[freqs==label])
                        frq.append(num_instances)
                        start = offset_per_label[label]
                        end = offset_per_label[label]+num_instances
                        _x = data['x'][start:end] 
                        _y = data['y'][start:end]
                        offset_per_label[label] = end 
                        num_instances_per_task = int(len(_x)/self.opt.num_tasks)
                        for tid in range(self.opt.num_tasks):
                            start = num_instances_per_task * tid
                            end = num_instances_per_task * (tid+1)
                            x_unlabeled[tid] = _x[start:end] if len(x_unlabeled[tid])==0 else np.concatenate([x_unlabeled[tid], _x[start:end]], axis=0)
                            y_unlabeled[tid] = _y[start:end] if len(y_unlabeled[tid])==0 else np.concatenate([y_unlabeled[tid], _y[start:end]], axis=0)
                    print('>>>> frq', frq)
                    for tid in range(self.opt.num_tasks):
                        x_task = x_unlabeled[tid]
                        y_task = y_unlabeled[tid]
                        x_task, y_task = self.shuffle(x_task, y_task)
                        self.save_task({
                            'x': x_task,
                            'y': tf.keras.utils.to_categorical(y_task, len(self.labels)),
                            'name': 'u_{}_{}_{}'.format(self.did_to_dname[self.dataset_id], cid, tid),
                            'labels': np.unique(y_task)
                        })
                else:
                    # batch-imbalanced
                    x_unlabeled = []
                    y_unlabeled = []
                    dist_type = cid%len(labels)
                    freqs = np.random.choice(labels, num_u_per_client, p=z[dist_type])
                    frq = []
                    for label, data in u.items():
                        num_instances = len(freqs[freqs==label])
                        frq.append(num_instances)
                        start = offset_per_label[label]
                        end = offset_per_label[label]+num_instances
                        x_unlabeled = data['x'][start:end] if len(x_unlabeled)==0 else np.concatenate([x_unlabeled, data['x'][start:end]], axis=0)
                        y_unlabeled = data['y'][start:end] if len(y_unlabeled)==0 else np.concatenate([y_unlabeled, data['y'][start:end]], axis=0)
                        offset_per_label[label] = end

                    # x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)

                    self.save_task({
                        'x': x_unlabeled,
                        'y': tf.keras.utils.to_categorical(y_unlabeled, len(self.labels)),
                        'name': 'u_{}_{}'.format(self.did_to_dname[self.dataset_id], cid),
                        'labels': np.unique(y_unlabeled)
                    })    
                    print('>>>> frq', frq)
        else:
            # batch-iid
            for cid in range(self.opt.num_clients):
                x_unlabeled = []
                y_unlabeled = []
                for label, data in u.items():
                    # print('>>> ', label, len(data['x']))
                    num_unlabels_per_class = int(len(data['x'])/self.opt.num_clients)
                    start = num_unlabels_per_class * cid
                    end = num_unlabels_per_class * (cid+1)
                    x_unlabeled = data['x'][start:end] if len(x_unlabeled)==0 else np.concatenate([x_unlabeled, data['x'][start:end]], axis=0)
                    y_unlabeled = data['y'][start:end] if len(y_unlabeled)==0 else np.concatenate([y_unlabeled, data['y'][start:end]], axis=0)

                # x_unlabeled, y_unlabeled = self.shuffle(x_unlabeled, y_unlabeled)

                self.save_task({
                    'x': x_unlabeled,
                    'y': tf.keras.utils.to_categorical(y_unlabeled, len(self.labels)),
                    'name': 'u_{}_{}'.format(self.did_to_dname[self.dataset_id], cid),
                    'labels': np.unique(y_unlabeled)
                })

    def save_task(self, data):
        save_task(base_dir=self.base_dir, filename=data['name'], data=data)
        print('filename:{}, labels:[{}], num_examples:{}'.format(data['name'],','.join(map(str, data['labels'])), len(data['x'])))
    
    def shuffle(self, x, y):
        idx = np.arange(len(x))
        random_shuffle(self.opt.seed, idx)
        return x[idx], y[idx]