import gc
import pdb
import cv2
import time
import random
import tensorflow as tf 
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K
from sklearn.cluster import KMeans
import sklearn
from utils.misc import *
from collections import defaultdict

from modules.log_manager import LogManager
from modules.data_manager import DataManager
from modules.model_manager import ModelManager
from modules.train_manager import TrainManager

class Client:

    def __init__(self, gid, opt):
        self.opt = opt
        self.state = {'gpu_id': gid}
        self.kl_divergence = tf.keras.losses.KLDivergence()
        self.cross_entropy = tf.keras.losses.CategoricalCrossentropy()
        self.log_manager = LogManager(self.opt) 
        self.data_manager = DataManager(self.opt, self.log_manager)
        self.model_manager = ModelManager(self.opt, self.log_manager)
        self.train_manager = TrainManager(self.opt, self.log_manager)
        self.init_model()

    def init_model(self):
        if self.opt.base_network in ['resnet9']:
            self.local_model = self.model_manager.build_resnet9_decomposed()


            # self.helpers = [self.model_manager.build_resnet9_plain() for _ in range(self.opt.num_helpers)]


        self.sigma = self.model_manager.get_sigma()
        self.psi = self.model_manager.get_psi()


        # for h in self.helpers:
        #     h.trainable = False


        self.log_manager.print('networks have been built')

    def switch_state(self, client_id):
        if self.is_new(client_id):
            self.log_manager.init_state(client_id)
            self.data_manager.init_state(client_id)
            self.model_manager.init_state(client_id)
            self.train_manager.init_state(client_id)
            self.init_state(client_id)
        else: # load_state
            self.log_manager.load_state(client_id)
            self.data_manager.load_state(client_id)
            self.model_manager.load_state(client_id)
            self.train_manager.load_state(client_id)
            self.load_state(client_id)

    def is_new(self, client_id):
        return not os.path.exists(os.path.join(self.opt.state_dir, '{}_client.npy'.format(client_id)))
    
    def init_state(self, client_id):
        self.state['client_id'] = client_id
        self.state['done'] = False
        self.state['curr_task'] =  -1
        self.state['task_names'] = []
        self.train_manager.set_details({
            'loss_s': self.loss_s,
            'loss_u': self.loss_u,
            'model': self.local_model,
            'trainables_s': self.sigma,
            'trainables_u': self.psi,
            'batch_size': self.opt.batch_size_client,
            'num_epochs': self.opt.num_epochs_client,
        })

    def load_state(self, client_id):
        self.state = np_load(os.path.join(self.opt.state_dir, '{}_client.npy'.format(client_id))).item()

    def save_state(self):
        np_save(self.opt.state_dir, '{}_client.npy'.format(self.state['client_id']), self.state)
        self.data_manager.save_state()
        self.model_manager.save_state()
        self.train_manager.save_state()


    #def train_one_round(self, client_id, curr_round, sigma, psi, helpers=None):#sigma和psi都是server的


    def train_one_round(self, client_id, curr_round, sigma, psi):  # sigma和psi都是server的
        #######################
        self.switch_state(client_id)


        # self.train_manager.check_s2c(curr_round, sigma, psi, helpers)


        #######################
        if self.state['curr_task'] < 0:
            self.init_new_task()
        else:
            is_last_task = (self.state['curr_task']==self.opt.num_tasks-1)
            is_last_round = (self.state['round_cnt']%self.opt.num_rounds==0 and self.state['round_cnt']!=0)
            is_last = is_last_task and is_last_round
            if is_last_round or self.train_manager.state['early_stop']:
                if is_last_task:
                    if self.train_manager.state['early_stop']:
                        self.train_manager.evaluate()
                    self.stop()
                    return
                else:
                    self.init_new_task()
            else:
                self.load_data()
        self.state['round_cnt'] += 1
        self.state['curr_round'] = curr_round
        self.set_weights(sigma, psi)
        self.train_manager.train_one_round(self.state['curr_round'], client_id ,self.state['round_cnt'], self.state['curr_task'])
        self.log_manager.save_current_state({
            'scores': self.train_manager.get_scores()
        })
        #
        #     # 's2c': self.train_manager.get_s2c(),
        #     # 'c2s': self.train_manager.get_c2s(),

        #######################
        self.save_state()
        #######################


        #return (self.get_weights(), self.get_train_size(), self.state['client_id'], self.train_manager.get_c2s(), self.train_manager.get_s2c())


        return (self.get_weights(), self.get_train_size(), self.state['client_id'], client_id)

    def loss_s(self, x, y):
        x = self.data_manager.rescale(x)
        y_pred = self.local_model(x)
        loss_s = self.cross_entropy(y, y_pred) * self.opt.lambda_s
        return y_pred, loss_s

    def loss_u(self, x):
        #print(x.shape)
        #half = int(len(x)/2)
        loss_u = 0
        y_pred = self.local_model(self.data_manager.rescale(self.data_manager.augment(x,R=True)))
        y_hard = self.local_model(self.data_manager.rescale(self.data_manager.augment(x, soft=True)))
        #front=rear=half
        #if len(x)%2 == 1:
        #        rear=half+1
        #compute = []
        #for i in range(len(x)):
        #        compute.append(np.concatenate(np.concatenate(x[i])))
        #k = sklearn.cluster.k_means(compute,10,max_iter=10000)
        #index = defaultdict(list)
        #for i in range(len(k[1])):
        #        index[k[1][i]].append(i)
        #for j in range(10):
        #        x_1 = []
        #        x_2 = []
        #        for i in range(0,len(index[j]),2):
        #                if i+2 <= len(index[j]):
        #                        x_1.append(x[index[j][i]])
        #                        x_2.append(x[index[j][i]+1])
        #        x_1 = np.array(x_1)
        #        x_2 = np.array(x_2)
                #print(x_1.shape,x_2.shape)
        #        y_pred_1 = self.local_model(self.data_manager.rescale(x_1))
        #        y_pred_2 = self.local_model(self.data_manager.rescale(x_2))
        #for i in range(len(y_pred_1)):
        loss_u += tf.math.reduce_sum(tf.math.square(y_pred - y_hard)) * self.opt.lambda_2 
        conf = np.where(np.max(y_pred.numpy(), axis=1)>=self.opt.confidence)[0]
        if len(conf)>0:
            x_conf = self.data_manager.rescale(x[conf])
            y_pred = K.gather(y_pred, conf)

            # if self.is_helper_available:
            #     y_preds = [rm(x_conf).numpy() for rid, rm in enumerate(self.helpers)]
            #     if self.state['curr_round']>0:
            #         #inter-client consistency loss
            #         for hid, pred in enumerate(y_preds):
            #             loss_u += (self.kl_divergence(pred, y_pred)/len(y_preds))*self.opt.lambda_i
            # else:
            #     y_preds = None
            y_pseu = np.zeros([len(y_pred),len(y_pred[0])])
            for i in range(5):
                y_pseu += self.local_model(self.data_manager.rescale(self.data_manager.augment(x[conf], soft=True)))
            #锐化
            y_pseu_numpy = y_pseu.numpy()
            for i in range(len(y_pseu)):
                y_pseu_numpy[i] = np.where(y_pseu_numpy[i] >= np.max(y_pseu_numpy[i]), y_pseu_numpy[i], 0)
                y_pseu_numpy[i][np.argmax(y_pseu_numpy[i])] = 1
            y_pseu = tf.convert_to_tensor(y_pseu_numpy)
            loss_u += self.cross_entropy(y_pseu, y_pred) * self.opt.lambda_1
        # additional regularization
        for lid, psi in enumerate(self.psi):
            # l2 regularization
            loss_u += tf.math.reduce_sum(tf.math.square(self.sigma[lid]-psi)) * self.opt.lambda_l2
        return y_pred, loss_u, len(conf)


    # def agreement_based_labeling(self, y_pred, y_preds=None):
    #     y_pseudo = np.array(y_pred)
    #     if self.is_helper_available:
    #         y_vote = tf.keras.utils.to_categorical(np.argmax(y_pseudo, axis=1), self.opt.num_classes)
    #         y_votes = np.sum([tf.keras.utils.to_categorical(np.argmax(y_rm, axis=1), self.opt.num_classes) for y_rm in y_preds], axis=0)
    #         y_vote = np.sum([y_vote, y_votes], axis=0)
    #         y_pseudo = tf.keras.utils.to_categorical(np.argmax(y_vote, axis=1), self.opt.num_classes)
    #     else:
    #         y_pseudo = tf.keras.utils.to_categorical(np.argmax(y_pseudo, axis=1), self.opt.num_classes)
    #     return y_pseudo


    def init_new_task(self):
        self.state['curr_task'] += 1
        self.state['round_cnt'] = 0
        self.load_data()

    def load_data(self):
        if self.opt.scenario == 'labels-at-client' or self.opt.scenario == 'labels-at-all':#形式一样的嘛
            if 'simb' in self.opt.task and self.state['curr_task']>0:
                self.x_unlabeled, self.y_unlabeled, task_name = self.data_manager.get_u_by_id(self.state['client_id'], self.state['curr_task'])
            else:
                self.x_labeled, self.y_labeled, task_name = self.data_manager.get_s_by_id(self.state['client_id'], self.state['curr_task'])
                self.x_unlabeled, self.y_unlabeled, task_name = self.data_manager.get_u_by_id(self.state['client_id'], self.state['curr_task'])
        elif self.opt.scenario == 'labels-at-server':
            self.x_labeled, self.y_labeled = None, None
            self.x_unlabeled, self.y_unlabeled, task_name = self.data_manager.get_u_by_id(self.state['client_id'], self.state['curr_task'])
        self.x_test, self.y_test =  self.data_manager.get_test()


        # self.x_valid, self.y_valid =  self.data_manager.get_valid()


        self.x_test = self.data_manager.rescale(self.x_test)


        # self.x_valid = self.data_manager.rescale(self.x_valid)


        # self.train_manager.set_task({
        #     'task_name': task_name.replace('u_',''),
        #     'x_valid':self.x_valid,
        #     'y_valid':self.y_valid,
        #     'x_test':self.x_test,
        #     'y_test':self.y_test,
        #     'x_labeled':self.x_labeled,
        #     'y_labeled':self.y_labeled,
        #     'x_unlabeled':self.x_unlabeled,
        #     'y_unlabeled':self.y_unlabeled,
        # })


        self.train_manager.set_task({
            'task_name':task_name.replace('u',''),
            'x_test':self.x_test,
            'y_test':self.y_test,
            'x_labeled':self.x_labeled,
            'y_labeled':self.y_labeled,
            'x_unlabeled':self.x_unlabeled,
            'y_unlabeled':self.y_unlabeled,
        })



    # def restore_helpers(self, helper_weights):
    #     for hid, hwgts in enumerate(helper_weights):
    #         wgts = self.helpers[hid].get_weights()
    #         for i in range(len(wgts)):
    #             wgts[i] = self.sigma[i].numpy() + hwgts[i] # sigma + psi
    #         self.helpers[hid].set_weights(wgts)


    def get_weights(self):
        if self.opt.scenario == 'labels-at-client' or self.opt.scenario == 'labels-at-all':
            sigs = [sig.numpy() for sig in self.sigma]
            psis = [psi.numpy() for psi in self.psi] 
            return np.concatenate([sigs,psis], axis=0)
        elif self.opt.scenario == 'labels-at-server':
            return [psi.numpy() for psi in self.psi]

    def set_weights(self, sigma, psi):
        # initialization of theta_sup from global model
        for i, sig in enumerate(sigma):
            self.sigma[i].assign(sig)
        for i, p in enumerate(psi):
            self.psi[i].assign(p)
    
    def get_train_size(self):
        train_size = len(self.x_unlabeled)
        if self.opt.scenario == 'labels-at-client' or self.opt.scenario == 'labels-at-all':
            train_size += len(self.x_labeled)
        return train_size

    def get_task_id(self):
        return self.state['curr_task']

    def get_client_id(self):
        return self.state['client_id']

    def stop(self):
        self.log_manager.print('finished learning all tasks')
        self.log_manager.print('done.')
        self.done = True
