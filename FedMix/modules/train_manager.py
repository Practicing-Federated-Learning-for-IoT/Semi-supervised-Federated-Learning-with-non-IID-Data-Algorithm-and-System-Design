import pdb
import time
import math
import numpy as np
import tensorflow as tf 
import tensorflow.keras.metrics as tf_metrics

from utils.misc import *

lss = np.zeros(100, dtype=float)
freq = np.zeros(100,dtype=float)
class TrainManager:

    def __init__(self, opt, log_manager):
        self.opt = opt
        self.log_manager = log_manager
        self.metrics = {
            'train_lss': tf_metrics.Mean(name='train_lss'),
            'train_acc': tf_metrics.CategoricalAccuracy(name='train_acc'),
            # 'valid_lss': tf_metrics.Mean(name='valid_lss'),
            # 'valid_acc': tf_metrics.CategoricalAccuracy(name='valid_acc'),
            'test_lss' : tf_metrics.Mean(name='test_lss'),
            'test_acc' : tf_metrics.CategoricalAccuracy(name='test_acc')
        }

    def init_state(self, client_id):
        self.state = {
            'client_id': client_id,
            'scores': {
                'train_loss': [],
                'train_acc': [],
                # 'valid_loss': [],
                # 'valid_acc': [],
                'test_loss': [],
                'test_acc': [],
            },

            # 's2c': {
            #     'ratio': [],
            #     'sig_ratio': [],
            #     'psi_ratio': [],
            #     'hlp_ratio': [],
            # },
            # 'c2s': {
            #     'ratio': [],
            #     'psi_ratio': [],
            #     'sig_ratio': [],
            # },

            'total_num_params': 0
        }
        self.init_learning_rate()

    def load_state(self, client_id):
        self.state = np_load(os.path.join(self.opt.state_dir, '{}_train_manager.npy'.format(client_id))).item()

    def save_state(self):
        np_save(self.opt.state_dir, '{}_train_manager.npy'.format(self.state['client_id']), self.state)

    def init_optimizer(self, curr_lr):
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=curr_lr)

    def init_learning_rate(self):
        # init learning rate
        self.state['early_stop'] = False
        self.state['lowest_lss'] = np.inf
        self.state['curr_lr'] = self.opt.lr
        self.state['curr_lr_patience'] = self.opt.lr_patience
        self.init_optimizer(self.opt.lr)

    def adaptive_lr_decay(self, vlss):
        if vlss<self.state['lowest_lss']:
            self.state['lowest_lss'] = vlss
            self.state['curr_lr_patience'] = self.opt.lr_patience
        else:
            self.state['curr_lr_patience']-=1
            if self.state['curr_lr_patience']<=0:
                prev = self.state['curr_lr']
                self.state['curr_lr']/=self.opt.lr_factor
                self.log_manager.print('epoch:%d, learning rate has been dropped from %.5f to %.5f' \
                                                    %(self.state['curr_epoch'], prev, self.state['curr_lr']))
                if self.state['curr_lr']<self.opt.lr_min:
                    self.log_manager.print('epoch:%d, early-stopped as minium lr reached to %.5f'%(self.state['curr_epoch'], self.state['curr_lr']))
                    self.state['early_stop'] = True
                self.state['curr_lr_patience'] = self.opt.lr_patience
                self.init_optimizer(self.state['curr_lr'])

    def train(self, curr_round, round_cnt, num_epochs=None):
        num_epochs = self.params['num_epochs'] if num_epochs == None else num_epochs
        self.state['curr_round'] = curr_round
        self.state['round_cnt'] = round_cnt
        self.num_train = len(self.task['x_train'])
        self.num_test = len(self.task['x_test'])
        start_time = time.time()            
        for epoch in range(num_epochs): 
            self.state['curr_epoch'] = epoch
            for i in range(0, len(self.task['x_train']), self.params['batch_size']): 
                x_batch = self.task['x_train'][i:i+self.params['batch_size']]
                y_batch = self.task['y_train'][i:i+self.params['batch_size']]
                with tf.GradientTape() as tape:
                    _, loss = self.params['loss_s'](x_batch, y_batch)
                gradients = tape.gradient(loss, self.params['trainables'])
                self.optimizer.apply_gradients(zip(gradients, self.params['trainables']))
            
            # vlss, vacc = self.validate()
            tlss, tacc = self.evaluate()
            #self.adaptive_lr_decay(tlss)
            self.log_manager.print('rnd:{}, ep:{}, n_train:{}, n_test:{} tlss:{}, tacc:{} ({}, {}s) '
                     .format(self.state['curr_round'], self.state['curr_epoch'], self.num_train, self.num_test, \
                            round(tlss, 4), round(tacc, 4), self.task['task_name'], round(time.time()-start_time,1)))
            
            if self.state['early_stop']:
                break

    def train_one_round(self, curr_round, client_id ,round_cnt, curr_task):#lss->client_id
        tf.keras.backend.set_learning_phase(1)
        self.state['curr_round'] = curr_round
        self.state['round_cnt'] = round_cnt
        self.state['curr_task'] = curr_task
        if self.opt.scenario == 'labels-at-client' or self.opt.scenario == 'labels-at-all':
            bsize_s = self.params['batch_size']
            num_steps = 5#round(len(self.task['x_labeled'])/bsize_s)
            bsize_u = math.ceil(len(self.task['x_unlabeled'])/num_steps)
            self.log_manager.print('num_steps:{}, bsize_s:{}, bsize_u:{}'.format(num_steps, bsize_s, bsize_u))
        else:
            num_steps = 2
            bsize_u = math.ceil(len(self.task['x_unlabeled'])/num_steps)
            self.log_manager.print('num_steps:{}, bsize_u:{}'.format(num_steps, bsize_u))

        self.num_labeled = 0 if not isinstance(self.task['x_labeled'], np.ndarray) else len(self.task['x_labeled'])
        self.num_unlabeled = 0 if not isinstance(self.task['x_unlabeled'], np.ndarray) else len(self.task['x_unlabeled'])
        self.num_train = self.num_labeled + self.num_unlabeled
        self.num_test = len(self.task['x_test'])
        
        start_time = time.time()
        for epoch in range(self.params['num_epochs']):
            loss_s = 0
            self.state['curr_epoch'] = epoch
            self.num_confident = 0 
            for i in range(num_steps):
                if 'fedmatch' in self.opt.model:
                    if self.opt.scenario == 'labels-at-client' or self.opt.scenario == 'labels-at-all':
                        bsize_s = self.params['batch_size']
                        ######################################
                        #         supervised learning    
                        ######################################
                        x_labeled = self.task['x_labeled'][i*bsize_s:(i+1)*bsize_s]
                        y_labeled = self.task['y_labeled'][i*bsize_s:(i+1)*bsize_s]
                        with tf.GradientTape() as tape:
                            _, loss_s = self.params['loss_s'](x_labeled, y_labeled)
                        gradients = tape.gradient(loss_s, self.params['trainables_s']) 
                        self.optimizer.apply_gradients(zip(gradients, self.params['trainables_s'])) 
                    
                    x_unlabeled = self.task['x_unlabeled'][i*bsize_u:(i+1)*bsize_u] 
                    with tf.GradientTape() as tape:
                        ######################################
                        #       unsupervised learning    
                        ######################################
                        _, loss_u, num_conf = self.params['loss_u'](x_unlabeled)
                    gradients = tape.gradient(loss_u, self.params['trainables_u']) 
                    self.optimizer.apply_gradients(zip(gradients, self.params['trainables_u'])) 
                    self.num_confident += num_conf
                    lss[client_id] = loss_s*35 + loss_u * self.opt.lambda_u
                else:
                    # base models: fixmatch & uda
                    x_unlabeled = self.task['x_unlabeled'][i*bsize_u:(i+1)*bsize_u] 
                    if len(x_unlabeled)>0:
                        with tf.GradientTape() as tape:
                            loss_final = 0
                            if self.opt.scenario == 'labels-at-client':
                                x_labeled = self.task['x_labeled'][i*bsize_s:(i+1)*bsize_s]
                                y_labeled = self.task['y_labeled'][i*bsize_s:(i+1)*bsize_s]        
                                _, loss_s = self.params['loss_s'](x_labeled, y_labeled)
                                loss_final += loss_s
                            _, loss_u, num_conf = self.params['loss_u'](x_unlabeled)
                            loss_final += loss_u
                        if loss_final>0:
                            gradients = tape.gradient(loss_final, self.params['trainables']) 
                            self.optimizer.apply_gradients(zip(gradients, self.params['trainables'])) 
                        self.num_confident += num_conf

            # vlss, vacc = self.validate()
            tlss, tacc = self.evaluate()
            if 'fedmatch' in self.opt.model:
                # self.check_c2s()
                self.log_manager.print('r:{},e:{},n_train:{}(s:{},u:{},c:{}),n_test:{},lss:{},acc:{},task:{},{}s) '
                 .format(self.state['curr_round'], self.state['curr_epoch'], self.num_train, self.num_labeled, self.num_unlabeled, self.num_confident,  \
                                    self.num_test, round(tlss, 4), round(tacc, 4),self.task['task_name'], round(time.time()-start_time,1)))
            else:
                self.log_manager.print('rnd:{}, ep:{}, n_train:{} (s:{}, u:{}, c:{}), n_test:{} tlss:{}, tacc:{} ({}, {}s) '
                     .format(self.state['curr_round'], self.state['curr_epoch'], self.num_train, self.num_labeled, self.num_unlabeled, self.num_confident,  \
                                        self.num_test, round(tlss, 4), round(tacc, 4), self.task['task_name'], round(time.time()-start_time,1)))
            
            if 'forgetting' in self.opt.task:
                flss, facc = self.evaluate_forgetting()
                self.log_manager.print('flss:{}, facc:{}'
                     .format(round(flss, 4), round(facc, 4)))
            
            #self.adaptive_lr_decay(tlss)
            if self.state['early_stop']:
                break

        
    # def validate(self):
    #     #     tf.keras.backend.set_learning_phase(0)
    #     #     for i in range(0, len(self.task['x_valid']), self.opt.batch_size_test):
    #     #         x_batch = self.task['x_valid'][i:i+self.opt.batch_size_test]
    #     #         y_batch = self.task['y_valid'][i:i+self.opt.batch_size_test]
    #     #         y_pred = self.params['model'](x_batch)
    #     #         loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred)
    #     #         self.add_performance('valid_lss', 'valid_acc', loss, y_batch, y_pred)
    #     #     vlss, vacc = self.measure_performance('valid_lss', 'valid_acc')
    #     #     self.state['scores']['valid_loss'].append(vlss)
    #     #     self.state['scores']['valid_acc'].append(vacc)
    #     #     return vlss, vacc
        
    def evaluate(self):
        tf.keras.backend.set_learning_phase(0)
        for i in range(0, len(self.task['x_test']), self.opt.batch_size_test):
            x_batch = self.task['x_test'][i:i+self.opt.batch_size_test]
            y_batch = self.task['y_test'][i:i+self.opt.batch_size_test]
            y_pred = self.params['model'](x_batch)
            loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred) 
            self.add_performance('test_lss', 'test_acc', loss, y_batch, y_pred)
        tlss, tacc = self.measure_performance('test_lss', 'test_acc')
        self.state['scores']['test_loss'].append(tlss)
        self.state['scores']['test_acc'].append(tacc)
        return tlss, tacc

    def evaluate_forgetting(self):
        tf.keras.backend.set_learning_phase(0)
        x_labeled = self.rescale(self.task['x_labeled'])
        for i in range(0, len(x_labeled), self.opt.batch_size_test):
            x_batch = x_labeled[i:i+self.opt.batch_size_test]
            y_batch = self.task['y_labeled'][i:i+self.opt.batch_size_test]
            y_pred = self.params['model'](x_batch)
            loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred) 
            self.add_performance('test_lss', 'test_acc', loss, y_batch, y_pred)
        flss, facc = self.measure_performance('test_lss', 'test_acc')
        if not 'forgetting_acc' in self.state['scores']:
            self.state['scores']['forgetting_acc'] = []
        if not 'forgetting_loss' in self.state['scores']:
            self.state['scores']['forgetting_loss'] = []
        self.state['scores']['forgetting_loss'].append(flss)
        self.state['scores']['forgetting_acc'].append(facc)
        return flss, facc

    def evaluate_after_aggr(self):
        tf.keras.backend.set_learning_phase(0)
        for i in range(0, len(self.task['x_test']), self.opt.batch_size_test):
            x_batch = self.task['x_test'][i:i+self.opt.batch_size_test]
            y_batch = self.task['y_test'][i:i+self.opt.batch_size_test]
            y_pred = self.params['model'](x_batch)
            loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred) 
            self.add_performance('test_lss', 'test_acc', loss, y_batch, y_pred)
        lss, acc = self.measure_performance('test_lss', 'test_acc')
        if not 'aggr_acc' in self.state['scores']:
            self.state['scores']['aggr_acc'] = []
        if not 'aggr_lss' in self.state['scores']:
            self.state['scores']['aggr_lss'] = []
        self.state['scores']['aggr_acc'].append(acc)
        self.state['scores']['aggr_lss'].append(lss)
        self.log_manager.print('aggr_lss:{}, aggr_acc:{}'.format(round(lss, 4), round(acc, 4)))

    def add_performance(self, lss_name, acc_name, loss, y_true, y_pred):
        self.metrics[lss_name](loss)
        self.metrics[acc_name](y_true, y_pred)

    def measure_performance(self, lss_name, acc_name):
        lss = float(self.metrics[lss_name].result())
        acc = float(self.metrics[acc_name].result())
        self.metrics[lss_name].reset_states()
        self.metrics[acc_name].reset_states()
        return lss, acc

    # def mask(self, weights):
    #     hard_threshold = tf.cast(tf.greater(tf.abs(weights), self.opt.l1_thres), tf.float32)
    #     return hard_threshold
    #
    # def sparsify(self, weights):
    #     hard_threshold = tf.cast(tf.greater(tf.abs(weights), self.opt.l1_thres), tf.float32)
    #     return tf.multiply(weights, hard_threshold)
    #
    # def threshold(self, weights):
    #     hard_threshold = tf.cast(tf.greater(tf.abs(weights), self.opt.delta_thres), tf.float32)#看是否超过阈值了
    #     return tf.multiply(weights, hard_threshold)

    def set_server_weights(self, sig, psi):
        self.sig_server = sig
        self.psi_server = psi

    def set_server_weights_2(self, sig_diff, psi_diff):
        self.sig_server = [ sig + self.params['trainables_s'][lid].numpy() for lid, sig in enumerate(sig_diff)] 
        self.psi_server = [ psi + self.params['trainables_u'][lid].numpy() for lid, psi in enumerate(psi_diff)] 


    def set_details(self, details):
        self.params = details

    def set_task(self, task):
        self.task = task

    def get_scores(self):
        return self.state['scores']
    
    def get_train_size(self):
        train_size = len(self.task['x_unlabeled'])
        if self.opt.scenario == 'labels-at-client' or self.opt.scenario == 'labels-at-all':
            train_size += len(self.task['x_labeled'])
        return train_size

    def aggregate(self, updates):
        self.log_manager.print('aggregating client-weights by {} ...'.format(self.opt.fed_method))
        if self.opt.fed_method == 'fedavg':
            return self.fedavg(updates)
        elif self.opt.fed_method == 'fedprox':
            return self.fedprox(updates)
        else:
            print('no correct fedmethod was given: {}'.format(self.opt.fed_method))
            os._exit(0)
    
    def fedavg(self, updates):
        client_weights = [u[0] for u in updates]
        client_sizes = [u[1] for u in updates]
        new_weights = [np.zeros_like(w) for w in client_weights[0]]
        total_size = np.sum(client_sizes)
        for c in range(len(client_weights)): # by client
            _client_weights = client_weights[c]
            for i in range(len(new_weights)): # by layer
                new_weights[i] += _client_weights[i] * float(client_sizes[c]/total_size)
        return new_weights

    def fedprox(self, updates):
        client_weights = [u[0] for u in updates]
        client_sizes = [u[1] for u in updates]
        new_weights = [np.zeros_like(w) for w in client_weights[0]]
        total_size = np.sum(client_sizes)
        for c in range(len(client_weights)): # by client
            _client_weights = client_weights[c]
            for i in range(len(new_weights)): # by layer
                new_weights[i] += _client_weights[i] * float(1/len(updates))
        return new_weights

    def fedavg_2(self, updates):
        client_weights = [u[0] for u in updates]
        new_weights = [np.zeros_like(w) for w in client_weights[0]]
        for c in range(len(client_weights)): # by client
            _client_weights = client_weights[c]
            for i in range(len(new_weights)): # by layer
                new_weights[i] += _client_weights[i] * (1/(self.opt.num_clients*self.opt.frac_clients))
        return new_weights

    def fedloss(self, updates):
        sum = 0
        client_weights = [u[0] for u in updates]
        client_id = [client[3] for client in updates]
        length_client = len(client_id)
        new_weights = [np.zeros_like(w) for w in client_weights[0]]
        frac = np.zeros(100, dtype = float)
        for i in range(length_client):
            sum += lss[client_id[i]]
        for i in range(length_client):
            frac[client_id[i]] = 1.0 - float(lss[client_id[i]]/sum)
        sum = 0
        for i in range(length_client):
            sum += frac[client_id[i]]
        for i in range(length_client):
            if frac[client_id[i]] > 0.09*sum:
                frac[client_id[i]] = float(frac[client_id[i]]/sum)
            else:
                frac[client_id[i]] = 0
        for c in range(len(client_weights)): # by client
            _client_weights = client_weights[c]
            this_id = client_id[c]
            for i in range(len(new_weights)): # by layer
                new_weights[i] += _client_weights[i] * float(frac[this_id])
        return new_weights

    def fedfreq(self,updates):
        sum = 0
        frac = np.zeros(100)
        client_weights = [u[0] for u in updates]
        new_weights = [np.zeros_like(w) for w in client_weights[0]]
        #client_avg = [np.zeros_like(w) for w in client_weights[0]]
        client_id = [client[3] for client in updates]
        length_client = len(client_id)
        for i in range(length_client):
            freq[client_id[i]] += 1
            sum += freq[client_id[i]]
        for i in range(length_client):
            frac[client_id[i]] = 1.0 - float(freq[client_id[i]]/sum)
        sum = 0
        for i in range(length_client):
            sum += frac[client_id[i]]
        for i in range(length_client):
            frac[client_id[i]] = float(frac[client_id[i]] / sum)
        for c in range(len(client_weights)):  # by client
            _client_weights = client_weights[c]
            this_id = client_id[c]
            for i in range(len(new_weights)):  # by layer
                new_weights[i] += _client_weights[i] * float(frac[this_id])
                #client_avg[i] += _client_weights[i] * 0.2
        #for i in range(len(new_weights)):
           # new_weights[i] = self.opt.lambda_global * new_weights[i] + self.opt.lambda_psi * client_avg[i] + self.opt.lambda_sig * sig[i]
        return new_weights
