import os
import pdb
from par import Parser
from datetime import datetime


from utils.misc import *
from modules.data_generator import DataGenerator

def main(opt):
    
    if opt.job == 'data':
        opt = set_data_config(opt)
        dg = DataGenerator(opt)
        dg.generate_data()

    elif opt.job == 'train':
        opt = set_config(opt)
        if opt.model == 'fedmatch': 
            from models.fedmatch.server import Server
            server = Server(opt)
            server.run()
        else:
            print('incorrect model was given: {}'.format(opt.model))
            os._exit(0)

def set_config(opt):
    
    os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu
    
    # make log dir
    now = datetime.now().strftime("%Y%m%d-%H%M")
    opt.log_dir = '/home/nx/msy/FL/fl/logs/{}-{}-{}'.format(now, opt.model, opt.task) # <-- set new path
    opt.state_dir = '/home/nx/msy/FL/fl/states/{}-{}-{}'.format(now, opt.model, opt.task) # <-- set new path

    if not opt.experiment == None:
        opt.log_dir += '-{}'.format(opt.experiment)
        opt.state_dir += '-{}'.format(opt.experiment)

    if not os.path.isdir(opt.log_dir):
        os.makedirs(opt.log_dir)

    if not os.path.isdir(opt.state_dir):
        os.makedirs(opt.state_dir)
    
    # data configuration
    opt = set_data_config(opt)

    # train details
    if opt.scenario  == 'labels-at-client':
        opt.lr_factor = 3
        opt.lr_patience = 5
        opt.lr_min = 1e-20
    elif opt.scenario  == 'labels-at-server':
        opt.lr_factor = 5
        opt.lr_patience = 5
        opt.lr_min = 1e-20
    elif opt.scenario  == 'labels-at-all':
        opt.lr_factor = 3
        opt.lr_patience = 5
        opt.lr_min = 1e-20

    if opt.base_network in ['resnet9']:
        opt.lr = 1e-3
        opt.wd = 1e-4

    if 'fedmatch' in opt.model:
        opt.num_helpers = 0
        opt.confidence = 0.80
        if opt.scenario == 'labels-at-client':
            opt.lambda_s = 10  # supervised
            opt.lambda_u = 1
            opt.lambda_i = 0  # inter-client consistency
            opt.lambda_psi = 0.5
            opt.lambda_sig = 0.2
            opt.lambda_sig_server = 0.3
            opt.lambda_1 = 1e-1
            opt.lambda_2 = 1e-1
            opt.l1_thres = 0
            opt.lambda_l2 = 20
            opt.delta_thres = 0
            
        elif opt.scenario == 'labels-at-server':
            opt.lambda_s = 10  # supervised
            opt.lambda_u = 1
            opt.lambda_i = 0  # inter-client consistency
            opt.lambda_psi = 0.5
            opt.lambda_sig = 0.2
            opt.lambda_global = 0.3
            opt.lambda_1 = 1
            opt.lambda_2 = 1e-1
            opt.l1_thres = 0
            opt.lambda_l2 = 0
            opt.delta_thres = 0

        elif opt.scenario == 'labels-at-all':
            opt.lambda_s = 10 # supervised
            opt.lambda_u = 1
            opt.lambda_i = 0 # inter-client consistency
            opt.lambd_psi = 0.4
            opt.lambda_sig = 0.4
            opt.lambda_sig_server = 0.3
            opt.lambda_1 = 1e-1
            opt.lambda_2 = 1e-1
            opt.l1_thres = 0
            opt.lambda_l2 =15
            opt.delta_thres = 0

    if 'simb' in opt.task:
        opt.lambda_s = 1
        opt.delta_thres = 1e-5

    return opt

def set_data_config(opt):
    opt.mixture_fname = 'saved_mixture.npy'
    # dataset id: CIFAR10(0), CIFAR100(1), MNIST(2), SVHN(3), 
    # F-MNIST(4), TrafficSign(5), FaceScrub(6), N-MNIST(7)
    if 'c10' in opt.task:
        opt.datasets = [0]
        opt.num_classes = 10
        opt.num_test = 2000
        #opt.num_valid = 2000
        opt.batch_size_test = 100
    elif 'fmnist' in opt.task:
        opt.datasets = [4]
        opt.num_classes = 10
        opt.num_test = 2000
        # opt.num_valid = 2000
        opt.batch_size_test = 100

    # labels-at-client, labels-at-server
    if 'lc' in opt.task:
        opt.scenario = 'labels-at-client'
        opt.num_labels_per_class = 5
        opt.num_epochs_client = 1 
        opt.batch_size_client = 10 # for labeled set
        opt.num_epochs_server = 0
        opt.batch_size_server = 0
        opt.num_epochs_server_pretrain = 0
    elif 'ls' in opt.task:
        opt.scenario = 'labels-at-server'
        opt.num_labels_per_class = 100
        opt.num_epochs_client = 1 
        opt.batch_size_client = 64
        opt.batch_size_server = 64
        opt.num_epochs_server = 1
        opt.num_epochs_server_pretrain = 1
    elif 'lall' in opt.task:
        opt.scenario = 'labels-at-all'
        opt.num_labels_per_class_client = 4
        opt.num_epochs_client = 1
        opt.batch_size_client = 20  # for labeled set
        opt.num_labels_per_class_server = 100
        opt.batch_size_server = 64
        opt.num_epochs_server = 1
        opt.num_epochs_server_pretrain = 1

    # batch-iid, batch-imbalanced, streaming-imbalanced
    if 'biid' in opt.task or 'bimb' in opt.task: # batch-iid
        opt.sync = False
        opt.num_tasks = 1
        opt.num_clients = 100
        opt.num_rounds =150 # per task

    elif 'simb' in opt.task:
        opt.sync = True
        opt.num_tasks = 10
        opt.num_clients = 10
        opt.num_rounds = 100 # per task

    else:
        print('no correct task was given: {}'.format(opt.task))
        os._exit(0)
    return  opt

if __name__ == '__main__':
    main(Parser().parse())
