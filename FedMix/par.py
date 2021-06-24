import argparse

SEED = 1
NGT = 5
JOB = 'train'
BASE_NETWORK = 'resnet9' 
DATASET_PATH = '/home/nx/msy/FL/fl' # <-- set path

class Parser:#类

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()
       
    def set_arguments(self):
        self.parser.add_argument('-g', '--gpu', type=str, help='to set gpu ids to use e.g. 0,1,2,...')
        self.parser.add_argument('-gc', '--gpu-clients', type=str, help='to set number of clients per gpu e.g. 3,3,3,...')
        self.parser.add_argument('-m', '--model', type=str, default=fedmatch, help='to set model to experiment')
        self.parser.add_argument('-t', '--task', type=str, help='to set task to experiment')
        self.parser.add_argument('-f', '--frac-clients', type=float, help='to set fraction of clients per round')
        self.parser.add_argument('-fm', '--fed-method', type=str, help='to set fedprox or fedavg')
        self.parser.add_argument('-su', '--su', type=str, help='to set for sl model to learn on s or su')
        self.parser.add_argument('-sp', '--sp', type=int, help='to set for fedmatch to be lite or full')

        self.parser.add_argument('-j', '--job', type=str, default=JOB, help='to set job to execute e.g. data, train, test, etc.')
        self.parser.add_argument('-ngt', '--num-gt', type=int, default=NGT, help='to set num ground truth per class')
        self.parser.add_argument('-e', '--experiment', type=str, help='to set experiment name')
        self.parser.add_argument('--base-network', type=str, default=BASE_NETWORK, help='to set base networks alexnet-like, etc.')
        self.parser.add_argument('--task-path', type=str, default=DATASET_PATH, help='to set task path')
        self.parser.add_argument('--seed', type=int, default=SEED, help='to set fixed random seed')
        
    def parse(self):
        args, unparsed  = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args
