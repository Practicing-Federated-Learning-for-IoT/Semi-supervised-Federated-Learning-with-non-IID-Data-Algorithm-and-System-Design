import os
import pdb
import glob
import numpy as np

from PIL import Image
from scipy.ndimage.interpolation import rotate, shift
from third_party.rand_augment.randaug import RandAugment

from utils.misc import *

class DataManager:

    def __init__(self, opt, log_manager):
        self.opt = opt
        self.log_manager = log_manager
        self.rand_augment = RandAugment()
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

    def init_state(self, client_id):
        self.state = {
            'client_id': client_id,
            'tasks': []
        }
        self.load_tasks()

    def load_state(self, client_id):
        self.state = np_load(os.path.join(self.opt.state_dir, '{}_data_manager.npy'.format(client_id))).item()

    def save_state(self):
        np_save(self.opt.state_dir, '{}_data_manager'.format(self.state['client_id']), self.state)

    def load_tasks(self):
        for d in self.opt.datasets:
            path = os.path.join(self.base_dir, self.did_to_dname[d]+'_'+str(self.state['client_id'])+'_*')
            self.tasks = [os.path.basename(p) for p in glob.glob(path)]
        self.tasks = sorted(self.tasks)

    def get_s_by_id(self, client_id, task_id):
        self.state['client_id'] = client_id
        for d in self.opt.datasets:
            path = os.path.join(self.base_dir, 's_{}_{}*'.format(self.did_to_dname[d],str(self.state['client_id'])))
            self.tasks = sorted([os.path.basename(p) for p in glob.glob(path)])
        task = load_task(self.base_dir, self.tasks[task_id]).item()
        return task['x'], task['y'], task['name']

    def get_u_by_id(self, client_id, task_id):
        self.state['client_id'] = client_id
        for d in self.opt.datasets:
            path = os.path.join(self.base_dir, 'u_{}_{}*'.format(self.did_to_dname[d],str(self.state['client_id'])))
            self.tasks = sorted([os.path.basename(p) for p in glob.glob(path)])
        task = load_task(self.base_dir, self.tasks[task_id]).item()
        return task['x'], task['y'], task['name']

    def get_s_server(self):
        tasks = []
        for d in self.opt.datasets:
            path = os.path.join(self.base_dir, 's_{}.npy'.format(self.did_to_dname[d]))
            tasks += [os.path.basename(p) for p in glob.glob(path)]
        task = load_task(self.base_dir, tasks[-1]).item()
        return task['x'], task['y'], task['name']

    def get_test(self):
        tasks = []
        for d in self.opt.datasets:
            path = os.path.join(self.base_dir, 't_{}*'.format(self.did_to_dname[d]))
            tasks += [os.path.basename(p) for p in glob.glob(path)]
        task = load_task(self.base_dir, tasks[-1]).item()
        return task['x'], task['y']

    # def get_valid(self):
    #     tasks = []
    #     for d in self.opt.datasets:
    #         path = os.path.join(self.base_dir, 'v_{}*'.format(self.did_to_dname[d]))
    #         tasks += [os.path.basename(p) for p in glob.glob(path)]
    #     task = load_task(self.base_dir, tasks[-1]).item()
    #     return task['x'], task['y']

    def rescale(self, images):
        return images.astype(np.float32)/255.

    def augment(self, images, soft=True, R=None):
        if soft:
            indices = np.arange(len(images)).tolist() 
            sampled = random.sample(indices, int(round(0.5*len(indices)))) # flip horizontally 50% 
            images[sampled] = np.fliplr(images[sampled])
            sampled = random.sample(sampled, int(round(0.25*len(sampled)))) # flip vertically 25% from above
            images[sampled] = np.flipud(images[sampled])
            return np.array([shift(img, [random.randint(-2, 2), random.randint(-2, 2), 0]) for img in images]) # random shift
        elif R:
            #indices = np.arange(len(images)).tolist()
            #sampled = random.sample(indices, int(round(0.5 * len(indices))))  # flip horizontally 50%
            #images[sampled] = np.fliplr(images[sampled])
            #sampled = random.sample(sampled, int(round(0.25 * len(sampled))))  # flip vertically 25% from above
            #images[sampled] = np.flipud(images[sampled])
            return np.array([rotate(img, 66) for img in images])
        else:
            return np.array([np.array(self.rand_augment(Image.fromarray(np.reshape(img, (32,32,3))), M=random.randint(2,5))) for img in images])
