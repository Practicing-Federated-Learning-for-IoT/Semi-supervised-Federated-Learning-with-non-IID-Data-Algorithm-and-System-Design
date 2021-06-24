
from datetime import datetime

from utils.misc import *

class LogManager:

    def __init__(self, opt, client_id=None):
        self.opt = opt
        self.init_state(client_id)

    def init_state(self, client_id):
        self.name = 'server' if client_id==None else 'client-{}'.format(client_id)

    def load_state(self, client_id):
        self.name = 'server' if client_id==None else 'client-{}'.format(client_id)

    def print(self, message):
	    print('[%s][%s][%s][%s] %s' %(datetime.now().strftime("%Y/%m/%d-%H:%M:%S"), self.opt.model, self.opt.task, self.name, message))

    def save_current_state(self, current_state):
    	current_state['options'] = vars(self.opt)
    	write_file(self.opt.log_dir, '{}.txt'.format(self.name), current_state)
