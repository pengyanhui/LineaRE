import json

import logging
import torch


class Configure(object):
	def __init__(self, config_path):
		with open(config_path, 'r') as fjson:
			json_config = json.load(fjson)
		self.dim = json_config['dim']
		self.norm_p = json_config['norm_p']
		self.alpha = json_config['alpha']
		self.beta = json_config['beta']
		self.gamma = json_config['gamma']
		self.learning_rate = json_config['learning_rate']
		self.decay_rate = json_config['decay_rate']
		self.batch_size = json_config['batch_size']
		self.neg_size = json_config['neg_size']
		self.regularization = json_config['regularization']
		self.drop_rate = json_config['drop_rate']
		self.test_batch_size = json_config['test_batch_size']
		self.data_path = json_config['data_path']
		self.save_path = json_config['save_path']
		self.max_step = json_config['max_step']
		self.valid_step = json_config['valid_step']
		self.log_step = json_config['log_step']
		self.test_log_step = json_config['test_log_step']
		self.optimizer = json_config['optimizer']
		self.init_checkpoint = json_config['init_checkpoint']
		self.use_old_optimizer = json_config['use_old_optimizer']
		self.sampling_rate = json_config['sampling_rate']
		self.sampling_bias = json_config['sampling_bias']
		self.device = torch.device(json_config['device'])
		self.multi_gpu = json_config['multiGPU']
		if self.multi_gpu:
			if torch.cuda.device_count() == 0:
				logging.error('no GPUs!!!\n')
				exit(-1)
			if torch.cuda.device_count() == 1:
				logging.error('only one GPU!!!\nwill use only one GPU!!!')

	def setting(self, new_path):
		self.__init__(new_path)


config = Configure('config/config_FB15k.json')
