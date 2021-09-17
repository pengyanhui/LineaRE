import copy
import json
import logging
import pathlib
from os import path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from configure import config
from data import KG, BiOneShotIterator
from lineare import LineaRE


class TrainTest:
	def __init__(self, kg: KG):
		self.__kg = kg
		self.__model = LineaRE(kg.num_ents, kg.num_rels).to(config.device)
		self.__cal_model = nn.DataParallel(self.__model) if config.multi_gpu else self.__model

	def train(self):
		logging.info('Start Training...')
		data_shoter = BiOneShotIterator(*self.__kg.train_data_iterator())
		optimizer, init_step, current_lr = self._get_optimizer()
		scheduler = ExponentialLR(optimizer=optimizer, gamma=config.decay_rate)

		max_mrr = 0.0
		training_logs = []
		# Training Loop
		for step in range(init_step, config.max_step + 1):
			log = LineaRE.train_step(self.__cal_model, optimizer, data_shoter.next())
			training_logs.append(log)
			# log
			if step % config.log_step == 0:
				metrics = {}
				for metric in training_logs[0].keys():
					metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
				self._log_metrics('Training', step, metrics)
				training_logs.clear()
			# valid
			if step % config.valid_step == 0:
				logging.info(f'---------- Evaluating on Valid Dataset ----------')
				metrics = LineaRE.test_step(self.__cal_model, self.__kg.test_data_iterator(test='valid'), True)
				self._log_metrics('Valid', step, metrics)
				logging.info('-----------------------------------------------')
				if metrics[0]['MRR'] >= max_mrr:
					max_mrr = metrics[0]['MRR']
					save_variable_list = {
						'step': step,
						'current_lr': current_lr,
					}
					self._save_model(optimizer, save_variable_list)
					logging.info(f'Find a better model, it has been saved in \'{config.save_path}\'!')
				if step / config.max_step in [0.2, 0.5, 0.8]:
					scheduler.step()
					current_lr *= config.decay_rate
					logging.info(f'Change learning_rate to {current_lr} at step {step}')
		logging.info('Training Finished!')

	def test(self):
		# load best model state
		checkpoint = torch.load(path.join(config.save_path, 'checkpoint'))
		self.__model.load_state_dict(checkpoint['model_state_dict'])
		step = checkpoint['step']
		# relation patterns
		test_datasets_str = ['Symmetry', 'Inversion', 'Composition', 'Other']
		for dataset_str in test_datasets_str:
			test_data_list = self.__kg.test_data_iterator(test=dataset_str)
			if len(test_datasets_str[0]) == 0:
				continue
			logging.info(f'---------- Evaluating on {dataset_str} Dataset ----------')
			metrics = LineaRE.test_step(self.__cal_model, test_data_list)
			self._log_metrics(dataset_str, step, metrics)
		# finally test
		test_data_list = self.__kg.test_data_iterator(test='test')
		logging.info('----------Evaluating on Test Dataset----------')
		metrics = LineaRE.test_step(self.__cal_model, test_data_list, True)
		self._log_metrics('Test', step, metrics)

		def _get_optimizer(self):  # add Optimizer you wanted here
		current_lr = config.learning_rate
		if config.optimizer == 'Adam':
			Optimizer = torch.optim.Adam
		elif config.optimizer == 'Adagrad':
			Optimizer = torch.optim.Adagrad
		elif config.optimizer == 'SGD':
			Optimizer = torch.optim.SGD
		else:
			raise ValueError(f'optimizer {config.optimizer} not supported')
		optimizer = Optimizer(
			filter(lambda p: p.requires_grad, self.__model.parameters()),
			lr=current_lr
		)

		if config.init_checkpoint:
			logging.info('Loading checkpoint...')
			checkpoint = torch.load(path.join(config.save_path, 'checkpoint'), map_location=config.device)
			init_step = checkpoint['step'] + 1
			self.__model.load_state_dict(checkpoint['model_state_dict'])
			if config.use_old_optimizer:
				current_lr = checkpoint['current_lr']
				optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		else:
			init_step = 1
		return optimizer, init_step, current_lr

	def _save_model(self, optimizer, save_vars):
		# 保存 config
		save_path = pathlib.Path(config.save_path)
		config_dict = vars(copy.deepcopy(config))
		del config_dict['device']
		with open(save_path / 'config.json', 'w') as fjson:
			json.dump(config_dict, fjson)
		# 保存某些变量、模型参数、优化器参数
		torch.save(
			{
				**save_vars,
				'model_state_dict': self.__model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()
			},
			save_path / 'checkpoint'
		)
		# 保存 numpy embedding
		param_dict = {
			'ent_embd': self.__model.ent_embd,
			'rel_embd': self.__model.rel_embd,
			'wrh': self.__model.wrh,
			'wrt': self.__model.wrt
		}
		for name, param in param_dict.items():
			param = param.weight.detach().cpu().numpy()
			np.save(str(save_path / name), param)

	@staticmethod
	def _log_metrics(dataset_str, step, metrics):
		def log_metrics(metrics_dict: Dict[str, float]):
			for metric, value in metrics_dict.items():
				logging.info(f'{dataset_str} {metric} at step {step:d}: {value:f}')

		if isinstance(metrics, dict):
			log_metrics(metrics)
		elif isinstance(metrics, list):
			log_metrics(metrics[0])
			cnt_mode_rtp = 1
			for mode in ('Prediction Head', 'Prediction Tail'):
				for rtp in ('1-1', '1-M', 'M-1', 'M-M'):
					logging.info(f'---------- {mode}... {rtp} ----------')
					log_metrics(metrics[cnt_mode_rtp])
					cnt_mode_rtp += 1
