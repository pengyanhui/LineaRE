import logging
from os import path
from queue import Queue
from threading import Thread

import torch

from configure import config


def set_logger():
	logging.basicConfig(
		format='%(asctime)s | %(levelname)s:  %(message)s',
		level=logging.INFO,
		datefmt='%Y-%m-%d %H:%M:%S',
		filename=path.join(config.save_path, 'train.log'),
		filemode='w'
	)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	console.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s:  %(message)s'))
	logging.getLogger().addHandler(console)


class PreDataLoader:
	"""
	@Author: Yuwei from https://www.zhihu.com/people/aewil-zheng, with few changes

	** 包装 torch.utils.data.DataLoader, 接收该类的一个实例 loader, 启动一个线程 t, 创建一个队列 q
	t 将 loader 中的数据预加载到队列 q 中, 以在模型计算时也能启动启动数据加载程序, 节省数据加载时间

	** 若提供了 cuda device, 数据将直接被加载到 GPU 上
	"""

	def __init__(self, loader, device=None, queue_size=2):
		"""
		:param loader: torch.utils.data.DataLoader
		:param device: torch.device('cuda' or 'cpu'), to use cpu, set None
		:param queue_size: the number of samples to be preloaded
		"""
		self.__loader = loader
		self.__device = device
		self.__queue_size = queue_size

		self.__load_stream = torch.cuda.Stream(device=device) \
			if str(device).startswith('cuda') else None  # 如果提供了 cuda device, 则创建 cuda 流

		self.__queue = Queue(maxsize=self.__queue_size)
		self.__idx = 0
		self.__worker = Thread(target=self._load_loop)
		self.__worker.setDaemon(True)
		self.__worker.start()

	def _load_loop(self):
		""" 不断的将数据加载到队列里 """
		if str(self.__device).startswith('cuda'):
			logging.info(f'>>> data will be preloaded into device \'{self.__device}\'')
			logging.info(f'>>> this may cost more GPU memory!!!')
			# The loop that will load into the queue in the background
			torch.cuda.set_device(self.__device)
			while True:
				for sample in self.__loader:
					self.__queue.put(self._load_instance(sample))
		else:
			while True:
				for sample in self.__loader:
					self.__queue.put(sample)

	def _load_instance(self, sample):
		""" 将 batch 数据从 CPU 加载到 GPU 中 """
		if torch.is_tensor(sample):
			with torch.cuda.stream(self.__load_stream):
				return sample.to(self.__device, non_blocking=True)
		elif sample is None or type(sample) == str:
			return sample
		elif isinstance(sample, dict):
			return {k: self._load_instance(v) for k, v in sample.items()}
		else:
			return [self._load_instance(s) for s in sample]

	def __iter__(self):
		self.__idx = 0
		return self

	def __next__(self):
		# 加载线程挂了
		if not self.__worker.is_alive() and self.__queue.empty():
			self.__idx = 0
			self.__queue.join()
			self.__worker.join()
			raise StopIteration
		# 一个 epoch 加载完了
		elif self.__idx >= len(self.__loader):
			self.__idx = 0
			raise StopIteration
		# 下一个 batch
		else:
			out = self.__queue.get()
			self.__queue.task_done()
			self.__idx += 1
		return out

	def next(self):
		return self.__next__()

	def __len__(self):
		return len(self.__loader)

	@property
	def sampler(self):
		return self.__loader.sampler

	@property
	def dataset(self):
		return self.__loader.dataset
