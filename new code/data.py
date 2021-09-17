import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import utils
from configure import config


def read_elements(file_path):
	elements2id = {}
	with open(file_path, 'r') as f:
		for line in f:
			e_id, e_str = line.strip().split('\t')
			elements2id[e_str] = int(e_id)
	return elements2id


class KG(object):
	def __init__(self):
		data_path = Path(config.data_path)

		self.__ent2id = read_elements(data_path / 'entities.dict')
		self.__rel2id = read_elements(data_path / 'relations.dict')
		self.num_ents = len(self.__ent2id)
		self.num_rels = len(self.__rel2id)

		self.__train_triples = self._read_triples(data_path / 'train.txt')
		self.__valid_triples = self._read_triples(data_path / 'valid.txt')
		self.__test_triples = self._read_triples(data_path / 'test.txt')
		self.__all_true_triples = self.__train_triples + self.__valid_triples + self.__test_triples

		self.__test_dict = {
			'test': self.__test_triples,
			'valid': self.__valid_triples,
			'symmetry': self._read_triples(data_path / 'symmetry_test.txt'),
			'inversion': self._read_triples(data_path / 'inversion_test.txt'),
			'composition': self._read_triples(data_path / 'composition_test.txt'),
			'other': self._read_triples(data_path / 'other_test.txt')
		}
		self.__all_true_heads, self.__all_true_tails = self._get_true_ents()
		self.__r_tp = self._get_rel_type()

		self._logger()

	def train_data_iterator(self):
		return tuple(
			utils.PreDataLoader(
				loader=DataLoader(
					dataset=dataset(self.__train_triples, self.num_ents),
					batch_size=config.batch_size,
					shuffle=True,
					num_workers=4,
					collate_fn=TrainDataset.collate_fn,
					persistent_workers=True
				),
				device=config.device
			) for dataset in [HeadDataset, TailDataset]
		)

	def test_data_iterator(self, test='test'):
		try:
			triples = self.__test_dict[test.lower()]
			return [
				(
					DataLoader(
						dataset(triples, self.num_ents, true_ents, self.__r_tp),
						batch_size=config.test_batch_size,
						num_workers=4,
						collate_fn=TestDataset.collate_fn
					),
					ht
				) for dataset, true_ents, ht in zip(
					[TestHead, TestTail],
					[self.__all_true_heads, self.__all_true_tails],
					['head-batch', 'tail-batch']
				)
			]
		except KeyError:
			logging.error(f'No triple file named {test}')
			exit(-1)

	def _read_triples(self, file_path):
		triples = []
		if not file_path.is_file():
			return []
		with open(file_path, 'r') as f:
			for line in f:
				h, r, t = line.strip().split('\t')
				triples.append((self.__ent2id[h], self.__rel2id[r], self.__ent2id[t]))
		return triples

	def _get_true_ents(self):
		true_heads = {}
		true_tails = {}
		for h, r, t in self.__all_true_triples:
			if (r, t) not in true_heads:
				true_heads[(r, t)] = []
			true_heads[(r, t)].append(h)
			if (h, r) not in true_tails:
				true_tails[(h, r)] = []
			true_tails[(h, r)].append(t)
		for rt in true_heads:
			true_heads[rt] = np.array(true_heads[rt])
		for hr in true_tails:
			true_tails[hr] = np.array(true_tails[hr])
		return true_heads, true_tails

	def _get_rel_type(self):
		count_r = {}
		count_h = {}
		count_t = {}
		for h, r, t in self.__train_triples:
			if r not in count_r:
				count_r[r] = 0
				count_h[r] = set()
				count_t[r] = set()
			count_r[r] += 1
			count_h[r].add(h)
			count_t[r].add(t)
		r_tp = {}
		for r in range(len(count_r)):
			tph = count_r[r] / len(count_h[r])
			hpt = count_r[r] / len(count_t[r])
			if hpt < 1.5:
				if tph < 1.5:
					r_tp[r] = 1  # 1-1
				else:
					r_tp[r] = 2  # 1-M
			else:
				if tph < 1.5:
					r_tp[r] = 3  # M-1
				else:
					r_tp[r] = 4  # M-M
		return r_tp

	def _logger(self):
		logging.info(f'#ent_num: {self.num_ents:d}')
		logging.info(f'#rel_num: {self.num_rels:d}')
		logging.info(f'#train triple num: {len(self.__train_triples):d}')
		logging.info(f'#valid triple num: {len(self.__valid_triples):d}')
		logging.info(f'#test triple num: {len(self.__test_triples):d}')


class TrainDataset(Dataset):
	def __init__(self, triples, num_ents):
		super(TrainDataset, self).__init__()
		self._triples = triples
		self.__num_ents = num_ents
		self.__neg_sample_size = config.neg_size + config.neg_size // 3

	def __len__(self):
		return len(self._triples)

	def __getitem__(self, idx):
		raise NotImplementedError

	def _get_neg_sample(self, true_ents):
		neg_list = []
		neg_num = 0
		while neg_num < config.neg_size:  # 采 neg_size 个负样本
			neg_sample = np.random.randint(self.__num_ents, size=self.__neg_sample_size)
			mask = np.in1d(neg_sample, true_ents, assume_unique=True, invert=True)
			neg_sample = neg_sample[mask]  # 滤除 False 处的值
			neg_list.append(neg_sample)  # 内容为 array
			neg_num += neg_sample.size
		neg_sample = np.concatenate(neg_list)[:config.neg_size]  # 合并, 去掉多余的
		return neg_sample

	@staticmethod
	def collate_fn(data):
		pos_sample = torch.tensor([_[0] for _ in data])
		neg_sample = torch.tensor([_[1] for _ in data])
		weight = torch.tensor([_[2] for _ in data])
		return pos_sample, neg_sample, weight

	@staticmethod
	def _get_true_ents(triples, mode):
		true_ents = {}
		weights = {}
		for triple in triples:
			if mode == 'head-batch':
				e2, r, e1 = triple
			elif mode == 'tail-batch':
				e1, r, e2 = triple
			else:
				raise ValueError(f'Training batch mode {mode} not supported')
			if (e1, r) not in true_ents:
				true_ents[(e1, r)] = []
			true_ents[(e1, r)].append(e2)
		for e1r, e2s in true_ents.items():
			true_ents[e1r] = np.array(e2s)
			weights[e1r] = 1 / (len(e2s) + config.sampling_bias) ** config.sampling_rate
		return true_ents, weights


class HeadDataset(TrainDataset):
	def __init__(self, triples, num_ents):
		super(HeadDataset, self).__init__(triples, num_ents)
		self.__true_ents, self.__weights = TrainDataset._get_true_ents(self._triples, 'head-batch')

	def __getitem__(self, idx):
		pos_sample = self._triples[idx]
		h, r, t = pos_sample
		neg_heads = self._get_neg_sample(self.__true_ents[(t, r)])
		return pos_sample, neg_heads, self.__weights[(t, r)]


class TailDataset(TrainDataset):
	def __init__(self, triples, num_ents):
		super(TailDataset, self).__init__(triples, num_ents)
		self.__true_ents, self.__weights = TrainDataset._get_true_ents(self._triples, 'tail-batch')

	def __getitem__(self, idx):
		pos_sample = self._triples[idx]
		h, r, t = pos_sample
		neg_tails = self._get_neg_sample(self.__true_ents[(h, r)])
		return pos_sample, neg_tails, self.__weights[(h, r)]


class TestDataset(Dataset):
	def __init__(self, triples, ent_num, true_ents, r_tp):
		self._triples = triples
		self._ent_num = ent_num
		self._true_ents = true_ents
		self._r_tp = r_tp  # 关系类型

	def __len__(self):
		return len(self._triples)

	def __getitem__(self, idx):
		raise NotImplementedError

	@staticmethod
	def collate_fn(data):
		pos_sample = torch.tensor([_[0] for _ in data])
		filter_bias = torch.tensor([_[1] for _ in data])
		rel_tp = np.array([_[2] for _ in data])
		return pos_sample, filter_bias, rel_tp


class TestHead(TestDataset):
	def __init__(self, triples, ent_num, true_ents, r_tp):
		super(TestHead, self).__init__(triples, ent_num, true_ents, r_tp)

	def __getitem__(self, idx):
		sample = self._triples[idx]
		h, r, t = sample
		filter_bias = np.zeros(self._ent_num, dtype=np.float32)
		filter_bias[self._true_ents[(r, t)]] = 10000.0
		filter_bias[h] = 0.0
		return sample, filter_bias, self._r_tp[r]


class TestTail(TestDataset):
	def __init__(self, triples, ent_num, true_ents, r_tp):
		super(TestTail, self).__init__(triples, ent_num, true_ents, r_tp)

	def __getitem__(self, idx):
		sample = self._triples[idx]
		h, r, t = sample
		filter_bias = np.zeros(self._ent_num, dtype=np.float32)
		filter_bias[self._true_ents[(h, r)]] = 10000.0
		filter_bias[t] = 0.0
		return sample, filter_bias, self._r_tp[r]


class BiOneShotIterator:
	def __init__(self, dataloader_head, dataloader_tail):
		self.__iterator_head = self._one_shot_iterator(dataloader_head)
		self.__iterator_tail = self._one_shot_iterator(dataloader_tail)
		self.__step = 0

	def __next__(self):
		self.__step += 1
		if self.__step % 2 == 0:  # head 和 tail 交替返回
			return next(self.__iterator_head), 'head-batch'
		else:
			return next(self.__iterator_tail), 'tail-batch'

	def next(self):
		return self.__next__()

	@staticmethod
	def _one_shot_iterator(dataloader):
		while True:
			for data in dataloader:
				yield data
