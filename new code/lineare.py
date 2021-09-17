import logging

import numpy as np
import torch
from torch import nn

from configure import config


class LineaRE(nn.Module):
	def __init__(self, num_ents, num_rels):
		super(LineaRE, self).__init__()
		self.register_buffer('gamma', torch.tensor(config.gamma))
		self.register_buffer('ents', torch.arange(num_ents).unsqueeze(dim=0))
		self.ent_embd = nn.Embedding(num_ents, config.dim, max_norm=None if config.multi_gpu else 1.0, sparse=True)
		self.rel_embd = nn.Embedding(num_rels, config.dim, max_norm=None if config.multi_gpu else 1.0, sparse=True)
		self.wrh = nn.Embedding(num_rels, config.dim)
		self.wrt = nn.Embedding(num_rels, config.dim)
		nn.init.xavier_normal_(self.ent_embd.weight)
		nn.init.xavier_normal_(self.rel_embd.weight)
		nn.init.zeros_(self.wrh.weight)
		nn.init.zeros_(self.wrt.weight)
		self.__dropout = nn.Dropout(config.drop_rate)
		self.__softplus = nn.Softplus(beta=config.beta)
		self.__softmax = nn.Softmax(dim=-1)

		self._log_params()

	def forward(self, sample, w_or_fb, ht, neg_ents=None):
		if neg_ents is not None:
			return self._train(sample, w_or_fb, ht, neg_ents)
		else:
			return self._test(sample, w_or_fb, ht)

	def _train(self, sample, weight, ht, neg_ent):
		h, r, t, wh, wt = self._get_pos_embd(sample)
		neg_embd = self.ent_embd(neg_ent)

		score = self.__dropout(wh * h + r - wt * t)
		pos_score = torch.norm(score, p=config.norm_p, dim=-1) - self.gamma
		pos_score = self.__softplus(torch.squeeze(pos_score, dim=-1))

		if ht == 'head-batch':
			score = self.__dropout(wh * neg_embd + (r - wt * t))
		elif ht == 'tail-batch':
			score = self.__dropout((wh * h + r) - wt * neg_embd)
		else:
			raise ValueError(f'mode {ht} not supported')
		neg_score = self.gamma - torch.norm(score, p=config.norm_p, dim=-1)
		neg_prob = self.__softmax(neg_score * config.alpha).detach()
		neg_score = torch.sum(neg_prob * self.__softplus(neg_score), dim=-1)

		pos_loss = weight * pos_score
		neg_loss = weight * neg_score
		ent_reg, rel_reg = self._regularize()

		return ent_reg, rel_reg, pos_loss, neg_loss

	def _test(self, sample, filter_bias, ht):
		h, r, t, wh, wt = self._get_pos_embd(sample)
		if ht == 'head-batch':
			score = wh * self.ent_embd.weight + (r - wt * t)
		elif ht == 'tail-batch':
			score = (wh * h + r) - wt * self.ent_embd.weight
		else:
			raise ValueError(f'mode {ht} not supported')
		score = torch.norm(score, p=config.norm_p, dim=-1) + filter_bias
		return torch.argsort(score)

	def _regularize(self):
		ent_reg = torch.norm(self.ent_embd.weight, p=2, dim=-1)
		rel_reg = torch.norm(self.rel_embd.weight, p=2, dim=-1)
		return ent_reg, rel_reg

	def _get_pos_embd(self, pos_sample):
		h = self.ent_embd(pos_sample[:, 0]).unsqueeze(dim=1)
		r = self.rel_embd(pos_sample[:, 1]).unsqueeze(dim=1)
		t = self.ent_embd(pos_sample[:, 2]).unsqueeze(dim=1)
		wh = self.wrh(pos_sample[:, 1]).unsqueeze(dim=1)
		wt = self.wrt(pos_sample[:, 1]).unsqueeze(dim=1)
		return h, r, t, wh, wt

	def _log_params(self):
		logging.info('>>> Model Parameter Configuration:')
		for name, param in self.named_parameters():
			logging.info(f'Parameter {name}: {str(param.size())}, require_grad = {str(param.requires_grad)}')

	@staticmethod
	def train_step(model, optimizer, data):
		model.train()
		optimizer.zero_grad()
		batch, ht = data
		sample, neg_ents, weight = batch
		ent_reg, rel_reg, pos_loss, neg_loss = model(sample, weight, ht, neg_ents)
		weight_sum = torch.sum(weight)
		pos_loss = torch.sum(pos_loss) / weight_sum
		neg_loss = torch.sum(neg_loss) / weight_sum
		loss = (pos_loss + neg_loss) / 2
		loss += torch.cat([ent_reg ** 2, rel_reg ** 2]).mean() * config.regularization
		loss.backward()
		optimizer.step()
		log = {
			'ent_reg': ent_reg.mean().item(),
			'rel_reg': rel_reg.mean().item(),
			'pos_sample_loss': pos_loss.item(),
			'neg_sample_loss': neg_loss.item(),
			'loss': loss.item()
		}
		return log

	@staticmethod
	def test_step(model, test_dataset_list, detail=False):
		def get_result(ranks_):
			return {
				'MR': np.mean(ranks_),
				'MRR': np.mean(np.reciprocal(ranks_)),
				'HITS@1': np.mean(ranks_ <= 1.0),
				'HITS@3': np.mean(ranks_ <= 3.0),
				'HITS@10': np.mean(ranks_ <= 10.0),
			}

		model.eval()
		mode_ents = {'head-batch': 0, 'tail-batch': 2}
		step = 0
		total_step = sum([len(dataset[0]) for dataset in test_dataset_list])
		ranks = []
		mode_rtps = []
		metrics = []
		with torch.no_grad():
			for test_dataset, mode in test_dataset_list:
				rtps = []
				for pos_sample, filter_bias, rel_tp in test_dataset:
					pos_sample = pos_sample.to(config.device)
					filter_bias = filter_bias.to(config.device)
					sort = model(pos_sample, filter_bias, mode)
					true_ents = pos_sample[:, mode_ents[mode]].unsqueeze(dim=-1)
					batch_ranks = torch.nonzero(torch.eq(sort, true_ents), as_tuple=False)
					ranks.append(batch_ranks[:, 1].detach().cpu().numpy())
					rtps.append(rel_tp)
					if step % config.test_log_step == 0:
						logging.info(f'Evaluating the model... ({step:d}/{total_step:d})')
					step += 1
				mode_rtps.append(rtps)
			ranks = np.concatenate(ranks).astype(np.float32) + 1.0
			result = get_result(ranks)
			if not detail:
				return result
			metrics.append(result)
			mode_ranks = [ranks[:ranks.size // 2], ranks[ranks.size // 2:]]
			for i in range(2):
				mode_ranks_i = mode_ranks[i]
				rtps = np.concatenate(mode_rtps[i])
				for j in range(1, 5):
					ranks_tp = mode_ranks_i[rtps == j]
					result = get_result(ranks_tp)
					metrics.append(result)
		return metrics
