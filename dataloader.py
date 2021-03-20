import numpy as np
import torch
from torch.utils.data import Dataset

from config import config


class TrainDataset(Dataset):
    def __init__(self, triples, ent_num, neg_size, mode):
        self.triples = triples
        self.ent_num = ent_num
        self.neg_size = neg_size
        self.neg_sample_size = neg_size + neg_size // 3
        self.mode = mode
        self.true_ents, self.weights = self.get_true_ents(self.triples, mode)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        pos_sample = self.triples[idx]
        if self.mode == "head-batch":
            _, r, e = pos_sample
        else:
            e, r, _ = pos_sample

        neg_list = []
        neg_num = 0
        while neg_num < self.neg_size:  # 采 neg_size 个负样本
            neg_sample = np.random.randint(self.ent_num, size=self.neg_sample_size)
            mask = np.in1d(neg_sample, self.true_ents[(e, r)], assume_unique=True, invert=True)
            neg_sample = neg_sample[mask]  # 滤除 False 处的值
            neg_list.append(neg_sample)  # 内容为 array
            neg_num += neg_sample.size

        neg_sample = np.concatenate(neg_list)[:self.neg_size]  # 合并, 去掉多余的
        neg_sample = torch.from_numpy(neg_sample).long()
        pos_sample = torch.tensor(pos_sample)

        return pos_sample, neg_sample, self.weights[(e, r)], self.mode

    @staticmethod
    def collate_fn(data):
        pos_sample = torch.stack([_[0] for _ in data])
        neg_sample = torch.stack([_[1] for _ in data])
        weight = torch.tensor([_[2] for _ in data])
        return pos_sample, neg_sample, torch.reciprocal(weight), data[0][3]

    @staticmethod
    def get_true_ents(triples, mode):
        true_ents = {}
        weights = {}
        for triple in triples:
            if mode == "head-batch":
                e2, r, e1 = triple
            elif mode == "tail-batch":
                e1, r, e2 = triple
            else:
                raise ValueError("Training batch mode %s not supported" % mode)
            if (e1, r) not in true_ents:
                true_ents[(e1, r)] = []
            true_ents[(e1, r)].append(e2)
        for e1r in true_ents:
            true_ents[e1r] = np.array(true_ents[e1r])
            weights[e1r] = true_ents[e1r].size ** config.sampling_rate
        return true_ents, weights


class TestDataset(Dataset):
    def __init__(self, triples, true_ents, ent_num, mode, r_tp):
        self.triples = triples
        self.ent_num = ent_num
        self.true_ents = true_ents
        self.mode = mode
        self.r_tp = r_tp  # 关系类型

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        sample = self.triples[idx]
        h, r, t = sample
        tmp = np.zeros(self.ent_num, dtype=np.float32)
        if self.mode == "head-batch":
            for hh in self.true_ents[(r, t)]:
                tmp[hh] = 100.0
            tmp[h] = 0.0
        elif self.mode == "tail-batch":
            for tt in self.true_ents[(h, r)]:
                tmp[tt] = 100.0
            tmp[t] = 0.0
        else:
            raise ValueError("negative batch mode %s not supported" % self.mode)
        filter_bias = torch.from_numpy(tmp)
        sample = torch.tensor(sample)
        return sample, filter_bias, self.mode, self.r_tp[r]

    @staticmethod
    def collate_fn(data):
        pos_sample = torch.stack([_[0] for _ in data])
        filter_bias = torch.stack([_[1] for _ in data])
        rel_tp = np.array([_[3] for _ in data])
        return pos_sample, filter_bias, data[0][2], rel_tp


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:  # head 和 tail 交替返回
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data
