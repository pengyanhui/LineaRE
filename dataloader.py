import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    cache_heads = {}
    cache_tails = {}

    def __init__(self, triples, ent_num, rel_num, neg_size, mode):
        self.triples = triples
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.neg_size = neg_size
        self.mode = mode
        self.count_hr, self.count_rt = self.count_frequency(self.triples)
        self.true_heads, self.true_tails = self.get_true_head_and_tail(self.triples)
        TrainDataset.cache_heads, TrainDataset.cache_tails = self.get_cache_head_and_tail(self.triples)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        pos_sample = self.triples[idx]
        head, relation, tail = pos_sample
        if self.mode == "head-batch":
            weight = self.count_rt[(relation, tail)]
            cache = TrainDataset.cache_heads[(relation, tail)]
        else:
            weight = self.count_hr[(head, relation)]
            cache = TrainDataset.cache_tails[(head, relation)]
        weight = torch.sqrt(torch.tensor([1.0 / weight ** 0.5]))
        neg_list = []
        neg_num = 0
        while neg_num < self.neg_size:  # 采 neg_size 个负样本
            neg_sample = np.random.randint(self.ent_num, size=self.neg_size * 2)
            if self.mode == "head-batch":
                mask = np.in1d(  # neg_sample 在 true_head 中, 则相应的位置为 True
                    neg_sample,
                    self.true_heads[(relation, tail)],  # np.array 的意义
                    assume_unique=True,
                    invert=True  # True 变 False, False 变 True
                )
            elif self.mode == "tail-batch":
                mask = np.in1d(
                    neg_sample,
                    self.true_tails[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError("Training batch mode %s not supported" % self.mode)
            neg_sample = neg_sample[mask]  # 滤除 False 处的值
            neg_list.append(neg_sample)  # 内容为 array
            neg_num += neg_sample.size

        neg_sample = np.concatenate(neg_list)[:self.neg_size]  # 合并, 去掉多余的
        if cache.size == 0:
            cache = neg_sample
        neg_sample = np.concatenate([neg_sample, cache])
        neg_sample = torch.from_numpy(neg_sample).long()
        pos_sample = torch.tensor(pos_sample)

        return pos_sample, neg_sample, weight, self.mode

    @staticmethod
    def collate_fn(data):
        pos_sample = torch.stack([_[0] for _ in data], dim=0)
        neg_sample = torch.stack([_[1] for _ in data], dim=0)
        weight = torch.cat([_[2] for _ in data], dim=0)
        return pos_sample, neg_sample, weight, data[0][3]

    @staticmethod
    def count_frequency(triples, start=4):
        """
        The frequency will be used for subsampling like word2vec
        """
        count_hr = {}
        count_rt = {}
        for head, relation, tail in triples:
            if (head, relation) not in count_hr:
                count_hr[(head, relation)] = start
            else:
                count_hr[(head, relation)] += 1

            if (relation, tail) not in count_rt:
                count_rt[(relation, tail)] = start
            else:
                count_rt[(relation, tail)] += 1
        return count_hr, count_rt

    @staticmethod
    def get_true_head_and_tail(triples):
        true_head = {}
        true_tail = {}
        # 统计 {hr:true_tails, rt:true_heads}
        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = set()
            true_tail[(head, relation)].add(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = set()
            true_head[(relation, tail)].add(head)
        # 变 np.array, 利于过滤负采样中的正样本
        for rt in true_head:
            true_head[rt] = np.array(list(true_head[rt]))
        for hr in true_tail:
            true_tail[hr] = np.array(list(true_tail[hr]))
        return true_head, true_tail

    @staticmethod
    def get_cache_head_and_tail(triples):
        true_head = {}
        true_tail = {}
        # 统计 {hr:true_tails, rt:true_heads}
        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = np.array([])
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = np.array([])
        # 变 np.array, 利于过滤负采样中的正样本
        for rt in true_head:
            true_head[rt] = np.array(true_head[rt])
        for hr in true_tail:
            true_tail[hr] = np.array(true_tail[hr])
        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, ent_num, mode, rtp):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.ent_num = ent_num
        self.mode = mode
        self.rtp = rtp  # 关系类型

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        if self.mode == "head-batch":
            tmp = [
                0.0
                if (rand_head, relation, tail) not in self.triple_set
                else 100.0
                for rand_head in range(self.ent_num)
            ]
            tmp[head] = 0.0
        elif self.mode == "tail-batch":
            tmp = [
                0.0
                if (head, relation, rand_tail) not in self.triple_set
                else 100.0
                for rand_tail in range(self.ent_num)
            ]
            tmp[tail] = 0.0
        else:
            raise ValueError("negative batch mode %s not supported" % self.mode)

        filter_bias = torch.tensor(tmp)
        pos_sample = torch.tensor((head, relation, tail))

        return pos_sample, filter_bias, self.mode, torch.tensor([self.rtp[relation]])

    @staticmethod
    def collate_fn(data):
        pos_sample = torch.stack([_[0] for _ in data], dim=0)
        filter_bias = torch.stack([_[1] for _ in data], dim=0)
        rel_tp = torch.cat([_[3] for _ in data], dim=0)
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
        """
        顾名思义, 一次发射一个 batch
        """
        while True:
            for data in dataloader:
                yield data
