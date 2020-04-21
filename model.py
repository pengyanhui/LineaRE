import torch
import torch.nn as nn
from math import pi


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(config.gamma), requires_grad=False)
        self.ents = nn.Parameter(torch.arange(config.ent_num), requires_grad=False)
        self.ent_embd = nn.Embedding(config.ent_num, config.dim)
        self.rel_embd = nn.Embedding(config.rel_num, config.dim)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.ent_embd.weight)
        nn.init.kaiming_uniform_(self.rel_embd.weight)

    def get_pos_embd(self, pos_sample):
        head = self.ent_embd(pos_sample[:, 0]).unsqueeze(dim=1)
        relation = self.rel_embd(pos_sample[:, 1]).unsqueeze(dim=1)
        tail = self.ent_embd(pos_sample[:, 2]).unsqueeze(dim=1)
        return head, relation, tail

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        raise NotImplementedError


class TransE(Model):
    def __init__(self, config):
        super(TransE, self).__init__(config)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        head, relation, tail = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score = neg_embd + (relation - tail)
            elif mode == "tail-batch":
                score = (head + relation) - neg_embd
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            score = head + relation - tail
        score = torch.norm(score, p=1, dim=-1) - self.gamma
        return score


class TransH(Model):
    def __init__(self, config):
        super(TransH, self).__init__(config)
        self.wr = nn.Embedding(config.rel_num, config.dim)
        nn.init.kaiming_uniform_(self.wr.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        head, relation, tail, w = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            wr_neg = (w * neg_embd).sum(dim=-1).unsqueeze(dim=-1)
            wr_neg_wr = wr_neg * w
            if mode == "head-batch":
                wrt = (w * tail).sum(dim=-1).unsqueeze(dim=-1)
                wrtwr = wrt * w
                score = (neg_embd - wr_neg_wr) + (relation - (tail - wrtwr))
            elif mode == "tail-batch":
                wrh = (w * head).sum(dim=-1).unsqueeze(dim=-1)
                wrhwr = wrh * w
                score = ((head - wrhwr) + relation) - (neg_embd - wr_neg_wr)
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            wrh = (w * head).sum(dim=-1).unsqueeze(dim=-1)
            wrhwr = wrh * w
            wrt = (w * tail).sum(dim=-1).unsqueeze(dim=-1)
            wrtwr = wrt * w
            score = (head - wrhwr) + relation - (tail - wrtwr)
        score = torch.norm(score, p=1, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        head, relation, tail = super(TransH, self).get_pos_embd(pos_sample)
        w = self.wr(pos_sample[:, 1]).unsqueeze(dim=1)
        return head, relation, tail, w


class TransR(Model):
    def __init__(self, config):
        super(TransR, self).__init__(config)
        self.rel_embd = nn.Embedding(config.rel_num, config.dim1)
        self.mr = nn.Embedding(config.rel_num, config.dim * config.dim1)
        nn.init.kaiming_uniform_(self.rel_embd.weight)
        nn.init.kaiming_uniform_(self.mr.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        head, relation, tail, m = self.get_pos_embd(pos_sample)
        m = m.view(m.size(0), m.size(1), relation.size(-1), head.size(-1))
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            mr_neg = torch.matmul(m, neg_embd.unsqueeze(dim=-1)).squeeze(dim=-1)
            if mode == "head-batch":
                mrt = torch.matmul(m, tail.unsqueeze(dim=-1)).squeeze(dim=-1)
                score = mr_neg + (relation - mrt)
            elif mode == "tail-batch":
                mrh = torch.matmul(m, head.unsqueeze(dim=-1)).squeeze(dim=-1)
                score = (mrh + relation) - mr_neg
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            mrh = torch.matmul(m, head.unsqueeze(dim=-1)).squeeze(dim=-1)
            mrt = torch.matmul(m, tail.unsqueeze(dim=-1)).squeeze(dim=-1)
            score = mrh + relation - mrt
        score = torch.norm(score, p=1, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        head, relation, tail = super(TransR, self).get_pos_embd(pos_sample)
        m = self.mr(pos_sample[:, 1]).unsqueeze(dim=1)
        return head, relation, tail, m


class TransD(Model):
    def __init__(self, config):
        super(TransD, self).__init__(config)
        self.ent_p = nn.Embedding(config.ent_num, config.dim)
        self.rel_p = nn.Embedding(config.rel_num, config.dim)
        nn.init.kaiming_uniform_(self.ent_p.weight)
        nn.init.kaiming_uniform_(self.rel_p.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        head, relation, tail, hp, rp, tp = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd, np = self.get_neg_embd(neg_sample)
            npn = (np * neg_embd).sum(dim=-1).unsqueeze(dim=-1)
            npnrp = npn * rp
            npnrp_n = npnrp + neg_embd
            if mode == "head-batch":
                tpt = (tp * tail).sum(dim=-1).unsqueeze(dim=-1)
                tptrp = tpt * rp
                tptrp_t = tptrp + tail
                score = npnrp_n + (relation - tptrp_t)
            elif mode == "tail-batch":
                hph = (hp * head).sum(dim=-1).unsqueeze(dim=-1)
                hphrp = hph * rp
                hphrp_h = hphrp + head
                score = (hphrp_h + relation) - npnrp_n
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            hph = (hp * head).sum(dim=-1).unsqueeze(dim=-1)
            hphrp = hph * rp
            hphrp_h = hphrp + head
            tpt = (tp * tail).sum(dim=-1).unsqueeze(dim=-1)
            tptrp = tpt * rp
            tptrp_t = tptrp + tail
            score = hphrp_h + relation - tptrp_t
        score = torch.norm(score, p=1, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        head, relation, tail = super(TransD, self).get_pos_embd(pos_sample)
        hp = self.ent_embd(pos_sample[:, 0]).unsqueeze(dim=1)
        rp = self.rel_embd(pos_sample[:, 1]).unsqueeze(dim=1)
        tp = self.ent_embd(pos_sample[:, 2]).unsqueeze(dim=1)
        return head, relation, tail, hp, rp, tp

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample), self.ent_p(neg_sample)


class STransE(Model):
    def __init__(self, config):
        super(STransE, self).__init__(config)
        self.mr1 = nn.Embedding(config.rel_num, config.dim ** 2)
        self.mr2 = nn.Embedding(config.rel_num, config.dim ** 2)
        nn.init.kaiming_uniform_(self.mr1.weight)
        nn.init.kaiming_uniform_(self.mr2.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        head, relation, tail, m1, m2 = self.get_pos_embd(pos_sample)
        m1 = m1.view(m1.size(0), m1.size(1), relation.size(-1), relation.size(-1))
        m2 = m2.view(m2.size(0), m2.size(1), relation.size(-1), relation.size(-1))
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                mr_neg = torch.matmul(m1, neg_embd.unsqueeze(dim=-1)).squeeze(dim=-1)
                mrt = torch.matmul(m2, tail.unsqueeze(dim=-1)).squeeze(dim=-1)
                score = mr_neg + (relation - mrt)
            elif mode == "tail-batch":
                mrh = torch.matmul(m1, head.unsqueeze(dim=-1)).squeeze(dim=-1)
                mr_neg = torch.matmul(m2, neg_embd.unsqueeze(dim=-1)).squeeze(dim=-1)
                score = (mrh + relation) - mr_neg
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            mrh = torch.matmul(m1, head.unsqueeze(dim=-1)).squeeze(dim=-1)
            mrt = torch.matmul(m2, tail.unsqueeze(dim=-1)).squeeze(dim=-1)
            score = mrh + relation - mrt
        score = torch.norm(score, p=1, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        head, relation, tail = super(STransE, self).get_pos_embd(pos_sample)
        m1 = self.mr1(pos_sample[:, 1]).unsqueeze(dim=1)
        m2 = self.mr2(pos_sample[:, 1]).unsqueeze(dim=1)
        return head, relation, tail, m1, m2


class WTransE(Model):
    def __init__(self, config):
        super(WTransE, self).__init__(config)
        self.wrh = nn.Embedding(config.rel_num, config.dim)
        self.wrt = nn.Embedding(config.rel_num, config.dim)
        nn.init.zeros_(self.wrh.weight)
        nn.init.zeros_(self.wrt.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        head, relation, tail, wh, wt = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score = wh * neg_embd + (relation - wt * tail)
            elif mode == "tail-batch":
                score = (wh * head + relation) - wt * neg_embd
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            score = wh * head + relation - wt * tail
        score = torch.norm(score, p=1, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        head, relation, tail = super(WTransE, self).get_pos_embd(pos_sample)
        wh = self.wrh(pos_sample[:, 1]).unsqueeze(dim=1)
        wt = self.wrt(pos_sample[:, 1]).unsqueeze(dim=1)
        return head, relation, tail, wh, wt


class DistMult(Model):
    def __init__(self, config):
        super(DistMult, self).__init__(config)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        head, relation, tail = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score = neg_embd * (relation * tail)
            elif mode == "tail-batch":
                score = (head * relation) * neg_embd
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            score = head * relation * tail
        return score.sum(dim=-1)


class ComplEx(Model):
    def __init__(self, config):
        super(ComplEx, self).__init__(config)
        self.ent_embd_im = nn.Embedding(config.ent_num, config.dim)
        self.rel_embd_im = nn.Embedding(config.rel_num, config.dim)
        nn.init.kaiming_uniform_(self.ent_embd_im.weight)
        nn.init.kaiming_uniform_(self.rel_embd_im.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        head_re, rel_re, tail_re, head_im, rel_im, tail_im = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_re, neg_im = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score_re = tail_re * rel_re + tail_im * rel_im
                score_im = tail_im * rel_re - tail_re * rel_im
                score = neg_re * score_re - neg_im * score_im
            elif mode == "tail-batch":
                score_re = head_re * rel_re - head_im * rel_im
                score_im = head_re * rel_im + head_im * rel_re
                score = score_re * neg_re + score_im * neg_im
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            score_re = head_re * rel_re - head_im * rel_im
            score_im = head_re * rel_im + head_im * rel_re
            score = score_re * tail_re + score_im * tail_im
        return score.sum(dim=2)

    def get_pos_embd(self, pos_sample):
        head_re, relation_re, tail_re = super(ComplEx, self).get_pos_embd(pos_sample)
        head_im = self.ent_embd(pos_sample[:, 0]).unsqueeze(dim=1)
        relation_im = self.rel_embd(pos_sample[:, 1]).unsqueeze(dim=1)
        tail_im = self.ent_embd(pos_sample[:, 2]).unsqueeze(dim=1)
        return head_re, relation_re, tail_re, head_im, relation_im, tail_im

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample), self.ent_embd_im(neg_sample)


class RotatE(Model):
    def __init__(self, config):
        super(RotatE, self).__init__(config)
        self.ent_embd_im = nn.Embedding(config.ent_num, config.dim)
        nn.init.kaiming_uniform_(self.ent_embd_im.weight)
        nn.init.uniform_(self.rel_embd.weight, a=-pi, b=pi)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        head_re, head_im, relation, tail_re, tail_im = self.get_pos_embd(pos_sample)
        rel_re = torch.cos(relation)
        rel_im = torch.sin(relation)
        if neg_sample is not None:
            neg_embd_re, neg_embd_im = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score_re = tail_re * rel_re + tail_im * rel_im
                score_im = tail_im * rel_re - tail_re * rel_im
            elif mode == "tail-batch":
                score_re = head_re * rel_re - head_im * rel_im
                score_im = head_re * rel_im + head_im * rel_re
            else:
                raise ValueError("mode %s not supported" % mode)
            score_re = score_re - neg_embd_re
            score_im = score_im - neg_embd_im
        else:
            score_re = head_re * rel_re - head_im * rel_im
            score_im = head_re * rel_im + head_im * rel_re
            score_re = score_re - tail_re
            score_im = score_im - tail_im
        score = torch.stack([score_re, score_im], dim=0).norm(dim=0)
        score = score.sum(dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        head_re, relation, tail_re = super(RotatE, self).get_pos_embd(pos_sample)
        head_im = self.ent_embd_im(pos_sample[:, 0]).unsqueeze(1)
        tail_im = self.ent_embd_im(pos_sample[:, 2]).unsqueeze(1)
        return head_re, head_im, relation, tail_re, tail_im

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample), self.ent_embd_im(neg_sample)


class RotatE2(Model):
    def __init__(self, config):
        super(RotatE2, self).__init__(config)
        self.ent_embd_im = nn.Embedding(config.ent_num, config.dim)
        self.rel_weight = nn.Embedding(config.rel_num, config.dim)
        nn.init.kaiming_uniform_(self.ent_embd_im.weight)
        nn.init.uniform_(self.rel_embd.weight, a=-pi, b=pi)
        nn.init.ones_(self.rel_weight.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        head_re, head_im, relation, tail_re, tail_im, w = self.get_pos_embd(pos_sample)
        rel_re = torch.cos(relation)
        rel_im = torch.sin(relation)

        if neg_sample is not None:
            neg_embd_re, neg_embd_im = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score_re = tail_re * rel_re + tail_im * rel_im
                score_im = tail_im * rel_re - tail_re * rel_im
            elif mode == "tail-batch":
                score_re = head_re * rel_re - head_im * rel_im
                score_im = head_re * rel_im + head_im * rel_re
            else:
                raise ValueError("mode %s not supported" % mode)
            score_re = score_re - neg_embd_re
            score_im = score_im - neg_embd_im
        else:
            score_re = head_re * rel_re - head_im * rel_im
            score_im = head_re * rel_im + head_im * rel_re
            score_re = score_re - tail_re
            score_im = score_im - tail_im

        score = torch.cat([score_re, score_im], dim=-1)
        score = torch.norm(score, p=1, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        head_re, relation, tail_re = super(RotatE2, self).get_pos_embd(pos_sample)
        head_im = self.ent_embd_im(pos_sample[:, 0]).unsqueeze(1)
        tail_im = self.ent_embd_im(pos_sample[:, 2]).unsqueeze(1)
        w = self.rel_weight(pos_sample[:, 1]).unsqueeze(1)
        return head_re, head_im, relation, tail_re, tail_im, w

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample), self.ent_embd_im(neg_sample)
