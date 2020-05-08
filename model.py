import torch
import torch.nn as nn
from math import pi

from config import config


class Model(nn.Module):
    def __init__(self, ent_num, rel_num):
        super(Model, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(config.gamma), requires_grad=False)
        self.ents = nn.Parameter(torch.arange(ent_num).unsqueeze(dim=0), requires_grad=False)
        self.ent_embd = nn.Embedding(ent_num, config.ent_dim)
        self.rel_embd = nn.Embedding(rel_num, config.rel_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embd.weight)
        nn.init.xavier_uniform_(self.rel_embd.weight)

    def get_pos_embd(self, pos_sample):
        h = self.ent_embd(pos_sample[:, 0]).unsqueeze(dim=1)
        r = self.rel_embd(pos_sample[:, 1]).unsqueeze(dim=1)
        t = self.ent_embd(pos_sample[:, 2]).unsqueeze(dim=1)
        return h, r, t

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        raise NotImplementedError


class TransE(Model):
    def __init__(self, ent_num, rel_num):
        super(TransE, self).__init__(ent_num, rel_num)
        self.ent_embd = nn.Embedding(ent_num, config.ent_dim)
        nn.init.xavier_uniform_(self.ent_embd.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h, r, t = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score = neg_embd + (r - t)
            elif mode == "tail-batch":
                score = (h + r) - neg_embd
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            score = h + r - t
        score = torch.norm(score, p=config.norm_p, dim=-1) - self.gamma
        return score


class TransH(Model):
    def __init__(self, ent_num, rel_num):
        super(TransH, self).__init__(ent_num, rel_num)
        self.wr = nn.Embedding(rel_num, config.rel_dim)
        nn.init.xavier_uniform_(self.wr.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h, r, t, w = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            wr_neg = (w * neg_embd).sum(dim=-1, keepdim=True)
            wr_neg_wr = wr_neg * w
            if mode == "head-batch":
                wr_t = (w * t).sum(dim=-1, keepdim=True)
                wr_t_wr = wr_t * w
                score = (neg_embd - wr_neg_wr) + (r - (t - wr_t_wr))
            elif mode == "tail-batch":
                wr_h = (w * h).sum(dim=-1, keepdim=True)
                wr_h_wr = wr_h * w
                score = ((h - wr_h_wr) + r) - (neg_embd - wr_neg_wr)
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            wr_h = (w * h).sum(dim=-1, keepdim=True)
            wr_h_wr = wr_h * w
            wr_t = (w * t).sum(dim=-1, keepdim=True)
            wr_t_wr = wr_t * w
            score = (h - wr_h_wr) + r - (t - wr_t_wr)
        score = torch.norm(score, p=config.norm_p, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        h, r, t = super(TransH, self).get_pos_embd(pos_sample)
        w = self.wr(pos_sample[:, 1]).unsqueeze(dim=1)
        return h, r, t, w


class TransR(Model):
    def __init__(self, ent_num, rel_num):
        super(TransR, self).__init__(ent_num, rel_num)
        self.mr = nn.Embedding(rel_num, config.ent_dim * config.rel_dim)
        nn.init.xavier_uniform_(self.mr.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h, r, t, m = self.get_pos_embd(pos_sample)
        m = m.view(-1, 1, config.rel_dim, config.ent_dim)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            mr_neg = torch.matmul(m, neg_embd.unsqueeze(dim=-1)).squeeze(dim=-1)
            if mode == "head-batch":
                mr_t = torch.matmul(m, t.unsqueeze(dim=-1)).squeeze(dim=-1)
                score = mr_neg + (r - mr_t)
            elif mode == "tail-batch":
                mr_h = torch.matmul(m, h.unsqueeze(dim=-1)).squeeze(dim=-1)
                score = (mr_h + r) - mr_neg
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            mr_h = torch.matmul(m, h.unsqueeze(dim=-1)).squeeze(dim=-1)
            mr_t = torch.matmul(m, t.unsqueeze(dim=-1)).squeeze(dim=-1)
            score = mr_h + r - mr_t
        score = torch.norm(score, p=config.norm_p, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        h, r, t = super(TransR, self).get_pos_embd(pos_sample)
        m = self.mr(pos_sample[:, 1])
        return h, r, t, m


class TransD(Model):
    def __init__(self, ent_num, rel_num):
        super(TransD, self).__init__(ent_num, rel_num)
        self.ent_p = nn.Embedding(ent_num, config.ent_dim)
        self.rel_p = nn.Embedding(rel_num, config.rel_dim)
        nn.init.xavier_uniform_(self.ent_p.weight)
        nn.init.xavier_uniform_(self.rel_p.weight)

    @staticmethod
    def ent_p_rel(ent):
        if config.ent_dim == config.rel_dim:
            return ent
        elif config.ent_dim < config.rel_dim:
            cat = torch.zeros(ent.shape[0], ent.shape[1], config.rel_dim - config.ent_dim)
            if config.cuda:
                cat = cat.cuda()
            return torch.cat([ent, cat], dim=-1)
        else:
            return ent[:, :, :config.rel_dim]

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h, r, t, hp, rp, tp = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd, np = self.get_neg_embd(neg_sample)
            np_neg = (np * neg_embd).sum(dim=-1, keepdim=True)
            np_neg_rp = np_neg * rp
            np_neg_rp_n = np_neg_rp + TransD.ent_p_rel(neg_embd)
            if mode == "head-batch":
                tp_t = (tp * t).sum(dim=-1, keepdim=True)
                tp_t_rp = tp_t * rp
                tp_t_rp_t = tp_t_rp + TransD.ent_p_rel(t)
                score = np_neg_rp_n + (r - tp_t_rp_t)
            elif mode == "tail-batch":
                hp_h = (hp * h).sum(dim=-1, keepdim=True)
                hp_h_rp = hp_h * rp
                hp_h_rp_h = hp_h_rp + TransD.ent_p_rel(h)
                score = (hp_h_rp_h + r) - np_neg_rp_n
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            hp_h = (hp * h).sum(dim=-1, keepdim=True)
            hp_h_rp = hp_h * rp
            hp_h_rp_h = hp_h_rp + TransD.ent_p_rel(h)
            tp_t = (tp * t).sum(dim=-1, keepdim=True)
            tp_t_rp = tp_t * rp
            tp_t_rp_t = tp_t_rp + TransD.ent_p_rel(t)
            score = hp_h_rp_h + r - tp_t_rp_t
        score = torch.norm(score, p=config.norm_p, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        h, r, t = super(TransD, self).get_pos_embd(pos_sample)
        hp = self.ent_p(pos_sample[:, 0]).unsqueeze(dim=1)
        rp = self.rel_p(pos_sample[:, 1]).unsqueeze(dim=1)
        tp = self.ent_p(pos_sample[:, 2]).unsqueeze(dim=1)
        return h, r, t, hp, rp, tp

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample), self.ent_p(neg_sample)


class STransE(Model):
    def __init__(self, ent_num, rel_num):
        super(STransE, self).__init__(ent_num, rel_num)
        self.mr1 = nn.Embedding(rel_num, config.ent_dim * config.rel_dim)
        self.mr2 = nn.Embedding(rel_num, config.ent_dim * config.rel_dim)
        nn.init.xavier_uniform_(self.mr1.weight)
        nn.init.xavier_uniform_(self.mr2.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h, r, t, m1, m2 = self.get_pos_embd(pos_sample)
        m1 = m1.view(-1, 1, config.rel_dim, config.ent_dim)
        m2 = m2.view(-1, 1, config.rel_dim, config.ent_dim)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                mr_neg = torch.matmul(m1, neg_embd.unsqueeze(dim=-1)).squeeze(dim=-1)
                mr_t = torch.matmul(m2, t.unsqueeze(dim=-1)).squeeze(dim=-1)
                score = mr_neg + (r - mr_t)
            elif mode == "tail-batch":
                mr_h = torch.matmul(m1, h.unsqueeze(dim=-1)).squeeze(dim=-1)
                mr_neg = torch.matmul(m2, neg_embd.unsqueeze(dim=-1)).squeeze(dim=-1)
                score = (mr_h + r) - mr_neg
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            mr_h = torch.matmul(m1, h.unsqueeze(dim=-1)).squeeze(dim=-1)
            mr_t = torch.matmul(m2, t.unsqueeze(dim=-1)).squeeze(dim=-1)
            score = mr_h + r - mr_t
        score = torch.norm(score, p=config.norm_p, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        h, r, t = super(STransE, self).get_pos_embd(pos_sample)
        m1 = self.mr1(pos_sample[:, 1]).unsqueeze(dim=1)
        m2 = self.mr2(pos_sample[:, 1]).unsqueeze(dim=1)
        return h, r, t, m1, m2


class LineaRE(Model):
    def __init__(self, ent_num, rel_num):
        super(LineaRE, self).__init__(ent_num, rel_num)
        self.wrh = nn.Embedding(rel_num, config.rel_dim)
        self.wrt = nn.Embedding(rel_num, config.rel_dim)
        nn.init.zeros_(self.wrh.weight)
        nn.init.zeros_(self.wrt.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h, r, t, wh, wt = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score = wh * neg_embd + (r - wt * t)
            elif mode == "tail-batch":
                score = (wh * h + r) - wt * neg_embd
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            score = wh * h + r - wt * t
        score = torch.norm(score, p=config.norm_p, dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        h, r, t = super(LineaRE, self).get_pos_embd(pos_sample)
        wh = self.wrh(pos_sample[:, 1]).unsqueeze(dim=1)
        wt = self.wrt(pos_sample[:, 1]).unsqueeze(dim=1)
        return h, r, t, wh, wt


class DistMult(Model):
    def __init__(self, ent_num, rel_num):
        super(DistMult, self).__init__(ent_num, rel_num)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h, r, t = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score = neg_embd * (r * t)
            elif mode == "tail-batch":
                score = (h * r) * neg_embd
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            score = h * r * t
        return torch.sum(score, dim=-1)


class ComplEx(Model):
    def __init__(self, ent_num, rel_num):
        super(ComplEx, self).__init__(ent_num, rel_num)
        self.ent_embd_im = nn.Embedding(ent_num, config.ent_dim)
        self.rel_embd_im = nn.Embedding(rel_num, config.rel_dim)
        nn.init.xavier_uniform_(self.ent_embd_im.weight)
        nn.init.xavier_uniform_(self.rel_embd_im.weight)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h_re, r_re, t_re, h_im, r_im, t_im = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_re, neg_im = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score_re = t_re * r_re + t_im * r_im
                score_im = t_re * r_im - t_im * r_re
                score = neg_re * score_re - neg_im * score_im
            elif mode == "tail-batch":
                score_re = h_re * r_re - h_im * r_im
                score_im = h_re * r_im + h_im * r_re
                score = score_re * neg_re + score_im * neg_im
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            score_re = h_re * r_re - h_im * r_im
            score_im = h_re * r_im + h_im * r_re
            score = score_re * t_re + score_im * t_im
        return torch.sum(score, dim=-1)

    def get_pos_embd(self, pos_sample):
        h_re, r_re, t_re = super(ComplEx, self).get_pos_embd(pos_sample)
        h_im = self.ent_embd_im(pos_sample[:, 0]).unsqueeze(dim=1)
        r_im = self.rel_embd_im(pos_sample[:, 1]).unsqueeze(dim=1)
        t_im = self.ent_embd_im(pos_sample[:, 2]).unsqueeze(dim=1)
        return h_re, r_re, t_re, h_im, r_im, t_im

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample), self.ent_embd_im(neg_sample)


class RotatE(Model):
    def __init__(self, ent_num, rel_num):
        super(RotatE, self).__init__(ent_num, rel_num)
        self.ent_embd_im = nn.Embedding(ent_num, config.ent_dim)
        nn.init.xavier_uniform_(self.ent_embd_im.weight)
        nn.init.uniform_(self.rel_embd.weight, a=-pi, b=pi)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        h_re, h_im, r, t_re, t_im = self.get_pos_embd(pos_sample)
        rel_re = torch.cos(r)
        rel_im = torch.sin(r)
        if neg_sample is not None:
            neg_embd_re, neg_embd_im = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score_re = t_re * rel_re + t_im * rel_im
                score_im = t_im * rel_re - t_re * rel_im
            elif mode == "tail-batch":
                score_re = h_re * rel_re - h_im * rel_im
                score_im = h_re * rel_im + h_im * rel_re
            else:
                raise ValueError("mode %s not supported" % mode)
            score_re = score_re - neg_embd_re
            score_im = score_im - neg_embd_im
        else:
            score_re = h_re * rel_re - h_im * rel_im
            score_im = h_re * rel_im + h_im * rel_re
            score_re = score_re - t_re
            score_im = score_im - t_im
        score = torch.stack([score_re, score_im], dim=0).norm(dim=0)
        score = score.sum(dim=-1) - self.gamma
        return score

    def get_pos_embd(self, pos_sample):
        h_re, r, t_re = super(RotatE, self).get_pos_embd(pos_sample)
        h_im = self.ent_embd_im(pos_sample[:, 0]).unsqueeze(1)
        t_im = self.ent_embd_im(pos_sample[:, 2]).unsqueeze(1)
        return h_re, h_im, r, t_re, t_im

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample), self.ent_embd_im(neg_sample)
