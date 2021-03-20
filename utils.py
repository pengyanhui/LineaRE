import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as func
from torch.utils.data import DataLoader

from config import config
from dataloader import TrainDataset, BidirectionalOneShotIterator, TestDataset


def read_elements(file_path):
    with open(file_path, 'r') as f:
        elements2id = {}
        for line in f:
            e_id, e_str = line.strip().split('\t')
            elements2id[e_str] = int(e_id)
    return elements2id


def read_triples(file_path, ent2id, rel2id):
    triples = []
    with open(file_path, 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            triples.append((ent2id[h], rel2id[r], ent2id[t]))
    return triples


def get_true_ents(triples):
    true_heads = {}
    true_tails = {}
    for h, r, t in triples:
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


def rel_type(triples):
    count_r = {}
    count_h = {}
    count_t = {}
    for h, r, t in triples:
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


def save_model(model, optimizer, save_vars):
    # 保存 config
    config_dict = vars(config)
    with open(os.path.join(config.save_path, "config.json"), 'w') as fjson:
        json.dump(config_dict, fjson)
    # 保存某些变量、模型参数、优化器参数
    torch.save(
        {
            **save_vars,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        },
        os.path.join(config.save_path, "checkpoint")
    )
    # 保存 embedding
    ent_embd = model.ent_embd.weight.detach().cpu().numpy()
    np.save(
        os.path.join(config.save_path, "ent_embd"),
        ent_embd
    )
    rel_embd = model.rel_embd.weight.detach().cpu().numpy()
    np.save(
        os.path.join(config.save_path, "rel_embd"),
        rel_embd
    )

    if config.model == "TransH":
        wr = model.wr.weight.detach().cpu().numpy()
        np.save(
            os.path.join(config.save_path, "wr"),
            wr
        )
    elif config.model == "TransR":
        mr = model.mr.weight.detach().cpu().numpy()
        np.save(
            os.path.join(config.save_path, "mr"),
            mr
        )
    elif config.model == "TransD":
        ent_p = model.ent_p.weight.detach().cpu().numpy()
        np.save(
            os.path.join(config.save_path, "ent_p"),
            ent_p
        )
        rel_p = model.rel_p.weight.detach().cpu().numpy()
        np.save(
            os.path.join(config.save_path, "rel_p"),
            rel_p
        )
    elif config.model == "STransE":
        mr1 = model.mr1.weight.detach().cpu().numpy()
        np.save(
            os.path.join(config.save_path, "mr1"),
            mr1
        )
        mr2 = model.mr2.weight.detach().cpu().numpy()
        np.save(
            os.path.join(config.save_path, "mr2"),
            mr2
        )
    elif config.model == "LineaRE":
        wrh = model.wrh.weight.detach().cpu().numpy()
        np.save(
            os.path.join(config.save_path, "wrh"),
            wrh
        )
        wrt = model.wrt.weight.detach().cpu().numpy()
        np.save(
            os.path.join(config.save_path, "wrt"),
            wrt
        )
    elif config.model == "ComplEx":
        ent_embd_im = model.ent_embd_im.weight.detach().cpu().numpy()
        np.save(
            os.path.join(config.save_path, "ent_embd_im"),
            ent_embd_im
        )
        rel_embd_im = model.rel_embd_im.weight.detach().cpu().numpy()
        np.save(
            os.path.join(config.save_path, "rel_embd_im"),
            rel_embd_im
        )
    elif "RotatE" in config.model:
        ent_embd_im = model.ent_embd_im.weight.detach().cpu().numpy()
        np.save(
            os.path.join(config.save_path, "ent_embd_im"),
            ent_embd_im
        )


def set_logger():
    log_file = os.path.join(config.save_path, "train.log")
    logging.basicConfig(
        format="%(asctime)s %(levelname)-5s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
        filemode="w"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    for metric in metrics:
        logging.info("%s %s at step %d: %f" % (mode, metric, step, metrics[metric]))


def train_data_iterator(train_triples, ent_num):
    modes = ["head-batch", "tail-batch"]
    datasets = [
        DataLoader(
            TrainDataset(train_triples, ent_num, config.neg_size, mode),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=TrainDataset.collate_fn
        )
        for mode in modes
    ]
    return BidirectionalOneShotIterator(datasets[0], datasets[1])


def test_data_sets(test_triples, true_all_heads, true_all_tails, ent_num, r_tp):
    modes = ["head-batch", "tail-batch"]
    mode_ht = {"head-batch": true_all_heads, "tail-batch": true_all_tails}
    test_dataset_list = [
        DataLoader(
            TestDataset(test_triples, mode_ht[mode], ent_num, mode, r_tp),
            batch_size=config.test_batch_size,
            num_workers=4,
            collate_fn=TestDataset.collate_fn
        )
        for mode in modes
    ]
    return test_dataset_list


def get_optim(optim_method, model, lr):
    if optim_method == "Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )
    elif optim_method == "SGD":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )
    else:
        raise ValueError("optimizer %s not supported" % optim_method)
    return optimizer


def train_step(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    pos_sample, neg_sample, weight, mode = data
    pos_sample = pos_sample.cuda()
    neg_sample = neg_sample.cuda()
    weight = weight.cuda()

    pos_score = model(pos_sample)
    neg_score = model(pos_sample, neg_sample, mode)
    # neg_score = (func.softmax(-neg_score * config.alpha, dim=-1).detach()
    #              * func.softplus(-neg_score, beta=config.beta)).sum(dim=-1)
    neg_score = func.softplus(-neg_score, beta=config.beta)
    adv = neg_score.detach()
    neg_score = torch.sum(adv * neg_score, dim=-1) / torch.sum(adv, dim=-1)

    pos_score = pos_score.squeeze(dim=-1)
    pos_score = func.softplus(pos_score, beta=config.beta)

    if config.uni_weight:
        pos_sample_loss = pos_score.mean()
        neg_sample_loss = neg_score.mean()
    else:
        pos_sample_loss = (weight * pos_score).sum() / weight.sum()
        neg_sample_loss = (weight * neg_score).sum() / weight.sum()
    loss = (pos_sample_loss + neg_sample_loss) / 2

    regularization_log = {}
    if config.model == "TransE":
        ent_reg = torch.abs(model.ent_embd.weight.norm(dim=-1) - 1).mean()
        loss += ent_reg * config.regularization
        regularization_log["ent_reg"] = ent_reg.item()
    elif config.model == "TransH":
        ent_reg = torch.relu(model.ent_embd.weight.norm(dim=-1) - 1).mean()
        wr_norm_1 = torch.abs(model.wr.weight.norm(dim=-1) ** 2 - 1.0).mean()
        wrr = func.relu(
            (
                    (model.wr.weight * model.rel_embd.weight).sum(dim=-1)
                    / model.rel_embd.weight.norm(dim=-1)
            ) ** 2
            - 0.0001
        ).mean()
        loss += wr_norm_1 * config.regularization
        loss += wrr * config.regularization
        regularization_log["ent_reg"] = ent_reg.item()
        regularization_log["wr_norm_1"] = wr_norm_1.item()
        regularization_log["wrr_reg"] = wrr.item()
    elif config.model == "DistMult":
        ent_reg = torch.sum(model.ent_embd.weight ** 2, dim=-1).mean()
        rel_reg = torch.sum(model.rel_embd.weight ** 2, dim=-1).mean()
        loss += ent_reg * config.regularization
        loss += rel_reg * config.regularization
        regularization_log["ent_reg"] = ent_reg.item()
        regularization_log["rel_reg"] = rel_reg.item()
    elif config.model == "ComplEx":
        ent_reg = torch.sum(model.ent_embd.weight ** 2, dim=-1).mean()
        rel_reg = torch.sum(model.rel_embd.weight ** 2, dim=-1).mean()
        ent_im_reg = torch.sum(model.ent_embd_im.weight ** 2, dim=-1).mean()
        rel_im_reg = torch.sum(model.rel_embd_im.weight ** 2, dim=-1).mean()
        loss += ent_reg * config.regularization
        loss += rel_reg * config.regularization
        loss += ent_im_reg * config.regularization
        loss += rel_im_reg * config.regularization
        regularization_log["ent_reg"] = ent_reg.item()
        regularization_log["rel_reg"] = rel_reg.item()
        regularization_log["ent_im_reg"] = ent_im_reg.item()
        regularization_log["rel_im_reg"] = rel_im_reg.item()
    elif config.model == "TransD":
        ent_reg = torch.sum(model.ent_embd.weight ** 2, dim=-1)
        rel_reg = torch.sum(model.rel_embd.weight ** 2, dim=-1)
        reg = torch.cat([ent_reg, rel_reg]).mean()
        loss += reg * config.regularization
        ent_p_reg = torch.sum(model.ent_p.weight ** 2, dim=-1)
        rel_p_reg = torch.sum(model.rel_p.weight ** 2, dim=-1)
        reg = torch.cat([ent_p_reg, rel_p_reg]).mean()
        loss += reg * config.regularization
        regularization_log["ent_reg"] = ent_reg.mean().item()
        regularization_log["rel_reg"] = rel_reg.mean().item()
        regularization_log["ent_p_reg"] = ent_p_reg.mean().item()
        regularization_log["rel_p_reg"] = rel_p_reg.mean().item()
    elif config.model == "TransIJ":
        ent_reg = torch.sum(model.ent_embd.weight ** 2, dim=-1)
        rel_reg = torch.sum(model.rel_embd.weight ** 2, dim=-1)
        reg = torch.cat([ent_reg, rel_reg]).mean()
        loss += reg * config.regularization
        ent_p_reg = torch.sum(model.ent_p.weight ** 2, dim=-1)
        reg = ent_p_reg.mean()
        loss += reg * config.regularization
        regularization_log["ent_reg"] = ent_reg.mean().item()
        regularization_log["rel_reg"] = rel_reg.mean().item()
        regularization_log["ent_p_reg"] = ent_p_reg.mean().item()
    elif config.model in ["LineaRE"]:
        ent_reg = torch.sum(model.ent_embd.weight ** 2, dim=-1)
        rel_reg = torch.sum(model.rel_embd.weight ** 2, dim=-1)
        reg = torch.cat([ent_reg, rel_reg]).mean()
        loss += reg * config.regularization
        regularization_log["ent_reg"] = ent_reg.mean().item()
        regularization_log["rel_reg"] = rel_reg.mean().item()

    loss.backward()
    optimizer.step()

    log = {
        **regularization_log,
        "pos_sample_loss": pos_sample_loss.item(),
        "neg_sample_loss": neg_sample_loss.item(),
        "loss": loss.item()
    }
    return log


def test_step(model, test_dataset_list, detail=False):
    model.eval()

    mode_ents = {"head-batch": 0, "tail-batch": 2}
    step = 0
    total_step = sum([len(dataset) for dataset in test_dataset_list])
    ranks = []
    mode_rtps = []
    metrics = []
    with torch.no_grad():
        for test_dataset in test_dataset_list:
            rtps = []
            for pos_sample, filter_bias, mode, rel_tp in test_dataset:
                pos_sample = pos_sample.cuda()
                filter_bias = filter_bias.cuda()
                all_scores = model(pos_sample, model.ents, mode) + filter_bias
                sort = torch.argsort(all_scores)
                true_ents = pos_sample[:, mode_ents[mode]].unsqueeze(dim=-1)
                batch_ranks = torch.nonzero(torch.eq(sort, true_ents), as_tuple=False)
                ranks.append(batch_ranks[:, 1].detach().cpu().numpy())
                rtps.append(rel_tp)
                if step % config.test_log_step == 0:
                    logging.info("Evaluating the model... (%d/%d)" % (step, total_step))
                step += 1
            mode_rtps.append(rtps)
        ranks = np.concatenate(ranks).astype(np.float) + 1.0
        reciprocal_ranks = np.reciprocal(ranks)
        result = {
            "MR": np.mean(ranks),
            "MRR": np.mean(reciprocal_ranks),
            "HITS@1": np.mean(ranks <= 1.0),
            "HITS@3": np.mean(ranks <= 3.0),
            "HITS@10": np.mean(ranks <= 10.0),
        }
        if not detail:
            return result
        metrics.append(result)
        mode_ranks = [ranks[:ranks.size // 2], ranks[ranks.size // 2:]]
        for i in range(2):
            ranks = mode_ranks[i]
            rtps = np.concatenate(mode_rtps[i])
            for j in range(1, 5):
                mm_ranks = ranks[rtps == j]
                reciprocal_ranks = np.reciprocal(mm_ranks)
                result = {
                    "MR": np.mean(mm_ranks),
                    "MRR": np.mean(reciprocal_ranks),
                    "HITS@1": np.mean(mm_ranks <= 1.0),
                    "HITS@3": np.mean(mm_ranks <= 3.0),
                    "HITS@10": np.mean(mm_ranks <= 10.0),
                }
                metrics.append(result)
    return metrics
