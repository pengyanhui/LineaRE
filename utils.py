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
    rtp = {}
    for r in range(len(count_r)):
        tph = count_r[r] / len(count_h[r])
        hpt = count_r[r] / len(count_t[r])
        if hpt < 1.5:
            if tph < 1.5:
                rtp[r] = 1  # 1-1
            else:
                rtp[r] = 2  # 1-M
        else:
            if tph < 1.5:
                rtp[r] = 3  # M-1
            else:
                rtp[r] = 4  # M-M
    return rtp


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
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
        filemode="w"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    for metric in metrics:
        logging.info("%s %s at step %d: %f" % (mode, metric, step, metrics[metric]))


def train_data_iterator(train_triples, ent_num):
    dataloader_head = DataLoader(
        TrainDataset(train_triples, ent_num, config.neg_size, "head-batch"),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=max(0, config.cpu_num // 3),
        collate_fn=TrainDataset.collate_fn
    )
    dataloader_tail = DataLoader(
        TrainDataset(train_triples, ent_num, config.neg_size, "tail-batch"),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=max(0, config.cpu_num // 3),
        collate_fn=TrainDataset.collate_fn
    )
    return BidirectionalOneShotIterator(dataloader_head, dataloader_tail)


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
    if config.cuda:
        pos_sample = pos_sample.cuda()
        neg_sample = neg_sample.cuda()
        weight = weight.cuda()

    pos_score = model(pos_sample)
    neg_score = model(pos_sample, neg_sample, mode)
    if config.adversarial:
        neg_score = (func.softmax(-neg_score * config.adversarial_temperature, dim=-1).detach()
                     * func.softplus(-neg_score, beta=config.beta)).sum(dim=-1)
    else:
        neg_score = func.softplus(-neg_score, beta=config.beta).mean(dim=-1)
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
    if config.regularization != 0.0:
        if config.model == "TransE":
            ent_reg = torch.abs(model.ent_embd.weight.norm(p=2, dim=-1) - 1.0).mean()
            loss += ent_reg * config.regularization
            regularization_log["ent_reg"] = ent_reg.item()
        if config.model == "TransH":
            ent_reg = torch.relu(model.ent_embd.weight.norm(p=2, dim=-1) - 1.0).mean()
            wr_norm_1 = torch.abs(model.wr.weight.norm(p=2, dim=-1) - 1.0).mean()
            wrr = func.relu(
                (
                        (model.wr.weight * model.rel_embd.weight).sum(dim=-1)
                        / model.rel_embd.weight.norm(p=2, dim=-1)
                ) ** 2
                - 0.0001
            ).mean()
            loss += ent_reg * config.regularization
            loss += wr_norm_1 * config.regularization
            loss += wrr * config.regularization
            regularization_log["ent_reg"] = ent_reg.item()
            regularization_log["wr_norm_1"] = wr_norm_1.item()
            regularization_log["wrr_reg"] = wrr.item()
        elif config.model == "TransR" or config.model == "TransD" or config.model == "STransE":
            ent_reg = torch.relu(model.ent_embd.weight.norm(p=2, dim=-1) - 1.0).mean()
            rel_reg = torch.relu(model.rel_embd.weight.norm(p=2, dim=-1) - 1.0).mean()
            loss += ent_reg * config.regularization
            loss += rel_reg * config.regularization
            regularization_log["ent_reg"] = ent_reg.item()
            regularization_log["rel_reg"] = rel_reg.item()
        elif config.model == "DistMult":
            ent_reg = torch.mean(model.ent_embd.weight ** 2)
            rel_reg = torch.mean(model.rel_embd.weight ** 2)
            loss += ent_reg * config.regularization
            loss += rel_reg * config.regularization
            regularization_log["ent_reg"] = ent_reg.item()
            regularization_log["rel_reg"] = rel_reg.item()
        elif config.model == "ComplEx":
            ent_reg = torch.mean(model.ent_embd.weight ** 2)
            rel_reg = torch.mean(model.rel_embd.weight ** 2)
            ent_im_reg = torch.mean(model.ent_embd_im.weight ** 2)
            rel_im_reg = torch.mean(model.rel_embd_im.weight ** 2)
            loss += ent_reg * config.regularization
            loss += ent_im_reg * config.regularization
            loss += rel_reg * config.regularization
            loss += rel_im_reg * config.regularization
            regularization_log["ent_reg"] = ent_reg.item()
            regularization_log["ent_im_reg"] = ent_im_reg.item()
            regularization_log["rel_reg"] = rel_reg.item()
            regularization_log["rel_im_reg"] = rel_im_reg.item()
        elif config.model == "LineaRE":
            ent_reg = model.ent_embd.weight.norm(p=2, dim=-1).mean()
            loss += ent_reg * config.regularization
            regularization_log["ent_reg"] = ent_reg.item()

    loss.backward()
    optimizer.step()

    log = {
        **regularization_log,
        "pos_sample_loss": pos_sample_loss.item(),
        "neg_sample_loss": neg_sample_loss.item(),
        "loss": loss.item()
    }

    return log


def test_step(model, test_triples, all_true_triples, ent_num, rtp):
    model.eval()

    test_dataloader_head = DataLoader(
        TestDataset(
            triples=test_triples,
            all_true_triples=all_true_triples,
            ent_num=ent_num,
            mode="head-batch",
            rtp=rtp
        ),
        batch_size=config.test_batch_size,
        num_workers=max(0, config.cpu_num // 3),
        collate_fn=TestDataset.collate_fn
    )
    test_dataloader_tail = DataLoader(
        TestDataset(
            triples=test_triples,
            all_true_triples=all_true_triples,
            ent_num=ent_num,
            mode="tail-batch",
            rtp=rtp
        ),
        batch_size=config.test_batch_size,
        num_workers=max(0, config.cpu_num // 3),
        collate_fn=TestDataset.collate_fn
    )
    test_dataset_list = [test_dataloader_head, test_dataloader_tail]

    logs = []
    logs_h_1 = []
    logs_h_2 = []
    logs_h_3 = []
    logs_h_4 = []
    logs_t_1 = []
    logs_t_2 = []
    logs_t_3 = []
    logs_t_4 = []
    step = 0
    total_step = sum([len(dataset) for dataset in test_dataset_list])
    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for pos_sample, filter_bias, mode, rel_tp in test_dataset:
                if config.cuda:
                    pos_sample = pos_sample.cuda()
                    filter_bias = filter_bias.cuda()

                batch_size = pos_sample.size(0)

                score = model(pos_sample, model.ents, mode)
                score += filter_bias
                # Explicitly sort all the entities to ensure that there is no test exposure bias
                argsort = torch.argsort(score, dim=1, descending=False)

                if mode == "head-batch":
                    pos_arg = pos_sample[:, 0]
                elif mode == "tail-batch":
                    pos_arg = pos_sample[:, 2]
                else:
                    raise ValueError("mode %s not supported" % mode)

                for i in range(batch_size):
                    # Notice that argsort is not ranking
                    ranking = (argsort[i, :] == pos_arg[i]).nonzero()
                    # ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    result = {
                        "MRR": 1.0 / ranking,
                        "MR": float(ranking),
                        "HITS@1": 1.0 if ranking <= 1 else 0.0,
                        "HITS@3": 1.0 if ranking <= 3 else 0.0,
                        "HITS@10": 1.0 if ranking <= 10 else 0.0,
                    }
                    logs.append(result)

                    if mode == "head-batch":
                        if rel_tp[i].item() == 1:
                            logs_h_1.append(result)
                        elif rel_tp[i].item() == 2:
                            logs_h_2.append(result)
                        elif rel_tp[i].item() == 3:
                            logs_h_3.append(result)
                        else:
                            logs_h_4.append(result)
                    else:
                        if rel_tp[i].item() == 1:
                            logs_t_1.append(result)
                        elif rel_tp[i].item() == 2:
                            logs_t_2.append(result)
                        elif rel_tp[i].item() == 3:
                            logs_t_3.append(result)
                        else:
                            logs_t_4.append(result)

                if step % config.test_log_step == 0:
                    logging.info("Evaluating the model... (%d/%d)" % (step, total_step))
                step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        metrics1 = {}
        for metric in logs_h_1[0].keys():
            metrics1[metric] = sum([log[metric] for log in logs_h_1]) / len(logs_h_1)
        metrics2 = {}
        for metric in logs_h_2[0].keys():
            metrics2[metric] = sum([log[metric] for log in logs_h_2]) / len(logs_h_2)
        metrics3 = {}
        for metric in logs_h_3[0].keys():
            metrics3[metric] = sum([log[metric] for log in logs_h_3]) / len(logs_h_3)
        metrics4 = {}
        for metric in logs_h_4[0].keys():
            metrics4[metric] = sum([log[metric] for log in logs_h_4]) / len(logs_h_4)
        metrics5 = {}
        for metric in logs_t_1[0].keys():
            metrics5[metric] = sum([log[metric] for log in logs_t_1]) / len(logs_t_1)
        metrics6 = {}
        for metric in logs_t_2[0].keys():
            metrics6[metric] = sum([log[metric] for log in logs_t_2]) / len(logs_t_2)
        metrics7 = {}
        for metric in logs_t_3[0].keys():
            metrics7[metric] = sum([log[metric] for log in logs_t_3]) / len(logs_t_3)
        metrics8 = {}
        for metric in logs_t_4[0].keys():
            metrics8[metric] = sum([log[metric] for log in logs_t_4]) / len(logs_t_4)

    return metrics, metrics1, metrics2, metrics3, metrics4, metrics5, metrics6, metrics7, metrics8
