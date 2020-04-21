import logging
import os

import torch

from config import Config
from model import TransE, TransH, TransR, TransD, STransE, WTransE, DistMult, ComplEx, RotatE, RotatE2
from utils import set_logger, read_entities_and_relations, read_triple, log_metrics, save_model, train_data_iterator, \
    get_optim, train_step, test_step, rel_type


def train(config, model, triples):
    logging.info("Start Training...")
    logging.info("batch_size = %d" % config.batch_size)
    logging.info("dim = %d" % config.dim)
    logging.info("gamma = %f" % config.gamma)

    current_lr = config.lr
    train_triples, valid_triples, test_triples = triples
    all_true_triples = train_triples + valid_triples + test_triples
    rtp = rel_type(train_triples)

    optimizer = get_optim("Adam", model, current_lr)
    train_iterator = train_data_iterator(train_triples, config)

    if config.init_checkpoint:
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(os.path.join(config.save_path, "checkpoint"))
        init_step = checkpoint["step"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        if config.old_optimizer:
            current_lr = checkpoint["current_lr"]
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        init_step = 1

    max_hit1 = 0.0
    max_mrr = 0.0
    training_logs = []
    # Training Loop
    for step in range(init_step, config.max_step):
        log = train_step(model, optimizer, next(train_iterator), config)
        training_logs.append(log)

        # log
        if step % config.log_step == 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
            metrics["gamma"] = model.gamma.item()
            log_metrics("Training average", step, metrics)
            training_logs = []

        # valid
        if step % config.valid_step == 0:
            logging.info("Evaluating on Test Dataset...")
            metrics = test_step(model, test_triples, all_true_triples, config, rtp=rtp)
            metrics, metrics1, metrics2, metrics3, metrics4, metrics5, metrics6, metrics7, metrics8 = metrics
            log_metrics("Valid", step, metrics)
            logging.info("-----------Prediction Head... 1-1 -------------")
            log_metrics("Valid", step, metrics1)
            logging.info("-----------Prediction Head... 1-M -------------")
            log_metrics("Valid", step, metrics2)
            logging.info("-----------Prediction Head... M-1 -------------")
            log_metrics("Valid", step, metrics3)
            logging.info("-----------Prediction Head... M-M -------------")
            log_metrics("Valid", step, metrics4)
            logging.info("-----------Prediction Tail... 1-1 -------------")
            log_metrics("Valid", step, metrics5)
            logging.info("-----------Prediction Tail... 1-M -------------")
            log_metrics("Valid", step, metrics6)
            logging.info("-----------Prediction Tail... M-1 -------------")
            log_metrics("Valid", step, metrics7)
            logging.info("-----------Prediction Tail... M-M -------------")
            log_metrics("Valid", step, metrics8)

            if metrics["HITS@1"] >= max_hit1 or metrics["MRR"] >= max_mrr:
                if metrics["HITS@1"] > max_hit1:
                    max_hit1 = metrics["HITS@1"]
                if metrics["MRR"] > max_mrr:
                    max_mrr = metrics["MRR"]
                save_variable_list = {
                    "step": step,
                    "current_lr": current_lr,
                }
                save_model(model, optimizer, save_variable_list, config)
            elif current_lr > 0.0000011:
                current_lr *= 0.1
                logging.info("Change learning_rate to %f at step %d" % (current_lr, step))
                optimizer = get_optim("Adam", model, current_lr)


def main(config):
    set_logger(config)

    # load data
    ent_path = os.path.join(config.data_path, "entities.dict")
    rel_path = os.path.join(config.data_path, "relations.dict")
    ent2id, rel2id = read_entities_and_relations(ent_path, rel_path)
    config.ent_num = len(ent2id)
    config.rel_num = len(rel2id)
    train_triples = read_triple(os.path.join(config.data_path, "train.txt"), ent2id, rel2id)
    valid_triples = read_triple(os.path.join(config.data_path, "valid.txt"), ent2id, rel2id)
    test_triples = read_triple(os.path.join(config.data_path, "test.txt"), ent2id, rel2id)
    logging.info("#ent_num: %d" % config.ent_num)
    logging.info("#rel_num: %d" % config.rel_num)
    logging.info("#train triple num: %d" % len(train_triples))
    logging.info("#valid triple num: %d" % len(valid_triples))
    logging.info("#test triple num: %d" % len(test_triples))
    logging.info("#Model: %s" % config.model)

    # 创建模型
    kge_model = TransE(config)
    if config.model == "TransH":
        kge_model = TransH(config)
    elif config.model == "TransR":
        kge_model = TransR(config)
    elif config.model == "TransD":
        kge_model = TransD(config)
    elif config.model == "STransE":
        kge_model = STransE(config)
    elif config.model == "WTransE":
        kge_model = WTransE(config)
    elif config.model == "DistMult":
        kge_model = DistMult(config)
    elif config.model == "ComplEx":
        kge_model = ComplEx(config)
    elif config.model == "RotatE":
        kge_model = RotatE(config)
    elif config.model == "RotatE2":
        kge_model = RotatE2(config)

    if config.cuda:
        kge_model = kge_model.cuda()
    logging.info("Model Parameter Configuration:")
    for name, param in kge_model.named_parameters():
        logging.info("Parameter %s: %s, require_grad = %s" % (name, str(param.size()), str(param.requires_grad)))

    # 训练
    train(
        config,
        model=kge_model,
        triples=(train_triples, valid_triples, test_triples)
    )
    print(kge_model.gamma)


if __name__ == "__main__":
    main(Config("./config/config.json"))
