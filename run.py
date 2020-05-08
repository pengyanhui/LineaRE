import logging
import os

import torch

from config import config
from model import TransE, TransH, TransR, TransD, STransE, LineaRE, DistMult, ComplEx, RotatE
from utils import set_logger, read_elements, read_triples, log_metrics, save_model, train_data_iterator, \
    get_optim, train_step, test_step, rel_type


def train(model, triples, ent_num):
    logging.info("Start Training...")
    logging.info("batch_size = %d" % config.batch_size)
    logging.info("dim = %d" % config.ent_dim)
    logging.info("gamma = %f" % config.gamma)

    current_lr = config.learning_rate
    train_triples, valid_triples, test_triples = triples
    all_true_triples = train_triples + valid_triples + test_triples
    rtp = rel_type(train_triples)

    optimizer = get_optim("Adam", model, current_lr)
    train_iterator = train_data_iterator(train_triples, ent_num)

    if config.init_checkpoint:
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(os.path.join(config.save_path, "checkpoint"))
        init_step = checkpoint["step"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        if config.use_old_optimizer:
            current_lr = checkpoint["current_lr"]
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        init_step = 1

    max_hit1 = 0.0
    max_mrr = 0.0
    training_logs = []
    # Training Loop
    for step in range(init_step, config.max_step):
        log = train_step(model, optimizer, next(train_iterator))
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
            logging.info("---------------Evaluating on Valid Dataset---------------")
            metrics = test_step(model, valid_triples, all_true_triples, ent_num, rtp)
            metrics, metrics1, metrics2, metrics3, metrics4, metrics5, metrics6, metrics7, metrics8 = metrics
            logging.info("----------------Overall Results----------------")
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
                save_model(model, optimizer, save_variable_list)
            elif current_lr > 0.0000011:
                current_lr *= 0.1
                logging.info("Change learning_rate to %f at step %d" % (current_lr, step))
                optimizer = get_optim("Adam", model, current_lr)
            else:
                logging.info("-------------------Training End-------------------")
                break
    # best state
    checkpoint = torch.load(os.path.join(config.save_path, "checkpoint"))
    model.load_state_dict(checkpoint["model_state_dict"])
    step = checkpoint["step"]
    logging.info("-----------------Evaluating on Test Dataset-------------------")
    metrics = test_step(model, test_triples, all_true_triples, ent_num, rtp)
    metrics, metrics1, metrics2, metrics3, metrics4, metrics5, metrics6, metrics7, metrics8 = metrics
    logging.info("----------------Overall Results----------------")
    log_metrics("Test", step, metrics)
    logging.info("-----------Prediction Head... 1-1 -------------")
    log_metrics("Test", step, metrics1)
    logging.info("-----------Prediction Head... 1-M -------------")
    log_metrics("Test", step, metrics2)
    logging.info("-----------Prediction Head... M-1 -------------")
    log_metrics("Test", step, metrics3)
    logging.info("-----------Prediction Head... M-M -------------")
    log_metrics("Test", step, metrics4)
    logging.info("-----------Prediction Tail... 1-1 -------------")
    log_metrics("Test", step, metrics5)
    logging.info("-----------Prediction Tail... 1-M -------------")
    log_metrics("Test", step, metrics6)
    logging.info("-----------Prediction Tail... M-1 -------------")
    log_metrics("Test", step, metrics7)
    logging.info("-----------Prediction Tail... M-M -------------")
    log_metrics("Test", step, metrics8)


def run():
    set_logger()

    # load data
    ent_path = os.path.join(config.data_path, "entities.dict")
    rel_path = os.path.join(config.data_path, "relations.dict")
    ent2id = read_elements(ent_path)
    rel2id = read_elements(rel_path)
    ent_num = len(ent2id)
    rel_num = len(rel2id)
    train_triples = read_triples(os.path.join(config.data_path, "train.txt"), ent2id, rel2id)
    valid_triples = read_triples(os.path.join(config.data_path, "valid.txt"), ent2id, rel2id)
    test_triples = read_triples(os.path.join(config.data_path, "test.txt"), ent2id, rel2id)
    logging.info("#ent_num: %d" % ent_num)
    logging.info("#rel_num: %d" % rel_num)
    logging.info("#train triple num: %d" % len(train_triples))
    logging.info("#valid triple num: %d" % len(valid_triples))
    logging.info("#test triple num: %d" % len(test_triples))
    logging.info("#Model: %s" % config.model)

    # 创建模型
    kge_model = TransE(ent_num, rel_num)
    if config.model == "TransH":
        kge_model = TransH(ent_num, rel_num)
    elif config.model == "TransR":
        kge_model = TransR(ent_num, rel_num)
    elif config.model == "TransD":
        kge_model = TransD(ent_num, rel_num)
    elif config.model == "STransE":
        kge_model = STransE(ent_num, rel_num)
    elif config.model == "LineaRE":
        kge_model = LineaRE(ent_num, rel_num)
    elif config.model == "DistMult":
        kge_model = DistMult(ent_num, rel_num)
    elif config.model == "ComplEx":
        kge_model = ComplEx(ent_num, rel_num)
    elif config.model == "RotatE":
        kge_model = RotatE(ent_num, rel_num)

    if config.cuda:
        kge_model = kge_model.cuda()
    logging.info("Model Parameter Configuration:")
    for name, param in kge_model.named_parameters():
        logging.info("Parameter %s: %s, require_grad = %s" % (name, str(param.size()), str(param.requires_grad)))

    # 训练
    train(
        model=kge_model,
        triples=(train_triples, valid_triples, test_triples),
        ent_num=ent_num
    )


if __name__ == "__main__":
    run()
