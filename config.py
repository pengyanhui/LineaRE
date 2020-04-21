import json


class Config(object):
    def __init__(self, config_path):
        with open(config_path, 'r') as fjson:
            config = json.load(fjson)
        self.model = config["model"]
        self.dim = config["dim"]
        self.dim1 = config["dim1"]
        self.gamma = config["gamma"]
        self.beta = config["beta"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.neg_size = config["neg_size"]
        self.regularization = config["regularization"]
        self.test_batch_size = config["test_batch_size"]
        self.data_path = config["data_path"]
        self.save_path = config["save_path"]
        self.cuda = config["cuda"]
        self.max_step = config["max_step"]
        self.valid_step = config["valid_step"]
        self.log_step = config["log_step"]
        self.test_log_step = config["test_log_step"]
        self.init_checkpoint = config["init_checkpoint"]
        self.old_optimizer = config["old_optimizer"]
        self.adversarial = config["adversarial"]
        self.adversarial_temperature = config["adversarial_temperature"]
        self.uni_weight = config["uni_weight"]
        self.cpu_num = config["cpu_num"]
        self.ent_num = config["ent_num"]
        self.rel_num = config["rel_num"]
