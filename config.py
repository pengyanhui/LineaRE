import json


class Config(object):
    def __init__(self, config_path):
        with open(config_path, 'r') as fjson:
            json_config = json.load(fjson)
        self.model = json_config["model"]
        self.ent_dim = json_config["ent_dim"]
        self.rel_dim = json_config["rel_dim"]
        self.norm_p = json_config["norm_p"]
        self.beta = json_config["beta"]
        self.gamma = json_config["gamma"]
        self.learning_rate = json_config["learning_rate"]
        self.batch_size = json_config["batch_size"]
        self.neg_size = json_config["neg_size"]
        self.regularization = json_config["regularization"]
        self.test_batch_size = json_config["test_batch_size"]
        self.data_path = json_config["data_path"]
        self.save_path = json_config["save_path"]
        self.cuda = json_config["cuda"]
        self.max_step = json_config["max_step"]
        self.valid_step = json_config["valid_step"]
        self.log_step = json_config["log_step"]
        self.test_log_step = json_config["test_log_step"]
        self.init_checkpoint = json_config["init_checkpoint"]
        self.use_old_optimizer = json_config["use_old_optimizer"]
        self.adversarial = json_config["adversarial"]
        self.adversarial_temperature = json_config["adversarial_temperature"]
        self.uni_weight = json_config["uni_weight"]
        self.cpu_num = json_config["cpu_num"]


config = Config("config/config.json")
