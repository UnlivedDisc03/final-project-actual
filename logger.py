import wandb
import yaml
from config import Config

class Logger:
    def __init__(self, logger_name='Experiment:', project='Final Individual Project'):
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        #systematically generate experiment name to avoid manual labor
        configs = Config()
        yolo_v = "YOLOv" +str(configs.yolo)
        if configs.weighted_dataset:
            balance = "Rebalanced-("+str(configs.agg_function)+")"
        else:
            balance = "unbalanced"

        experiment_name = (yolo_v + "-" + balance)

        logger_name = f'{logger_name}-{experiment_name}'
        logger = wandb.init(project=project, name=logger_name, config=config)
        self.logger = logger
        return

    def get_logger(self):
        return self.logger