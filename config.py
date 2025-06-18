import yaml

class Config:
    def __init__(self):
        with open('config.yaml', 'r') as settings:
            configs = yaml.safe_load(settings)
            #train settings
            self.agg_function = configs['train']['agg_func']
            self.epochs = configs['train']['epochs']
            self.weighted_dataset = configs['train']['weighted_dataset']