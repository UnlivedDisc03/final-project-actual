import yaml

class Config:
    def __init__(self):
        with open('config.yaml', 'r') as settings:
            configs = yaml.safe_load(settings)

            #train settings
            self.agg_function = configs['train']['agg_func']
            self.epochs = configs['train']['epochs']
            self.weighted_dataset = configs['train']['weighted_dataset']
            self.image_size = configs['train']['image_size']
            self.yolo = configs['train']['yolo']
            self.train_tomato = configs['train']['train_tomato']

            #augmentation settings
            self.seed = configs['augmentations']['seed']
            self.degrees = configs['augmentations']['degrees']
            self.translate = configs['augmentations']['translate']
            self.scale = configs['augmentations']['scale']
            self.shear = configs['augmentations']['shear']
            self.perspective = configs['augmentations']['perspective']
            self.fliplr = configs['augmentations']['fliplr']
            self.mosaic = configs['augmentations']['mosaic']
            self.erasing = configs['augmentations']['erasing']
            self.hsv_h = configs['augmentations']['hsv_h']
            self.hsv_s = configs['augmentations']['hsv_s']
            self.hsv_v = configs['augmentations']['hsv_v']