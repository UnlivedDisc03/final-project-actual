import torch
import numpy as np
from torch.nn.functional import dropout
from torch.optim import AdamW
from torchvision.transforms.v2.functional import perspective
from dataset import Dataset, DiseaseDataset
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import os
import matplotlib.pyplot as plt
from collections import Counter
from utilities import verify_class_balance, plot_class_balance
from logger import Logger
from config import Config
from dataset import DiseaseDataset

#issue with multiple OpenMP instances, this bypasses it as well as importing torch before numpy each time.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

configs = Config()

#TRY WITH DIFFERENT FRUITS AND VEGGIES at the end
#Think of something thatll introduce novelty

def training_tomato():
    logger = Logger(logger_name='Tomato' if configs.train_tomato else "Disease") #sets appropriate name for logger and initializes it.
    if configs.train_tomato:
        dataset = Dataset()
    else:
        dataset = DiseaseDataset()
    #model = YOLO("yolov8n.pt", task="detect")
    model = YOLO("yolo12n.pt", task="detect")
    dataset_path = os.path.join(os.getcwd(), 'dataset', 'dataset.yaml') if configs.train_tomato else os.path.join(os.getcwd(), 'disease data', 'data.yaml')

    results = model.train(
        data=dataset_path,
        epochs=configs.epochs,
        imgsz=configs.image_size,
        optimizer='AdamW',
        dropout=0.15, #stronger dropout,
        minimum_epochs=100, #custom additon made to YOLO's trainer.py to allow minimum training epochs
        patience=3,
        lr0=0.001,#initial learning rate, 0.001 needed for AdamW, reduced to prevent overfitting. All previois experience dictates lowerd learning rate with L2 always ahd the best effects on overfitting
        lrf=0.00001,#final learning rate
        weight_decay=0.00075, #slightly stronger weight decay
        workers=4,
        device=0,
        project='Tomato Training Output' if configs.train_tomato else 'Disease Training Output',
        seed=configs.seed,
        degrees=configs.degrees,
        translate=configs.translate,
        scale=configs.scale,
        shear=configs.shear,
        perspective=configs.perspective,
        fliplr=configs.fliplr,
        mosaic=configs.mosaic,
        erasing=configs.erasing,
        hsv_h=configs.hsv_h,
        hsv_s=configs.hsv_s,
        hsv_v=configs.hsv_v
    )

    # confirms which dataset was used (weighted or standard)
    if configs.train_tomato:
        print(model.trainer.train_loader.dataset)
        if configs.weighted_dataset:
            model.trainer.train_loader.dataset.weights = model.trainer.train_loader.dataset.calculate_weights()
            model.trainer.train_loader.dataset.probabilities = model.trainer.train_loader.dataset.calculate_probabilities()
            # Get class counts in weighted mode
            model.trainer.train_loader.dataset.train_mode = True
            weighted_counts = verify_class_balance(model.trainer.train_loader.dataset)
            # Get class counts in default mode
            model.trainer.train_loader.dataset.train_mode = False
            default_counts = verify_class_balance(model.trainer.train_loader.dataset)
            # Plot the comparison
            plot_class_balance(weighted_counts, default_counts, list(model.trainer.train_loader.dataset.data["names"].values()))

#-------------------------------DISEASE TRAINING-----------------------------------

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Selected Device: " + str(device))
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)

    training_tomato()