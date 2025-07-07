import torch
import numpy as np
from torchvision.transforms.v2.functional import perspective
from dataset import Dataset
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import os
import matplotlib.pyplot as plt
from collections import Counter
from utilities import verify_class_balance, plot_class_balance
from logger import Logger
from config import Config
from customTrainer import CustomTrainer

#issue with multiple OpenMP instances, this bypasses it as well as importing torch before numpy each time.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

configs = Config()

#TRY WITH DIFFERENT FRUITS AND VEGGIES at the end
#Think of something thatll introduce novelty

def training():
    logger = Logger()
    dataset = Dataset()
    model = YOLO("yolov8n.pt", task="detect")
    dataset_path = os.path.join(os.getcwd(), 'dataset', 'dataset.yaml')

    results = model.train(
        data=dataset.dataset_path,
        epochs=configs.epochs,
        imgsz=configs.image_size,
        device=0,
        project=logger.logger.project,
        seed=42,
        degrees=10, #to account for slight rotations during growth as well as human error in holding phone slightly diagonally
        translate=0.20,
        scale=0.25, #most images appear as close-ups therefore scaling is required to both account for farther shots and close shots.
        shear=0.25, #slight shearing of image improves generalization of viewing angles and tilts
        perspective=0.00005, #slight perspective added to improve robustness with varying camera angles but not too much.
        fliplr=0.5, #good augmentation method for more data to train on
        mosaic=0.33, #combines numerous images together, slicing them up. Could lead to improvements in detecting partially obscured tomatoes
        erasing=0.5, #erases parts of the image, helps reduce over-reliance on certain features increasing model robustness.
        hsv_h=0.0, #disables hue adjustments as colours play a crucial role for deciding ripeness
        hsv_s=0.0,#disables sautration adjustements for same reason
        hsv_v=0.1,#adds slight brightness adjustment to account for sun and shade.
        )

    # confirms which dataset was used (weighted or standard)
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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Selected Device: " + str(device))
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)

    training()