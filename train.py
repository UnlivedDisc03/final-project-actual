import torch
import numpy as np
from dataset import Dataset
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import os
import matplotlib.pyplot as plt
from collections import Counter
from utilities import verify_class_balance, plot_class_balance

#issue with multiple OpenMP instances, this bypasses it as well as importing torch before numpy each time.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Selected Device: " + str(device))
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)

    dataset = Dataset()

    #selects yolov8 model architecture and trains it
    model = YOLO("yolov8n.pt")
    results = model.train(data=dataset.dataset_path, epochs=1, imgsz=640, device=0)

    #confirms which dataset was used (weighted or standard)
    print(model.trainer.train_loader.dataset)
    model.trainer.train_loader.dataset.weights = model.trainer.train_loader.dataset.calculate_weights()
    model.trainer.train_loader.dataset.probabilities = model.trainer.train_loader.dataset.calculate_probabilities()

    # Get class counts in weighted mode
    model.trainer.train_loader.dataset.train_mode = True
    weighted_counts = verify_class_balance(model.trainer.train_loader.dataset, num_samples=1000)

    # Get class counts in default mode
    model.trainer.train_loader.dataset.train_mode = False
    default_counts = verify_class_balance(model.trainer.train_loader.dataset, num_samples=1000)

    # Plot the comparison
    plot_class_balance(weighted_counts, default_counts, list(model.trainer.train_loader.dataset.data["names"].values()))
