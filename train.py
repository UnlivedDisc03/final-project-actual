import torch
import numpy as np
from dataset import Dataset
from ultralytics import YOLO
import os

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

    model = YOLO("yolov8n.pt")
    results = model.train(data=dataset.dataset_path, epochs=25, imgsz=640, device=0)
