import os
from ultralytics import YOLO
import random
import torch

def inference():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    cwd = os.getcwd()
    desired_model = 6 #run number (for testing simplicity sake) 6=unbalanced 100 epoch images, 7 = rebalanced
    model_path = os.path.join(cwd, 'runs', 'detect', f'train{desired_model}', 'weights', 'best.pt')

    model = YOLO(model_path)
    model.to(device)

    test_images_path = os.path.join(cwd, 'test_images')

    image_list = []
    for image in os.listdir(test_images_path):
        image_list.append(image)

    chosen_image = os.path.join(test_images_path, random.choice(image_list))
    results = model(chosen_image)

    results[0].show()




inference()
