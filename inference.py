import os
from ultralytics import YOLO
import random
import torch
import cv2

def replay_latest():
    cwd = os.getcwd()
    prediction_path = os.path.join(cwd, 'prediction_results', 'latest_prediction', 'trackedResult.mp4')
    print(prediction_path)
    capture = cv2.VideoCapture(prediction_path)

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        original_height, original_width = frame.shape[:2]
        resized_frame = cv2.resize(frame, (original_width // 3, original_height // 3))
        cv2.imshow("YOLOv8 Output", resized_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

def inference():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    cwd = os.getcwd()
    desired_model = 22 #run number (for testing simplicity sake) 6=unbalanced 100 epoch images, 7 = rebalanced
    model_path = os.path.join(cwd, 'runs', 'detect', f'train{desired_model}', 'weights', 'best.pt')

    model = YOLO(model_path)
    model.to(device)

    test_images_path = os.path.join(cwd, 'test_images')
    test_video_path = os.path.join(cwd, 'test_videos')

    image_list = []
    for image in os.listdir(test_images_path):
        image_list.append(image)

    video_list = []
    for vid in os.listdir(test_video_path):
        video_list.append(vid)

    video = True

    if not video:
        chosen_image = os.path.join(test_images_path, random.choice(image_list))
        results = model(chosen_image)
        results[0].show()
    else:
        #chosen_video = os.path.join(test_video_path, random.choice(video_list))
        chosen_video = os.path.join(test_video_path, '17.mp4')
        results = model(source=chosen_video, stream=False, save=True, project='prediction_results', name='latest_prediction', exist_ok=True)

        replay_latest()

#inference()
replay_latest()
