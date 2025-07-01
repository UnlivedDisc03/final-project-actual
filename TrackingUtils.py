import yolox
import os
import numpy as np
import supervision as sv
from ultralytics import YOLO
import matplotlib
import ultralytics

cwd = os.getcwd()
#sys.path.append(f"{cwd}/ByteTrack")
print("yolox.__version__:", yolox.__version__)
print("supervision.__version__:", sv.__version__)


# settings
desired_model = 22 #run number (for testing simplicity sake) 6=unbalanced 100 epoch images, 7 = rebalanced
model_path = os.path.join(cwd, 'runs', 'detect', f'train{desired_model}', 'weights', 'best.pt')
model = YOLO(model_path)

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
print(CLASS_NAMES_DICT)

# the class names we have chosen
SELECTED_CLASS_NAMES = ['b_fully_ripened', 'b_half_ripened', 'b_green', 'l_fully_ripened', 'l_half_ripened', 'l_green']

# class ids matching the class names we have chosen
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name
    in SELECTED_CLASS_NAMES
]

# create frame generator
test_video_path = os.path.join(cwd, 'test_videos')
chosen_video = os.path.join(test_video_path, '17.mp4')

# create frame generator
generator = sv.get_video_frames_generator(chosen_video)
# create instance of BoxAnnotator and LabelAnnotator
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK)
# acquire first video frame
iterator = iter(generator)
frame = next(iterator)
# model prediction on single frame and conversion to supervision Detections
results = model(frame, verbose=False)[0]

# convert to Detections
detections = sv.Detections.from_ultralytics(results)
# only consider class id from selected_classes define above
detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]

# format custom labels
labels = [
    f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
    for confidence, class_id in zip(detections.confidence, detections.class_id)
]

# annotate and display frame
annotated_frame = frame.copy()
annotated_frame = box_annotator.annotate(
scene=annotated_frame, detections=detections)
annotated_frame = label_annotator.annotate(
    scene=annotated_frame, detections=detections, labels=labels)

sv.plot_image(annotated_frame, (16, 16))
