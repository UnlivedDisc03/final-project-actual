import os
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Setup paths and model
cwd = os.getcwd()
desired_model = 7  # run number (for testing simplicity sake) 6 = unbalanced 100 epoch images, 7 = rebalanced
model_path = os.path.join(cwd, 'runs', 'detect', f'train{desired_model}', 'weights', 'best.pt')
model = YOLO(model_path)

# dict mapping class_id to class_name
CLASS_NAMES_DICT = model.model.names
print(CLASS_NAMES_DICT)

# the class names we have chosen
SELECTED_CLASS_NAMES = [
    'b_fully_ripened', 'b_half_ripened', 'b_green',
    'l_fully_ripened', 'l_half_ripened', 'l_green'
]

# class ids matching the class names we have chosen
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name in SELECTED_CLASS_NAMES
]

# settings
SOURCE_VIDEO_PATH = os.path.join(cwd, "test_videos", "TomatosAneta2.mp4")
TARGET_VIDEO_PATH = cwd + "/prediction_results/latest_prediction/trackedResult.mp4"

# create BYTETracker instance
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25, #confidence needed to start tracking
    lost_track_buffer=60, #for how many frames to keep including lost tracks after dissapearance, match with fps, Good for tracks that dissapear.
    minimum_matching_threshold=0.70,
    frame_rate=60, #frame rate of tracker to match 60fps of vide
    minimum_consecutive_frames=4 #minimum amount of frames a track must exist for to be valid
)
byte_tracker.reset()

# create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# create instance of BoxAnnotator, LabelAnnotator, and TraceAnnotator
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK)
#trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50) #trace annotator creates a trailing line to show movement progress. Inapplicable when camera position moves.

# define callback function to be used in video processing
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    # model prediction on single frame and conversion to supervision Detections
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    # only consider class id from selected_classes defined above
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    # tracking detections
    detections = byte_tracker.update_with_detections(detections)
    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]
    annotated_frame = frame.copy()
    # annotated_frame = trace_annotator.annotate(
    #     scene=annotated_frame, detections=detections)
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)

    # return frame with box and label annotated result
    return annotated_frame

# process the whole video
sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback
)
