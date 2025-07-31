import os
import numpy as np
import supervision as sv
from ultralytics import YOLO
import pyodbc

use_sql = True
#for examiner purposes if they do not wish to set up sql workbench and its relevant database info when running the code

#-------------------------------------connecting to sql-----------------------------
#---------------------------------------------------------------------------------------

def sql_connect():
    driver_name = 'ODBC Driver 17 for SQL Server'
    server_name = r'(localdb)\tomato'
    database_name = 'Tomato'

    connection_string = f"""
        DRIVER={{{driver_name}}};
        SERVER={server_name};
        DATABASE={database_name};
        Trust_Connection=yes;
"""
    connection = pyodbc.connect(connection_string)
    print(connection)

    return connection

#-------------------------------------------main----------------------------------
#---------------------------------------------------------------------------------------

def main():
    #connect to sql server
    current_scan_id = 0
    if use_sql:
        connection = sql_connect()
        cursor = connection.cursor()
        cursor.execute("select TOP 1 ScanID FROM Tomatoes ORDER BY ScanID DESC;") #selects the top scan ID from a descending order of scan ID's. Tells user which scan the results are a part of.
        latest_scan_id = cursor.fetchall()[0][0]
        current_scan_id = latest_scan_id + 1
        print(f"Current ScanID: {current_scan_id}")

    cwd = os.getcwd()

    #get and load first model (TOMATO DETECTION)
    desired_model_1 = ""  # run number (for testing simplicity sake) 6 = unbalanced 100 epoch images, 7 = rebalanced
    model_path_1 = os.path.join(cwd, 'runs', 'detect', f'train{desired_model_1}', 'weights', 'best.pt')
    model1 = YOLO(model_path_1)

    #get and load second model (DISEASE DETECTION)
    desired_model_2 = 15  # run number (for testing simplicity sake) 6 = unbalanced 100 epoch images, 7 = rebalanced
    model_path_2 = os.path.join(cwd, 'Disease Training Output', f'train{desired_model_2}', 'weights', 'best.pt')
    model2 = YOLO(model_path_2)

    #--------------------- Create dict of  classes TOMATO ----------------------

    # dict mapping class_id to class_name
    CLASS_NAMES_DICT_1 = model1.model.names
    print(CLASS_NAMES_DICT_1)

    # the class names we have chosen
    SELECTED_CLASS_NAMES_1 = [
        'b_fully_ripened', 'b_half_ripened', 'b_green',
        'l_fully_ripened', 'l_half_ripened', 'l_green'
    ]

    # class ids matching the class names we have chosen
    SELECTED_CLASS_IDS_1 = [
        {value: key for key, value in CLASS_NAMES_DICT_1.items()}[class_name]
        for class_name in SELECTED_CLASS_NAMES_1
    ]

    #--------------------- Create dict of  classes Disease ----------------------

    # dict mapping class_id to class_name
    CLASS_NAMES_DICT_2 = model2.model.names
    print(CLASS_NAMES_DICT_2)

    # the class names we have chosen
    SELECTED_CLASS_NAMES_2 = [
        'Early Blight', 'Late Blight', 'Leaf Miner',
        'Leaf Mold', 'Mosaic Virus', 'Septoria', 'Spider Mites', 'Yellow Leaf Curl Virus'
    ]

    # class ids matching the class names we have chosen
    SELECTED_CLASS_IDS_2 = [
        {value: key for key, value in CLASS_NAMES_DICT_2.items()}[class_name]
        for class_name in SELECTED_CLASS_NAMES_2
    ]

    #---------------------------------------- Video selection and ByteTracker formulation -----------------------------

    # settings
    #SOURCE_VIDEO_PATH = os.path.join(cwd, "test_videos", "diseaseTest.mp4") #input disease
    SOURCE_VIDEO_PATH = os.path.join(cwd, "test_videos", "TomatosAneta.mp4") #input just tomato
    TARGET_VIDEO_PATH = cwd + "/prediction_results/latest_prediction/result100ESEpochAdamW(tomatoes2).mp4" #save output

    # create BYTETracker instance
    byte_tracker1 = sv.ByteTrack(
        track_activation_threshold=0.25, #confidence needed to start tracking
        lost_track_buffer=60, #for how many frames to keep including lost tracks after dissapearance, match with fps, Good for tracks that dissapear.
        minimum_matching_threshold=0.70,
        frame_rate=60, #frame rate of tracker to match 60fps of vide
        minimum_consecutive_frames=4 #minimum amount of frames a track must exist for to be val
    )
    byte_tracker1.reset()

    byte_tracker2 = sv.ByteTrack(
        track_activation_threshold=0.25, #confidence needed to start tracking
        lost_track_buffer=60, #for how many frames to keep including lost tracks after dissapearance, match with fps, Good for tracks that dissapear.
        minimum_matching_threshold=0.70,
        frame_rate=60, #frame rate of tracker to match 60fps of vide
        minimum_consecutive_frames=4 #minimum amount of frames a track must exist for to be val
    )
    byte_tracker2.reset()

    # create VideoInfo instance
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    # create frame generator
    generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    # create instance of BoxAnnotator, LabelAnnotator, and TraceAnnotator
    box_annotator1 = sv.BoxAnnotator(thickness=4)
    box_annotator2 = sv.BoxAnnotator(thickness=4, color=sv.Color.BLUE) #creates a 2nd box annotator for disease with high contrast blue color
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK)
    #trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50) #trace annotator creates a trailing line to show movement progress. Inapplicable when camera position moves.

    all_tomatoes = {}
    all_diseases = {}
    frames_running = []

    # define callback function to be used in video processing
    def callback(frame: np.ndarray, index: int,) -> np.ndarray:
        annotated_frame = frame.copy()

    #------------------------ TOMATO TRACKING --------------------
        # model prediction on single frame and conversion to supervision Detections
        results = model1(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        # only consider class id from selected_classes defined above
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS_1)]
        # tracking detections
        detections = byte_tracker1.update_with_detections(detections)

        #Extracts valuable info from each track id (each tracked tomato) which I can use to store on the database for insights.
        for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id): #zip combines 3 lists into one for easier accessing
            if tracker_id not in all_tomatoes: #if the tomato with a specific id isnt already in the dictionary, add it
                all_tomatoes[tracker_id] = {
                "track_id": tracker_id,
                "class_id": class_id,
                "class_name": model1.model.names[class_id],
                "confidence": confidence,
                "frame_count": 1}#frame count keeps track how many times its been in frame for average confidence
            else:#if tomato with said ID already exists, sum the confidence and frame count over the frames its visible in.
                all_tomatoes[tracker_id]["confidence"] += confidence #each frame adds confidence scores for a total
                all_tomatoes[tracker_id]["frame_count"] += 1 #increases frame count to keep track of what to divide confidence by to find average

        labels_1 = [
            f"#{tracker_id} {model1.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id, tracker_id
            in zip(detections.confidence, detections.class_id, detections.tracker_id)
        ]
        # annotated_frame = trace_annotator.annotate(
        #     scene=annotated_frame, detections=detections)
        annotated_frame = box_annotator1.annotate(
            scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels_1)

    #-------------------- DISEASE TRACKING --------------------------

        # model prediction on single frame and conversion to supervision Detections
        results2 = model2(frame, verbose=False)[0]
        detections2 = sv.Detections.from_ultralytics(results2)
        # only consider class id from selected_classes defined above
        detections2 = detections2[np.isin(detections2.class_id, SELECTED_CLASS_IDS_2)]
        # tracking detections
        detections2 = byte_tracker2.update_with_detections(detections2)

        # Extracts valuable info from each track id (each tracked disease) which I can use to store on the database for insights.
        for confidence, class_id, tracker_id in zip(detections2.confidence, detections2.class_id,detections2.tracker_id):  # zip combines 3 lists into one for easier accessing
            if tracker_id not in all_diseases:  # if the disease with a specific id isnt already in the dictionary, add it
                all_diseases[tracker_id] = {
                    "track_id": tracker_id,
                    "class_id": class_id,
                    "class_name": model2.model.names[class_id],
                    "confidence": confidence,
                    "frame_count": 1,
                    "appeared_at": round(len(frames_running)/60)}  # frame count keeps track how many times its been in frame for average confidence
            else:  # if disease with said ID already exists, sum the confidence and frame count over the frames its visible in.
                all_diseases[tracker_id]["confidence"] += confidence  # each frame adds confidence scores for a total
                all_diseases[tracker_id][
                    "frame_count"] += 1  # increases frame count to keep track of what to divide confidence by to find average

        labels_2 = [
            f"#{tracker_id} {model2.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id, tracker_id
            in zip(detections2.confidence, detections2.class_id, detections2.tracker_id)
        ]
        # annotated_frame = trace_annotator.annotate(
        #     scene=annotated_frame, detections=detections)
        annotated_frame = box_annotator2.annotate(
            scene=annotated_frame, detections=detections2)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections2, labels=labels_2)

        frames_running.append("x")#increase count of current video time

        # return frame with box and label annotated result
        return annotated_frame

    # process the whole video
    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=callback
    )

    #goes over each dictionary value and updates it with the averaged confidence
    for track_id, tomato in all_tomatoes.items(): #track id correlates to the id of each tomato, tomatoes correlates to the data stored in all_tomatoes
        average_confidence = tomato["confidence"] / tomato["frame_count"]
        tomato["confidence"] = average_confidence
        if use_sql: #if sql tracking is enabled, the record of that tomato will be added
            insert_statement = f"""INSERT INTO Tomatoes (ScanID, TomatoID, TomatoType, ConfidenceTomato)
    VALUES ({current_scan_id}, {tomato["track_id"]}, '{tomato["class_name"]}', {tomato["confidence"]});"""
            print(insert_statement)
            cursor.execute(insert_statement)

    for track_id, disease in all_diseases.items():
        average_confidence = disease["confidence"] / disease["frame_count"]
        disease["confidence"] = average_confidence
        if use_sql:
            insert_statement = f"""INSERT INTO Diseases (ScanID, DiseaseID, DiseaseType, ConfidenceDisease, SecondsIntoVideo)
    VALUES ({current_scan_id}, {disease["track_id"]}, '{disease["class_name"]}', {disease["confidence"]}, {disease["appeared_at"]});"""
            print(insert_statement)
            cursor.execute(insert_statement)

    #commits all changes made to database and closes the cursor
    if use_sql:
        connection.commit()

main()