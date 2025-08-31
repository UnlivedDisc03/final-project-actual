import os
import numpy as np
import supervision as sv
from ultralytics import YOLO
import pyodbc
from collections import Counter
import cv2
from PIL import Image
import imagehash
import time
import matplotlib.pyplot as plt
import pandas as pd

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

    start = time.time()
    #wipes last runs tomato images
    tomato_folder = os.path.join(os.getcwd(), "DetectedTomatoes")
    for image in os.listdir(tomato_folder): #for each image in tomatofolder
        image_path = os.path.join(tomato_folder, image) #image path is the path of the image
        os.remove(image_path)#delete image

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

    # get and load the tomato detector model.
    desired_model_1 = 'V12-1K' #detector
    model_path_1 = os.path.join(cwd, 'runs', 'detect', f'train{desired_model_1}', 'weights', 'best.pt')
    model1 = YOLO(model_path_1)

    #get and load second model (DISEASE DETECTION)
    desired_model_2 = 15  # run number (for testing simplicity sake) 6 = unbalanced 100 epoch images, 7 = rebalanced
    model_path_2 = os.path.join(cwd, 'Disease Training Output', f'train{desired_model_2}', 'weights', 'best.pt')
    model2 = YOLO(model_path_2)

    # get and load first model (TOMATO RIPENESS DETECTION)
    desired_model_3 = "V12-1K"  # run number (for testing simplicity sake) ""=best, 8=1280 v8, v12 = V12, V12-1K = 1000imgsz
    model_path_3 = os.path.join(cwd, 'runs', 'detect', f'train{desired_model_3}', 'weights', 'best.pt')
    model3 = YOLO(model_path_3)

    #takes class names of model, chooses which ones to use (omits healthy leaf detection)
    CLASS_NAMES_DICT_2 = model2.model.names
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
    SOURCE_VIDEO_PATH = os.path.join(cwd, "test_videos", "TomatoesMixedTRIM.mp4") #input just tomato
    TARGET_VIDEO_PATH = cwd + "/prediction_results/latest_prediction/ImprovedBytetrack.mp4" #save output

    # create BYTETracker instance for tomato
    byte_tracker1 = sv.ByteTrack(
        track_activation_threshold=0.25, #confidence needed to start tracking
        lost_track_buffer=45, #for how many frames to keep including lost tracks after dissapearance, match with fps + 50% Good for tracks that dissapear.
        minimum_matching_threshold=0.70,
        frame_rate=30, #frame rate of tracker to match 60fps of vide
        minimum_consecutive_frames=4 #minimum amount of frames a track must exist for to be val
    )
    byte_tracker1.reset()

    byte_tracker2 = sv.ByteTrack(
        track_activation_threshold=0.70, #confidence needed to start tracking
        lost_track_buffer=45, #for how many frames to keep including lost tracks after dissapearance, match with fps + 50%, Good for tracks that dissapear.
        minimum_matching_threshold=0.40,
        frame_rate=30, #frame rate of tracker to match 60fps of vide
        minimum_consecutive_frames=4 #minimum amount of frames a track must exist for to be val
    )
    byte_tracker2.reset()

    smoother1 = sv.DetectionsSmoother(length=5) #smoother keeps history of detections and provides smoothed predictions for detections
    smoother2 = sv.DetectionsSmoother(length=5) #smoother for disease
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
    tomato_history = {}  # a dictionary to keep track of the mode ripeness.

#-------------------------CALLBACK-------------------------------
    mot_results = []

    # define callback function to be used in video processing
    def callback(frame: np.ndarray, index: int,) -> np.ndarray:
        annotated_frame = frame.copy()

    #------------------------ TOMATO TRACKING --------------------

        # model prediction on single frame and conversion to supervision Detections
        results = model1(frame, verbose=False, iou=0.7, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(results)

        height, width, _ = frame.shape
        margin = int(0.10 * width) #creates a region
        region_of_interest = (margin, 0, width - margin, height) #x, y minimum and x, y maximum values

        #draws the rectangle on the video for user to see region of interest, used for debugging
        x1, y1, x2, y2 = map(int, region_of_interest)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        #takes the x coordiantes of region of interest and bounding box
        def inside_region(bounding_box, region):
            rx1, _, rx2, _ = region_of_interest #establishes coordinates of region of interest corners, skips over y as its not necessary
            x1, _, x2, _ = bounding_box #establishes bounding box coordinates

            #checks if box overlaps region of interest
            return x1 < rx2 and x2 > rx1 #returns true or false based on result

        mask = [inside_region(box, region_of_interest) for box in detections.xyxy]
        detections = detections[mask]

        # tracking detections
        detections = byte_tracker1.update_with_detections(detections)
        detections = smoother1.update_with_detections(detections) #adds smoothing which reduces jittering of detection boxes.

        def save_tomatoes(frame, detections, tracker_id):
            for number, box in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(int, box) #converts float coordinates to int
                crop = frame[y1:y2, x1:x2]  #crop the bounding box region in the frame
                filename = os.path.join("DetectedTomatoes", f"tomato_{tracker_id}.png") #sets name of tomato image
                cv2.imwrite(filename, crop) #saves image

        #Extracts valuable info from each track id (each tracked tomato) which I can use to store on the database for insights.
        for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id): #zip combines 3 lists into one for easier accessing
            x1, y1, x2, y2 = detections.xyxy[0]
            w, h = x2 - x1, y2 - y1
            mot_results.append(f"{index + 1},{tracker_id},{x1},{y1},{w},{h},{confidence:.2f},{class_id+1},1.0\n")#frame, track id, x, y, w, h, confidence for MOT analysis
            if tracker_id not in all_tomatoes: #if the tomato with a specific id isnt already in the dictionary, add it
                all_tomatoes[tracker_id] = {
                "track_id": tracker_id,
                "class_id": class_id,
                "class_name": model1.model.names[class_id],
                "confidence": confidence,
                "frame_count": 1}#frame count keeps track how many times its been in frame for average confidence

                save_tomatoes(frame, detections, tracker_id) #saves tomato image only on the first frame of detection, labels as a

                tomato_history[tracker_id] = [class_id]  # initializes tomato history dict
            else:  # if tomato with said ID already exists, sum the confidence and frame count over the frames its visible in.
                all_tomatoes[tracker_id]["confidence"] += confidence  # each frame adds confidence scores for a total
                all_tomatoes[tracker_id]["frame_count"] += 1  # increases frame count to keep track of what to divide confidence by to find average
                tomato_history[tracker_id].append(class_id)  # adds current detection class to the list

        labels_1 = [
            f"#{tracker_id} {model1.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id)]

        annotated_frame = box_annotator1.annotate(
            scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels_1)

    #-------------------- DISEASE TRACKING --------------------------

        # model prediction on single frame and conversion to supervision Detections
        results2 = model2(frame, verbose=False)[0]
        detections2 = sv.Detections.from_ultralytics(results2)

        #strips detections off of healthy leaves
        detections2 = detections2[np.isin(detections2.class_id, SELECTED_CLASS_IDS_2)]

        mask2 = [inside_region(box, region_of_interest) for box in detections2.xyxy]
        detections2 = detections2[mask2]

        # only consider class id from selected_classes defined above
        # tracking detections
        detections2 = byte_tracker2.update_with_detections(detections2)
        detections2 = smoother2.update_with_detections(detections2)

        # Extracts valuable info from each track id (each tracked disease) which I can use to store on the database for insights.
        for confidence, class_id, tracker_id in zip(detections2.confidence, detections2.class_id,detections2.tracker_id):  # zip combines 3 lists into one for easier accessing
            if tracker_id not in all_diseases:  # if the disease with a specific id isnt already in the dictionary, add it
                all_diseases[tracker_id] = {
                    "track_id": tracker_id,
                    "class_id": class_id,
                    "class_name": model2.model.names[class_id],
                    "confidence": confidence,
                    "frame_count": 1,
                    "appeared_at": round(len(frames_running)/30)}  # frame count keeps track how many times its been in frame for average confidence

            else:  # if disease with said ID already exists, sum the confidence and frame count over the frames its visible in.
                all_diseases[tracker_id]["confidence"] += confidence  # each frame adds confidence scores for a total
                all_diseases[tracker_id]["frame_count"] += 1  # increases frame count to keep track of what to divide confidence by to find average

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

    #writes mot format file to be used to compare against ground truth
    mot_save_path = os.path.join(cwd, "datasets", "trackers", "mot_challenge", "ByteTrack", "data", "TomatoesMixedTRIM.txt")
    with open(mot_save_path, "w") as f:
        f.writelines(mot_results)
    print("Saved MOT results to TomatoesMixedTRIM.txt")

#----------------------- DUPLICATE REMOVAL ------------------------------------------------
    old_image_list = os.listdir(os.path.join(os.getcwd(), "DetectedTomatoes"))

    #sets up counter for how many duplicate images were deleted
    duplicate_images_count = 0
    old_len_images = len(os.listdir(os.path.join(os.getcwd(), "DetectedTomatoes")))

    orb = cv2.ORB.create()

    def remove_duplicates(image1, next_image, duplicate_images, i, j):
        delete_image = False

        pil_image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        pil_next_image = Image.fromarray(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))

        hash1 = imagehash.phash(pil_image1)
        hash2 = imagehash.phash(pil_next_image)

        diff = hash1 - hash2

        print(f"pHash difference between image {i + 1} and image {i + 1 + j} is: {diff}")

        if diff <= 18:  # based on experiments, I havn't received a single 19, but every 18 result was the same
            # code for saving images to folder for visual inspection of duplicates
            # save_folder = os.path.join(os.getcwd(), 'tempPHASH')
            # image1_save_path = os.path.join(save_folder, f"tomato_{i + 1}.png")
            # next_image_save_path = os.path.join(save_folder, f"tomato_{i + 1 + j}.png")
            #
            # cv2.imwrite(image1_save_path, image1)
            # cv2.imwrite(next_image_save_path, next_image)

            duplicate_images += 1
            delete_image = True

        return duplicate_images, delete_image


    # --------------------------------- comparison of image a to b the next 10 id images to remove duplicate tomato detecions

    for i in range(old_len_images):  # for amount of images in DetectedTomatoes
        image_1_path = os.path.join(os.getcwd(), "DetectedTomatoes", f"tomato_{i + 1}.png")
        for j in range(1, 16):
            next_image_path = os.path.join(os.getcwd(), "DetectedTomatoes", f"tomato_{i + 1 + j}.png")
            if os.path.exists(image_1_path) and os.path.exists(next_image_path):
                image1 = cv2.imread(image_1_path)
                next_image = cv2.imread(next_image_path)

                duplicate_images_count, delete_image = remove_duplicates(image1, next_image, duplicate_images_count, i,
                                                                         j)
                if delete_image:
                    os.remove(image_1_path)

    new_len_images = len(os.listdir(os.path.join(os.getcwd(), "DetectedTomatoes")))
    print(f"Amount of images initially: {old_len_images}, Amount of images after cleaning: {new_len_images}, Duplicates removed: {duplicate_images_count}")

    # finds the difference of items in both lists and maintains those which dont appear in list 2
    # this obtains a list of tomato images which need to be deleted from all_tomatoes
    new_image_list = os.listdir(os.path.join(os.getcwd(), "DetectedTomatoes"))
    image_list = [x for x in old_image_list if x not in new_image_list]

    print(image_list)
    print(all_tomatoes)

    #remove from all_tomatoes id tracks found in the list
    id_list = []
    for item in image_list:
        item = item.replace("tomato_", "")
        item = item.replace(".png", "")
        id_list.append(int(item))

    #replaces all_tomatoes with itself only with items whos track number is not in id_list
    #print(f"before: {len(all_tomatoes)}")
    all_tomatoes = {k: v for k, v in all_tomatoes.items()if int(v['track_id']) not in id_list}
    #print(f"after: {len(all_tomatoes)}")


#----------------------- UPDATING CONFIDENCES AND INSERTING TO SQL ---------------------------

    for track_id, tomato in all_tomatoes.items():  # track id correlates to the id of each tomato, tomatoes correlates to the data stored in all_tomatoes
        majority_class_id = Counter(tomato_history[track_id]).most_common(1)[0][0]#finds most common appearing tomato class.
        tomato["class_id"] = majority_class_id #sets the class id of that tomato as the most common one, a work around for class jittering
        tomato["class_name"] = model1.model.names[int(majority_class_id)]
        average_confidence = tomato["confidence"] / tomato["frame_count"]
        tomato["confidence"] = average_confidence
        if use_sql: #if sql tracking is enabled, the record of that tomato will be added
            insert_statement = f"""INSERT INTO Tomatoes (ScanID, TomatoID, TomatoType, ConfidenceTomato)
    VALUES ({current_scan_id}, {tomato["track_id"]}, '{tomato["class_name"]}', {tomato["confidence"]});"""
            #print(insert_statement)
            cursor.execute(insert_statement)

    for track_id, disease in all_diseases.items():
        average_confidence = disease["confidence"] / disease["frame_count"]
        disease["confidence"] = average_confidence
        if use_sql:
            insert_statement = f"""INSERT INTO Diseases (ScanID, DiseaseID, DiseaseType, ConfidenceDisease, SecondsIntoVideo)
    VALUES ({current_scan_id}, {disease["track_id"]}, '{disease["class_name"]}', {disease["confidence"]}, {disease["appeared_at"]});"""
            #print(insert_statement)
            cursor.execute(insert_statement)

    #commits all changes made to database and closes the cursor
    if use_sql:
        connection.commit()

        #visualises past data.
        select_statement = f"""SELECT ScanID, ScanDate, TomatoType, COUNT(TomatoType) as Counts FROM Tomatoes GROUP BY ScanID, TomatoType, ScanDate ORDER BY ScanID;"""
        df = pd.read_sql_query(select_statement, connection)

        # scan_ids = []
        # scan_dates = []
        # types = []
        # counts = []
        #
        # for row in select_results:
        #     scan_ids.append(row[0])
        #     scan_dates.append(row[1])
        #     types.append(row[2])
        #     counts.append(row[3])

        # print(select_results[:5])
        # print(len(select_results[0]))

        # df = pd.DataFrame(select_results, columns=["ScanID", "ScanDate", "TomatoType", "Counts"])
        df["TomatoType"] = df["TomatoType"].fillna("None")#just in case no tomatoes are detected in a run
        df["ScanDate"] = pd.to_datetime(df["ScanDate"])

        tomato_types = df["TomatoType"].unique()

        plt.figure(figsize=(10, 10))

        for tomato_type in tomato_types:
            subset = df[df["TomatoType"] == tomato_type]
            plt.plot(subset["ScanDate"], subset["Counts"], marker='o', label=tomato_type)

        plt.xlabel("Scan Date")
        plt.ylabel("Tomato Count")
        plt.title("Tomato Counts Over Time By Ripeness")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")


main()