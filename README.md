# Final-Individual-Project
IMPORTANT: 
The models were trained on google colab using the controller file, It is imperitive to train via colab as the corresponding code for altering yolov12 is in the controller.

ByteTrack requires python 3.10 to run, while the requirements are provided in requirements.txt file, trying to install them all will most probably fail. Modules regarding bytetracker 
such as ONX and pyWin may require windows software to be upgraded. As ByteTrack was modified, it is reccomended to use the provided ZIP's ByteTrack folder as it is modified and 
ready to be used. Alternatively it can be downloaded via: https://github.com/FoundationVision/ByteTrack but will lack ImprovedBytetrack changes made.

To perform metric analysis such as: MOTA, IDF1, MOTP; the motmetrics module needs to be cloned from: https://github.com/cheind/py-motmetrics 

The tomato dataset: https://www.kaggle.com/datasets/nexuswho/laboro-tomato and its contents should be extracted into the 'dataset' folder following the structure of:

Dataset
-annotations
-train
-val
-dataset.yaml

The disease dataset: https://www.kaggle.com/datasets/diemhuongnt12/tomato-leaf-diseases-yolov11/data and its contents should be extracted into the 'disease data' folder following the structure of:

disease data
-test
-train
-val
-data.yaml
-README.dataset.txt
-README.robloflow.txt

Test data sits in the "test_videos" folder and its output goes into "prediction_results".

