# Import packages
from customlib.centroidtracker import CentroidTracker
import customlib.myvisualize as vis
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

class KalmanFilter:
    def __init__(self) :
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # kf.statePre = np.array([[401], [588]])
    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        self.measured = np.array([[np.float32(coordX-initState[0])], [np.float32(coordY-initState[1])]])
        correct = self.kf.correct(self.measured)
        predicted = self.kf.predict()
        predicted[0],predicted[1]=predicted[0]+initState[0],predicted[1]+initState[1]
        return predicted

file = open ("data.txt","a+")


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'videotest.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 3

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

ct = CentroidTracker()
(H, W) = (None, None)
score_thresh = 0.6 
# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi',fourcc, 20.0, (1920,1080))
passed = 0
count = 0
PredictedCoords = np.zeros((2, 1), np.float32)
kalman =  KalmanFilter()
print("\n\n")
array_centroid_x = []
array_centroid_y = []
array_cenkalman_x = []
array_cenkalman_y = []
array_lastcenkalman_x = []
array_lastcenkalman_y = []
lastPredictedCoords = np.zeros((2, 1), np.float32)
initState = (401,588)

h_UAV = 100
# realityHeight = tan(70/2)*2*h_UAV 
# size_pixel_reality = realityHeight/1920,2



while(video.isOpened()):
    start = time.time()
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    # Perform the actual detection by running the model with the image as input
    detections = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    rects = []
    centroids = []
    classes = []
    for i in range(int(detections[3][0])):

        if detections[1][0][i] > score_thresh:
            box = detections[0][0][i] * np.array([H, W, H, W]) 
            startY, startX, endY, endX = box.astype("int")
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            
            # predictedCoords.append(PredictedCoords)
            # cv2.rectangle(frame, (predictedCoords[i][0] - int(abs(startX - endX)/2), predictedCoords[i][1] - int(abs(startY - endY)/2)), (predictedCoords[i][0] + int(abs(startX - endX)/2), predictedCoords[i][1] + int(abs(startY - endY)/2)),
            #    (0, 255, 255), 2)
            #if (cY >= 0.1 * H) & (cY <= 0.65 * H):
            if True: 
                frame = vis.drawBox(frame,detections[2][0][i],box)
                classes.append(detections[2][0][i])
                centroids.append((cX,cY))
                rects.append(box.astype("int"))

    # cập nhật bộ theo dõi tâm sử dụng
    objects,rectangles, speeds = ct.update(rects,count%5)
    Kalman_track = objects[4]
    #print(rectangles)
    _rectangle = rectangles[4]
    start_K_y , start_K_x , end_K_y , end_K_x = _rectangle
    
    if len(Kalman_track.shape)==1:
        centroid_x = Kalman_track[0]
        centroid_y = Kalman_track[1]
    else:
        centroid_x = Kalman_track[-1][0] 
        centroid_y = Kalman_track[-1][1]
        PredictedCoords = kalman.Estimate(centroid_x, centroid_y)
        # cv2.rectangle(frame, (start_K_x, start_K_y),(end_K_x,end_K_y),
        #     (0, 255, 255), 2)

        cv2.rectangle(frame, ((PredictedCoords[0] - abs(start_K_x - end_K_x)/2), (PredictedCoords[1] - abs(start_K_y - end_K_y)/2)), ((PredictedCoords[0] + abs(start_K_x - end_K_x)/2), (PredictedCoords[1] + abs(start_K_y - end_K_y)/2)),
            (0, 255, 255), 2)
        print("measured : " ,centroid_x,centroid_y)
        print("predicted : " ,PredictedCoords[0],PredictedCoords[1])
        file.write(str(centroid_x)+"\n")
        file.write(str(centroid_y)+"\n")
        file.write(str(PredictedCoords[0])+"\n")
        file.write(str(PredictedCoords[1])+"\n")

        frame = vis.drawOrbit(frame,5,Kalman_track)
        lastPredictedCoords[0],lastPredictedCoords[1] = PredictedCoords[0],PredictedCoords[1]
        if(lastPredictedCoords[0] !=0 and lastPredictedCoords[1] !=0):
            array_lastcenkalman_x.append(lastPredictedCoords[0])
            array_lastcenkalman_y.append(lastPredictedCoords[1])
            array_cenkalman_x.append(PredictedCoords[0])
            array_cenkalman_y.append(PredictedCoords[1])
        for i in range(1,len(array_cenkalman_x)):
            cv2.line(frame,(array_lastcenkalman_x[i-1],array_lastcenkalman_y[i-1]), (array_cenkalman_x[i-1],array_cenkalman_y[i-1]),(255,255,0),6)


    '''
    for (objeqctID, centroid) in objects.items():
        if len(centroid.shape) == 1:
            continue
        else:
            if int(0.6 * H) in range(centroid[-2][1],centroid[-1][1]):
                passed += 1 
    '''
    #frame = vis.drawSystem1(frame, objects, classes, centroids, passed)
    frame = vis.drawSystem2(frame, objects, speeds)

    out.write(frame)
    frame = cv2.resize(frame,(960,540))
    # cv2.imshow('Object tracking', frame)
    count += 1
    #print(count)
    # Press 'q' to quit
    end = time.time()
    print(end-start)
    if cv2.waitKey(1) == ord('q'):
        break
# Clean up
video.release()
out.release()
cv2.destroyAllWindows()
