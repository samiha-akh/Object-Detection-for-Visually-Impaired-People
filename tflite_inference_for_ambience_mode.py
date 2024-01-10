# This code is a modified version of edje electronic's tflite inference code written by "Evan Juras"
# All the variables and functions have been manipulated vastly according to our needs
# We also added our own function for Audio Feedback
# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
from threading import Thread
import time
import pygame

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Define Audio Feedback function that calls on the appropriate audio file 
# according to the scene being detected and plays it in real-time
def audioFeedback(path_to_audio,filename,extension):
    file = os.path.join(path_to_audio,filename+extension)
    pygame.mixer.init()
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue


#required variables
MODEL_NAME = 'ambience_mode/'
GRAPH_NAME = 'tflite_model_5_quant.tflite'
LABELMAP_NAME = 'labelmap.txt'
VIDEO_NAME = 'video_test/video_test_5.mp4'

#Audio Feedback Variables
path_to_audio = 'ambience_mode/audio_feedback' #path of the audio folder
filename = '' # name of the audio file
prev_filename = '' # name of the previous file that was played
extension = '.mp3' #file type

#Frame Variables
resolution = '1280x720'
resW, resH = resolution.split('x')
imW, imH = int(resW), int(resH)
min_conf_threshold = float(0.7)

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, 
# else import from regular tensorflow
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file (if inferencing from a video)
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

    
# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print(output_details)
scale,zero_point = output_details[0]['quantization']
print('Scale: '+str(scale)+' | Zero Point: '+str(zero_point))


# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# for ambience mode models
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5


# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

# Loop over every image and perform detection
while (True):
    
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    
    # grab frame from video stream
    frame1 = videostream.read()
    
    # Load frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    frameH, frameW = int(imH),int(imW) 

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    
    # Retrieve detection results
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
    scores = (scores-zero_point)*scale #normalizing confidence score
    
    label_index = np.argmax(scores, axis = 0) # select the index of the highest confidence score
    
    # Select the one over all detections that has the highest confidence score and check if it is above minimum threshold
    if ((scores[label_index] > min_conf_threshold) and (scores[label_index] <= 1.0)):
        # Set bounding box coordinates and draw box in this case it is going to be static
        ymin = 9
        xmin = 9
        ymax = frameH-9
        xmax = frameW-9
            
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            
        # Draw label
        object_name = labels[label_index] # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, int(scores[label_index]*100)) # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        
        # Audio feedback
        filename = object_name #Set filename to the object that is being detected
        
        # If the scene has already been described then do not describe again
        if (filename!=prev_filename):
            audioFeedback(path_to_audio,filename,extension)
        
        prev_filename = filename
        
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # The result has been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
    
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press any key to continue to next image, or press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()