import cv2
import time
from ultralytics import YOLO
import torch.multiprocessing as mp

import os
import glob

def process_video_read_multiprocessing(group_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_jump_unit * group_number)

    #Loop throgh the incomming data
    while cap.isOpened():
        model = YOLO("Selcet YOLO model")
        
        while proc_frames < frame_jump_unit:
            #Read incomming data
            success, frame = cap.read()

            if success:
                #Run YOLO inference on the frame
                results = model(frame, verbose=False)
                proc_frames = proc_frames + 1
            else:
                #break the loop of the end of the video is reached
                cap.release()
                break
        else:
            cap.release()

#Open the incomming data file
video_path = "Path to your data"
cap = cv2.VideoCapture(video_path) 

#Number of available cores
num_processes = 1

#Total frame counts of the incomming data
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#Calculate the frame_jump_unit
frame_jump_unit = frame_count // num_processes

def multi_process():
    #Parallel the execution of a function
    pool = mp.Pool(num_processes)
    pool.map(process_video_read_multiprocessing, range(num_processes))
    pool.close()
    pool.join()

multi_process()
