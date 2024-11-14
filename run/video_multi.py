import cv2
import time
from ultralytics import YOLO
import torch.multiprocessing as mp

import os
import glob

proc_frames = 0
numWhileLoop = 0
video_read_time = 0
model_load_time = 0

YOLOv8StartTime = time.time()
def process_video_read_multiprocessing(group_number):
    print(group_number)
    global numWhileLoop, proc_frames
    global video_read_time
    global model_load_time

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_jump_unit * group_number)

    #Loop throgh the video frames
    while cap.isOpened():
        model_load_start_time = time.time()
        print("KSH: START Model Load")
        model = YOLO('yolov8n.pt')
        model_load_end_time = time.time()
        model_load_time = model_load_end_time - model_load_start_time
        print("KSH: model_load_time: ", model_load_time)
        print("KSH: END Model Load")

        while proc_frames < frame_jump_unit:
            print("KSH: while loop count: ", numWhileLoop)
            numWhileLoop = numWhileLoop + 1
            #Read a frame form the video
            video_read_start_time = time.time()
            success, frame = cap.read()
            video_read_end_time = time.time()
            video_read_time += video_read_end_time - video_read_start_time
            print("KSH: video_read_time: ", video_read_time)
            if success:
                #Run YOLOv8 inference on the frame
                results = model(frame, verbose=False)
                proc_frames = proc_frames + 1
            else:
                #break the loop of the end of the video is reached
                cap.release()
                break
                #cap.release()
        else:
            cap.release()

print("KSH: START video Open")
#Open the video file
video_path = "/mnt/data/elefant.mp4"
cap = cv2.VideoCapture(video_path) 
print("KSH: END video Open")

#Number of parallel processes
num_processes = 1

#Total frame counts of the video file
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#Calculate the frame_jump_unit
frame_jump_unit = frame_count // num_processes

def multi_process():
    #Parallel the execution of a function
    pool = mp.Pool(num_processes)
    pool.map(process_video_read_multiprocessing, range(num_processes))
    pool.close()
    pool.join()

print("KSH: START Function multi_process")
multi_process()
#process_video_read_multiprocessing(1)

print("KSH: END Function multi_process")

YOLOv8EndTime = time.time()
YOLOv8TotalTime = YOLOv8EndTime - YOLOv8StartTime
print("KSH: YOLOv8TotalTime ", YOLOv8TotalTime )
print("KSH: video_read_time ", video_read_time )

