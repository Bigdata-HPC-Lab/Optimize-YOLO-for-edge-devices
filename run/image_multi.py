import cv2
import time
from ultralytics import YOLO
import torch.multiprocessing as mp

import os
from PIL import Image

proc_frames = 0
numWhileLoop = 0
image_read_time = 0

YOLOv8StartTime = time.time()
def image_multiprocessing(group_number):
    print(group_number)
    global numWhileLoop, proc_frames
    global image_read_time

    for i in range(frame_jump_unit * group_number, min(frame_jump_unit * (group_number + 1), len(image_files))):
        #Set the model
        model = YOLO('yolov8n.pt')

        print("KSH: while loop count: ", numWhileLoop)
        numWhileLoop = numWhileLoop + 1
        
        #Connect to COCO directory
        image_file = os.path.join(image_path, image_files[i])
        
        #Read frames from COCO directory
        image_read_start_time = time.time()
        frame = cv2.imread(image_file)
        image_read_end_time = time.time()
        image_read_time += image_read_end_time - image_read_start_time
        
        #Actually inference performs
        results = model(frame, verbose=False)

        proc_frames = proc_frames + 1
        
print("KSH: START video Open")
#Open the image file
image_path = "/mnt/data/JPEGImages/"
image_files = os.listdir(image_path)
print("KSH: END video Open")

#Number of parallel processes
num_processes = 6

print("KSH: image_files_len : ", len(image_files))

#Total frame counts of the video file
frame_count = int(len(image_files))

#Calculate the frame_jump_unit
frame_jump_unit = frame_count // num_processes

def multi_process():
    #Parallel the execution of a function
    pool = mp.Pool(num_processes)
    pool.map(image_multiprocessing, range(num_processes))
    pool.close()
    pool.join()

print("KSH: START Function multi_process")
multi_process()
#process_video_read_multiprocessing(1)

print("KSH: END Function multi_process")

YOLOv8EndTime = time.time()
YOLOv8TotalTime = YOLOv8EndTime - YOLOv8StartTime
print("KSH: YOLOv8TotalTime ", YOLOv8TotalTime )
print("KSH: image_read_time ", image_read_time )
