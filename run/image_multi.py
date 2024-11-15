import cv2
import time
from ultralytics import YOLO
import torch.multiprocessing as mp

import os
from PIL import Image

def image_multiprocessing(group_number):
    for i in range(frame_jump_unit * group_number, min(frame_jump_unit * (group_number + 1), len(image_files))):
        #Set the model
        model = YOLO('yolov8n.pt')
        
        #Connect to incomming data
        image_file = os.path.join("Path to your data", image_files[i])
        
        #Read incomming data
        frame = cv2.imread(image_file)
        
        #Actually inference performs
        results = model(frame, verbose=False)

        proc_frames = proc_frames + 1
        
#Open the image file
image_path = "Path to your data"
image_files = os.listdir(image_path)

#Number of parallel processes
num_processes = "Available number of cores"

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

multi_process()
