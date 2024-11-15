import cv2
import time
from ultralytics import YOLO

import os
from PIL import Image

def image_singleprocessing():
    image_path = "Path to your data"
    image_files = os.listdir(image_path)

    for i in range(0, len(image_files)):
        #Bring the model YOLO
        model = YOLO("Select YOLO model")
    
        #Open the incomming data file
        image_path = "Path to your data"
        imgae_files = os.listdir(image_path)
        image_file = os.path.join(image_path, image_files[i])

        #Read incomming data
        image = cv2.imread(image_file)
        
        #Actually inference performs
        results = model(image, verbose=False)

image_singleprocessing()
