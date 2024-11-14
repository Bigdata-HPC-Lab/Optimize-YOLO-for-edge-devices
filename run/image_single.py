#Bring time value from ultralytics(YOLO)
import ultralytics.models.yolo.detect
import ultralytics.engine.predictor
import ultralytics.nn.tasks

import cv2
import time
from ultralytics import YOLO

import os
from PIL import Image

numWhileLoop = 0
image_read_time = 0
model_load_time = 0

YOLOv8_start_time = time.time()
def image_singleprocessing():
    global numWhileLoop
    global image_read_time
    global model_load_time

    #Open the image file
    image_path = "/mnt/data/imagenet/test/"
    image_files = os.listdir(image_path)

    for i in range(0, len(image_files)):
        #Bring the model YOLO
        model_load_start_time = time.time()
        model = YOLO('yolov8n.pt')
        model_load_end_time = time.time()
        model_load_time += model_load_end_time - model_load_start_time
    
        #Open the image file
        image_path = "/mnt/data/imagenet/test/"
        imgae_files = os.listdir(image_path)
        image_file = os.path.join(image_path, image_files[i])

        #Read images from COCO directory
        image_read_start_time = time.time()
        image = cv2.imread(image_file)
        image_read_end_time = time.time()
        image_read_time += image_read_end_time - image_read_start_time
    
        #Actually inference performs
        results = model(image, verbose=False)

        print("KSH: image_inference_count: ", numWhileLoop)
        numWhileLoop = numWhileLoop + 1

image_singleprocessing()

YOLOv8_end_time = time.time()
YOLOv8_Total_time = YOLOv8_end_time - YOLOv8_start_time

print("KSH: YOLOv8_Total_time: ", YOLOv8_Total_time )
print("KSH: model_load_time: ", model_load_time )
print("KSH: image_read_time: ", image_read_time )
print("KSH: inference_time: ", ultralytics.nn.tasks.inference_time )
print("KSH: postprocess_time: ", ultralytics.models.yolo.detect.predict.postprocess_time ) 
print("KSH: preprocess_time: ", ultralytics.engine.predictor.preprocess_time )

