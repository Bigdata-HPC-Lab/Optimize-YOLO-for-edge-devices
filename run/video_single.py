#this contains time variables
import ultralytics.engine.model
import ultralytics.engine.predictor
import ultralytics.nn.tasks
import ultralytics.nn.modules.head
import ultralytics.models
import ultralytics.data
import ultralytics.hub.utils
import ultralytics.engine.results
import ultralytics.cfg
import ultralytics.utils.ops
import ultralytics.nn.modules.conv
#import torch
import time
#import pdb
#from mpi4py import MPI

YOLOv8StartTime = time.time()

print("KSH: START import cv2")
import_cv2_start_time = time.time()
import cv2
import_cv2_end_time = time.time()
print("KSH: END import cv2")
import_cv2_time = import_cv2_end_time - import_cv2_start_time

import_YOLO_start_time = time.time()
print("KSH: START import YOLO")
from ultralytics import YOLO
print("KSH: END import YOLO")
import_YOLO_end_time = time.time()
import_YOLO_time = import_YOLO_end_time - import_YOLO_start_time

#import multiprocessing as mp
#import torch.multiprocessing as mp


video_open_time = video_open_count = 0
single_frame_process_time = single_frame_process_count = 0
model_load_time = model_load_count = 0  
frame_read_time = frame_read_count = 0
inference_time = inference_count = 0

#torch.set_num_threads(1)


def video_inference():

    global video_open_time, video_open_count
    global model_load_time, model_load_count
    global single_frame_process_time, single_frame_process_load_count
    global frame_read_time, frame_read_count
    global inference_time, inference_count

    proc_frames = 0
    
    model_load_total_time = 0
    inference_total_time = 0
    one_iteration_inference_time = 0


    #Open the video file
    video_open_start_time = time.time()
    video_path = "/mnt/data/polar_bear.mp4"
    cap = cv2.VideoCapture(video_path)
    video_open_end_time = time.time()
    video_open_time += video_open_end_time - video_open_start_time
    video_open_count += 1
    
    #Calculate frame count
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    #Loop throgh the video frames
    while cap.isOpened():
        
        #comm = MPI.COMM_WORLD
        model_load_start_time = time.time()
        model = YOLO('yolov8n.pt')
        model_load_end_time = time.time()
        model_load_time += model_load_end_time - model_load_start_time
        model_load_count +=1
        print("KSH: model_load_count ", model_load_count)
        #print(type(model))
        #print(model)
        #exit(0)

        numFrame = 0
        while numFrame < frame_count:
         #SKIM: for measuing entire time for a single frame proccessing
            single_frame_process_start_time = time.time() 

            print("KSH: numFrame: ", numFrame)

            numFrame += 1
            

            frame_read_start_time = time.time()
            success, frame = cap.read()
            frame_read_end_time = time.time()
            frame_read_time += frame_read_end_time - frame_read_start_time
            frame_read_count += 1
            print("KSH: frame_read_count ", frame_read_count )

            if success:
                #print("KSH: START inference")
                inference_start_time = time.time()
                #pdb.set_trace()
                results = model(frame, verbose=True)
                inference_end_time = time.time()
                #print("KSH: END inference")
                inference_time += inference_end_time - inference_start_time
                inference_count += 1
                
                single_frame_process_end_time = time.time()
                single_frame_process_time += single_frame_process_end_time - single_frame_process_start_time
            else:
                #break the loop of the end of the video is reached
                single_frame_process_end_time = time.time()
                single_frame_process_time += single_frame_process_end_time - single_frame_process_start_time
                cap.release()
                break
                #cap.release()


        else:
            cap.release()

video_inference()
YOLOv8EndTime = time.time()

YOLOv8TotalTime = YOLOv8EndTime - YOLOv8StartTime

#SKIM: percentage of total runtime is not necessary, delete them if necessary!
#SKIM: BUT TIME and COUNT is crucial 
print("\nKSH ---- END of EXECUTION ---\n")
print("KSH: YOLOv8TotalTime ", YOLOv8TotalTime )
print("KSH: model_load_time: ", model_load_time, ", model_load_count: ", model_load_count, " ( %.2f" % (model_load_time*100/YOLOv8TotalTime) , "%)" )
print("KSH: frame_read_time: ", frame_read_time, ", frame_read_count: ", frame_read_count, " ( %.2f" % (frame_read_time*100/YOLOv8TotalTime) , "%)")
print("KSH: inference_time: ", inference_time, ", inference_count: ", inference_count, " ( %.2f" % (inference_time*100/YOLOv8TotalTime) , "%)")
print("KSH: video_open_time: ", video_open_time, ", video_open_count: ", video_open_count, " ( %.2f" % (video_open_time*100/YOLOv8TotalTime) , "%)")
print("KSH: postprocess_time: ", ultralytics.models.yolo.detect.predict.postprocess_time) 
print("KSH: preprocess_time: ", ultralytics.engine.predictor.preprocess_time)
print("KSH -- IMPORT ---")
print("KSH: import_cv2_time: ", import_cv2_time, " ( %.2f" % (import_cv2_time*100/YOLOv8TotalTime) , "%)")
print("KSH: import_YOLO_time: ", import_YOLO_time, " ( %.2f" % (import_YOLO_time*100/YOLOv8TotalTime) , "%)")
print("KSH -- START ---")
print("KSH: video_open_time: ", video_open_time, ", video_open_count: ", video_open_count, " ( %.2f" % (video_open_time*100/YOLOv8TotalTime) , "%)")
print("KSH: model_load_time: ", model_load_time, ", model_load_count: ", model_load_count, " ( %.2f" % (model_load_time*100/YOLOv8TotalTime) , "%)" )
print("KSH -- WHILE LOOP per Frame ---")
print("KSH: single_frame_process_time: ", single_frame_process_time, " ( %.2f" % (single_frame_process_time*100/YOLOv8TotalTime) , "%)")
print("KSH: frame_read_time: ", frame_read_time, ", frame_read_count: ", frame_read_count, " ( %.2f" % (frame_read_time*100/YOLOv8TotalTime) , "%)")
print("KSH: inference_time: ", inference_time, ", inference_count: ", inference_count, " ( %.2f" % (inference_time*100/YOLOv8TotalTime) , "%)")

print("\nKSH -- FROM ultralytics/engine/model.py ---")
print("KSH: model_init_time: ", ultralytics.engine.model.model_init_time )
print("KSH: call_time ", ultralytics.engine.model.call_time )
print("KSH: predict_time: ", ultralytics.engine.model.predict_time )
print("KSH: smart_load_time:  ", ultralytics.engine.model.smart_load_time )

print("\nKSH -- FROM ultralytics/engine/predictor.py ---")
print("KSH: init_time: ", ultralytics.engine.predictor.init_time )
print("KSH: stream_inference_time: ", ultralytics.engine.predictor.stream_inference_time )
print("KSH: stream_inference_time_a: ", ultralytics.engine.predictor.stream_inference_time_a )
print("KSH: stream_inference_time_b: ", ultralytics.engine.predictor.stream_inference_time_b )
print("KSH: stream_inference_time_c: ", ultralytics.engine.predictor.stream_inference_time_c )
print("KSH: stream_inference_time_d: ", ultralytics.engine.predictor.stream_inference_time_d )
print("KSH: stream_inference_time_e: ", ultralytics.engine.predictor.stream_inference_time_e )
print("KSH: setup_source_time: ", ultralytics.engine.predictor.setup_source_time )
print("KSH: preprocess_time: ", ultralytics.engine.predictor.preprocess_time )
print("KSH: pre_transform_time: ", ultralytics.engine.predictor.pre_transform_time )
print("KSH: post_process_time: ", ultralytics.engine.predictor.post_process_time )
print("KSH: inference_visualize_time: ", ultralytics.engine.predictor.inference_visualize_time )
print("KSH: inference_else_time: ", ultralytics.engine.predictor.inference_else_time )
print("KSH: setup_model_time: ", ultralytics.engine.predictor.setup_model_total_time )

print("\nKSH: -- FROM ultralytics/nn/tasks.py ---")
print("KSH: predict_time: ", ultralytics.nn.tasks.predict_time )
print("KSH: predict_once_time: ", ultralytics.nn.tasks.predict_once_time )
print("KSH: predict_once_num: ", ultralytics.nn.tasks.predict_once_num )
print("KSH: predict_once_a_time: ", ultralytics.nn.tasks.predict_once_a_time )
print("KSH: predict_once_b_time: ", ultralytics.nn.tasks.predict_once_b_time )
print("KSH: predict_once_b_num: ", ultralytics.nn.tasks.predict_once_b_num )
print("KSH: predict_once_c_time: ", ultralytics.nn.tasks.predict_once_c_time )
print("KSH: predict_once_tensor_split_time: ", ultralytics.nn.tasks.predict_once_tensor_split_time )
print("KSH: predict_once_tensor_split_num: ", ultralytics.nn.tasks.predict_once_tensor_split_num )
print("KSH: predict_once_splitted_tensor_convolution_time: ", ultralytics.nn.tasks.predict_once_splitted_tensor_convolution_time )
print("KSH: predict_once_splitted_tensor_convolution_num: ", ultralytics.nn.tasks.predict_once_splitted_tensor_convolution_num )
print("KSH: predict_once_original_convolution_time: ", ultralytics.nn.tasks.predict_once_original_convolution_time )
print("KSH: predict_once_tensor_merge_time: ", ultralytics.nn.tasks.predict_once_tensor_merge_time )
print("KSH: predict_once_split_time: ", ultralytics.nn.tasks.predict_once_split_time )
print("KSH: predict_once_pass_time: ", ultralytics.nn.tasks.predict_once_pass_time )

print("\nKSH: -- FROM ultralytics/nn/modules/head.py ---")
print("KSH: forward_time: ", ultralytics.nn.modules.head.forward_time )
print("KSH: forward_time_a: ", ultralytics.nn.modules.head.forward_time_a )
print("KSH: forward_time_a_num: ", ultralytics.nn.modules.head.forward_time_a_num )
print("KSH: forward_time_b: ", ultralytics.nn.modules.head.forward_time_b )
print("KSH: forward_time_c: ", ultralytics.nn.modules.head.forward_time_c )
print("KSH: forward_time_d: ", ultralytics.nn.modules.head.forward_time_d )
print("KSH: forward_num: ", ultralytics.nn.modules.head.forward_num )

print("\nKSH: -- FROM ultralytics/nn/modules/conv.py ---")
print("KSH: conv_forward_fuse_time: ", ultralytics.nn.modules.conv.forward_fuse_time )
print("KSH: conv_forward_fuse_num: ", ultralytics.nn.modules.conv.forward_fuse_num )
print("KSH: Concat_forward_time: ", ultralytics.nn.modules.conv.Concat_forward_time )
print("KSH: Concat_forward_num: ", ultralytics.nn.modules.conv.Concat_forward_num )


print("\nKSH: -- FROM ultralytics/nn/modules/block.py ---")
print("KSH: Bottleneck_forward_time: ", ultralytics.nn.modules.block.Bottleneck_forward_time )
print("KSH: Bottleneck_forward_num: ", ultralytics.nn.modules.block.Bottleneck_forward_num )
print("KSH: C2f_forward_time: ", ultralytics.nn.modules.block.C2f_forward_time )
print("KSH: C2f_forward_num: ", ultralytics.nn.modules.block.C2f_forward_num )
print("KSH: DFL_forward_time: ", ultralytics.nn.modules.block.DFL_forward_time )
print("KSH: DFL_forward_num: ", ultralytics.nn.modules.block.DFL_forward_num )
print("KSH: SPPF_forward_time: ", ultralytics.nn.modules.block.SPPF_forward_time )
print("KSH: SPPF_forward_num: ", ultralytics.nn.modules.block.SPPF_forward_num )

print("\nKSH -- FROM ultralytics/models/yolo/detect/predict.py ---")
#print("KSH: postprocess_time: ", ultralytics.models.yolo.detect.predict.postprocess_time )

print("\nKSH -- FROM ultralytics/data/build.py ---")
print("KSH: load_inference_source_time: ", ultralytics.data.build.load_inference_source_time )

print("\nKSH -- FROM ultralytics/data/loaders.py ---")
print("KSH: init_time: ", ultralytics.data.loaders.init_time )
print("KSH: single_check_time: ", ultralytics.data.loaders.single_check_time )
print("KSH: len_time: ", ultralytics.data.loaders.len_time )
print("KSH: iter_time: ", ultralytics.data.loaders.iter_time )
print("KSH: next_time: ", ultralytics.data.loaders.next_time )

print("\nKSH -- FROM ultralytics/engine/results.py ---")
print("KSH: Results_init_time: ", ultralytics.engine.results.Results_init_time )
print("KSH: Boxes_init_time: ", ultralytics.engine.results.Boxes_init_time )
print("KSH: BaseTensor_init_time: ", ultralytics.engine.results.BaseTensor_init_time )

print("\nKSH -- FROM ultralytics/hub/utils.py ---")
print("KSH: call_time: ", ultralytics.hub.utils.call_time )

print("\nKSH -- FROM ultralytics/utils/ops.py ---")
print("KSH: nms_time: ", ultralytics.utils.ops.nms_time )


print("\nKSH -- FROM ultralytics/cfg/init.py ---")
#print("KSH: get_cfg_time: ", ultralytics.cfg.__init__.get_cfg_time )
#print("KSH: cfg2dict_load_dict_time: ", ultralytics.cfg.__init__.cfg_load_dict_time )
#print("KSH: cfg2dict_convert_dict_time: ", ultralytics.cfg.__init__.cfg_convert_dict_time )
#print("KSH: check_dict_alignment_time: ", ultralytics.cfg.__init__.check_dict_alignmnet_time )
#print("KSH: handle_deprecation_time: ", ultralytics.cfg.__init__.handle_deprecation_time )"""


