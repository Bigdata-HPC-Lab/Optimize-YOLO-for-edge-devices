# Optimize-YOLO-for-edge-devices
Improving performance of real-time object detection in edge device through concurrent multi-frame processing (Submitted to the IEEE ACCESS journal)


## Introduction
ML inference is a latency-intensive task, and executing it in a cloud environment introduces significant communication overhead.
Consequently, offloading this task to edge environments has become critical to mitigate such overhead and improve overall system performance.

As demonstrated in our paper, in the process of running the YOLO model on an edge device, it was observed that only one CPU core is being utilized due to serial processing, while the rest of the resources remain underutilized.

In order to address this, our proposed scheme divides the incoming video frames or image dataset into groups based on the available number of cores, allocates each group to an individual core, and performs object detection asynchronously on each core.

For evaluation setup, we use state-of-the-art edge device NVIDIA Jetson Orin Nano (8GB), with YOLO (YOLOv5n.pt, YOLOv8n.pt) as the object detection model. The video datasets include real-world data such as animals and car traffic videos, while the image dataset used are MS-COCO, ImageNet, Pascal VOC, and DOTA.

## The goals of this paper
- Grouping Input Data: Group the input data according to the number of available cores.
- Parallel Inference Execution: Process the grouped data asynchronously in parallel.
- Resource Evaluation: Compare the original and proposed schemes based on runtime, memory usage, and power consumption, and also compare with SOTA ML model optimization technique.

## How to run
First, Install

```
# Install ultralytics
pip install -r requirements.txt
pip install .
```

Second, Set configuration (Data path and YOLO model)

```
cd ./run
# Open the python file you want to run and modify configuration
```

Third, Run the python file
```
# For original (Set optimized number of threads according to your deivce)
OMP_NUM_THREADS=6 python video_single.py
# For proposed (Set optimized number of threads according to your deivce)
OMP_NUM_THREADS=1 python video_multi.py
```
