import cv2

def video_inference():
    proc_frames = 0

    #Open the incomming data file
    video_path = "Path to your data"
    cap = cv2.VideoCapture(video_path)

    #Calculate data count
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #Loop throgh the incomming data
    while cap.isOpened():
        model = YOLO("Selcet YOLO model")
        
        numFrame = 0
        while numFrame < frame_count:
            #Read incomming data
            success, frame = cap.read()
            
            if success:
                #Actual inference task
                results = model(frame, verbose=True)
            else:
                #break the loop of the end of the video is reached
                cap.release()
                break
        else:
            cap.release()

video_inference()
