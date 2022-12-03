import os
import sys
import cv2
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
import time
import shutil
from math import sqrt
import numpy as np
from PIL import Image
import io
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder


# User parameters
TO_PREDICT_PATH         = "./Images/Prediction_Images/To_Predict/"
PREDICTED_PATH          = "./Images/Prediction_Images/Predicted_Images/"
MIN_SCORE               = 0.5
ROBOFLOW_MODEL          = "MODEL_NAME/MODEL_VERSION"
ROBOFLOW_API_KEY        = "API_KEY"


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )


def deleteDirContents(dir):
    # Deletes photos in path "dir"
    # # Used for deleting previous cropped photos from last run
    for f in os.listdir(dir):
        full_path = os.path.join(dir, f)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)


def make_appropriate_directory():
    if not os.path.exists(TO_PREDICT_PATH):
        os.makedirs(TO_PREDICT_PATH)
    if not os.path.exists(PREDICTED_PATH):
        os.makedirs(PREDICTED_PATH)


def writes_text(text, start_point_index, font, font_scale, color, thickness):
    start_point = (image_b4_color.shape[1]-300, 
                   50 + 30*start_point_index
                   )
    cv2.putText(image_b4_color, text,  start_point, 
                font, font_scale, color, thickness)



# Starting stopwatch to see how long process takes
start_time = time.time()

# If prediction folders don't exist, create them
make_appropriate_directory()

# Deletes images already in "Predicted_Images" folder
deleteDirContents(PREDICTED_PATH)\


# Start FPS timer
fps_start_time = time.time()

close_counters = 0 # Sum of how many times a person is close to a machine per frame
ii = 0
# Goes through each video in TO_PREDICT_PATH
for video_name in os.listdir(TO_PREDICT_PATH):
    video_path = os.path.join(TO_PREDICT_PATH, video_name)
    
    video_capture = cv2.VideoCapture(video_path)
    
    # Video frame count and fps needed for VideoWriter settings
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = round( video_capture.get(cv2.CAP_PROP_FPS) )
    video_fps = int(video_fps/5)
    
    # If successful and image of frame
    success, image_b4_color = video_capture.read()
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_out = cv2.VideoWriter(PREDICTED_PATH + video_name, fourcc, video_fps, 
                                (int(image_b4_color.shape[1]), 
                                 int(image_b4_color.shape[0])
                                 )
                                )
    
    workers_in_frame_list = []
    count = 1
    while success:
        success, image_b4_color = video_capture.read()
        if not success:
            break
        
        # Inference through Roboflow section
        # -----------------------------------------------------------------------------
        # Load Image with PIL
        image = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(image)
        
        # Convert to JPEG Buffer
        buffered = io.BytesIO()
        pilImage.save(buffered, quality=100, format="JPEG")
        
        # Construct the URL
        upload_url = "".join([
            "https://detect.roboflow.com/",
            ROBOFLOW_MODEL,
            "?api_key=",
            ROBOFLOW_API_KEY,
            "&confidence=",
            str(MIN_SCORE)
            # "&format=image",
            # "&stroke=5"
        ])
        
        # Build multipart form and post request
        m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})
        
        response = requests.post(upload_url, 
                                 data=m, 
                                 headers={'Content-Type': m.content_type},
                                 )
        
        predictions = response.json()['predictions']
        # -----------------------------------------------------------------------------
        
        # Creates lists from inferenced frames
        prev_machine_list = []
        
        data_list = []
        machine_base_data_list = []
        machine_whole_data_list = []
        person_data_list = []
        
        for prediction in predictions:
            label = prediction['class']
            midx = prediction['x']
            midy = prediction['y']
            width = prediction['width']
            height = prediction['height']
            x1 = midx - width/2
            y1 = midy - height/2
            x2 = x1 + prediction['width']
            y2 = y1 + prediction['height']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            midx, midy = int(midx), int(midy)
            data_list.append([label, x1, y1, x2, y2, midx, midy, width, height])
            
            if label == "Machinery-Base":
                machine_base_data_list.append(data_list[-1])
            elif label == "Machinery-Whole":
                machine_whole_data_list.append([midx, midy, "Inactive"])
            elif label == "Person":
                person_data_list.append(data_list[-1])
            
        
        # Sees if person close to machine base
        # ---------------------------START-1-------------------------------------
        for index, person_data in enumerate(person_data_list):
            is_person_close = False # Presets if person too close to machine-base
            
            for machine_base_data in machine_base_data_list:
                # Gets minimum width and height between person's feet and machine-base
                rect1 = {'x':person_data[1], 'y':person_data[4]+1, 
                         'w':person_data[7], 'h':(1)
                         }
                rect2 = {'x':machine_base_data[1], 'y':machine_base_data[2], 
                         'w':machine_base_data[7], 'h':machine_base_data[8]
                         }
                
                min_width = int(min(rect1['x']+rect1['w']-rect2['x'],rect2['x']+rect2['w']-rect1['x']))
                min_height = int(min(rect1['y']+rect1['h']-rect2['y'],rect2['y']+rect2['h']-rect1['y']))
                
                # If person's feet inside machine-base's bounding box, then to flag
                if min_width > 0 and min_height > 0:
                    is_person_close = True
                else:
                    hypotenuse = sqrt(min_width**2 + min_height**2)
                    
                    # if person's feet is close to machine-base, then to flag
                    if hypotenuse < (person_data[8]*2/3):
                        is_person_close = True
                        
                
                if is_person_close:
                    # If person is close to machine-base, then to rename label to "CAUTION"
                    person_data_list[index][0] = "CAUTION"
                    close_counters += 1
                    break
            # -----------------------------END-1------------------------------------
            
            
        # Sees if machine is active
        # ---------------------------START-2-------------------------------------
        if count == 1:
            # Copies over detached center_machine_list to prevent changes 
            #  to new one when original changes
            prev_machine_whole_data_list = machine_whole_data_list.copy()
            prev_prev_machine_whole_data_list = prev_machine_whole_data_list.copy()
        else:
            # Checks to see if bounding box matches with previous frames and if it has moved
            for machine_whole_data in machine_whole_data_list:
                is_active = False
                
                # Checks last frame if machine active
                for prev_machine_whole_data in prev_machine_whole_data_list:
                    # Gets x, y difference in center of bounding box lists 
                    #  from previous frame to current
                    diff_hor = (machine_whole_data[0] - prev_machine_whole_data[0])
                    diff_ver = (machine_whole_data[1] - prev_machine_whole_data[1])
                    # Checks to see if the bounding box (BB) of machine in previous 
                    #  frame matches with current BB list
                    # Value of 150 is arbitrary. Choose whatever is reasonable
                    if diff_hor < 150 and diff_ver < 150:
                        # Now we know that the two BB in list match, let's check
                        #  to see if there has been slight movement in machine
                        if diff_hor > 30 or diff_ver > 30:
                            is_active = True
                            break
                        
                # If didn't catch any matching, then checks 2 frames ago
                if not is_active:
                    for prev_prev_machine_whole_data in prev_prev_machine_whole_data_list:
                        # Gets x, y difference in center of bounding box lists 
                        #  from previous frame to current
                        diff_hor = (machine_whole_data[0] - prev_prev_machine_whole_data[0])
                        diff_ver = (machine_whole_data[1] - prev_prev_machine_whole_data[1])
                        # Checks to see if the bounding box (BB) of machine in previous 
                        #  frame matches with current BB list
                        # Value of 150 is arbitrary. Choose whatever is reasonable
                        if diff_hor < 150 and diff_ver < 150:
                            # Now we know that the two BB in list match, let's check
                            #  to see if there has been slight movement in machine
                            if diff_hor > 30 or diff_ver > 30:
                                is_active = True
                                break
                
                if is_active:
                    machine_whole_data_list[-1][2] = "Active"
                
            # Copies over detached center_machine_list to prevent changes 
            #  to new one when original changes
            prev_prev_machine_whole_data_list = prev_machine_whole_data_list.copy()
            prev_machine_whole_data_list = machine_whole_data_list.copy()
        # -----------------------------END-2------------------------------------
        
        # Gets worker and machinery info
        workers_in_frame_list.append(len(person_data_list))
        min_workers = min(workers_in_frame_list)
        max_workers = max(workers_in_frame_list)
        avg_workers = round(np.mean(workers_in_frame_list))
        
        # Writes worker and machinery info on top right of video
        # ----------------------------Start-3----------------------------------
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 0, 255)
        thickness = 1
        
        # Writes text: workers in frame
        text = "Workers in Frame: " + str(workers_in_frame_list[-1])
        writes_text(text, 0, font, font_scale, color, thickness)
        
        # Writes text: min workers
        text = "Min Workers: " + str(min_workers)
        writes_text(text, 1, font, font_scale, color, thickness)
        
        # Writes text: max workers
        text = "Max Workers: " + str(max_workers)
        writes_text(text, 2, font, font_scale, color, thickness)
        
        # Writes text: avg workers
        text = "Avg Workers: " + str(avg_workers)
        writes_text(text, 3, font, font_scale, color, thickness)
        
        # Writes person close counters to machines
        text = "Close Counters: " + str(close_counters)
        writes_text(text, 4, font, font_scale, color, thickness)
        
        # Writes active machines
        active_machine_count = 0
        for machine_whole_data in machine_whole_data_list:
            if machine_whole_data[2] == "Active":
                active_machine_count += 1
        text = "Active Machines: " + str(active_machine_count)
        writes_text(text, 5, font, font_scale, color, thickness)
        
        # ----------------------------End-3-------------------------------------
        
        
        # Draws bounding box's (BB) and Writes BB's identity text on top left of BB
        # Machinery-Base
        for machine_base_data in machine_base_data_list:
            # Bounding Box Section
            # -------------------------------------------------------------
            start_point = (int(machine_base_data[1]), int(machine_base_data[2]) )
            end_point = (int(machine_base_data[3]), int(machine_base_data[4]) )
            color = (255, 0, 0)
            thickness = 1
            cv2.rectangle(image_b4_color, start_point, end_point, color, thickness)
            # -------------------------------------------------------------
            
            
            # Text Section
            # -------------------------------------------------------------
            text = "Machinery"
            start_point_text = (machine_base_data[1], max(machine_base_data[2]-5,0) )
            color = (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.30
            thickness = 1
            
            cv2.putText(image_b4_color, text, 
                        start_point_text, font, fontScale, color, thickness)
            # -------------------------------------------------------------
        
        # Person
        for person_data in person_data_list:
            # Bounding Box Section
            # -------------------------------------------------------------
            start_point = (int(person_data[1]), int(person_data[2]) )
            end_point = (int(person_data[3]), int(person_data[4]) )
            color = (255, 0, 255)
            thickness = 1
            cv2.rectangle(image_b4_color, start_point, end_point, color, thickness)
            # -------------------------------------------------------------
            
            
            # Text Section
            # -------------------------------------------------------------
            text = person_data[0]
            start_point_text = (person_data[1], max(person_data[2]-5,0) )
            color = (255, 255, 255)
            fontScale = 0.30
            thickness = 1
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            if "CAUTION" in text:
                color = (0, 100, 255)
                fontScale = 0.60
                thickness = 2
            
            cv2.putText(image_b4_color, text, 
                        start_point_text, font, fontScale, color, thickness)
            # -------------------------------------------------------------
        
        # Writes active on machineas that are moving
        for machine_whole_data in machine_whole_data_list:
            if machine_whole_data[2] == "Active":
                mid_x = machine_whole_data[0]
                mix_y = machine_whole_data[1]
                
                cv2.putText(image_b4_color, "Active", (mid_x, mix_y), 
                            font, 0.6, (0,255,100), 2)
            
        
        # Saves video with bounding boxes
        video_out.write(image_b4_color)
        
        
        # Just prints out how fast video is inferencing and how much time left
        # ---------------------------------------------------------------------
        tenScale = 10
        ii += 1
        if ii % tenScale == 0:
            fps_end_time = time.time()
            fps_time_lapsed = fps_end_time - fps_start_time
            fps = round(tenScale/fps_time_lapsed, 2)
            time_left = round( (frame_count-ii)/fps )
            
            mins = time_left // 60
            sec = time_left % 60
            hours = mins // 60
            mins = mins % 60
            
            sys.stdout.write('\033[2K\033[1G')
            print("  " + str(ii) + " of " 
                  + str(frame_count), 
                  "-", fps, "FPS",
                  "-", "{}m:{}s".format(int(mins), round(sec) ),
                  end="\r", flush=True
                  )
            fps_start_time = time.time()
        # ---------------------------------------------------------------------
        
        count += 1
        # If you want to stop after so many frames to debug, uncomment below
        if count == 300:
            break
    
    video_out.release()


print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)