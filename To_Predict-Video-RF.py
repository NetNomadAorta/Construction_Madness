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
    if not os.path.exists("./Models/"):
        os.makedirs("./Models/")
    if not os.path.exists("./Training_Data/"):
        os.makedirs("./Training_Data/")


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
        coordinates = []
        labels_found = []
        confidence_level_list = []
        for prediction in predictions:
            x1 = prediction['x'] - prediction['width']/2
            y1 = prediction['y'] - prediction['height']/2
            x2 = x1 + prediction['width']
            y2 = y1 + prediction['height']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            coordinates.append([x1, y1, x2, y2])
            
            label = prediction['class']
            labels_found.append(label)
        
        
        labels_list = []
        machine_list = []
        prev_machine_list = []
        
        coordinates_temp = coordinates.copy()
        center_machine_list = []
        active_machine_list = []
        
        ppl_coordinates = [coordinates[index] 
            for index, label in enumerate(labels_found) if label == "Person"]
        machine_base_coordinates = [coordinates[index] 
            for index, label in enumerate(labels_found) if label == "Machinery-Base"]
        
        # Sees if person close to machine base
        # ---------------------------START-1-------------------------------------
        for index, label in enumerate(labels_found):
            if label == "Person": 
                left_coordinate_person = coordinates[index][0]
                right_coordinate_person = coordinates[index][2]
                top_coordinate_person = coordinates[index][1]
                bottom_coordinate_person = coordinates[index][3]
                person_height = int(bottom_coordinate_person - top_coordinate_person)
                
                is_person_close = False # Presets if person too close to machine-base
                
                for machine_base_coordinate in machine_base_coordinates:
                    left_coordinate_machine = machine_base_coordinate[0]
                    right_coordinate_machine = machine_base_coordinate[2]
                    top_coordinate_machine = machine_base_coordinate[1]
                    bottom_coordinate_machine = machine_base_coordinate[3]
                    mid_vert_coord_machine = (top_coordinate_machine - bottom_coordinate_machine)/2
                    
                    # Gets minimum width and height between person's feet and machine-base
                    rect1 = {'x':left_coordinate_person, 'y':bottom_coordinate_person+1, 
                             'w':(right_coordinate_person-left_coordinate_person), 
                             'h':(1)}
                    
                    rect2 = {'x':left_coordinate_machine, 'y':top_coordinate_machine, 
                             'w':(right_coordinate_machine-left_coordinate_machine), 
                             'h':(bottom_coordinate_machine-top_coordinate_machine)}
                    
                    min_width = int(min(rect1['x']+rect1['w']-rect2['x'],rect2['x']+rect2['w']-rect1['x']))
                    min_height = int(min(rect1['y']+rect1['h']-rect2['y'],rect2['y']+rect2['h']-rect1['y']))
                    
                    
                    # If person's feet inside machine-base's bounding box, then to flag
                    if min_width > 0 and min_height > 0:
                        is_person_close = True
                    else:
                        hypotenuse = sqrt(min_width**2 + min_height**2)
                        
                        # if person's feet is close to machine-base, then to flag
                        if hypotenuse < (person_height*2/3):
                            is_person_close = True
                            
                    
                    if is_person_close:
                        # If person is close to machine-base, then to rename label to "CAUTION"
                        labels_list.append("CAUTION")
                        close_counters += 1
                        break
                
                if is_person_close == False:
                    labels_list.append( label )
            else:
                # We do not want "Machinery-Whole" bounding box to show in the video
                if label == "Machinery-Whole":
                    coordinates[index] = [0,0,0,0]
                
                labels_list.append( label )
            
            
            # Sees if machine is active
            # ---------------------------START-2-------------------------------------
            if label == "Machinery-Whole": # If class is machine-whole
                left_coord_machine = coordinates_temp[index][0]
                right_coord_machine = coordinates_temp[index][2]
                mid_hor_coord_machine = left_coord_machine + (right_coord_machine-left_coord_machine)/2
                top_coord_machine = coordinates_temp[index][1]
                bottom_coord_machine = coordinates_temp[index][3]
                mid_ver_coord_machine = top_coord_machine + (bottom_coord_machine-top_coord_machine)/2
                
                center_machine_list.append([mid_hor_coord_machine, mid_ver_coord_machine, index])
        
        if count == 1:
            # Copies over detached center_machine_list to prevent changes 
            #  to new one when original changes
            prev_center_machine_list = center_machine_list.copy()
        else:
            # Checks to see if bounding box matches with previous frames and if it has moved
            for center_machine in center_machine_list:
                is_active = False
                
                for prev_center_machine in prev_center_machine_list:
                    # Gets x,y difference in center of bounding box lists 
                    #  from previous frame to current
                    diff_hor = (center_machine[0] - prev_center_machine[0])
                    diff_ver = (center_machine[1] - prev_center_machine[1])
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
                    # [index, "Active", mid_hor_coord_machine, mid_ver_coord_machine]
                    active_machine_list.append([center_machine[2], "Active", 
                        center_machine[0], center_machine[1]]) 
                else:
                    # [index, "Active", mid_hor_coord_machine, mid_ver_coord_machine]
                    active_machine_list.append([center_machine[2], "Inactive", 
                        center_machine[0], center_machine[1]]) 
                
                # Copies over detached center_machine_list to prevent changes 
                #  to new one when original changes
                prev_center_machine_list = center_machine_list.copy()
                
            # -----------------------------END-2------------------------------------
        
        # -----------------------------END-1------------------------------------
        
        # Gets worker and machinery info
        workers_in_frame_list.append(len(ppl_coordinates))
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
        for active_machine in active_machine_list:
            if active_machine[1] == "Active":
                active_machine_count += 1
        text = "Active Machines: " + str(active_machine_count)
        writes_text(text, 5, font, font_scale, color, thickness)
        
        # ----------------------------End-3-------------------------------------
        
        
        # Draws bounding box's (BB) and Writes BB's identity text on top left of BB
        for coordinate_index, coordinate in enumerate(coordinates):
            # Bounding Box Section
            # -------------------------------------------------------------
            start_point = (int(coordinate[0]), int(coordinate[1]) )
            end_point = (int(coordinate[2]), int(coordinate[3]) )
            if labels_list[coordinate_index] == "Machinery-Base":
                color = (255, 0, 0)
            elif (labels_list[coordinate_index] == "Person" 
                  or labels_list[coordinate_index] == "CAUTION"):
                color = (255, 0, 255)
            thickness = 1
            cv2.rectangle(image_b4_color, start_point, end_point, color, thickness)
            # -------------------------------------------------------------
            
            
            # Text Section
            # -------------------------------------------------------------
            text = labels_list[coordinate_index]
            start_point = ( int(coordinate[0]), int(coordinate[1]) )
            color = (255, 255, 255)
            
            start_point_text = (start_point[0], max(start_point[1]-5,0) )
            font = cv2.FONT_HERSHEY_SIMPLEX
            if "CAUTION" in text:
                color = (0, 100, 255)
                fontScale = 0.60
                thickness = 2
            else:
                if text != "Person" and text != "Machinery-Base":
                    text = ""
                elif text == "Machinery-Base":
                    text = "Machinery"
                
                color = (255, 255, 255)
                fontScale = 0.30
                thickness = 1
            
            cv2.putText(image_b4_color, text, 
                        start_point_text, font, fontScale, color, thickness)
            # -------------------------------------------------------------
        
        # Writes active on machineas that are moving
        for active_machine in active_machine_list:
            if active_machine[1] == "Active":
                coordinates_temp[active_machine[0]]
                mid_x = int(active_machine[2])
                mix_y = int(active_machine[3])
                
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
        # if count == 300:
        #     break
    
    video_out.release()


print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)