# Import the necessary packages
import os
import cv2
import time
# TESTING SVD FROM NUMPY
import numpy as np
import json
import torch
from torchvision import models
import re
import albumentations as A  # our data augmentation library
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
# Now, we will define our transforms
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image
import shutil


# User parameters
SAVE_NAME_OD = "./Models/Construction.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models/",1)[1].split(".model",1)[0] +"/"
TO_PREDICT_PATH         = "./Images/Prediction_Images/To_Predict/"
PREDICTED_PATH          = "./Images/Prediction_Images/Predicted_Images/"
SAVE_ANNOTATED_IMAGES   = True
MIN_SCORE               = 0.6
NUMBER_TO_RUN = 1000
FRAMES_TO_SKIP = 59


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


# MAIN():
# =============================================================================
# Starting stopwatch to see how long process takes
start_time = time.time()

# Clears some of the screen for asthetics
print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

make_appropriate_directory()

# Deletes images already in "Predicted_Images" folder
deleteDirContents(PREDICTED_PATH)

# Deletes unwanted images in "To_Predict" path
for i in range(10000):
    if os.path.isfile(os.path.join(TO_PREDICT_PATH, str(i)+".jpg")):
        os.remove(os.path.join(TO_PREDICT_PATH, str(i)+".jpg"))


dataset_path = DATASET_PATH


#load classes
coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
categories = coco.cats
n_classes_1 = len(categories.keys())

classes_1 = [i[1]['name'] for i in categories.items()]
classes_1



# lets load the faster rcnn model
model_1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True
                                                   )
in_features = model_1.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model_1.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes_1)


# TESTING TO LOAD MODEL
if os.path.isfile(SAVE_NAME_OD):
    checkpoint = torch.load(SAVE_NAME_OD)
    model_1.load_state_dict(checkpoint)


device = torch.device("cuda") # use GPU to train
model_1 = model_1.to(device)

model_1.eval()
torch.cuda.empty_cache()

transforms_1 = A.Compose([
    # A.Resize(IMAGE_SIZE, IMAGE_SIZE), # our input size can be 600px
    # A.Rotate(limit=[90,90], always_apply=True),
    ToTensorV2()
])


# For Json file
image_names = []
image_ids = []
image_heights = []
image_widths = []

# For Json file
index = -1
ids = []
image_ids = []
category_id = []
bboxes = np.zeros([1, 4], np.int32)
bbox_areas = []
segmentations = []
iscrowd = []

# From object detection "To_Predict"
color_list =['green', 'red', 'magenta', 'blue', 'orange', 'cyan', 'lime', 'turquoise', 'yellow']
# Below for SMiPE4
# color_list =['white', 'gray', 'lime', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'gray']
pred_dict = {}
ii = 0

for video_index, video_name in enumerate(os.listdir(TO_PREDICT_PATH) ):
    video_path = os.path.join(TO_PREDICT_PATH, video_name)
    
    
    if "_annotations.coco.json" in video_name:
        continue
    
    
    video_capture = cv2.VideoCapture(video_path)
    
    # Video frame count and fps needed for VideoWriter settings
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = round( video_capture.get(cv2.CAP_PROP_FPS) )
    
    # If successful and image of frame
    success, image_b4_color = video_capture.read()
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_out = cv2.VideoWriter(PREDICTED_PATH + video_name, fourcc, video_fps, 
                                (int(image_b4_color.shape[1]), 
                                 int(image_b4_color.shape[0])
                                 )
                                )
    
    count = 1
    image_index = -1
    while success:
        success, image_b4_color = video_capture.read()
        if not success:
            break
        
        count += 1
        
        # I don't want to annotate every images so this skips a lot of frames
        if count % FRAMES_TO_SKIP != 0:
            continue
        
        image_index += 1
    
        # TESTING - Only completes up to index (NUMBER_TO_RUN-1)
        if image_index == NUMBER_TO_RUN:
            break
        
        image_name = str(image_index) + ".jpg"
        
        # Saves image to "To_Predict" folder
        cv2.imwrite(TO_PREDICT_PATH + image_name, image_b4_color)
        
        image = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2RGB)
        
        transformed_image = transforms_1(image=image)
        transformed_image = transformed_image["image"]
        
        # For Json file
        image_names.append(image_name)
        image_ids.append(image_index)
        image_heights.append(transformed_image.shape[1])
        image_widths.append(transformed_image.shape[2])
        
        if ii == 0:
            line_width = max(round(transformed_image.shape[1] * 0.002), 1)
        
        with torch.no_grad():
            prediction_1 = model_1([(transformed_image/255).to(device)])
            pred_1 = prediction_1[0]
        
        coordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
        class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE].tolist()
        # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
        scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE].tolist()
        labels_found = [classes_1[i] for i in class_indexes]
        
        if SAVE_ANNOTATED_IMAGES:
            predicted_image = draw_bounding_boxes(transformed_image,
                boxes = coordinates,
                # labels = [classes_1[i] for i in class_indexes], 
                labels = [str(round(i,2)) for i in scores], # SHOWS SCORE IN LABEL
                width = line_width,
                colors = [color_list[i] for i in class_indexes],
                font = "arial.ttf",
                font_size = 20
                )
            
            # Saves full image with bounding boxes
            if len(class_indexes) != 0:
                save_image((predicted_image/255), PREDICTED_PATH + image_name)
            
            # save_image((predicted_image/255), PREDICTED_PATH + image_name)
            
        
        # SAVE_CROPPED_IMAGES Section
        # --------------------------------------------------------------
        names = []
        box_count = 0 # Number of boxes made per full 100-objects image
        
        # Changes column names in names
        for box_index in range(len(coordinates)):
            limiter = 0
            x1 = max( int( coordinates[box_index][0] ), limiter)
            y1 = max( int( coordinates[box_index][1] ), limiter)
            x2 = min( int( coordinates[box_index][2] ), transformed_image.shape[2]-limiter)
            y2 = min( int( coordinates[box_index][3] ), transformed_image.shape[1]-limiter)
            
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            
            box_count += 1
            
            # JSON info
            index += 1
            ids.append(index)
            image_ids.append(image_ids[-1]) # image_id to place in annotations category
            category_id.append(class_indexes[box_index])
            if index == 0:
                bboxes[-1] = np.array([x1, y1, bbox_width, bbox_height], ndmin=2)
            else:
                bboxes = np.append(bboxes, [[x1, y1, bbox_width, bbox_height]], axis=0)
            bbox_areas.append(bbox_area)
            segmentations.append([])
            iscrowd.append(0)
        # --------------------------------------------------------------
        
        # =========================================================================
        
        
        if len(os.listdir(TO_PREDICT_PATH)) > 1000:
            tenScale = 1000
        else:
            tenScale = 100
        
        ii += 1
        if ii % tenScale == 0:
            print("  " + str(ii) + " of " + str(len(os.listdir(TO_PREDICT_PATH))))
    
        # ==================================================================================
    video_out.release()


# Creating JSON section
# ==================================================================================
data = {
    "info": {
        "year": "2022",
        "version": "1",
        "description": "Created own",
        "contributor": "Troy P.",
        "url": "",
        "date_created": "2022-02-13T01:11:34+00:00"
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0"
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": str(classes_1[0]),
            "supercategory": "none"
        }
    ]
}



# EXPERIMENTAL
# Updates "categories" section for each class/label
for label_index, label in enumerate(classes_1):
    if label_index != 0:
        to_update_with = {
            "categories": {
                "id": label_index,
                "name": str(classes_1[label_index]),
                "supercategory": str(classes_1[0])
            }
        }
        data["categories"].append(to_update_with["categories"])



# Updates "image" section oc coco.json
for image_index in image_ids:
    if image_index == 0:
        to_update_with = {
            "images": [
                {
                    "id": image_index,
                    "license": 1,
                    "file_name": image_names[image_index],
                    "height": image_heights[image_index],
                    "width": image_widths[image_index],
                    "date_captured": "2022-02-13T01:11:34+00:00"
                }
            ]
        }
        data.update(to_update_with)
    else:
        to_update_with = {
            "images": {
                    "id": image_index,
                    "license": 1,
                    "file_name": image_names[image_index],
                    "height": image_heights[image_index],
                    "width": image_widths[image_index],
                    "date_captured": "2022-02-13T01:11:34+00:00"
            }
        }
        data["images"].append(to_update_with["images"])


# Updates "annotations" section oc coco.json
for index in ids:
    if index == 0:
        to_update_with = {
            "annotations": [
                {
                    "id": index,
                    "image_id": image_ids[index],
                    "category_id": category_id[index],
                    "bbox": [
                        int(bboxes[index][0]),
                        int(bboxes[index][1]),
                        int(bboxes[index][2]),
                        int(bboxes[index][3])
                    ],
                    "area": bbox_areas[index],
                    "segmentation": segmentations[index],
                    "iscrowd": iscrowd[index]
                }
            ]
        }
        
        data.update(to_update_with)
        
    else:
        to_update_with = {
            "annotations": {
                    "id": index,
                    "image_id": image_ids[index],
                    "category_id": category_id[index],
                    "bbox": [
                        int(bboxes[index][0]),
                        int(bboxes[index][1]),
                        int(bboxes[index][2]),
                        int(bboxes[index][3])
                    ],
                    "area": bbox_areas[index],
                    "segmentation": segmentations[index],
                    "iscrowd": iscrowd[index]
            }
        }
        
        data["annotations"].append(to_update_with["annotations"])


with open(TO_PREDICT_PATH+'_annotations.coco.json', 'w') as f:
    json.dump(data, f, indent=4)

# ==================================================================================




print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)