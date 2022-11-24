import os
import torch
from torchvision import models
import re
import cv2
import albumentations as A  # our data augmentation library
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
import time
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
# Now, we will define our transforms
from albumentations.pytorch import ToTensorV2
import shutil
from math import sqrt


# User parameters
SAVE_NAME_OD = "./Models/Construction.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models/",1)[1].split(".model",1)[0] +"/"
TO_PREDICT_PATH         = "./Images/Prediction_Images/To_Predict/"
PREDICTED_PATH          = "./Images/Prediction_Images/Predicted_Images/"
SAVE_ANNOTATED_IMAGES   = True
SAVE_ORIGINAL_IMAGE     = False
SAVE_CROPPED_IMAGES     = False
DIE_SPACING_SCALE       = 0.99
MIN_SCORE               = 0.6 # Default 0.5


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


# Creates class folder
def makeDir(dir, classes_2):
    for classIndex, className in enumerate(classes_2):
        os.makedirs(dir + className, exist_ok=True)


def hypotenuse(a, b):
    c = sqrt(a**2 + b**2)
    return c



# Starting stopwatch to see how long process takes
start_time = time.time()

make_appropriate_directory()

# Deletes images already in "Predicted_Images" folder
deleteDirContents(PREDICTED_PATH)

dataset_path = DATASET_PATH



#load classes
coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
categories = coco.cats
n_classes_1 = len(categories.keys())
categories

classes_1 = [i[1]['name'] for i in categories.items()]



# lets load the faster rcnn model
model_1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True
                                                   )
in_features = model_1.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model_1.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes_1)


# Loads last saved checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

if os.path.isfile(SAVE_NAME_OD):
    checkpoint = torch.load(SAVE_NAME_OD, map_location=map_location)
    model_1.load_state_dict(checkpoint)

model_1 = model_1.to(device)

model_1.eval()
torch.cuda.empty_cache()

transforms_1 = A.Compose([
    # A.Resize(IMAGE_SIZE, IMAGE_SIZE), # our input size can be 600px
    # A.Rotate(limit=[90,90], always_apply=True),
    ToTensorV2()
])


# Start FPS timer
fps_start_time = time.time()

color_list =['green', 'red', 'blue', 'magenta', 'orange', 'cyan', 'lime', 'turquoise', 'yellow']
pred_dict = {}
ii = 0
for video_name in os.listdir(TO_PREDICT_PATH):
    video_path = os.path.join(TO_PREDICT_PATH, video_name)
    
    
    video_capture = cv2.VideoCapture(video_path)
    
    # Video frame count and fps needed for VideoWriter settings
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = round( video_capture.get(cv2.CAP_PROP_FPS) )
    video_fps = int(video_fps/4)
    
    # If successful and image of frame
    success, image_b4_color = video_capture.read()
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_out = cv2.VideoWriter(PREDICTED_PATH + video_name, fourcc, video_fps, 
                                (int(image_b4_color.shape[1]), 
                                 int(image_b4_color.shape[0])
                                 )
                                )
    
    count = 1
    while success:
        success, image_b4_color = video_capture.read()
        if not success:
            break
        
        # if count % 6 != 0:
        #     count += 1
        #     continue
        
        image = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2RGB)
        
        transformed_image = transforms_1(image=image)
        transformed_image = transformed_image["image"]
        
        if ii == 0:
            line_width = max(round(transformed_image.shape[1] * 0.002), 1)
        
        with torch.no_grad():
            prediction_1 = model_1([(transformed_image/255).to(device)])
            pred_1 = prediction_1[0]
        
        dieCoordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
        die_class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE]
        # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
        die_scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE]
        
        
        labels_found = []
        
        ppl_coordinates = dieCoordinates[die_class_indexes == 3]
        machine_base_coordinates = dieCoordinates[die_class_indexes == 1]
        machine_whole_coordinates = dieCoordinates[die_class_indexes == 2]
        
        # Sees if person close to machine base
        for index, class_index in enumerate(die_class_indexes):
            if class_index == 3: # If class is person
                left_coordinate_person = dieCoordinates[index][0]
                right_coordinate_person = dieCoordinates[index][2]
                mid_x_coordinate_person = right_coordinate_person-left_coordinate_person
                bottom_coordinate_person = dieCoordinates[index][3]
                person_height = int(dieCoordinates[index][1] - bottom_coordinate_person)
                
                is_person_close = False
                
                for machine_base_coordinate in machine_base_coordinates:
                    left_coordinate_machine = machine_base_coordinate[0]
                    right_coordinate_machine = machine_base_coordinate[2]
                    top_coordinate_machine = machine_base_coordinate[1]
                    bottom_coordinate_machine = machine_base_coordinate[3]
                    mid_vert_coord_machine = (top_coordinate_machine - bottom_coordinate_machine)/2
                    
                    
                    dist_ppl_mchn_left = left_coordinate_person - right_coordinate_machine
                    dist_ppl_mchn_right = left_coordinate_machine - right_coordinate_person
                    dist_ppl_mchn_mid_left = left_coordinate_machine - mid_x_coordinate_person
                    dist_ppl_mchn_mid_right = mid_x_coordinate_person - right_coordinate_machine
                    dist_ppl_mchn_top = top_coordinate_machine - bottom_coordinate_person
                    dist_ppl_mchn_bottom = bottom_coordinate_person - bottom_coordinate_machine
                    
                    if (dist_ppl_mchn_left < 0
                        and dist_ppl_mchn_right < 0
                        and dist_ppl_mchn_top < 0
                        and dist_ppl_mchn_bottom < 0
                        ):
                        is_person_close = True
                    else:
                        
                        if (dist_ppl_mchn_left < 0
                            and dist_ppl_mchn_right < 0
                            ):
                            dist_ppl_mchn_left = 0
                            dist_ppl_mchn_right = 0
                            dist_ppl_mchn_mid_left = 0
                            dist_ppl_mchn_mid_right = 0
                        
                        if (dist_ppl_mchn_top < 0
                            and dist_ppl_mchn_bottom < 0
                            ):
                            dist_ppl_mchn_top = 0
                            dist_ppl_mchn_bottom = 0
                        
                        # hypotenuse or distance between machine base and person's 
                        #  feet is related to c^2 = a^2 + b^2
                        hypotenuse_left = hypotenuse( 
                            abs(dist_ppl_mchn_left), 
                            abs(mid_vert_coord_machine - bottom_coordinate_person)
                            )
                        hypotenuse_right = hypotenuse( 
                            abs(dist_ppl_mchn_right), 
                            abs(mid_vert_coord_machine - bottom_coordinate_person)
                            )
                        hypotenuse_top = hypotenuse( 
                            abs(dist_ppl_mchn_mid_left), 
                            abs(dist_ppl_mchn_top)
                            )
                        hypotenuse_bottom = hypotenuse( 
                            abs(dist_ppl_mchn_mid_right), 
                            abs(dist_ppl_mchn_bottom)
                            )
                        
                        min_hypotenuse = min(hypotenuse_left, hypotenuse_right,
                                             hypotenuse_top, hypotenuse_bottom)
                    
                    
                        if min_hypotenuse < person_height:
                            is_person_close = True
                    
                    if is_person_close:
                        labels_found.append("Person TOO CLOSE")
                        break
                
                if is_person_close == False:
                    labels_found.append( str(classes_1[class_index]) )
            else:
                labels_found.append( str(classes_1[class_index]) )
        
        
        
        
        if SAVE_ANNOTATED_IMAGES:
            predicted_image = draw_bounding_boxes(transformed_image,
                boxes = dieCoordinates,
                # labels = [classes_1[i] for i in die_class_indexes], 
                # labels = [str(round(i,2)) for i in die_scores], # SHOWS SCORE IN LABEL
                width = line_width,
                colors = [color_list[i] for i in die_class_indexes],
                font = "arial.ttf",
                font_size = 10
                )
            
            predicted_image_cv2 = predicted_image.permute(1,2,0).contiguous().numpy()
            predicted_image_cv2 = cv2.cvtColor(predicted_image_cv2, cv2.COLOR_RGB2BGR)
            
            for dieCoordinate_index, dieCoordinate in enumerate(dieCoordinates):
                start_point = ( int(dieCoordinate[0]), int(dieCoordinate[1]) )
                # end_point = ( int(dieCoordinate[2]), int(dieCoordinate[3]) )
                color = (255, 255, 255)
                # thickness = 3
                # cv2.rectangle(predicted_image_cv2, start_point, end_point, color, thickness)
                
                start_point_text = (start_point[0], max(start_point[1]-5,0) )
                font = cv2.FONT_HERSHEY_SIMPLEX
                if labels_found[dieCoordinate_index] == "Person TOO CLOSE":
                    color = (0, 0, 255)
                    fontScale = 0.60
                    thickness = 2
                else:
                    color = (255, 255, 255)
                    fontScale = 0.30
                    thickness = 1
                cv2.putText(predicted_image_cv2, labels_found[dieCoordinate_index], 
                            start_point_text, font, fontScale, color, thickness)
            
            # Saves video with bounding boxes
            video_out.write(predicted_image_cv2)
        
        
        tenScale = 10
    
        ii += 1
        if ii % tenScale == 0:
            fps_end_time = time.time()
            fps_time_lapsed = fps_end_time - fps_start_time
            print("  " + str(ii) + " of " 
                  + str(frame_count), 
                  "-",  round(tenScale/fps_time_lapsed, 2), "FPS")
            fps_start_time = time.time()
        
        count += 1
        
    video_out.release()


print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)