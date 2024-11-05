import cv2
import time
import os
import random
import sys
import psutil
import csv
from PIL import Image

# 0-6 Monkey labels 7 unsure
def score_bb(frame):
    bb_value = -1.0

    frame = cv2.resize(frame, (800,800))
    cv2.imshow("show",frame)
    key = cv2.waitKeyEx(0)
    cv2.destroyAllWindows()

    key = int(key) - 48         # 0: skip, 1: value = 0, 2: value = 0.5, 3: value = 1 || 7 show image again

    if key == 1:
        bb_value = 0

    if key == 2:
        bb_value = 1

    if key == 3:
        bb_value = 2

    if key == 4:
        bb_value = 3

    if key == 5:
        bb_value = 4

    if key == 6:
        bb_value = 5

    if key == 7:
        bb_value = 6

    if key == 8:
        bb_value = 7

    if bb_value == -1:
        bb_value = None

    return bb_value

path = "../data/second_try/99_model_labeled_data/99_all_data_without_sort/"
dir = "images/"
file_dir_1 = path+"2k"
write_file = "labeled_2k"
num_images_video = 1000
step = 1
counter = 0

with open(file_dir_1, 'r') as file:
    lines = file.readlines()

lines = [ x.replace("\n","").split(",") for x in lines]
lines = [ [x[0], x[1], x[2], x[3]] for x in lines]

for x in lines:   
    image_name, frame_nr, track, id = x  
    frame_nr = int(frame_nr)
    track = int(track)
    id = int(id)

    print(id+1)

    image_path = os.path.join(path+dir+image_name)
    
    img = cv2.imread(image_path)

    score = score_bb(img)

    if score == None:
        with open(path+write_file,"a") as f:
            #print("written")
            f.write(image_name +","+ str(frame_nr) +"," +str(track)+ ","+ str(id) + "\n")
    else:
        with open(path+write_file,"a") as f:
            #print("changed: ",score+1)
            f.write(image_name +","+ str(frame_nr) +"," +str(track)+ ","+ str(score) + "\n")

    counter = counter + step
