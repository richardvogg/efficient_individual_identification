import cv2
import time
import os
import random
import sys
import psutil
import csv
from PIL import Image

# 0-6 Monkey labels 7 unsure
# the pressed buttons are currently id + 1
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


# path of directory with frames
frame_path = "other/frames/"
# path where the frames are written
annotation_file = "other/annotation_images/labels"
video_dir = "other/annotation_images/images/"

# threshold for loading frames
threshh = 0.9

# loading and removing of frames smaller than threshold
frames = os.listdir(frame_path)
frames = [ [x, x.replace(" ","").replace("\n","").split("_")[-1]] for x in frames ]
frames = [ x[0] for x in frames if float(x[1][:-4]) >= threshh ]


for frame in frames:
    img = cv2.imread(frame_path+frame)

    score = None
    
    while score == None:
        score = score_bb(img)


    with open(annotation_file, "a") as f:
        f.write( frame + "," + str(-1) +"," +str(-1)+ ","+ str(int(score)) + "\n")

    path =  video_dir + frame
    print(score)
    print(path)

    cv2.imwrite( path, img)
