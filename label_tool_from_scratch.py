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

    key = int(key) - 48         # 1: skip, 2: value = 0, 3: value = 1

    if key == 1:
        bb_value = -1

    if key == 2:
        bb_value = 0

    if key == 3:
        bb_value = 1

    if bb_value == -1:
        bb_value = None

    return bb_value

# use many videos at once here IMPORTANT
# file to read track data from
path_to_files = "videos/"
file_r1 = [f"R_e{str(exp)}_c{str(cam)}.txt" for exp in range(7,10) for cam in range(1,5)]
# file to read video from
file_v1 = [f"R_e{str(exp)}_c{str(cam)}.mp4" for exp in range(7,10) for cam in range(1,5)]

# path where files is written
annotation_file = "annotation_images/labels.txt"
video_dir = "annotation_images/images/"

# Create directories if they don't exist
#os.makedirs(annotation_file, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

lines = []
# read lines 
for x in range(len(file_r1)):
    if not os.path.exists(path_to_files + file_r1[x]):
        continue
    with open(path_to_files+file_r1[x], 'r') as file:
        video_lines = file.readlines()
        video_lines = [ x.replace("\n","").split(",") for x in video_lines]
        video_lines = [ [x, y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7] ] for y in video_lines]
        lines += video_lines

while True:
    x = lines[random.randint(0, len(lines))]
    video_nr, frame_nr, tracknr, posx, posy, width, height, certainty, ape = x

    if int(ape) != 1: 
        cap = cv2.VideoCapture(path_to_files+file_v1[video_nr])

        if (cap.isOpened()== False):
            print("Error opening video file")

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_nr)-1)
        ret, frame = cap.read()
        print(frame.shape)
        y1 = max(0,int(float(posy)))
        y2 = min(frame.shape[0], int(float(posy))+int(float(height)))
        x1 = max(0,int(float(posx)))
        x2 = min(frame.shape[1], int(float(posx))+int(float(width)))

        frame = frame[y1:y2, x1:x2]

        #score = None
        
        #while score == None:
        score = score_bb(frame)

        print(score)
        print(annotation_file)

        if score != None:
            with open(annotation_file, "a") as f:
                f.write(file_v1[video_nr][:-4] + "_" + str(int(frame_nr)) + "_" + str(int(tracknr)) + ".png" + "," + str(int(frame_nr)) +"," +str(int(tracknr))+ ","+ str(int(score)) + "\n")

            path =  video_dir + file_v1[video_nr][:-4] + "_" + str(int(frame_nr)) + "_" + str(int(tracknr)) + ".png"
            print(path)

            cv2.imwrite( path, frame)
