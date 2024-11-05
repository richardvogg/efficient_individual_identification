import os
import cv2
import torch
import time
import numpy as np
import torchvision

from torch import nn
from torchvision.models import get_model, list_models
from torchvision.transforms import v2
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader

data_path = "../data/validation_videos/data/"
groundtruth = "../data/validation_videos/groundtruth.csv"
ident_model_path = "save/99_label_all_data_2k_0.pth"
threshhold = 0.95

device = "cuda:0"
num_models = 1
batch_size = 30
classes = 8

# load resnet
ident_model = torchvision.models.resnet18(pretrained=True)
num_ftrs = ident_model.fc.in_features
ident_model.fc = nn.Linear(num_ftrs, 8)
dict = torch.load(ident_model_path, map_location=device)
prefix = 'cnn.'
n_clip = len(prefix)
adapted_dict = {k[n_clip:]: v for k, v in dict.items()
                if k.startswith(prefix)}
ident_model.load_state_dict(adapted_dict)
ident_model.to(device)
ident_model.eval()

files = os.listdir(data_path)
files = list(set(files))
files = set([x[:-4] for x in files])

sm = nn.Softmax(dim=1)

with open(groundtruth, 'r') as file:
    gt_lines = file.readlines()

for file in files:
    print(file)

    gt_tracks = [ x for x in gt_lines if file in x]
    gt_tracks = [ x.replace("\n","").split(",") for x in gt_tracks if file in x]

    gt_tracks = [ [ int(x[1]), int(x[2]) ] for x in gt_tracks]
    print("track - label")
    print(gt_tracks)

    vid = cv2.VideoCapture(data_path+file+".mp4")
    if not vid.isOpened:
        print("There has been an error opening the video")

    with open(data_path+file+".txt", 'r') as file:
        lines = file.readlines()
    
    lines = [x.replace("\n","").replace(" ","").split(",") for x in lines]
    lines = [x for x in lines if x[7] == "0"]
    lines = [ [ int(x[0]), int(x[1]), int(float(x[2])), int(float(x[3])), int(float(x[4])), int(float(x[5])), int(x[-1]) ] for x in lines]
    
    lines.sort(key=lambda x: x[1])

    save = lines[0][1]
    track_labels = []
    track_values = []
    images = []

    for line in lines:
        if save == line[1] and save in [ x[0] for x in gt_tracks]:
            vid.set(cv2.CAP_PROP_POS_FRAMES, line[0]-1)
            ret, orig = vid.read()
            full_frame = orig.copy()

            if not ret:
                break

            fd = line
            x_border = orig.shape[1]
            y_border = orig.shape[0]

            x = max(0,fd[2])
            y = max(0,fd[3])

            x = x - max(0, x+fd[4]-x_border)
            y = y - max(0, y+fd[5]-y_border)

            #print(str(y)+"-"+str(y+fd[5])+"-"+str(x)+"-"+str(x+fd[4]))
            
            orig = orig[y:y+fd[5], x:x+fd[4]]
        
            frame = cv2.resize(orig, (224, 224))

            frame = frame[:, :, ::-1].transpose(2, 0, 1)
            frame = np.ascontiguousarray(frame, dtype=np.float32)
            frame /= 255.0

            frame = torch.from_numpy(frame)
            frame = frame.to(device).unsqueeze(dim=0)    

            with torch.no_grad():
                pred = sm(ident_model(frame))
                ident = torch.argmax(pred).item()
                value = pred[0][ident].item()
                track_labels.append(ident)
                track_values.append(value)
                if len(images) < 2000:
                    images.append(orig)
                    if len(images) == 1999:
                        print("cut_video")
        else:
            if save in [ x[0] for x in gt_tracks]:
                track_labels = np.array(track_labels)
                track_values = np.array(track_values)
                
                unique, counts = np.unique(track_labels, return_counts=True)
                
                if len(unique) > 0:
                    print(save)
                    print(unique)
                    print(counts)
                    print()
                    
                if len(unique) > 0:
                    for i,image in enumerate(images):
                        #print(track_labels[i],end="\r")
                        cv2.imshow("",image)
                        key = cv2.waitKey(60)
                        if key != -1:
                            break
                # reset
                track_labels = []
                track_values = []
                images = []
            
            save = line[1]