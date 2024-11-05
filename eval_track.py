import os
import cv2
import torch
import time
import numpy as np

from torch import nn
from torchvision.models import get_model, list_models
from torchvision.transforms import v2
from sklearn.cluster import KMeans

def get_file_names(dir):
    files = os.listdir(dir)
    unique = []

    for f in files:
        if f[:-4] not in unique:
            unique.append(f[:-4])
    
    return unique

def create_model(efficient_net_nr, classes):
    model = get_model("resnet"+efficient_net_nr, weights="IMAGENET1K_V1")
    
    last_layer = nn.Sequential(
        nn.Linear(model.fc.in_features, classes),
        nn.Softmax()
    )

    model.fc = last_layer
    return model

def create_model_res(res):
    model = get_model("resnet"+res, weights="IMAGENET1K_V1")
    
    last_layer = nn.Sequential(
        nn.Linear(model.fc.in_features,3),
        nn.Softmax() )
    model.fc = last_layer

    return model

def read_txt(file):
    data = []

    with open(path_data+file+".txt", "r") as f:
        file_content = f.readlines()
    
    for line in file_content:
        line = line.replace("\n","").replace(",","").split(" ")
        
        if line[7] == "0":
            data.append( [int(line[0]), int(float(line[2])), int(float(line[3])), int(float(line[4])), int(float(line[5])), int(line[1])] )
    return data

# get frame, resize, normalise
def jump_extract(fd, vid):
    vid.set(cv2.CAP_PROP_POS_FRAMES, fd[0]-1)
    res, frame = vid.read()
    
    return frame

def simple_extract(vid):
    res, frame = vid.read()
    return frame

def mod_frame(frame, fd):
    x_border = frame.shape[1]
    y_border = frame.shape[0]

    x = max(0,fd[1])
    y = max(0,fd[2])

    x = x - max(0, x+fd[3]-x_border)
    y = y - max(0, y+fd[4]-y_border)

    frame = frame[y:y+fd[4], x:x+fd[3]]
    copy = frame.copy()

    # resize
    frame= cv2.resize(frame, (224,224), cv2.INTER_AREA)

    frame = torch.from_numpy(frame).float()
    frame = frame.permute(2,0,1)
    frame = frame/255
    
    return frame, copy

def eval_bb( frame, models ):
    X = frame

    with torch.no_grad():
        pred = models[0](X)

    for i in range(1, len(models)):
        with torch.no_grad():
            pred += models[i](X)
        
    return pred[:,1]/len(models), torch.argmax(pred,dim=1)

path_data = "../data/all_data_learning/"
device = "cuda:0"
num_models = 1
batch_size = 100
dir_save = "frames/"

models = []

print("start")

model = create_model("18",2)
model.to(device)
model.load_state_dict(torch.load("save/quality_model_binary.pth",map_location='cuda:0'))
model.eval()
models.append(model)

print("model loaded")

for file in get_file_names(path_data):
    print(file)
    data = read_txt(file)

    vid = cv2.VideoCapture(path_data+file+".mp4")
    if not vid.isOpened:
        print("There has been an error opening the video")

    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    good_frames = []
    breaking = False
    data_iter = iter(data)

    batch = []
    copy_frames = []
    fds = []
    counter = 0
    fd_last = -1
    copy_last = None
    
    eval_count = 0

    while True:
        fd = next(data_iter, None)

        if fd == None or len(batch) == batch_size:
            eval_count += 1
            print("eval: ", eval_count*batch_size)
            batch = torch.stack((batch))
            batch = batch.to(device)

            certainty, result  = eval_bb( batch, models )
            #print(result)
            certainty = torch.round(certainty,decimals=3)

            for x in range(batch.shape[0]):
                if result[x].item() == 1:
                    status = cv2.imwrite(dir_save +str(file)+"_"+str(fds[x][0])+"_"+str(fds[x][5])+"_"+str(round(certainty[x].item(),3))+'.jpg',copy_frames[x])
                    if not status:
                        print("Error writing !!!")
            batch = []
            copy_frames = []
            fds = []

            
        if fd == None:
            print("break")
            break
        
        if fd[0] - 1 == fd_last:
            frame = simple_extract(vid)
        elif fd[0] == fd_last:
            frame = copy_last.copy()
        else:
            frame = jump_extract(fd,vid)

        copy_last = frame.copy()
        frame, copy = mod_frame(frame, fd)
        batch.append(frame)
        copy_frames.append(copy.copy())
        fds.append(fd)
        fd_last = fd[0]

    break
