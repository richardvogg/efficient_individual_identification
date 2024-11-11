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

class GeorgDataset(Dataset):
    def __init__(self, root_dir_1, lines):
        self.root_dir_1 = root_dir_1
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        image_name, fr, label, id = self.lines[idx]
            
        image_path = os.path.join(self.root_dir_1, image_name)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        img = torch.from_numpy(img)
        
        label = torch.from_numpy(np.array(int(label))) 

        return img, label

# path
data_path = "other/videos/"
sorting_model_path = "other/models/binary_model.pth"
ident_model_path = "other/models/best1.pth"
write_path = "other/save_pseudo/"

# Threshholds for id and presorting module
threshhold_id = 0.9
threshhold_sort = 0

# flag to only apply to id module
only_apply_ID = False

device = "cpu"
num_models = 1
batch_size = 10
classes = 8

sm = nn.Softmax(dim=1)

# load ID module
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

# load presorting module
sorting_model = get_model("resnet18", weights="IMAGENET1K_V1")    
last_layer = nn.Sequential(
    nn.Linear(sorting_model.fc.in_features, 2),
    nn.Softmax()
)
sorting_model.fc = last_layer
sorting_model.to(device)
sorting_model.load_state_dict(torch.load(sorting_model_path,map_location=device))
sorting_model.eval()

files = os.listdir(data_path)
files = list(set(files))
files = set([x[:-4] for x in files])
labels = []

for file in files:
    with open(data_path+file+".txt", "r") as f:
        lines = f.readlines()
    
    lines = [x.replace("\n","").replace(" ","").split(",") for x in lines]
    lines = [[int(x[0]), int(x[1]), int(float(x[2])), int(float(x[3])), int(float(x[4])), int(float(x[5])) ] for x in lines if x[7] == '0']
    lines.sort(key=lambda x: x[0])

    vid = cv2.VideoCapture(data_path+"/"+file+".mp4")
    if not vid.isOpened:
        print("There has been an error opening the video")

    print(int(vid.get(cv2.CAP_PROP_FRAME_COUNT)))
  
    breaking = False

    data_iter = iter(lines)

    counter_vid = 0
    pos_lines = 0
    

    ident = 0
    stop = False

    while True:
        ret, orig = vid.read()

        if not ret or stop:
            break
        
        orig = orig.copy()
        
        fd = lines[pos_lines]
        x_border = orig.shape[1]
        y_border = orig.shape[0]

        x = max(0,fd[2])
        y = max(0,fd[3])

        x = x - max(0, x+fd[4]-x_border)
        y = y - max(0, y+fd[5]-y_border)

        orig = orig[y:y+fd[5], x:x+fd[4]]
    
        orig = cv2.resize(orig, (224, 224))

        while counter_vid == lines[pos_lines][0]:
            
            frame = orig.copy()
            frame = frame[:, :, ::-1].transpose(2, 0, 1)
            frame = np.ascontiguousarray(frame, dtype=np.float32)
            frame /= 255.0

            frame = torch.from_numpy(frame)
            frame = frame.to(device).unsqueeze(dim=0)    

            with torch.no_grad():
                pred = sm(ident_model(frame))
                ident = torch.argmax(pred).item()
                value = pred[0][ident].item()
            
            if torch.argmax(pred).item() != 7 and value > threshhold_id:
                if only_apply_ID:
                    frame = orig.copy()
                    cv2.imwrite(write_path+"images/"+file+"_"+str(fd[0])+"_"+str(fd[1])+"_"+str(ident)+".jpg",frame)
                    labels.append([file,str(fd[0]),str(fd[1]), ident])
                else:
                    frame = orig.copy()

                    frame = orig.copy()
                    frame = frame[:, :, ::-1].transpose(2, 0, 1)
                    frame = np.ascontiguousarray(frame, dtype=np.float32)
                    frame /= 255.0

                    frame = torch.from_numpy(frame)
                    frame = frame.to(device).unsqueeze(dim=0)       

                    with torch.no_grad():
                        pred_2 = sorting_model(frame)
                        ident_2 = torch.argmax(pred_2).item()
                        value = pred_2[0][ident_2].item()
                    
                    if torch.argmax(pred_2).item() == 1 and value > threshhold_sort:
                        frame = orig.copy()
                        
                        cv2.imwrite(write_path+"images/"+file+"_"+str(fd[0])+"_"+str(fd[1])+"_"+str(ident)+".jpg",frame)
                        labels.append([file,str(fd[0]),str(fd[1]), ident])
            
            pos_lines += 1
            if pos_lines >= len(lines):
                stop = True
                break
                  
        counter_vid += 1
        
f = open(write_path+"labels", "a")
for x in labels:
    f.write(str(x[0])+","+str(x[1])+","+str(x[2])+","+str(x[3])+"\n")
f.close()
labels = []

print("done")
