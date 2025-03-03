import torch 
import os
import cv2
import sys
import torchvision
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision.models import get_model
from torchvision.transforms import v2

class simple_dataset(Dataset):
    def __init__(self, X, y, train):
        self.X = X.copy()
        self.y = y.copy()
        self.train = train
        if len(X) != len(y):
            raise Exception("Not possible !!!")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = self.X[idx].copy()
        img = cv2.resize(img,(224,224), cv2.INTER_AREA)   

        transforms = v2.Compose([
            #v2.RandomResizedCrop(size=(224,224), scale=(0.8,1)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            #v2.RandomRotation(degrees=45),
        ])

        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)
        img = img / 255.0
        
        if self.train:
            img = transforms(img)

        return img.float(), torch.tensor(self.y[idx]).to(torch.float)

class EarlyStopping:
    def __init__(self, checkpoint_path, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.model_checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation Loss Decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_checkpoint_path)
        self.val_loss_min = val_loss

def load_txt(fn):
    with open(fn,"r") as f:
        file = f.readlines()

    bb = []
    for line in file:
       bb.append(line.replace("\n", "").replace(" ", "").split(","))

    return bb

def create_model(efficient_net_nr, classes):
    model = get_model("resnet"+efficient_net_nr, weights="IMAGENET1K_V1")
    
    last_layer = nn.Sequential(
        nn.Linear(model.fc.in_features, classes),
        nn.Softmax()
    )

    model.fc = last_layer
    return model

def accuracy(y, pred):  
    pred_max = torch.argmax(pred, dim=1)
    y_max = torch.argmax(y, dim=1)
    
    return (sum(y_max==pred_max)/len(y_max)).cpu()

def conv_matrix(pred,y):
    pred_max = torch.argmax(pred, dim=1)
    y_max = torch.argmax(y,dim=1)
    conv = np.zeros((2,2))

    for i,j in zip(pred_max,y_max):
        conv[i][j] += 1
    return conv

def train(dataloader, optimizer, model, loss_fn, device):   
    model.train()
    losses = []
    acc = []
    # Loop over each batch of data provided by the dataloader
    for X, y in dataloader:
        X, y = X.to(device), y.type(torch.FloatTensor).to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        losses.append(loss.item())
        acc.append(accuracy(y, pred))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("train loss: ",sum(losses) / len(losses), ", train acc:  ",sum(acc) / len(acc))
    return sum(losses) / len(losses), sum(acc) / len(acc)

def validate(dataloader, model, loss_fn, device):
    model.eval()
    losses = []
    acc = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.type(torch.FloatTensor).to(device)
            pred = model(X).squeeze()
            losses.append( loss_fn(pred, y).item() )
            acc.append(accuracy(y, pred))
    print("val loss: ",sum(losses) / len(losses), ", val acc:  ",sum(acc) / len(acc))
    return sum(losses) / len(losses), sum(acc) / len(acc)

def run_training(model, optimizer, loss_function, device, num_epochs, train_dataloader, val_dataloader, early_stopping):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print("Epoch: ",epoch)
        sys.stdout.flush()

        train_loss, train_acc = train( train_dataloader, optimizer, model, loss_function, device )
        
        val_loss, val_acc = validate( val_dataloader, model, loss_function, device )
        
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early Stopp !!!")
            break
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
    return train_losses, val_losses, train_accs, val_accs

def testing(dataloader, model, device):
    model.eval()
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()

    loss_mse = []
    loss_msa = []
    acc = []
    conv = np.zeros((2,2))
    preds = []
    ys = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            ys.append(y)
            pred = model(X)
            preds.append(pred)
            loss_msa.append( l1(pred, y).item() )
            loss_mse.append( l2(pred, y).item() )
            acc.append(accuracy(y,pred))
            conv += conv_matrix(pred, y)
            
    mse = sum(loss_mse) / len(loss_mse)
    msa = sum(loss_msa) / len(loss_msa)
    acc = sum(acc)/len(acc)

    return mse, msa, acc.item(), conv, (preds[0],ys[0])

device = "cuda"

# number of classes
classes = 2

# path where the model is written to 
if not os.path.exists("models"):
    os.makedirs("models")
model_path = "models/binary.pth"
model = create_model(str(18),classes)
model.to(device)

epochs = 300
lr = 0.0001
batch_size = 128

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

data_path = "annotation_images/"
with open(os.path.join(data_path, "labels.txt"), 'r') as f:
    lines = f.readlines()

early_stopping = EarlyStopping(model_path, patience=10, verbose=False, delta=0)
model.to(device)

y = []
X = []
# IDs -> binary labels
for x in lines:
    x = x.replace(" ","").replace("\n","").split(",")
    im = cv2.imread(os.path.join(data_path, "images/", x[0]))
    X.append(im)
    if int(x[3]) == 0:
        y.append([1,0])
    else:
        y.append([0,1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=41)

print("done")
train_loader = DataLoader(simple_dataset(X_train, y_train, True), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(simple_dataset(X_test, y_test, False), batch_size=batch_size, shuffle=True) 
val_loader = DataLoader(simple_dataset(X_val, y_val, False), batch_size=batch_size, shuffle=True)

train_losses, val_losses, train_accs, val_accs = run_training(model, optimizer, loss_function, device, epochs, train_loader, val_loader, early_stopping)

print("------------------------------------------")
#print(train_losses)
#print(val_losses)

model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)

mse, msa, acc, conv, e = testing(test_loader, model, device)

print(acc)
print(conv)
