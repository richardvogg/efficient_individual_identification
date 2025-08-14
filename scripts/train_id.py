import os

import argparse
import torch
import os
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms as T
from torchvision.transforms import Resize
from torch.utils.data import DataLoader, Dataset
import torchvision

import torch.nn.functional as Func

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="/usr/users/vogg/Labelling/Lemurs/labelling_app_indID/experiments/")
parser.add_argument('--output_path', type=str, default="models/id/")
parser.add_argument('--experiment', type=str, default="cluster_1000_5000")
parser.add_argument('--group', type=str, default="R1")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--gpus', type=int, nargs='+', default=[0])
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--patience', type=int, default=10)
args = parser.parse_args()


class IDDataset(Dataset):
    def __init__(self, root_path, txt_filename='cluster_1000_5000_R1.txt', train=True, transform=None, test_size=0.2, unknown_prop = 1, random_state=64):
        self.root_path = root_path
        self.image_dir = os.path.join(root_path, "images")
        self.txt_path = os.path.join(root_path, txt_filename)
        self.transform = transform

        # Read label file
        df = pd.read_csv(self.txt_path, header=None,
                         names=["filename", "x", "y", "w", "h", "id", "score"])

        # Extract experiment name for splitting (e.g. "R_e1_c1" from "R_e1_c1_31257.png")
        df["experiment"] = df["filename"].apply(lambda x: "_".join(x.split("_")[:-1]))

        if 0 < unknown_prop < 1:
            max_id = df["id"].max()
            max_id_rows = df[df["id"] == max_id]
            sampled_max_id_rows = max_id_rows.sample(frac=unknown_prop, random_state=random_state)
            df = pd.concat([df[df["id"] != max_id], sampled_max_id_rows]).reset_index(drop=True)

        # Split by experiment
        experiments = df["experiment"].unique()
        train_exps, test_exps = train_test_split(
            experiments, test_size=test_size, random_state=random_state)

        if train:
            self.df = df[df["experiment"].isin(train_exps)].reset_index(drop=True)
        else:
            self.df = df[df["experiment"].isin(test_exps)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")


        x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
        bbox = torch.tensor([int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)])
        image_tensor = T.ToTensor()(image).unsqueeze(0)
        cropped = self.crop_and_pad(image=image_tensor, bbox=bbox, output_size=(224, 224)).squeeze(0)


        label = int(row["id"])
        return cropped, label




    def crop_and_pad(self, image, bbox, output_size=(224, 224)):
        x1, y1, x2, y2 = [max(0, val) for val in bbox.squeeze().int().tolist()]
        
        cropped = image[:, :, y1:y2, x1:x2] 

        h, w = cropped.shape[2:]

        # Determine padding to make it square
        if h > w:
            padding = (h - w) // 2
            padding_dims = (padding, h - w - padding, 0, 0)  # Pad left/right equally, no padding for top/bottom
        else:
            padding = (w - h) // 2
            padding_dims = (0, 0, padding, w - h - padding)  # Pad top/bottom equally, no padding for left/right

        padded_square = Func.pad(cropped, padding_dims, value=0)

        resize_transform = Resize(output_size)
        resized = resize_transform(padded_square)

        return resized


def main(args):
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if args.gpus[0] >= 0 else 'cpu')

    print('Setting up data...')

    data_path = os.path.join(args.data_path, args.experiment, args.group)

    transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dset_train = IDDataset(data_path, args.experiment + "_" + args.group + ".txt", train=True, transform=transform, test_size=0.2, unknown_prop=1, random_state=64)
    dset_val = IDDataset(data_path, args.experiment + "_" + args.group + ".txt", train=False, transform=transform, test_size=0.2, unknown_prop=1, random_state=64)
    print("datasets loaded")
    
    train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(dset_val, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4)

    print('Creating model...')

    progress_file_path = os.path.join(args.output_path, args.group, args.experiment, "training_progress.txt")
    os.makedirs(os.path.dirname(progress_file_path), exist_ok=True)

    num_classes = max(dset_train.df['id']) + 1

    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr) #0.0001

    # Initialize variables for early stopping
    best_val_accuracy = 0.0
    counter = 0  # Counter for epochs without improvement

    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Use tqdm for progress bar in training
        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Compute statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        model.eval()  # Set model to evaluation mode
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        # Use tqdm for progress bar in validation
        with torch.no_grad():  # No need to compute gradients during validation
            for val_inputs, val_labels in tqdm(val_loader, desc="Validation", leave=False):
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                
                # Compute statistics
                val_running_loss += val_loss.item() * val_inputs.size(0)
                _, val_predicted = val_outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += val_predicted.eq(val_labels).sum().item()

        
        # Print statistics for the current epoch
        epoch_loss = running_loss / total
        epoch_accuracy = 100. * correct / total

        # Print statistics for the current epoch
        val_epoch_loss = val_running_loss / val_total
        val_epoch_accuracy = 100. * val_correct / val_total

        progress = f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.2f}%'
        print(progress)
        with open(progress_file_path, "a") as progress_file:
            progress_file.write(progress + '\n')

        # Save model checkpoint
        prefix = 'cnn.'  # Define the prefix you want to add

        # Get the state dictionary of the model
        model_state_dict = model.state_dict()

        # Create a new state dictionary with modified keys
        new_model_state_dict = {}
        for key, value in model_state_dict.items():
            new_key = prefix + key  # Add the prefix to the key
            new_model_state_dict[new_key] = value

        # Save the new state dictionary to the checkpoint
        torch.save(new_model_state_dict, f'{args.output_path}/{args.group}/{args.experiment}/model_checkpoint_{epoch+1}.pth')

        if val_epoch_accuracy > best_val_accuracy:
            best_val_accuracy = val_epoch_accuracy
            counter = 0
        else:
            counter += 1

        # Check if early stopping criteria are met
        if counter >= args.patience:
            print("Early stopping...")
            with open(progress_file_path, "a") as progress_file:
                progress_file.write("Early stopping...")
            break

        torch.cuda.empty_cache()




if __name__ == '__main__':
    torch.cuda.set_device(0)
    args = parser.parse_args()
    main(args)
