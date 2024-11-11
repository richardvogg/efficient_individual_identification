import os
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import random
import torchvision
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
import math
import sys

class GeorgDataset(Dataset):
    def __init__(self, lines,  transform, file_dir_1, file_dir_2, file_dir_3):
        self.lines = lines
        self.transform = transform
        self.file_dir_1 = file_dir_1
        self.file_dir_2 = file_dir_2
        self.file_dir_3 = file_dir_3

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        image_name, fr, label, id = self.lines[idx]

        if id == 1:
            dir = self.file_dir_1
        elif id == 2:
            dir = self.file_dir_2
        else:
            dir = self.file_dir_3

        image_path = os.path.join(dir+image_name)

        img = cv2.imread(image_path)
        if img is None:
            print(image_path)
        img = cv2.resize(img, (224, 224))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        
        img = torch.from_numpy(img)
        img = self.transform(img)
        img /= 255.0
        
        label = torch.from_numpy(np.array(int(label)))
       
        return img, label

for size_of_data in (3000,4000,5000,7500,9950):
    print("\n\n\n\n\n\n\n\n\n")
    for seed in range(3):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()

        device = "cuda:0"

        num_classes = 8 

        model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00005)

        # Initialize variables for early stopping
        best_val_loss = float('inf')
        patience = 10  # Number of epochs to wait for improvement
        counter = 0  # Counter for epochs without improvement

        # Step 4: Train the model
        num_epochs = 200

        model_name = "scaling_test_"+str(size_of_data)+"_"+str(seed)+".pth"

        file_dir_1 = "../data/second_try/99_model_labeled_data/99_all_data_without_sort/images/"
        file_dir_2 = "../data/second_try/A_frames_hand_labeled/images/"
        file_dir_3 = "../data/second_try/counter_example_model_labeled_data/images/"

        ds = 1
        if size_of_data > 1000:
            ds = 2
        if size_of_data > 2000:
            ds = 5
        if size_of_data > 5000:
            ds = 10
        
        labels_file_1 = "../data/second_try/99_model_labeled_data/99_all_data_without_sort/"+str(ds)+"k"
        labels_file_2 = "../data/second_try/A_frames_hand_labeled/labels"

        with open(labels_file_1, 'r') as file:
            lines = file.readlines()
        
        lines = [ x.replace("\n","").split(",") for x in lines]
        lines = [ [str(x[0]), int(x[1]), int(x[3]), 1] for x in lines]

        with open(labels_file_2, 'r') as file:
            lines_2 = file.readlines()

        lines_2 = [ x.replace("\n","").replace(" ","").split(",") for x in lines_2]
        lines_2 = [ [str(x[0]) + "_" + str(x[1]) + ".png", int(x[1]), int(x[2]), 2] for x in lines_2]

        lines_3 = os.listdir(file_dir_3)
        lines_3 = [ [x.replace("\n",""), 1, 7, 3] for x in lines_3]

        alle = lines + lines_3

        seven = [x for x in alle if x[2] == 7]
        not_seven = [x for x in alle if x[2] != 7]

        if size_of_data < len(not_seven):
            not_seven = random.sample(not_seven, size_of_data)
        seven = random.sample(seven, int(len(not_seven)/2))

        if len(not_seven)/2 - len(seven) > 0:
            seven += random.sample(lines_3, int(len(not_seven)/2 - len(seven)))
        
        alle = seven + not_seven

        if seed == 0:
            print("example: ",len(not_seven))
            print("counter_example: ",len(seven))
            sys.stdout.flush()
        
        X_train, X_test, _, _ = train_test_split(alle, [0 for x in range(len(alle))], test_size=0.2, random_state=seed)

        # use all augmentations here
        transform = v2.Compose([
            v2.RandomResizedCrop(size=(224,224), scale=(0.8,1)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=45),
        ])

        train_dataset = GeorgDataset(X_train, transform, file_dir_1,file_dir_2, file_dir_3)
        transform = lambda x: x
        val_dataset = GeorgDataset(X_test, transform, file_dir_1,file_dir_2, file_dir_3)

        batch_size = 128
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            model.to(device)
            model.train()  
            running_loss = 0.0
            correct = 0
            total = 0

            # Split the tensor into smaller batches
            num_batches = 1 

            # Process each batch separately
            for i in range(num_batches):

                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    inputs = inputs.to(device)
                    labels = labels.type(torch.LongTensor).to(device)

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

                with torch.no_grad():  # No need to compute gradients during validation
                    for val_inputs, val_labels in val_loader:
                        val_inputs = val_inputs.to(device)
                        val_labels = val_labels.type(torch.LongTensor).to(device)

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
                sys.stdout.flush()

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
            
            torch.save(new_model_state_dict, model_name)

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                counter = 0
            else:
                counter += 1

            # Check if early stopping criteria are met
            if counter >= patience:
                print("Early stopping...")
                break