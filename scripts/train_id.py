from __future__ import absolute_import, print_function
import torch
import _init_paths

import os
import json

import torch
import os
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import Resize
from torch.utils.data import DataLoader, TensorDataset, Dataset, ConcatDataset
import torchvision

import torch.nn.functional as Func




def crop_and_pad(image, bbox, output_size=(224, 224)):
    x1, y1, x2, y2 = [max(0, val) for val in bbox.squeeze().int().tolist()]
    
    # Extract the bounding box region from the image
    cropped = image[:, :, y1:y2, x1:x2]  # Cropping the image using bounding box coordinates

    # Get dimensions of the cropped region
    h, w = cropped.shape[2:]

    # Determine padding to make it square
    if h > w:
        padding = (h - w) // 2
        padding_dims = (padding, h - w - padding, 0, 0)  # Pad left/right equally, no padding for top/bottom
    else:
        padding = (w - h) // 2
        padding_dims = (0, 0, padding, w - h - padding)  # Pad top/bottom equally, no padding for left/right

    # Apply padding to make the cropped region square
    padded_square = Func.pad(cropped, padding_dims, value=0)


    # Resize the square to 224x224
    resize_transform = Resize(output_size)
    resized = resize_transform(padded_square)

    return resized


def main(opt):
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    #torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Setting up data...')

    dataset_folder_path = opt.output_path

    # Check if the dataset folder exists
    if not os.path.exists(dataset_folder_path):
        # If it doesn't exist, create it
        os.makedirs(dataset_folder_path)


    dset_train = dataset.JointDataset2(
        opt,
        root=opt.id_train_root,
        paths={
            "lemur_train": opt.id_train_file
        },
        augment=True, transforms = T.Compose([T.ToTensor()])
        )

    # dset_val = dataset.ImageLabelDataset( #validation images that are cut out from whole image
    #     root=opt.id_val_root,
    #     file_list=opt.id_val_file,
    #     resize=True
    #     )
    
    dset_val = dataset.JointDataset2(
        opt,
        root=opt.id_val_root,
        paths={
            "lemur_val": opt.id_val_file
        },
        augment=False, transforms = T.Compose([T.ToTensor()])
        )
    print("datasets loaded")
    
    val_labels = []
    cut_out_objects = []

    for i in range(dset_val.__len__()):

        output = dset_val.__getitem__(i)
        bbox = output['bbox']
        input_image = output['input'].clone().detach().unsqueeze(0)
        box_lemur = output['box_lemur_class'].numpy()
        labels = output['gc'].numpy()
        gt_bbox = bbox[:len(box_lemur)][box_lemur == 0] #only predict class 0
        # gt_bbox = bbox
        bboxes = []
        new_bboxes = []
        for bbox in gt_bbox:
            x1, y1, x2, y2 = bbox.tolist()
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            max_side = max(width, height)
            
            new_bbox = [x1, y1, x2, y2]
            new_bboxes.append(new_bbox)


        if len(new_bboxes) != len(labels):
            print(len(new_bboxes))
            print(len(labels))
        for new_bbox in new_bboxes:
            bboxes.append(torch.tensor(new_bbox))
        val_labels.append(labels)

        for bbox in bboxes:
            roi_output = crop_and_pad(input_image, bbox * 4, output_size=(224,224))
            #roi_output = torchvision.ops.roi_align(input_image, [bbox * 4 for bbox in bboxes], output_size=(224,224))
            cut_out_objects.append(roi_output)

    print("val loop done")    
    val_tensor = torch.cat(cut_out_objects, dim=0)
    val_labels = torch.tensor(np.concatenate(val_labels))
    print(val_tensor.shape)
    print(val_labels.shape)

    if len(val_labels) < 1024:
        val_tensor = val_tensor.to(device)
        val_labels = val_labels.to(device)
        val_dataset = TensorDataset(val_tensor, val_labels)

    else:
        data_batch_size = 128
        # Create a list of batches
        data_batches = [val_tensor[i:i+data_batch_size] for i in range(0, len(val_tensor), data_batch_size)]
        label_batches = [val_labels[i:i+data_batch_size] for i in range(0, len(val_labels), data_batch_size)]

        # Create and concatenate batches on GPU
        val_datasets = []
        with torch.no_grad():
            for data_batch, label_batch in zip(data_batches, label_batches):
                data_tensor = data_batch.to(device)
                label_tensor = label_batch.to(device)
                val_datasets.append(dataset.CustomDataset(data_tensor, label_tensor))

                del data_tensor, label_tensor
                torch.cuda.empty_cache()

        # Concatenate datasets
        val_dataset = ConcatDataset(val_datasets)
    print("val dataset done")
    batch_size = 64
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)



    
    print('Creating model...')
        
    progress_file_path = os.path.join(opt.output_path, "training_progress.txt")

    zoom_min = 1.0
    zoom_max = 1.2
    move_px = 5
    
    num_classes = len(opt.gc_cls_names.split(','))

    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr) #0.0001

    # Initialize variables for early stopping
    best_val_accuracy = 0.0
    patience = 20  # Number of epochs to wait for improvement
    counter = 0  # Counter for epochs without improvement

    num_epochs = opt.num_epochs



    for epoch in range(num_epochs):
        cut_out_objects = []
        all_labels = []

        for i in range(dset_train.__len__()):
            output = dset_train.__getitem__(i)
            bbox = output['bbox']
            input_image = output['input'].clone().detach().unsqueeze(0)
            box_lemur = output['box_lemur_class'].numpy()
            labels = output['gc'].numpy()
            gt_bbox = bbox[:len(box_lemur)][box_lemur == 0] #ignore foodboxes
            bboxes = []
            new_bboxes = []
            for bbox in gt_bbox:
                x1, y1, x2, y2 = bbox.tolist()
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Determine the longer side
                max_side = max(width, height)
                
                jitter = random.uniform(zoom_min, zoom_max) # this makes the squared bounding boxes slightly larger or smaller 
                new_width = max_side * jitter
                new_height = max_side * jitter

                move = random.uniform(-move_px, move_px)
                center_x = center_x + move
                center_y = center_y + move
                
                # Calculate new coordinates
                new_x1 = center_x - new_width / 2
                new_y1 = center_y - new_height / 2
                new_x2 = center_x + new_width / 2
                new_y2 = center_y + new_height / 2

                
                new_bbox = [new_x1, new_y1, new_x2, new_y2]
                new_bboxes.append(new_bbox)

            for new_bbox in new_bboxes:
                bboxes.append(torch.tensor(new_bbox))
            #bboxes.append(torch.tensor(new_bboxes))
            all_labels.append(labels)

            for bbox in bboxes:
                roi_output = crop_and_pad(input_image, bbox * 4, output_size=(224,224))
                cut_out_objects.append(roi_output)
            

        full_tensor = torch.cat(cut_out_objects, dim=0)
        full_labels = torch.tensor(np.concatenate(all_labels))

        if len(full_labels) < 1024:
            full_tensor = full_tensor.to(device)
            full_labels = full_labels.to(device)
            train_dataset = TensorDataset(full_tensor, full_labels)

        else:
            data_batch_size = 128
            # Create a list of batches
            print(full_tensor.shape)
            data_batches = [full_tensor[i:i+data_batch_size] for i in range(0, len(full_tensor), data_batch_size)]
            label_batches = [full_labels[i:i+data_batch_size] for i in range(0, len(full_labels), data_batch_size)]

            # Create and concatenate batches on GPU
            train_datasets = []
            with torch.no_grad():
                for data_batch, label_batch in zip(data_batches, label_batches):
                    data_tensor = data_batch.to(device)
                    label_tensor = label_batch.to(device)
                    train_datasets.append(dataset.CustomDataset(data_tensor, label_tensor))

                    del data_tensor, label_tensor
                    torch.cuda.empty_cache()

            # Concatenate datasets
            train_dataset = ConcatDataset(train_datasets)

        

        batch_size = 128
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        


        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0


        # Split the tensor into smaller batches
        num_batches = 1 #(full_tensor.size(0) + batch_size - 1) // batch_size

        for i in range(num_batches):
            
            for inputs, labels in train_loader:
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

            with torch.no_grad():  # No need to compute gradients during validation
                for val_inputs, val_labels in val_loader:
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
        torch.save(new_model_state_dict, f'{opt.output_path}/model_checkpoint_{epoch+1}.pth')

        if val_epoch_accuracy > best_val_accuracy:
            best_val_accuracy = val_epoch_accuracy
            counter = 0
        else:
            counter += 1

        # Check if early stopping criteria are met
        if counter >= patience:
            print("Early stopping...")
            with open(progress_file_path, "a") as progress_file:
                progress_file.write("Early stopping...")
            break

        torch.cuda.empty_cache()




if __name__ == '__main__':
    torch.cuda.set_device(0)
    opt = opts().parse()
    main(opt)
