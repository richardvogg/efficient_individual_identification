import os

import argparse
import torch
import torchvision
from torch.utils.data import Dataset
import cv2
import torchvision.transforms.functional as F
import torch.nn.functional as Func
from torchvision.transforms import Resize
import torch.nn as nn
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('--annotation_dir', type=str, default="/usr/users/vogg/sfb1528s3/B06/Annotations/Exp1_lifting/individual_id/")
parser.add_argument('--experiment_name', type=str, default="A_e7_c1")
parser.add_argument('--video_path', type=str, default="/usr/users/vogg/sfb1528s3/B06/2023april-july/NewBoxesClosed/Converted/")
parser.add_argument('--model_path', type=str, default="models/id/Alpha/cluster_1000_5000/model_checkpoint_11.pth")
args = parser.parse_args()

class LoadVideoAndLabels(Dataset):
    def __init__(self, annotation_dir, video_dir, transform=None):
        self.annotation_dir = annotation_dir
        self.video_dir = video_dir
        self.transform = transform
        self.samples = []
        self._parse_annotations()

    def _parse_annotations(self):
        filename = f"{args.experiment_name}_identification.txt"
        
        if args.experiment_name[0] == 'A':
            group = "Alpha"
        elif args.experiment_name[0] == 'R':
            group = "R1"
        else:
            group = args.experiment_name[0]

        annotation_path = os.path.join(self.annotation_dir, filename)
        video_filename = args.experiment_name + ".mp4"
        video_path = os.path.join(self.video_dir, group, video_filename)

        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            parts = line.split(',')
            if len(parts) < 10:
                continue

            if parts[9].strip() != '1':
                continue

            frame_num = int(float(parts[0].strip()))
            x = float(parts[2].strip())
            y = float(parts[3].strip())
            w = float(parts[4].strip())
            h = float(parts[5].strip())
            label = int(parts[8].strip())  # nameOrder

            # Convert to top-left bottom-right format
            tlbr = torch.Tensor([x, y, x + w, y + h])
            self.samples.append((video_path, frame_num, tlbr, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_num, tlbr, label = self.samples[idx]


        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Could not read frame {frame_num} from {video_path}")

        # Convert frame to RGB tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(frame_rgb).unsqueeze(0)  # shape: [3, H, W], float32 in [0,1]

        # Crop using your custom function
        cropped = self.crop_and_pad(image_tensor, tlbr).squeeze(0)

        if self.transform:
            cropped = self.transform(cropped)

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
    dataset = LoadVideoAndLabels(args.annotation_dir, args.video_path)
    num_classes = 13 # Alpha = 13, B = 10, J = 9, R1 = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    # Load checkpoint and remove "cnn." prefix from keys if present
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint
    # If checkpoint is a dict with 'state_dict' key, use that
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("cnn."):
            new_k = k[len("cnn."):]
        else:
            new_k = k
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
   
    
    for i in range(len(dataset)):
        cropped_image, label = dataset[i]
        cropped_image = cropped_image.to(device)
        model.eval()
        with torch.no_grad():
            output = model(cropped_image.unsqueeze(0))
            print(f"Sample {i}: Model Output: {output}", f"Label: {label}")  

    # Here you can add code to use the dataset with a DataLoader or model training

if __name__ == '__main__':
    torch.cuda.set_device(0)
    args = parser.parse_args()
    main(args)
