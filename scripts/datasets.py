import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision.transforms import transforms as T
from torchvision.transforms import Resize
import torch.nn.functional as Func
import torchvision.transforms.functional as F



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