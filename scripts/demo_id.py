import os

import argparse
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms.functional as F
import torch.nn.functional as Func
from torchvision import transforms
from torchvision.transforms import Resize
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
from scipy.stats import entropy



parser = argparse.ArgumentParser()
parser.add_argument('--annotation_dir', type=str, default="data/id_annotation/")
parser.add_argument('--experiment_name', type=list, default=["A_e7_c1", "A_e7_c3", "A_e8_c1", "A_e8_c3", "A_e9_c1", "A_e9_c3", "A_e10_c2", "A_e10_c3"]) #"A_e8_c1", 
parser.add_argument('--video_path', type=str, default="/usr/users/vogg/sfb1528s3/B06/2023april-july/NewBoxesClosed/Converted/")
parser.add_argument('--model_path', type=str, default="models/id/Alpha/limit10_1000_1000/model_checkpoint_29.pth")
args = parser.parse_args()

class LoadVideoAndLabels(Dataset):
    def __init__(self, annotation_dir, video_dir, experiment_name, transform=None):
        self.annotation_dir = annotation_dir
        self.video_dir = video_dir
        self.experiment_name = experiment_name
        self.transform = transform
        self.samples = []
        self._parse_annotations()

    def _parse_annotations(self):
        filename = f"{self.experiment_name}_identification.txt"

        if self.experiment_name[0] == 'A':
            group = "Alpha"
        elif self.experiment_name[0] == 'R':
            group = "R1"
        else:
            group = self.experiment_name[0]

        first_char = self.experiment_name[0]
        if first_char == 'A':
            self.num_classes = 13
        elif first_char == 'B':
            self.num_classes = 10
        elif first_char == 'J':
            self.num_classes = 9
        elif first_char == 'R':
            self.num_classes = 8
        else:
            raise ValueError(f"Unknown experiment group '{first_char}' in experiment_name")

        annotation_path = os.path.join(self.annotation_dir, self.experiment_name, filename)
        video_filename = self.experiment_name + ".mp4"
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

            if int(parts[8].strip()) >= self.num_classes - 1:
                continue

            frame_num = int(parts[0].strip())
            track_id = int(parts[1].strip())
            x = float(parts[2].strip())
            y = float(parts[3].strip())
            w = float(parts[4].strip())
            h = float(parts[5].strip())
            label = int(parts[8].strip())  # nameOrder

            # Convert to top-left bottom-right format
            tlbr = torch.Tensor([x, y, x + w, y + h])
            self.samples.append((video_path, frame_num, track_id, tlbr, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_num, track_id, tlbr, label = self.samples[idx]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            # wait for 1 sec and try again
            time.sleep(1)
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
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


        return cropped, label, track_id, frame_num

    
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


class PrecomputedCropDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: path to precomputed dataset folder
            transform: torchvision transforms to apply
        """
        self.root_dir = root_dir
        self.metadata = pd.read_csv(os.path.join(root_dir, "val_labels.csv"))
        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.root_dir, "images", row['filename'])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        experiment = row['experiment']
        label = int(row['label'])
        track_id = int(row['track_id'])
        frame_num = int(row['frame_num'])

        return img, experiment, label, track_id, frame_num



def main(args):
    #dataset = LoadVideoAndLabels(args.annotation_dir, args.video_path)
    dataset = PrecomputedCropDataset(root_dir=f"data/video_valset_cropped", transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    # have a dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)


    # set num_classes based on experiment group
    first_char = args.experiment_name[0][0]
    if first_char == 'A':
        num_classes = 13  # Alpha
    elif first_char == 'B':
        num_classes = 10
    elif first_char == 'J':
        num_classes = 9
    elif first_char == 'R':
        num_classes = 8  # R1
    else:
        raise ValueError(f"Unknown experiment group '{first_char}' in experiment_name")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
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
    
    records = []
    
    for batch in dataloader:
        images, experiments, labels, track_ids, frame_nums = batch
        images = images.to(device)

        model.eval()
        with torch.no_grad():
            logits = model(images)                     # [B, C]
            probs = Func.softmax(logits[:, :], dim=1)        # [B, C] was -1, removing Unknown

        logits_np = logits.cpu().numpy()
        probs_np = probs.cpu().numpy()

        # ensure labels/ids/frame nums are numpy arrays for iteration
        if torch.is_tensor(labels):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = np.array(labels)

        if torch.is_tensor(track_ids):
            track_ids_np = track_ids.cpu().numpy()
        else:
            track_ids_np = np.array(track_ids)

        if torch.is_tensor(frame_nums):
            frame_nums_np = frame_nums.cpu().numpy()
        else:
            frame_nums_np = np.array(frame_nums)

        for tid, experiment, frame, lab, logit_vec, prob_vec in zip(track_ids_np, experiments, frame_nums_np, labels_np, logits_np, probs_np):
            records.append({
                'experiment': experiment,
                'track_id': int(tid),
                'frame_num': int(frame),
                'label': int(lab),
                'logits': logit_vec,
                'probs': prob_vec,
            })

    df = pd.DataFrame(records)
    # save this dataframe
    output_csv = f"{first_char}_identification_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved frame-level results to {output_csv}")



    def aggregate_track_predictions(
        track_df, 
        method='mean_logit', 
        threshold=0.6, 
        unsure_index=-1
    ):
        """
        Aggregate predictions for one track using various strategies.
        
        Parameters
        ----------
        track_df : pd.DataFrame
            Must contain columns: ['logits', 'probs', 'label'].
        method : str
            One of ['mean_logit', 'max_logit', 'threshold_count', 'entropy', 'weighted_confidence'].
        threshold : float
            Used for threshold-based voting.
        unsure_index : int
            Index of the "Unsure" class to ignore.
        """

        # Stack model outputs
        logits = np.stack(track_df['logits'].to_numpy())  # [T, n_classes]
        probs = np.stack(track_df['probs'].to_numpy())    # [T, n_classes]
        label = track_df['label'].iloc[0]

        #if unsure_index is not None:
            # Remove the "Unsure" column
            #logits = np.delete(logits, unsure_index, axis=1)
            #probs = np.delete(probs, unsure_index, axis=1)

        if method == 'mean_logit':
            mean_logits = logits.mean(axis=0)
            final_pred = mean_logits.argmax()

        elif method == 'max_logit':
            max_logits = logits.max(axis=0)
            final_pred = max_logits.argmax()

        elif method == 'threshold_count':
            # Count how many times each class exceeds threshold probability
            counts = (probs > threshold).sum(axis=0)
            final_pred = counts.argmax()
            if final_pred == unsure_index:
                sorted_idx = np.argsort(counts)[::-1]  # descending
                if len(sorted_idx) >= 2:
                    second_idx = int(sorted_idx[1])
                    if counts[second_idx] > 1:
                        final_pred = second_idx

        elif method == 'entropy':
            # Compute entropy per frame and select the most confident frames
            ent = entropy(probs.T)  # entropy over classes for each frame
            inv_ent = 1.0 / (ent + 1e-8)  # inverse entropy as confidence weight
            weighted_probs = (probs * inv_ent[:, None]).sum(axis=0) / inv_ent.sum()
            final_pred = weighted_probs.argmax()

        elif method == 'weighted_confidence':
            # Compute per-frame confidence (max prob)
            conf = probs.max(axis=1)
            weights = np.exp(9.2 * (conf - 0.5))  # exponential weighting
            weighted_probs = (probs * weights[:, None]).sum(axis=0) / weights.sum()
            final_pred = weighted_probs.argmax()
            # If the top prediction is the "Unsure" class, check the second best
            if final_pred == unsure_index:
                sorted_idx = np.argsort(weighted_probs)[::-1]  # descending
                if len(sorted_idx) >= 2:
                    second_idx = int(sorted_idx[1])
                    # If the second highest value exceeds threshold 1.0, choose it
                    if weighted_probs[second_idx] > 0.1:
                        final_pred = second_idx

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        return {'label': label, 'pred': final_pred}


        
    track_results = []

    for (experiment, track_id), group in df.groupby(['experiment', 'track_id']):
        result = aggregate_track_predictions(group, method='weighted_confidence', threshold=0.5, unsure_index=num_classes-1)
        result['track_id'] = track_id
        result['experiment'] = experiment
        track_results.append(result)

    track_df = pd.DataFrame(track_results)
    track_output_csv = f"{first_char}_track_identification_results.csv"
    track_df.to_csv(track_output_csv, index=False)
    print(f"Saved track-level results to {track_output_csv}")
    accuracy = (track_df['label'] == track_df['pred']).mean()
    print(f"Trackwise accuracy: {accuracy:.3f}")




def save_precomputed_crops(dataset, output_dir, prefix="crop", num_workers=4, batch_size=32):
    """
    Precompute and save cropped video frames with metadata.
    
    Args:
        dataset: an instance of LoadVideoAndLabels
        output_dir: folder where to save images and metadata.csv
        prefix: filename prefix for saved crops
    """
    os.makedirs(output_dir, exist_ok=True)
    records = []

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )

    idx = 0
    for i, batch in enumerate(tqdm(loader, desc="Saving crops")):
        crops, labels, track_ids, frame_nums = batch

        for b in range(len(crops)):
            crop = crops[b]
            label = int(labels[b])
            track_id = int(track_ids[b])
            frame_num = int(frame_nums[b])


            if crop.dtype != torch.float32:
                crop = crop.float()
            if crop.max() > 2:
                print("super alarm")
                crop = crop / 255.0

            filename = f"{prefix}_{frame_num:06d}_{track_id}.png"
            filepath = os.path.join(output_dir, filename)

            
            save_image(crop, filepath)

            records.append({
                'filename': filename,
                'experiment': prefix,
                'label': label,
                'track_id': track_id,
                'frame_num': frame_num,
            })
            idx += 1


    # Save metadata
    metadata_path = os.path.join(output_dir, "..", f"{prefix}_metadata.csv")
    pd.DataFrame(records).to_csv(metadata_path, index=False)
    print(f"✅ Saved {len(records)} crops to {output_dir}")



if __name__ == '__main__':
    torch.cuda.set_device(0)
    args = parser.parse_args()

    #for experiment_name in args.experiment_name:
    #    save_precomputed_crops(LoadVideoAndLabels(args.annotation_dir, args.video_path, experiment_name), 
    #                           output_dir=f"data/video_valset_cropped/images/", 
    #                           prefix=experiment_name)
    main(args)
