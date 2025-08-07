import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import torchvision.models as models
import torch.nn as nn
import cv2
import re




def load_data(label_root, group):
    columns = ['trackNumber', 'trackId', 'xCoord', 'yCoord', 'width', 'height', 'confidenceTrack', 'species', 'nameOrder', 'confidenceId', 'experiment']
    combined_df = pd.DataFrame(columns=columns)

    files_starting_with_A = [f for f in files_in_label_root if f.startswith(group[0])]

    for i, filename in enumerate(files_starting_with_A):

        file_path = f"{label_root}/{filename}"

        with open(file_path, "r") as file:
            file_content = file.readlines()

        metadata = [line.strip("#").strip() for line in file_content if line.startswith("#")]
        metadata_dict = dict(item.split(": ") for item in metadata)

        username = metadata_dict["username"]
        editDate = metadata_dict["editDate"]
        orderedNames = metadata_dict["orderedNames"].split(", ")
        if i > 0:
            if orderedNames != previous_orderedNames:
                print(f"Warning: orderedNames in {filename} do not match the previous file.")
        previous_orderedNames = orderedNames
        dataColumns = metadata_dict["dataColumns"].split(", ")

        df_id = pd.read_csv(file_path, skiprows=len(metadata), names=dataColumns).sort_values(["species", "trackId", "trackNumber"])

        filtered_df_id = df_id[df_id['species'] == 0]

        filtered_df_id = filtered_df_id.copy()
        filtered_df_id['experiment'] = '_'.join(filename.split('_', 3)[:3])

        # Append the filtered DataFrame to the combined DataFrame
        combined_df = pd.concat([combined_df, filtered_df_id], ignore_index=True)

    # Get the indices for 'Uns' and 'Unsure' from the orderedNames list
    uns_index = orderedNames.index('Uns')
    unsure_index = orderedNames.index('Unsure')
    # Remove 'Uns' from orderedNames
    orderedNames = [name for name in orderedNames if name != 'Uns']

    # Merge the nameOrder numbers that belong to Uns and Unsure
    combined_df['nameOrder'] = combined_df['nameOrder'].replace({uns_index: unsure_index})

    return combined_df, orderedNames


def get_dataset_statistics(combined_df, orderedNames):
    print("Dataset Statistics:")
    print(f"Total number of detections: {len(combined_df)}")
    print(f"Unique track IDs: {combined_df[['experiment', 'trackId']].nunique()}")
    print(f"Unique individuals: {combined_df['nameOrder'].nunique()}")
    
    # Count the occurrences of each name order
    name_order_counts = combined_df['nameOrder'].value_counts()
    print("Name Order Counts:")
    for name, count in name_order_counts.items():
        print(f"{orderedNames[name]} ({name}): {count}")

    return name_order_counts


def load_model(pretrained=True):
    print("Loading sorting model...")
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 2),
        nn.Softmax(dim=1)
    )
    if pretrained:
        state_dict = torch.load("models/binary.pth", map_location='cuda:0')
        model.load_state_dict(state_dict)
    model.eval()
    return model


def cluster_indices(combined_df, ind_id, n_cluster=10000):
    print(ind_id)

    subset_df = combined_df[combined_df['nameOrder'] == ind_id] #.reset_index(drop=True)

    # Convert the 'experiment' column to a numerical variable
    subset_df.loc[:, 'experiment_num'] = subset_df['experiment'].astype('category').cat.codes
    subset_df[['xCoord_norm', 'yCoord_norm', 'width_norm', 'height_norm']] = subset_df[['xCoord', 'yCoord', 'width', 'height']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    features = subset_df[['xCoord_norm', 'yCoord_norm', 'width_norm', 'height_norm', 'experiment_num']]
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(features)
    subset_df.loc[:, 'cluster'] = kmeans.labels_

    np.random.seed(0)
    # Group by cluster and sample one index from each cluster
    sampled_indices = subset_df.groupby('cluster').apply(lambda x: x.sample(1)).index.droplevel(0)
    sorted_sampled_indices = sampled_indices.sort_values()
    return sorted_sampled_indices


def sample_top_n_scores(combined_df, sorted_sampled_indices, model, top_n=1000, min_score=0.5):
    # Retrieve sorting model scores from video data
    path_to_videos = f"/usr/users/vogg/sfb1528s3/B06/2023april-july/NewBoxesClosed/Converted/{group}/"

    i = 0
    scores = []


    for i in range(len(sorted_sampled_indices)):

        index = sorted_sampled_indices[i]
        if i % 1000 == 0:
            print(f"Processing row {i}")
        row = combined_df.iloc[index]
        video_path = os.path.join(path_to_videos, f"{row['experiment']}.mp4")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
        else: 
            frame_number = int(row['trackNumber'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                # Crop the bounding box from the frame
                x, y, w, h = int(row['xCoord']), int(row['yCoord']), int(row['width']), int(row['height'])
                # Ensure the coordinates are within the frame dimensions
                x = max(0, x)
                y = max(0, y)
                h = min(frame.shape[0] - y, h)
                w = min(frame.shape[1] - x, w)

                cropped_frame = frame[y:y+h, x:x+w]

                # Resize the cropped frame to 224x224
                resized_frame = cv2.resize(cropped_frame, (224, 224), cv2.INTER_AREA)

                frame = torch.from_numpy(resized_frame).float()
                frame = frame.permute(2, 0, 1)
                img = frame / 255.0

                # Add a batch dimension and move the tensor to the same device as the model
                img = img.unsqueeze(0).to(next(model.parameters()).device)

                # Apply the model to the image
                with torch.no_grad():
                    output = model(img)
                    score = output.numpy()[0][1]

                scores.append(score)
    
    # Convert scores to numpy array
    scores_array = np.array(scores)
    # Filter indices where score >= min_score
    valid_indices = np.where(scores_array >= min_score)[0]
    # If there are fewer than top_n valid scores, take all of them
    if len(valid_indices) <= top_n:
        top_n_sorted_sampled_indices = sorted_sampled_indices[valid_indices]
    else:
        # Otherwise, take the top_n highest scores among the valid ones
        valid_scores = scores_array[valid_indices]
        top_indices_within_valid = np.argsort(valid_scores)[-top_n:]
        top_n_sorted_sampled_indices = sorted_sampled_indices[valid_indices[top_indices_within_valid]]

    subset_df = combined_df.loc[top_n_sorted_sampled_indices]

    return subset_df



def extract_images(label_root, group, path_to_videos, final_df, experiment_name="test"):

    group_folder = os.path.join(label_root, "..", "experiments", experiment_name, group)
    os.makedirs(group_folder, exist_ok=True)

    images_folder = os.path.join(group_folder, "images")
    #labels_with_ids_folder = os.path.join(group_folder, "labels_with_ids")

    os.makedirs(images_folder, exist_ok=True)
    #os.makedirs(labels_with_ids_folder, exist_ok=True)

    for i in range(len(final_df)):
        if i % 1000 == 0:
            print(f"Processing row {i}")
        row = final_df.iloc[i]

        # Normalize coordinates and dimensions
        xCoord = int(row['xCoord'] + row['width']/2)
        yCoord = int(row['yCoord'] + row['height']/2)
        width = int(row['width'])
        height = int(row['height'])

        filename = experiment_name + "_" + group
        
        file_path = os.path.join(group_folder, filename)

        if i == 0 or row['experiment'] != final_df.iloc[i-1]['experiment']:
            if i > 0:
                if cap.isOpened():
                    cap.release()
            video_path = os.path.join(path_to_videos, f"{row['experiment']}.mp4")
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
        else:
            frame_number = int(row['trackNumber'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                image_filename = f"{row['experiment']}_{row['trackNumber']}.png"
                image_path = os.path.join(images_folder, image_filename)
                cv2.imwrite(image_path, frame)
                with open(file_path, 'a') as f:
                    f.write(f"{row['experiment']}_{row['trackNumber']}.png,{xCoord},{yCoord},{width},{height},{row['nameOrder']}\n")
                #print(f"Frame {frame_number} saved as {image_path}")
            else:
                print(f"Error: Could not read frame {frame_number}")


label_root = "../Labelling/Lemurs/labelling_app_indID/raw_labels/"
files_in_label_root = os.listdir(label_root)

experiment_name = "cluster_1000_5000"
n_cluster = 5000

for group in ['R1', 'J', 'B', 'Alpha']:

    path_to_videos = f"/usr/users/vogg/sfb1528s3/B06/2023april-july/NewBoxesClosed/Converted/{group}/"
    #data_root = Path(f"/usr/users/vogg/Labelling/Lemurs/labelling_app_indID/richard_sorted/{group}")
    combined_df, orderedNames = load_data(label_root, group)
    name_order_counts = get_dataset_statistics(combined_df, orderedNames)
    model = load_model(pretrained=True)
    unsure_index = orderedNames.index('Unsure')
    under_n_index = name_order_counts[name_order_counts < n_cluster].index
    #under_n_index = under_n_names.map(lambda x: int(re.search(r'\((\d+)\)', x).group(1)))

    all_subsets = []

    for ind_id in [i for i in orderedNames if i not in ([unsure_index] + list(under_n_index))]:
        print(ind_id)
        sorted_sampled_indices = cluster_indices(combined_df, ind_id, n_cluster=n_cluster)
        subset_df = sample_top_n_scores(combined_df, sorted_sampled_indices, model, top_n=1000)
        all_subsets.append(subset_df)

    final_df = pd.concat(all_subsets, ignore_index=True)

    if len(under_n_index) > 0:
        under_n_samples = []
        
        for nameOrder in under_n_index:
            nameOrder_df = combined_df[combined_df['nameOrder'] == nameOrder]
            sorted_sampled_indices = nameOrder_df.index.values
            subset_df = sample_top_n_scores(combined_df, sorted_sampled_indices, model, top_n=1000)
            under_n_samples.append(subset_df)

        under_n_df = pd.concat(under_n_samples, ignore_index=True)

    unsure_df = combined_df[combined_df['nameOrder'] == unsure_index]
    unsure_sample = unsure_df.sample(5000, random_state=0)

    dfs_to_concat = [final_df, unsure_sample]
    if 'under_n_df' in locals():
        dfs_to_concat.append(under_n_df)
    final_df = pd.concat(dfs_to_concat, ignore_index=True)

    # Reorder sampled_df by the columns experiment and trackNumber
    final_df = final_df.sort_values(by=['experiment', 'trackNumber']).reset_index(drop=True)

    name_order_counts_after = get_dataset_statistics(final_df, orderedNames)
    extract_images(label_root, group, path_to_videos, final_df, experiment_name=experiment_name)

