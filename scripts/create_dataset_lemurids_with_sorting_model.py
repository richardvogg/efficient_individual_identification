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


label_root = "../../Labelling/Lemurs/labelling_app_indID/raw_labels/"
files_in_label_root = os.listdir(label_root)

group = "R1"


path_to_videos = f"/usr/users/vogg/sfb1528s3/B06/2023april-july/NewBoxesClosed/Converted/{group}/"
data_root = Path(f"/usr/users/vogg/Labelling/Lemurs/labelling_app_indID/richard_sorted/{group}")

# Data preparation
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

# Merge the nameOrder numbers that belong to Uns and Unsure
combined_df['nameOrder'] = combined_df['nameOrder'].replace({uns_index: unsure_index})

name_order_counts = combined_df['nameOrder'].value_counts()
name_order_counts.index = name_order_counts.index.map(lambda x: f"{orderedNames[x]} ({x})")
print(name_order_counts)


# Load the sorting model
print("Loading sorting model...")
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 2),
    nn.Softmax(dim=1)
)

state_dict = torch.load("../../efficient_individual_identification/models/binary.pth", map_location='cuda:0')
model.load_state_dict(state_dict)
model.eval();

