from collections import Counter
from PIL import Image
import random
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_images(samples, class_label, image_folder):
    #visualizes 16 images in a 4x4 grid from the selected class (0 or 1)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax, line in zip(axes.flatten(), samples):
        image_name = line.strip().split(',')[0]
        image_path = f"{image_folder}/{image_name}"
        image = Image.open(image_path).resize((224, 224))
        ax.imshow(image)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'assets/sample2_class_{class_label}.png')
    plt.close()


def plot_sample_images(file_path, image_folder):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    class_0 = [line for line in lines if line.strip().split(',')[-1] == '0']
    class_1 = [line for line in lines if line.strip().split(',')[-1] == '1']

    sample_class_0 = random.sample(class_0, 16, image_folder)
    sample_class_1 = random.sample(class_1, 16, image_folder)

    plot_images(sample_class_0, 0)
    plot_images(sample_class_1, 1)



            



def plot_sample_images_from_dir(base_folder, experiment, group):
    # plots examples for each ID in a given group directory
    os.makedirs(f"assets/{experiment}/{group}", exist_ok=True)
    label_folder = base_folder
    label_file = os.path.join(label_folder, f"{experiment}_{group}.txt")
    count_dict = {}
    image_dict = {}
    with open(label_file, 'r') as file:
        lines = file.readlines()
        random.shuffle(lines)
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 1:
                f, x, y, w, h, obj_id, confidence = parts[0], parts[1], parts[2], parts[3], parts[4], int(parts[5]), parts[6]
                if obj_id not in count_dict:
                    count_dict[obj_id] = 1
                    image_dict[obj_id] = []
                else:
                    count_dict[obj_id] += 1

                if count_dict[obj_id] <= 16:
                    image_dict[obj_id].append({"file": f, "bbox": (float(x), float(y), float(w), float(h)), "confidence": confidence})

    image_folder = os.path.join(base_folder, "images")
    for obj_id in image_dict:

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i, elem in enumerate(image_dict[obj_id]):
            file_name = elem["file"]
            bbox = elem["bbox"]
            image_path = os.path.join(image_folder, file_name)
            image = Image.open(image_path)
            #w, h = image.size
            #x_rel, y_rel, w_rel, h_rel = bbox
            #x = int(x_rel * w)
            #y = int(y_rel * h)
            #crop_w = int(w_rel * w)
            #crop_h = int(h_rel * h)
            x, y , crop_w, crop_h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cropped = image.crop((x - crop_w//2, y - crop_h//2, x + crop_w//2, y + crop_h//2))
            cropped = cropped.resize((224, 224))
            axes[i%4, i//4].imshow(cropped)
            axes[i%4, i//4].axis('off')
            conf = round(float(elem['confidence']), 3)
            axes[i%4, i//4].set_title(f"Conf: {conf}", fontsize=8)
        fig.suptitle(f"Object ID: {obj_id}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"assets/{experiment}/{group}/{obj_id}.png")
        plt.close()






def read_and_summarize_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    last_values = [line.strip().split(',')[-1] for line in lines]
    summary = Counter(last_values)

    for value, count in summary.items():
        print(f"{value}: {count}")

        

        plot_sample_images('annotation_images/labels.txt', 'annotation_images/images')

def show_id_samples(file_path, n_samples=10):
    max_id = 0
    labels_dir = os.path.join(file_path, "labels_with_ids")
    images_dir = os.path.join(file_path, "images")
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    sample_files = random.sample(label_files, min(100, len(label_files)))

    samples = []
    overview_dict = {}
    for label_file in sample_files:
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    x, y, w, h, obj_id = parts[2], parts[3], parts[4], parts[5], parts[6]
                    samples.append({
                        "file": label_file,
                        "x": float(x),
                        "y": float(y),
                        "w": float(w),
                        "h": float(h),
                        "id": int(obj_id)
                    })
                    if obj_id < max_id:
                        max_id = int(obj_id)

                    if obj_id not in overview_dict:
                        overview_dict[obj_id] = 1
                    else:
                        overview_dict[obj_id] += 1
                    



    print(f"Loaded {len(samples)} objects from {len(sample_files)} label files.")
    # Optionally, print a few samples
    for sample in samples[:10]:
        print(sample)

if __name__ == "__main__":
    #file_path = 'annotation_images/labels.txt'
    #read_and_summarize_labels(file_path)

    experiment = "limit10_1000_5000"
    for group in ["R1", "Alpha", "B", "J"]:
        id_path = f"/usr/users/vogg/Labelling/Lemurs/labelling_app_indID/experiments/{experiment}/{group}/"
        plot_sample_images_from_dir(id_path, experiment, group)
        #show_id_samples(id_path)