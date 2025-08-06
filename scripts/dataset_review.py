from collections import Counter
from PIL import Image
import random
import os
import matplotlib.pyplot as plt


def plot_images(samples, class_label, image_folder):
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



            



def plot_sample_images_from_dir(base_folder):
    label_folder = os.path.join(base_folder, "labels_with_ids")
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    count_dict = {}
    image_dict = {}
    for f in label_files:
        with open(os.path.join(label_folder, f), 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 7:
                    x, y, w, h, obj_id = parts[2], parts[3], parts[4], parts[5], int(parts[6])
                    if obj_id not in count_dict:
                        count_dict[obj_id] = 1
                        image_dict[obj_id] = []
                    else:
                        count_dict[obj_id] += 1

                    if count_dict[obj_id] <= 16:
                        image_dict[obj_id].append({"file": f, "bbox": (float(x), float(y), float(w), float(h))})

    image_folder = os.path.join(base_folder, "images")
    for obj_id in image_dict:

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i, elem in enumerate(image_dict[obj_id]):
            file_name = elem["file"].replace('.txt', '.png')
            bbox = elem["bbox"]
            image_path = os.path.join(image_folder, file_name)
            image = Image.open(image_path)
            w, h = image.size
            x_rel, y_rel, w_rel, h_rel = bbox
            x = int(x_rel * w)
            y = int(y_rel * h)
            crop_w = int(w_rel * w)
            crop_h = int(h_rel * h)
            cropped = image.crop((x, y, x + crop_w, y + crop_h))
            cropped = cropped.resize((224, 224))
            axes[i%4, i//4].imshow(cropped)
            axes[i%4, i//4].axis('off')
        fig.suptitle(f"Object ID: {obj_id}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"assets/{obj_id}.png")
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

    id_path = "/usr/users/vogg/Labelling/Lemurs/labelling_app_indID/R1/"
    plot_sample_images_from_dir(id_path)
    #show_id_samples(id_path)