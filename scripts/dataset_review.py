from collections import Counter
from PIL import Image
import random
import matplotlib.pyplot as plt

def read_and_summarize_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    last_values = [line.strip().split(',')[-1] for line in lines]
    summary = Counter(last_values)

    for value, count in summary.items():
        print(f"{value}: {count}")

        def plot_sample_images(file_path, image_folder):
            with open(file_path, 'r') as file:
                lines = file.readlines()

            class_0 = [line for line in lines if line.strip().split(',')[-1] == '0']
            class_1 = [line for line in lines if line.strip().split(',')[-1] == '1']

            sample_class_0 = random.sample(class_0, 16)
            sample_class_1 = random.sample(class_1, 16)

            def plot_images(samples, class_label):
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

            plot_images(sample_class_0, 0)
            plot_images(sample_class_1, 1)

        plot_sample_images('annotation_images/labels.txt', 'annotation_images/images')

if __name__ == "__main__":
    file_path = 'annotation_images/labels.txt'
    read_and_summarize_labels(file_path)