import os
import random
import shutil

# Path to the folder containing the images
source_folder = "/Users/mac/Desktop/bdd100k/bdd100k_in_yolo/train/images"

# Paths to the new folders
folder_500 = "/Users/mac/Desktop/proper_training"
folder_100 = "/Users/mac/Desktop/calibration"

# List all files in the source folder
images = os.listdir(source_folder)

# Randomly select 500 images
selected_500 = random.sample(images, 500)

# Copy the selected 500 images to folder_500
for image in selected_500:
    src = os.path.join(source_folder, image)
    dst = os.path.join(folder_500, image)
    shutil.copy(src, dst)

# Remove the selected 500 images from the list of all images
for image in selected_500:
    images.remove(image)

# Randomly select 100 images from the remaining images
selected_100 = random.sample(images, 100)

# Copy the selected 100 images to folder_100
for image in selected_100:
    src = os.path.join(source_folder, image)
    dst = os.path.join(folder_100, image)
    shutil.copy(src, dst)

print("Copying completed.")
