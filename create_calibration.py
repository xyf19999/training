import os
import random
import shutil

# Path to the folder containing the images
source_folder = "/Users/mac/Desktop/bdd100k/bdd100k_in_yolo/train/images"

# Paths to the new folders
folder_proper_training = "/Users/mac/Desktop/proper_training"
folder_calibration = "/Users/mac/Desktop/calibration"

# List all files in the source folder
images = os.listdir(source_folder)

# Randomly select 500 images
selected_200 = random.sample(images, 200)

# Copy the selected 500 images to folder_500
for image in selected_200:
    src = os.path.join(source_folder, image)
    dst = os.path.join(folder_proper_training, image)
    shutil.copy(src, dst)

# Remove the selected 500 images from the list of all images
for image in selected_200:
    images.remove(image)

# Randomly select 100 images from the remaining images
selected_1000 = random.sample(images, 100)

# Copy the selected 100 images to folder_100
for image in selected_1000:
    src = os.path.join(source_folder, image)
    dst = os.path.join(folder_calibration, image)
    shutil.copy(src, dst)
