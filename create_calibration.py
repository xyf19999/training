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
selected_2000 = random.sample(images, 2000)

# Copy the selected 500 images to folder_500
for image in selected_2000:
    src = os.path.join(source_folder, image)
    dst = os.path.join(folder_proper_training, image)
    shutil.copy(src, dst)

# Remove the selected 500 images from the list of all images
for image in selected_2000:
    images.remove(image)

# Randomly select 100 images from the remaining images
selected_1000 = random.sample(images, 1000)

# Copy the selected 100 images to folder_100
for image in selected_1000:
    src = os.path.join(source_folder, image)
    dst = os.path.join(folder_calibration, image)
    shutil.copy(src, dst)


def create_proper_training_calibration(dataset, proper_training_size: int = 2000, calibration_size: int = 1000):
    source_folder = dataset.data_folder
    folder_proper_training = source_folder + '/proper_training'
    folder_calibration = source_folder + '/calibration'
    images = os.listdir(source_folder+'/train')
    selected_proper_training = random.sample(images, proper_training_size)

    for image in selected_proper_training:
      src = os.path.join(source_folder, image)
      dst = os.path.join(folder_proper_training, image)
      shutil.copy(src, dst)

    for image in selected_proper_training:
       images.remove(image)

    selected_calibration = random.sample(images, calibration_size)

    for image in selected_calibration:
      src = os.path.join(source_folder, image)
      dst = os.path.join(folder_calibration, image)
      shutil.copy(src, dst)


