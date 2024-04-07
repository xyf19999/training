import os
import random
import shutil
import json

# Path to the folder containing the images
source_folder_train = "/Users/mac/Desktop/bdd100k/bdd100k_in_yolo/train/images"

# Paths to the new folders
folder_train_excluded = "/Users/mac/Desktop/test"

file_path = "/Users/mac/Desktop/bdd100k/test.txt"

# Initialize an empty list to store the items
excluded_image_name = []

# Open the file and read its contents line by line
with open(file_path, "r") as file:
    # Read each line and append it to the list
    for line in file:
        # Remove any leading or trailing whitespace and append the line to the list
        excluded_image_name.append(line.strip())

for image in excluded_image_name:
    src = os.path.join(source_folder_train, image)
    dst = os.path.join(folder_train_excluded, image)
    shutil.copy(src, dst)

