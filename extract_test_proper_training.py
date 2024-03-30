import os
import random
import shutil
import json

# Path to the folder containing the images
source_folder_train = "/home/yifei/bdd100k/train"

# Paths to the new folders
folder_train_excluded = "/home/yifei/bdd100k/train2017"

file_path = "/home/yifei/bdd100k/test_proper_training_calibration_occluded.txt"

# Initialize an empty list to store the items
excluded_image_name = []

# Open the file and read its contents line by line
with open(file_path, "r") as file:
    # Read each line and append it to the list
    for line in file:
        # Remove any leading or trailing whitespace and append the line to the list
        excluded_image_name.append(line.strip())

with open(f'/home/yifei/bdd100k/bdd100k_labels_images_train.json', 'r') as input_file:
    original_data = json.load(input_file)

chosen_name = [element['name'] for element in original_data if element['name'] not in excluded_image_name]

for image in chosen_name:
    src = os.path.join(source_folder_train, image)
    dst = os.path.join(folder_train_excluded, image)
    shutil.copy(src, dst)

modified_data = [element for element in original_data if element['name'] not in excluded_image_name]

with open('/home/yifei/bdd100k/annotations/instances_train2017.json', "w") as json_file:
    json.dump(modified_data, json_file, indent=2)

with open(f'/home/yifei/bdd100k/bdd100k_labels_images_val.json', 'r') as input_file:
    original_data = json.load(input_file)

with open('/home/yifei/bdd100k/annotations/instances_val2017.json', "w") as json_file:
    json.dump(original_data, json_file, indent=2)