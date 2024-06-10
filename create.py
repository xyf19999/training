import json


import os
import random
import shutil
import json

# Path to the folder containing the images
source_folder_train = "/home/yifei/bdd_coco/train"
source_folder_val = '/home/yifei/bdd_coco/val'

# Paths to the new folders
target_folder_train2017 = "/home/yifei/bdd_coco/train2017"
target_folder_val2017 = '/home/yifei/bdd_coco/val2017'
target_folder_test = '/home/yifei/bdd_coco/test'
target_folder_proper_training = '/home/yifei/bdd_coco/proper_training'
target_folder_calibration = '/home/yifei/bdd_coco/calibration'
target_folder_occluded = '/home/yifei/bdd_coco/occluded'


file_path_all = "/home/yifei/my_project/training/test_proper_training_calibration_occluded.txt"
file_path_test = '/home/yifei/my_project/training/test.txt'
file_path_proper_training = '/home/yifei/my_project/training/proper_training.txt'
file_path_calibration = '/home/yifei/my_project/training/calibration.txt'
file_path_occluded = '/home/yifei/my_project/training/occluded.txt'



# Initialize an empty list to store the items
all_ex = []

# Open the file and read its contents line by line
with open(file_path_all, "r") as file:
    # Read each line and append it to the list
    for line in file:
        # Remove any leading or trailing whitespace and append the line to the list
        all_ex.append(line.strip())

test = []
with open(file_path_test, "r") as file:
    # Read each line and append it to the list
    for line in file:
        # Remove any leading or trailing whitespace and append the line to the list
        test.append(line.strip())

proper_training = []
with open(file_path_proper_training, "r") as file:
    # Read each line and append it to the list
    for line in file:
        # Remove any leading or trailing whitespace and append the line to the list
        proper_training.append(line.strip())
print(len(proper_training))

calibration = []
with open(file_path_calibration, "r") as file:
    # Read each line and append it to the list
    for line in file:
        # Remove any leading or trailing whitespace and append the line to the list
        calibration.append(line.strip())

occluded = []
with open(file_path_occluded, "r") as file:
    # Read each line and append it to the list
    for line in file:
        # Remove any leading or trailing whitespace and append the line to the list
        occluded.append(line.strip())

with open(f'/home/yifei/bdd_coco/bdd_coco_train.json', 'r') as input_file:
    original_data_train = json.load(input_file)

with open(f'/home/yifei/bdd_coco/bdd_coco_val.json', 'r') as input_file:
    original_data_val = json.load(input_file)

modified_train = [element for element in original_data_train if element['name'] not in all_ex]
modified_val = original_data_val

val_name = [element['name'] for element in original_data_val]

for image in test:
    src = os.path.join(source_folder_train, image)
    dst = os.path.join(target_folder_test, image)
    shutil.copy(src, dst) 

for image in proper_training:
    src = os.path.join(source_folder_train, image)
    dst = os.path.join(target_folder_proper_training, image)
    shutil.copy(src, dst) 

for image in calibration:
    src = os.path.join(source_folder_train, image)
    dst = os.path.join(target_folder_calibration, image)
    shutil.copy(src, dst)

for image in occluded:
    src = os.path.join(source_folder_train, image)
    dst = os.path.join(target_folder_occluded, image)
    shutil.copy(src, dst)  


train_bdd_part = [element for element in modified_train if '-' in element['name']]
train_coco_part = [element for element in modified_train if '-' not in element['name']]
print(len(train_coco_part), 'coco length')

val_bdd_part = [element for element in modified_val if '-' in element['name']]
val_coco_part = [element for element in modified_val if '-' not in element['name']]

train_to_val_coco = random.sample(train_coco_part, 1813)
train_to_val_coco_name = [element['name'] for element in train_to_val_coco]

new_val_coco_part = val_coco_part + train_to_val_coco
new_train_coco_part = [element for element in train_coco_part if element['name'] not in train_to_val_coco_name]

new_train = new_train_coco_part + train_bdd_part
new_train_name = [element['name'] for element in new_train]
new_val = new_val_coco_part + val_bdd_part
new_val_name = [element['name'] for element in new_val]

with open('/home/yifei/bdd_coco/bdd_coco_train_after_exclusion.json', "w") as json_file:
    json.dump(new_train, json_file, indent=2)

with open('/home/yifei/bdd_coco/bdd_coco_val_after_exclusion.json', "w") as json_file:
    json.dump(new_val, json_file, indent=2)

for image in new_train_name:
    src = os.path.join(source_folder_train, image)
    dst = os.path.join(target_folder_train2017, image)
    shutil.copy(src, dst)

for image in val_name:
    src = os.path.join(source_folder_val, image)
    dst = os.path.join(target_folder_val2017, image)
    shutil.copy(src, dst)

for image in train_to_val_coco_name:
    src = os.path.join(source_folder_train, image)
    dst = os.path.join(target_folder_val2017, image)
    shutil.copy(src, dst)