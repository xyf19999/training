import os
import random
import shutil
import json

usecase = 'train'
# Path to the folder containing the images
source_folder_train = f"/home/yifei/bdd_coco/{usecase}2017"

# Paths to the new folders
folder_train_excluded = f"/home/yifei/bdd_daytime_no_bus/{usecase}2017"

with open(f'/home/yifei/bdd_coco/bdd_coco_{usecase}.json', 'r') as input_file:
    original_data = json.load(input_file)

print(len(original_data))

modified_data = [element for element in original_data if '-' in element['name']]
print(len(modified_data))

modified_data_name = [element['name'] for element in modified_data]

with open(f'/home/yifei/bdd_daytime_no_bus/bdd_sunny_no_bus_{usecase}.json', "w") as json_file:
    json.dump(original_data, json_file, indent=2)

for image in modified_data_name:
    src = os.path.join(source_folder_train, image)
    dst = os.path.join(folder_train_excluded, image)
    shutil.copy(src, dst)
