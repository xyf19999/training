import os
import random
import shutil
import json

usecase = 'train'
# Path to the folder containing the images


with open(f'/home/yifei/bdd_daytime_no_bus/bdd_daytime_no_bus_{usecase}.json', 'r') as input_file:
    original_data = json.load(input_file)

print(len(original_data))