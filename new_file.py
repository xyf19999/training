import json
import shutil
import os
import sys

def extract_first_column(file_path):
    result = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split by comma and take the first part
            index = line.split(',')[0].strip()
            index = eval(index)
            result.append(index)
    print(len(result))
    return result
included_image_val = extract_first_column('/home/yifei/monitizer/monitizer/benchmark_object_detection/specifications/bdd_coco/generated_OOD_region_val.txt')

usecase = 'val'

def remove_labels(data):
    for entry in data:
        if 'labels' in entry and isinstance(entry['labels'], list):
            entry['labels'] = [label for label in entry['labels'] if label.get('category') not in ['drivable area', 'lane']]
    return data

with open(f'/home/yifei/bdd_coco_old/bdd_coco_val.json', 'r') as input_file:
    data_val = json.load(input_file)

data_val = remove_labels(data_val)

data_remaining_name = [data_val[i]['name'] for i in included_image_val]
print(len(data_remaining_name))

source_folder = '/home/yifei/bdd_coco_old/val2017'
destination_path = '/home/yifei/bdd_coco_old/val_for_optimization'
for file in data_remaining_name:
    source_path_new = os.path.join(source_folder, file)
    destination_path_new = os.path.join(destination_path, file)
    shutil.copy(source_path_new, destination_path_new)

