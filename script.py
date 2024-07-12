import os
import glob

def collect_jpg_names(folder_path):
    """
    Collect all .jpg file names in the specified folder and return them as a list.

    :param folder_path: Path to the folder where .jpg files are to be collected
    :return: List of .jpg file names
    """
    # Ensure the folder path ends with a slash
    folder_path = os.path.join(folder_path, '')

    # Use glob to find all .jpg files in the folder
    jpg_files = glob.glob(folder_path + "*.jpg")

    # Extract file names from the paths
    jpg_file_names = [os.path.basename(file) for file in jpg_files]

    return jpg_file_names

def write_list_to_file(file_path, elements):
    """
    Write elements of a list to a file, one element per line.

    :param file_path: Path to the file where the elements will be written
    :param elements: List of elements to be written to the file
    """
    with open(file_path, 'w') as file:
        for element in elements:
            file.write(f"{element}\n")

#bdd100k
folder_path_train = '/home/yifei/bdd_day_no_bus/train2017'
folder_path_val = '/home/yifei/bdd_day_no_bus/val2017'
folder_path_test = '/home/yifei/bdd_day_no_bus/test'
folder_path_proper_training = '/home/yifei/bdd_day_no_bus/proper_training'
folder_path_calibration = '/home/yifei/bdd_day_no_bus/calibration'
folder_path_occluded = '/home/yifei/bdd_day_no_bus/occluded'
folder_path_val_for_optimization = '/home/yifei/bdd_day_no_bus/val_for_optimization'

whole_image = collect_jpg_names(folder_path_test) + collect_jpg_names(folder_path_proper_training) + collect_jpg_names(folder_path_calibration) + collect_jpg_names(folder_path_occluded) + collect_jpg_names(folder_path_val_for_optimization)
train_set = collect_jpg_names(folder_path_train)
val_set = collect_jpg_names(folder_path_val)

write_list_to_file(file_path='/home/yifei/bdd_day_no_bus_train.txt', elements=train_set)
write_list_to_file(file_path='/home/yifei/bdd_day_no_bus_val.txt', elements=val_set)
write_list_to_file(file_path='/home/yifei/bdd_day_no_bus_special_sets.txt', elements=whole_image)