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
folder_path_test = '/home/yifei/bdd100k/test'
folder_path_proper_training = '/home/yifei/bdd100k/proper_training'
folder_path_calibration = '/home/yifei/bdd100k/calibration'
folder_path_occluded = '/home/yifei/bdd100k/occluded'
folder_path_val_for_optimization = '/home/yifei/bdd100k/val_for_optimization'

whole_image = collect_jpg_names(folder_path_test) + collect_jpg_names(folder_path_proper_training) + \
+ collect_jpg_names(folder_path_calibration) + collect_jpg_names(folder_path_occluded) + \
collect_jpg_names(folder_path_val_for_optimization)

write_list_to_file(file_path='/home/yifei/bdd100k_special_sets.txt', elements=whole_image)