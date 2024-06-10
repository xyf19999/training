"""
help to convert bdd100k label format to coco label format
source: https://github.com/ucbdrive/bdd100k/blob/master/bdd100k/bdd2coco.py
"""
import os
import json
import argparse
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='BDD100K to COCO format')
    parser.add_argument(
          "-l", "--label_dir",
          default="/home/yifei/bdd_coco",
          help="root directory of BDD label Json files",
    )
    parser.add_argument(
          "-s", "--save_path",
          default="/home/yifei/bdd_coco/annotations",
          help="path to save coco formatted label file",
    )
    return parser.parse_args()


def bdd2coco_detection(id_dict, labeled_images, fn):

    images = list()
    annotations = list()

    counter = 0
    for i in tqdm(labeled_images):
        counter += 1
        image = dict()
        image['file_name'] = i['name']
        if image.get('height', None) == None:
           image['height'] = 1280
           image['width'] = 720

        image['id'] = counter

        empty_image = True

        for label in i['labels']:
            annotation = dict()
            if label['category'] in id_dict.keys():
                empty_image = False
                annotation["iscrowd"] = 0
                annotation["image_id"] = image['id']
                x1 = label['box2d']['x1']
                y1 = label['box2d']['y1']
                x2 = label['box2d']['x2']
                y2 = label['box2d']['y2']
                annotation['bbox'] = [x1, y1, x2-x1, y2-y1]
                annotation['area'] = float((x2 - x1) * (y2 - y1))
                annotation['category_id'] = id_dict[label['category']]
                annotation['ignore'] = 0
                annotation['id'] = label['id']
                annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                annotations.append(annotation)

        if empty_image:
            continue

        images.append(image)

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    print('saving...')
    json_string = json.dumps(attr_dict)
    with open(fn, "w") as file:
        file.write(json_string)


if __name__ == '__main__':

    args = parse_arguments()

    attr_dict = dict()
    attr_dict["categories"] = [
     {'supercategory': 'none', 'id': 1, 'name': 'person'},
     {'supercategory': 'none', 'id': 2, 'name': 'car'},
     {'supercategory': 'none', 'id': 3, 'name': 'traffic sign'},
     {'supercategory': 'none', 'id': 4, 'name': 'traffic light'},
     {'supercategory': 'none', 'id': 5, 'name': 'motor'},
     {'supercategory': 'none', 'id': 6, 'name': 'rider'},
     {'supercategory': 'none', 'id': 7, 'name': 'train'},
     {'supercategory': 'none', 'id': 8, 'name': 'truck'},
     {'supercategory': 'none', 'id': 9, 'name': 'bike'},
     {'supercategory': 'none', 'id': 10, 'name': 'bear'},
     {'supercategory': 'none', 'id': 11, 'name': 'bird'},
     {'supercategory': 'none', 'id': 12, 'name': 'cow'},
     {'supercategory': 'none', 'id': 13, 'name': 'cat'},
     {'supercategory': 'none', 'id': 14, 'name': 'horse'},
     {'supercategory': 'none', 'id': 15, 'name': 'dog'},
     {'supercategory': 'none', 'id': 16, 'name': 'giraffe'},
     {'supercategory': 'none', 'id': 17, 'name': 'sheep'},
     {'supercategory': 'none', 'id': 18, 'name': 'elephant'},
     {'supercategory': 'none', 'id': 19, 'name': 'zebra'}
]

    attr_id_dict = {i['name']: i['id'] for i in attr_dict['categories']}

    # create BDD training set detections in COCO format
    print('Loading training set...')
    with open(os.path.join(args.label_dir,
                           'bdd_coco_train_after_exclusion.json')) as f:
        train_labels = json.load(f)
    print('Converting training set...')

    out_fn = os.path.join(args.save_path,
                          'instances_train2017.json')
    bdd2coco_detection(attr_id_dict, train_labels, out_fn)

    print('Loading validation set...')
    # create BDD validation set detections in COCO format
    with open(os.path.join(args.label_dir,
                           'bdd_coco_val_after_exclusion.json')) as f:
        val_labels = json.load(f)
    print('Converting validation set...')

    out_fn = os.path.join(args.save_path,
                          'instances_val2017.json')
    bdd2coco_detection(attr_id_dict, val_labels, out_fn)