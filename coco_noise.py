import torchvision.datasets as dset


path2data="/home/yifei/coco_dataset/val2017"
path2json="/home/yifei/coco_dataset/annotations/instances_val2017.json"


coco_val = dset.CocoDetection(root = path2data,
                              annFile = path2json)

print('success')
