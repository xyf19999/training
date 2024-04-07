import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from typing import Optional, Callable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision.datasets import MNIST
import torch.nn.functional as F

# custom dataset BDD100K (basic version, only overwrite __len__ and __getitem__)
class BDD100K_basic(Dataset):
    def __init__(self, 
                 root: str, 
                 annFile: str, 
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None, 
                 transforms: Optional[Callable] = None,
                 noise_pertubation_in_region_name:str = None,
                 region = None,
                 transform_params: list = None):
        """
        Custom dataset for BDD100K object detection.

        Args:
        - root_folder (str): Root directory where images are stored.
        - annotation_file (str): Path to the annotation file (JSON format).
        - transform (callable, optional): A function/transform to apply to the images.
        """
        self.root = root
        self.annFile = annFile
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.noise_pertubation_in_region_name = noise_pertubation_in_region_name
        self.region = region
        self.transform_params = transform_params

        #Load BDD100K annotations
        with open(annFile, 'r') as f:
          self.annotations = json.load(f)

          
    def __len__(self):
        """"
        Get the number of samples in the dataset.

        Returns:
        - int: Number of samples.
        """

        return len(self.annotations) 

    def __getitem__(self, idx):
        """
        Get a specific sample from the dataset.

        Args:
        - idx (int): Index of the sample.

        Returns:
        - tuple: (image, target) where
          - image (PIL Image): The input image.
          - target (dict): Dictionary containing information about bounding boxes and labels.
        """
        # Get image filename and open the image
        img_filename = os.path.join(self.root, self.annotations[idx]['name'])
        image = Image.open(img_filename).convert('RGB')

        # Get bounding box annotations
        annotations = self.annotations[idx]
        # Format bounding box annotations

        target = annotations

        # Apply transformations if available
        if self.transform:
            image = self.transform(image)
            # yifei, target?

        return image, target

preprocessing_transformers = transforms.ToTensor()

def custom_collate_fn(batch):
    # Extract images and labels from the batch
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Stack images along a new batch dimension
    images = torch.stack(images, dim=0)

    return images, labels


model = torch.load('/home/yifei/bdd100k/faster_rcnn_bdd100k.pth')
#model = torch.load('/Users/mac/Desktop/tum/monitizer/example-networks/MNIST3x100')

model.eval()

model = model.to('cpu')

# Define your list of transforms
preprocessing_transformers = [ 
    transforms.ToTensor() # Add more transforms as needed
]

# Create a composed transform using transforms.Compose
transform = transforms.Compose(preprocessing_transformers)
""" 
bdd100k_trainset = BDD100K_basic(root='/Users/mac/Desktop/bdd100k/bdd100k_in_yolo/train/images', 
                                 annFile='/Users/mac/Desktop/bdd100k/labels_origin/bdd100k_labels_images_train.json',
                                 transform=transforms.Compose(preprocessing_transformers))

trainloader = DataLoader(bdd100k_trainset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn) """

coco_set = CocoDetection(root='/home/yifei/coco/coco/val2017', annFile='/home/yifei/coco/coco/annotations/instances_val2017.json', transform=transform)

coco_dataloader = DataLoader(coco_set, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)
""" 
mnist_set = MNIST(root='/Users/mac/Desktop/tum/monitizer/data', train=True, download=False, transform=transform)

mnist_dataloader = DataLoader(mnist_set, batch_size=64, shuffle=False)""" 

for idx, (images, labels) in enumerate(coco_dataloader):
    print('start')
    print(images.shape)
    prediction = model(images)
    print(prediction.shape)
    if idx == 2:
      break 

""" image, label = mnist_set[0]
with torch.no_grad():
  prediction = model(image)

predicted_class = torch.argmax(prediction, dim=1).item()

print(predicted_class) """
