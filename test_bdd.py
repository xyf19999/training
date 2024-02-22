import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision
# Assuming you have defined your model
model = fasterrcnn_resnet50_fpn(weights = None, num_classes = 11) # Your PyTorch model
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

checkpoint_path = '/home/yifei/bdd100k/network_faster_rcnn_11_classes/checkpoint.pth'

if os.path.exists(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Load to CPU if needed
    
    # Load the model state dictionary from the checkpoint
    model.load_state_dict(checkpoint["model"])


model.eval()

image = Image.open("/home/yifei/bdd100k/val2017/b1c81faa-3df17267.jpg")
#image = Image.open("/home/yifei/coco_dataset/train2017/000000311997.jpg")

# Define the transformation to convert the image to a PyTorch tensor
transform = transforms.ToTensor()

# Apply the transformation to the image
tensor_image = transform(image)

images = [tensor_image]

predictions = model(images)
print(predictions)


    # Now `model` contains the weights from the current model file (`model_{epoch}.pth`)
    # You can use `model` for inference or further training
