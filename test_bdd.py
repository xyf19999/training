import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision
import sys
# Assuming you have defined your model
model_faster = fasterrcnn_resnet50_fpn(weights = None, num_classes = 20) # Your PyTorch model
model_retina = retinanet_resnet50_fpn(weights= None, num_classes = 20)
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

checkpoint_path_faster = '/home/yifei/bdd_coco/faster_rcnn/checkpoint.pth'
checkpoint_path_retina = '/home/yifei/bdd_coco/retina/checkpoint.pth'

if os.path.exists(checkpoint_path_faster):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path_faster, map_location=torch.device('cpu'))  # Load to CPU if needed # map_location=torch.device('cpu')
    
    # Load the model state dictionary from the checkpoint
    model_faster.load_state_dict(checkpoint["model"])

if os.path.exists(checkpoint_path_retina):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path_retina, map_location=torch.device('cpu'))  # Load to CPU if needed # map_location=torch.device('cpu')
    
    # Load the model state dictionary from the checkpoint
    model_retina.load_state_dict(checkpoint["model"])

torch.save(model_faster, '/home/yifei/bdd_coco/faster_rcnn_bdd_coco.pth')
torch.save(model_retina, '/home/yifei/bdd_coco/retina_bdd_coco.pth')


sys.exit()

model.eval()

image = Image.open("/home/yifei/bdd100k/train2017/0000f77c-6257be58.jpg")
image = Image.open("/home/yifei/coco_dataset/train2017/000000311997.jpg")

# Define the transformation to convert the image to a PyTorch tensor
transform = transforms.ToTensor()

# Apply the transformation to the image
tensor_image = transform(image)

images = [tensor_image]

predictions = model(images)
print(predictions)


    # Now `model` contains the weights from the current model file (`model_{epoch}.pth`)
    # You can use `model` for inference or further training
