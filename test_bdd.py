import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Instantiate the pre-trained model without loading pre-trained weights
model = fasterrcnn_resnet50_fpn(pretrained=False)

# Load your custom weights from the .pth file
# Adjust the file path according to where your weights are stored
weights_path = "/home/yifei/bdd100k/network_faster_rcnn/model_25.pth"
model.load_state_dict(torch.load(weights_path))

# Make sure to set the model to evaluation mode
model.eval()

# Now you can use this model for inference
