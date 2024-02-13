import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Assuming you have defined your model
model = fasterrcnn_resnet50_fpn(weights = None) # Your PyTorch model
output_dir = "/home/yifei/bdd100k/network_faster_rcnn"  # Directory containing the model files

# Loop through each model file in the output directory
for epoch in range(26):  # Assuming you have model_0.pth to model_25.pth
    # Construct the path to the model file for the current epoch
    model_file_path = os.path.join(output_dir, f"model_{epoch}.pth")
    
    # Load the weights from the model file
    checkpoint = torch.load(model_file_path, map_location=torch.device('cpu'))  # Load to CPU if needed
    model.load_state_dict(checkpoint["model"])

    # Now `model` contains the weights from the current model file (`model_{epoch}.pth`)
    # You can use `model` for inference or further training
