import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision
import numpy as np
# Assuming you have defined your model
# Your PyTorch model
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '/home/yifei/monitizer/monitizer/benchmark_object_detection/networks/faster_rcnn_bdd100k.pth'
model = torch.load(model_path, map_location=device)

model.eval()

torch.set_default_dtype(torch.float)
#torch.set_printoptions(threshold=float('inf'))

def hook_fn(module, input, output):
    # Store the output of the layer
    if module.__class__.__name__ == 'FeaturePyramidNetwork':
      hook_fn.layer_outputs[module.__class__.__name__] = output

feature_map_tensor_zero = []
feature_map_tensor_one = []
feature_map_tensor_two = []
feature_map_tensor_three = []
feature_map_tensor_four = []

feature_map_dict = {
   0: feature_map_tensor_zero,
   1: feature_map_tensor_one,
   2: feature_map_tensor_two,
   3: feature_map_tensor_three,
   4: feature_map_tensor_four
}
# Register the hook to all modules in the model
hook_fn.layer_outputs = {}
for name, module in model.named_modules():
    module.register_forward_hook(hook_fn)

def get_image_names(folder_path):
    image_names = []
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            image_names.append(file)
    return image_names

image_paths = ['/Users/mac/Desktop/bdd100k/bdd100k_in_yolo/val/images/b1c81faa-3df17267.jpg']
#image_paths = get_image_names('/Users/mac/Desktop/bdd100k/bdd100k_in_yolo/val/images')

source_folder_calibration = '/home/yifei/bdd100k/calibration/'
images_calibration = os.listdir(source_folder_calibration)
images_calibration = [source_folder_calibration+image for image in images_calibration]

source_folder_proper_training = '/home/yifei/bdd100k/proper_training'
images_proper_training = os.listdir(source_folder_proper_training)
images_proper_training = [source_folder_proper_training+image for image in images_proper_training]



transform = transforms.Compose([
    transforms.ToTensor()
])
images = []

""" 
for name, param in model.named_parameters():
    print(name, param.shape) #param.detach().numpy()) """

# Loop through each image path
""" for image_path in image_paths:
  with torch.no_grad(): # Open the image
    image = Image.open(image_path)
    
    # Apply the transformation to the image
    tensor_image = transform(image)
    
    # Append the transformed image tensor to the list
    print(model)
    predictions = model([tensor_image])
    #print('prediction', predictions)

    for layer_name, output in hook_fn.layer_outputs.items():
        print(f"Output of layer '{layer_name}': {output['3'].shape}")
       

    hook_fn.layer_outputs.clear() """

for image_path in source_folder_proper_training:
    with torch.no_grad():
        image = Image.open(image_path)
        tensor_image = transform(image).to(device)  # Move tensor to GPU
        images.append(tensor_image)

for tensor_image in images:
    with torch.no_grad():
        predictions = model([tensor_image])
        for layer_name, output in hook_fn.layer_outputs.items():
            keys = list(output.keys())
            for idx, key in enumerate(keys):
                feature_map_dict[idx].append(output[key])
        hook_fn.layer_outputs.clear()

for i in range(5):
  print('start', i)
  current_feature_map_element = feature_map_dict[i]
  current_feature_map_element = torch.stack(current_feature_map_element).to(device)
  np.save(f'./arrays_{i}.npy', [1,2,3])
  mean_tensor = torch.mean(current_feature_map_element, dim=0)
  std_tensor = torch.std(current_feature_map_element, dim=0)
  mean_tensor_array = mean_tensor.cpu().numpy() 
  std_tensor_array = std_tensor.cpu().numpy()
  np.save(f'./arrays_{i}.npy', (mean_tensor_array, std_tensor_array))
