#!/usr/bin/env python
# coding: utf-8





# Read and display the contents of the annotate.txt file
with open('annotate.txt', 'r') as file:
    content = file.read()

print(content)


# In[6]:


import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

print(torch.__version__)
print(torchvision.__version__)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("device is :")
print(device)

# Define the class to index mapping
class_to_idx = {
    "ChickenCrown": 1,
    "ChickenEye": 2,
    "ChickenTail": 3,
    "ChickenFeet": 4
}

# Define the reverse mapping from index to class
idx_to_class = {v: k for k, v in class_to_idx.items()}

class CustomDataset(Dataset):
    def __init__(self, root, annotations, transform=None):
        self.root = root
        self.transform = transform
        with open(annotations) as f:
            self.data = [line.strip().split(',') for line in f.readlines()]

    def __getitem__(self, idx):
        img_path, xmin, ymin, xmax, ymax, label = self.data[idx]
        img_path = os.path.join(self.root, os.path.basename(img_path))  # Correctly join the path
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        boxes = torch.tensor([[int(xmin), int(ymin), int(xmax), int(ymax)]], dtype=torch.float32)
        labels = torch.tensor([class_to_idx[label]], dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels
        }

        return img, target

    def __len__(self):
        return len(self.data)

class CustomTestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = [os.path.join(root, img) for img in os.listdir(root) if img.endswith('.jpg')]

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_path

    def __len__(self):
        return len(self.images)

# Define the data transforms
transform = transforms.Compose([
    transforms.ToTensor()
])



# Create the datasets
train_dataset = CustomDataset(root='train_images', annotations='annotate.txt', transform=transform) # please set to correct directory
test_dataset = CustomTestDataset(root='test_images', transform=transform)



# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


# In[7]:


import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def get_model(num_classes):
    # Load a model pre-trained on COCO and then fine-tune it
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

# 4 classes + background
model = get_model(num_classes=5)


# In[8]:


import os

# List files in train_images directory
train_images_dir = 'train_images'
print("Training Images:")
for filename in os.listdir(train_images_dir):
    if filename.endswith(".jpg"):
        print(os.path.join(train_images_dir, filename))

# List files in test_images directory
test_images_dir = 'test_images'
print("\nTesting Images:")
for filename in os.listdir(test_images_dir):
    if filename.endswith(".jpg"):
        print(os.path.join(test_images_dir, filename))


# In[ ]:


import torch.optim as optim
from torch.utils.data import DataLoader

print("Initialize train_loader")
# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
print("Finish train_loader")


print("Start Training")

# Training parameters
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 1 # Change the number of epochs trained in each run, larger models are usually set to 100, but it's set to 1 for now to save time.

for epoch in range(num_epochs):
    model.train()
    i = 0
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {epoch}, Iteration {i}, Loss: {losses.item()}")
        i += 1
    print(f"Epoch {epoch} finished.")

print("Training complete.")


# In[ ]:


model_save_path = "frcnn_model3.pth"  # Specify your desired path
torch.save(model.state_dict(), model_save_path)
print(f"Model weights saved to {model_save_path}")


# In[ ]:


import os
import matplotlib.pyplot as plt

# Set up the folder for saving images with bounding boxes
output_folder = "test_images_WBB"
os.makedirs(output_folder, exist_ok=True)

model.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for images, img_paths in test_loader:
    images = list(image.to(device) for image in images)
    with torch.no_grad():
        outputs = model(images)

    for i, image in enumerate(images):
        img = image.permute(1, 2, 0).cpu().numpy()  # Convert the tensor to a numpy array
        plt.imshow(img)
        
        # Draw bounding boxes
        for box, label in zip(outputs[i]['boxes'].cpu().numpy(), outputs[i]['labels'].cpu().numpy()):
            color = 'g' if label == 1 else 'r' if label == 2 else 'b' if label == 3 else 'yellow'
            plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                              linewidth=1, edgecolor=color, facecolor='none'))

        # Set title and save the figure
        img_filename = os.path.basename(img_paths[i])
        output_path = os.path.join(output_folder, img_filename)
        plt.title(f'Image: {img_filename}')
        plt.axis('off')  # Turn off axis
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the plot to free memory

print("Images with bounding boxes have been saved to 'test_images_WBB'.")
# In[ ]:




