import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 確認 PyTorch 和 torchvision 版本
print("PyTorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)

# 檢查設備
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device:", device)

# 定義類別到索引的映射
class_to_idx = {
    "crown": 1,
    "eyes": 2,
    "tail": 3,
    "feet": 4
}
idx_to_class = {v: k for k, v in class_to_idx.items()}


# 定義自訂數據集類別
class CustomDataset(Dataset):
    def __init__(self, root, annotations, transform=None):
        self.root = root
        self.transform = transform
        with open(annotations) as f:
            self.data = [line.strip().split(',') for line in f.readlines()]

    def __getitem__(self, idx):
        img_path, xmin, ymin, xmax, ymax, label = self.data[idx]
        img_path = os.path.join(self.root, os.path.basename(img_path))
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

# 定義數據轉換
transform = transforms.Compose([
    transforms.ToTensor()
])

# 建立訓練和測試數據集
train_dataset = CustomDataset(root='train_images', annotations='annotate.txt', transform=transform)
test_dataset = CustomTestDataset(root='test_images', transform=transform)

# 建立數據加載器
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 定義模型
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

# 初始化模型
model = get_model(num_classes=5)
model.to(device)

# 訓練模型
optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 1  # 修改為 1 以加快測試速度
print("Start Training...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
print("Training complete.")

# 儲存模型
model_save_path = "frcnn_model3.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved at {model_save_path}")

# 測試模型並儲存帶有 Bounding Box 的圖片
output_folder = "test_images_WBB"
os.makedirs(output_folder, exist_ok=True)

model.eval()
print("Start Testing...")
with torch.no_grad():
    for images, img_paths in test_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for i, image in enumerate(images):
            img = image.permute(1, 2, 0).cpu().numpy()
            plt.imshow(img)

            for box, label in zip(outputs[i]['boxes'].cpu().numpy(), outputs[i]['labels'].cpu().numpy()):
                color = 'g' if label == 1 else 'r' if label == 2 else 'b' if label == 3 else 'yellow'
                plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                                  linewidth=2, edgecolor=color, facecolor='none'))
            output_path = os.path.join(output_folder, os.path.basename(img_paths[i]))
            plt.title(f"Image: {os.path.basename(img_paths[i])}")
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
print(f"Testing complete. Results saved in {output_folder}")
