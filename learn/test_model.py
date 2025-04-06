import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Конфигурация (должна совпадать с тренировочной)
DEVICE = torch.device("cpu")
NUM_CLASSES = 100
INPUT_SIZE = 128
DATA_DIR = "final_dataset"
MODEL_PATH = "best_model.pth"

# Определение архитектуры модели (должна совпадать с тренировочной)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (INPUT_SIZE//8)**2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Загрузка данных
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except:
            return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), -1

test_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = SafeImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transforms)
test_dataset = torch.utils.data.Subset(test_dataset, [i for i in range(len(test_dataset)) if test_dataset[i][1] != -1])

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Загрузка модели
model = SimpleCNN(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Получение предсказаний
all_preds = []
all_labels = []
class_names = test_dataset.dataset.classes

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Processing"):
        outputs = model(images.to(DEVICE))
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Генерация отчетов
print("\nClassification Report:")
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)

# Confusion Matrix
plt.figure(figsize=(20, 15))
cm = confusion_matrix(all_labels, all_preds)

# Визуализация (для больших матриц лучше сохранять в файл)
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=90, ha='right', fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

print("Results saved to classification_report.txt and confusion_matrix.png")