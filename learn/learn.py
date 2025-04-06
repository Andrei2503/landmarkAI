import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import os
from PIL import Image

# Конфигурация
DEVICE = torch.device("cpu")
NUM_CLASSES = 100
INPUT_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 0.0005
DATA_DIR = "final_dataset"

# Кастомный класс датасета с обработкой изображений
class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        
    def __getitem__(self, index):
        try:
            path, target = self.samples[index]
            img = Image.open(path).convert('RGB')
            
            # Принудительное изменение размера
            if self.transform is not None:
                img = self.transform(img)
                
            return img, target
        except Exception as e:
            print(f"Error loading {self.samples[index][0]}: {str(e)}")
            # Возвращаем черное изображение и метку -1
            return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), -1

# Аугментации с гарантированным изменением размера
train_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Загрузка данных с фильтрацией битых изображений
def create_dataset(path, transform):
    dataset = SafeImageFolder(path, transform=transform)
    
    # Фильтрация невалидных образцов
    valid_indices = [i for i in range(len(dataset)) 
                    if dataset[i][1] != -1]
    
    return torch.utils.data.Subset(dataset, valid_indices)

train_dataset = create_dataset(os.path.join(DATA_DIR, "train"), train_transforms)
val_dataset = create_dataset(os.path.join(DATA_DIR, "val"), val_test_transforms)
test_dataset = create_dataset(os.path.join(DATA_DIR, "test"), val_test_transforms)

# Кастомная функция для сборки батчей
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images.to(DEVICE), labels.to(DEVICE)

# DataLoader с отключенной многопоточностью
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=True
)

# Упрощенная модель
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

model = SimpleCNN(NUM_CLASSES).to(DEVICE)

# Инициализация обучения
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
criterion = nn.CrossEntropyLoss()

# Функции обучения и оценки
def train_epoch(loader):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Основной цикл обучения
best_acc = 0.0
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
    train_loss = train_epoch(train_loader)
    val_acc = evaluate(val_loader)
    scheduler.step(val_acc)
    
    print(f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Model saved!")

# Тестирование
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
test_acc = evaluate(test_loader)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

# Генерация отчетов
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.dataset.classes))

with open("results.txt", "w") as f:
    f.write(classification_report(all_labels, all_preds, target_names=train_dataset.dataset.classes))