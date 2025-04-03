import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import os

# Конфигурация
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 100
INPUT_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 30
LR = 0.0005
DATA_DIR = "final_dataset"  # Основная директория с данными

# Проверка наличия всех поддиректорий
for folder in ['train', 'val', 'test']:
    if not os.path.exists(os.path.join(DATA_DIR, folder)):
        raise ValueError(f"Directory {os.path.join(DATA_DIR, folder)} not found!")

# Аугментации
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Загрузка данных
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_test_transforms)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Проверка соответствия классов во всех наборах
assert train_dataset.classes == val_dataset.classes == test_dataset.classes, "Class labels mismatch between datasets!"

# Модель (остается без изменений)
class CustomCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes))
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Инициализация модели
model = CustomCNN(NUM_CLASSES).to(DEVICE)

# Функция для подсчета классов
def get_class_counts(dataset):
    counts = [0] * len(dataset.classes)
    for _, label in dataset:
        counts[label] += 1
    return counts

# Веса для loss (используем только train для расчета весов)
class_counts = get_class_counts(train_dataset)
class_weights = torch.tensor([1.0/count for count in class_counts], device=DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Оптимизатор и scheduler
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1, verbose=True)
scaler = torch.cuda.amp.GradScaler()

# MixUp аугментация
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

# Функция для вычисления точности
def evaluate(loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return acc, all_preds, all_labels

# Цикл обучения
best_val_acc = 0.0
for epoch in range(NUM_EPOCHS):
    model.train()
    progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    epoch_loss = 0.0
    
    for images, labels in progress:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # MixUp
        inputs, targets_a, targets_b, lam = mixup_data(images, labels)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        progress.set_postfix({'loss': loss.item()})
    
    # Валидация
    val_acc, _, _ = evaluate(val_loader)
    scheduler.step(val_acc)  # Обновляем learning rate на основе val accuracy
    
    print(f'Epoch {epoch+1}: Train Loss: {epoch_loss/len(train_loader):.4f}, Val Accuracy: {val_acc*100:.2f}%')
    
    # Сохранение лучшей модели
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, 'best_model.pth')
        print('New best model saved!')

# Загрузка лучшей модели для тестирования
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Финальное тестирование на test set
test_acc, test_preds, test_labels = evaluate(test_loader)

print('\nFinal Test Results:')
print(f'Test Accuracy: {test_acc*100:.2f}%')
print('Confusion Matrix:')
print(confusion_matrix(test_labels, test_preds))
print('\nClassification Report:')
print(classification_report(test_labels, test_preds, target_names=test_dataset.classes))

# Дополнительно: сохранение полного отчета
with open('test_results.txt', 'w') as f:
    f.write(f'Test Accuracy: {test_acc*100:.2f}%\n\n')
    f.write('Confusion Matrix:\n')
    f.write(str(confusion_matrix(test_labels, test_preds)))
    f.write('\n\nClassification Report:\n')
    f.write(classification_report(test_labels, test_preds, target_names=test_dataset.classes))