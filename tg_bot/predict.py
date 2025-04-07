import torch
from torchvision import transforms
from PIL import Image
import os
import json

# Конфигурация (должна совпадать с тренировочной)
DEVICE = torch.device("cpu")
NUM_CLASSES = 100
INPUT_SIZE = 128
MODEL_PATH = "../learn/best_model.pth"
TEST_DATA_DIR = "../learn/final_dataset/test"  # Путь к тестовой директории с классами

# Определение архитектуры модели
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * (INPUT_SIZE//8)**2, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Загрузка модели
model = SimpleCNN(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Получение списка классов
class_names = sorted([d.name for d in os.scandir(TEST_DATA_DIR) if d.is_dir()])

# Трансформации для изображения
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    try:
        # Загрузка и преобразование изображения
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Предсказание
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted_idx = torch.max(output, 1)
        
        return class_names[predicted_idx.item()]
    
    except Exception as e:
        return f"Error: {str(e)}"