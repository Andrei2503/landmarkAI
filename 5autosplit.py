import os
import shutil
from sklearn.model_selection import train_test_split

# Настройки
base_dir = "raw_dataset"
target_dir = "learn/final_dataset"
test_val_size = 0.3  # 30% на val + test

for class_name in os.listdir(base_dir):
    # Список всех файлов класса
    images = os.listdir(os.path.join(base_dir, class_name))
    
    # Разделение
    train, test_val = train_test_split(images, test_size=test_val_size, random_state=42)
    val, test = train_test_split(test_val, test_size=0.5, random_state=42)
    
    # Копирование
    for split, data in [("train", train), ("val", val), ("test", test)]:
        dest = os.path.join(target_dir, split, class_name)
        os.makedirs(dest, exist_ok=True)
        
        for img in data:
            src = os.path.join(base_dir, class_name, img)
            shutil.copy(src, dest)

print("Датасет структурирован!")