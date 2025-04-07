import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import imagehash

def find_duplicates(class_path, hash_threshold=5, hist_threshold=0.8):
    """Находит дубликаты в папке класса"""
    hashes = {}
    duplicates = []
    
    # Получить список изображений
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    for img_name in tqdm(images, desc=f"Обработка {os.path.basename(class_path)}"):
        img_path = os.path.join(class_path, img_name)
        
        try:
            # Вычисление хеша изображения
            with Image.open(img_path) as img:
                phash = imagehash.phash(img)
            
            # Вычисление гистограммы
            img = cv2.imread(img_path)
            hist = cv2.calcHist([img], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Поиск похожих изображений
            is_duplicate = False
            for existing in hashes.values():
                # Сравнение хешей
                hash_diff = phash - existing['phash']
                
                # Сравнение гистограмм
                hist_diff = cv2.compareHist(hist, existing['hist'], cv2.HISTCMP_CORREL)
                
                if hash_diff < hash_threshold and hist_diff > hist_threshold:
                    is_duplicate = True
                    break
                    
            if is_duplicate:
                duplicates.append(img_name)
            else:
                hashes[img_name] = {'phash': phash, 'hist': hist}
                
        except Exception as e:
            print(f"Ошибка обработки {img_name}: {str(e)}")
            continue
            
    return duplicates

# Основной цикл обработки
for class_dir in os.listdir("raw_dataset"):
    class_path = os.path.join("raw_dataset", class_dir)
    
    if not os.path.isdir(class_path):
        continue
        
    print(f"\nПоиск дубликатов в: {class_dir}")
    duplicates = find_duplicates(class_path)
    
    if duplicates:
        print(f"Найдено дубликатов: {len(duplicates)}")
        for dup in tqdm(duplicates, desc="Удаление"):
            os.remove(os.path.join(class_path, dup))
    else:
        print("Дубликаты не найдены")