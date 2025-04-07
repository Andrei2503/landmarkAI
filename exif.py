import os
from PIL import Image
import warnings

def check_exif_errors(directory):
    corrupt_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            try:
                with warnings.catch_warnings(record=True) as captured_warnings:
                    warnings.simplefilter("always")
                    
                    with Image.open(file_path) as img:
                        img.load()  # Явная загрузка изображения
                        
                        # Проверка EXIF (некоторые ошибки проявляются только при обращении)
                        if hasattr(img, 'tag_v2'):
                            _ = img.tag_v2
                        
                    # Проверяем перехваченные предупреждения
                    for warn in captured_warnings:
                        if "Corrupt EXIF data" in str(warn.message):
                            corrupt_files.append(file_path)
                            break
                            
            except Exception as e:
                print(f"Ошибка при обработке {file_path}: {str(e)}")

    return corrupt_files

if __name__ == "__main__":
    target_directory = "/home/andrei/landmarkAI/raw_dataset/Sagrada Familia"
    corrupt_files = check_exif_errors(target_directory)
    
    if corrupt_files:
        print("Файлы с проблемами EXIF:")
        for f in corrupt_files:
            print(f"• {f}")
    else:
        print("Файлы с повреждённым EXIF не найдены")