import os

def delete_webp_files(root_dir):
    deleted_count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.gif'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Удалён файл: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Ошибка при удалении {file_path}: {str(e)}")
    return deleted_count

if __name__ == "__main__":
    target_directory = "raw_dataset"
    
    if not os.path.exists(target_directory):
        print(f"Папка {target_directory} не найдена!")
        exit(1)
        
    print(f"Начинаю поиск .webp файлов в {target_directory}...")
    count = delete_webp_files(target_directory)
    print(f"Готово! Удалено файлов: {count}")