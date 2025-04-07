import os

def get_subfolders(parent_dir):
    subfolders = []
    try:
        for entry in os.scandir(parent_dir):
            if entry.is_dir():
                # Формируем путь с нормальными слешами для всех ОС
                path = os.path.join(parent_dir, entry.name).replace("\\", "/")
                subfolders.append(f'"{path}"')
    except FileNotFoundError:
        print(f"❌ Ошибка: Директория {parent_dir} не найдена")
    return sorted(subfolders)

if __name__ == "__main__":
    parent_folder = "raw_dataset"
    folders_list = get_subfolders(parent_folder)
    
    if folders_list:
        print("Список подпапок:\n[\n" + ",\n".join(folders_list) + "\n]")
    else:
        print("В указанной директории нет подпапок")