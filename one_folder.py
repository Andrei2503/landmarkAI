from cleanvision import Imagelab
import os
from PIL import Image
import warnings
import traceback
import tempfile
import shutil
from tqdm import tqdm

TARGET_FOLDER = "raw_dataset/Roman Forum"  # Укажите вашу папку

def process_with_cleanvision(file_path):
    """Обрабатывает один файл через CleanVision во временной папке"""
    try:
        # Создаем временную среду
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Копируем файл во временную папку
            tmp_file = os.path.join(tmp_dir, os.path.basename(file_path))
            shutil.copyfile(file_path, tmp_file)

            # Запускаем CleanVision
            imagelab = Imagelab(data_path=tmp_dir)
            imagelab.find_issues()

        return True, None

    except Exception as e:
        error_info = {
            'file': file_path,
            'error_type': type(e).__name__,
            'message': str(e),
            'traceback': traceback.format_exc()
        }
        return False, error_info

def check_image_integrity(file_path):
    """Проверяет базовую целостность файла"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception as e:
        return False

def main():
    if not os.path.exists(TARGET_FOLDER):
        print(f"Папка {TARGET_FOLDER} не существует!")
        return

    # Собираем все файлы
    all_files = []
    for root, _, files in os.walk(TARGET_FOLDER):
        all_files.extend([os.path.join(root, f) for f in files])

    print(f"\n🔍 Начинаем анализ {len(all_files)} файлов в {TARGET_FOLDER}")

    results = {'good': [], 'bad': [], 'cleanvision_errors': []}

    # Первичная проверка целостности
    for file_path in tqdm(all_files, desc="Базовая проверка"):
        if not check_image_integrity(file_path):
            results['bad'].append(file_path)

    # Проверка через CleanVision только для целых файлов
    clean_files = [f for f in all_files if f not in results['bad']]
    
    for file_path in tqdm(clean_files, desc="CleanVision анализ"):
        # Проверка через CleanVision
        success, error = process_with_cleanvision(file_path)
        
        if success:
            results['good'].append(file_path)
        else:
            results['cleanvision_errors'].append(error)

    # Вывод результатов
    print("\n📊 Результаты:")
    print(f"✅ Целые файлы: {len(results['good'])}")
    print(f"⛔ Битые файлы: {len(results['bad'])}")
    print(f"⚠️ Ошибки CleanVision: {len(results['cleanvision_errors'])}")

    # Детализация ошибок
    if results['cleanvision_errors']:
        print("\n🔥 Ошибки при анализе CleanVision:")
        for error in results['cleanvision_errors']:
            print(f"\n▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬")
            print(f"📁 Файл: {error['file']}")
            print(f"🚨 Ошибка: {error['error_type']}")
            print(f"📄 Сообщение: {error['message']}")
            print("🔍 Трассировка:")
            print(error['traceback'])

    if results['bad']:
        print("\n🗑️ Битые файлы:")
        for file in results['bad']:
            print(f"• {file}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
    print("\n✅ Анализ завершен")