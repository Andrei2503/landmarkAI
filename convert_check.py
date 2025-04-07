from cleanvision import Imagelab
import os
from PIL import Image
import warnings
import contextlib
import traceback

# Конфигурация
FOLDERS_TO_CHECK = [
#"raw_dataset/Acropolis of Athens",
#"raw_dataset/Alcázar of Seville",
#"raw_dataset/Alhambra",
#"raw_dataset/Amalfi Coast",
#"raw_dataset/Angkor Wat",
#"raw_dataset/Auschwitz-Birkenau",
# "raw_dataset/Banff National Park",
#"raw_dataset/Basilica di San Marco",
#"raw_dataset/Belém Tower",
#"raw_dataset/Big Ben",
#"raw_dataset/Brandenburg Gate",
#"raw_dataset/Buckingham Palace",
#"raw_dataset/Burj Al Arab",
#"raw_dataset/Burj Khalifa",
#"raw_dataset/CN Tower",
#"raw_dataset/Capri",
#"raw_dataset/Charles Bridge",
#"raw_dataset/Chichen Itza",
#"raw_dataset/Christ the Redeemer",
#"raw_dataset/Château de Chambord",
#"raw_dataset/Cinque Terre",
#"raw_dataset/Cliffs of Moher",
#"raw_dataset/Colosseum",
#"raw_dataset/Douro Valley",
#"raw_dataset/Dubrovnik Old Town",
#"raw_dataset/Eiffel Tower",
#"raw_dataset/Empire State Building",
#"raw_dataset/Florence Cathedral",
#"raw_dataset/Forbidden City",
#"raw_dataset/Giant's Causeway",
#"raw_dataset/Golden Gate Bridge",
#"raw_dataset/Grand Canyon",
#"raw_dataset/Great Wall of China",
#"raw_dataset/Guggenheim Museum Bilbao",
#"raw_dataset/Hagia Sophia",
#"raw_dataset/Hallstatt",
#"raw_dataset/Hermitage Museum",
#"raw_dataset/Hollywood Sign",
#"raw_dataset/Ibiza",
#"raw_dataset/Kraków Old Town",
#"raw_dataset/Kremlin",
#"raw_dataset/La Sagrada Familia",
#"raw_dataset/Leaning Tower of Pisa",
#"raw_dataset/Lisbon Cathedral",
#"raw_dataset/London Eye",
#"raw_dataset/Louvre Abu Dhabi",
#"raw_dataset/Machu Picchu",
#"raw_dataset/Matterhorn",
#"raw_dataset/Mesa Verde",
#"raw_dataset/Meteora",
#"raw_dataset/Mezquita of Córdoba",
#"raw_dataset/Milan Cathedral",
#"raw_dataset/Mont Saint-Michel",
#"raw_dataset/Mount Fuji",
#"raw_dataset/Mount Rushmore",
#"raw_dataset/Mount Teide",
#"raw_dataset/Neuschwanstein Castle",
#"raw_dataset/Niagara Falls",
#"raw_dataset/Notre-Dame Cathedral",
#"raw_dataset/Palm Jumeirah",
#"raw_dataset/Pantheon",
#"raw_dataset/Park Güell",
#"raw_dataset/Petra",
#"raw_dataset/Plaza de España",
#"raw_dataset/Plitvice Lakes",
#"raw_dataset/Pompeii",
#"raw_dataset/Prague Castle",
#"raw_dataset/Pyramids of Giza",
#"raw_dataset/Red Square",
#"raw_dataset/Rialto Bridge",
#"raw_dataset/Sagrada Familia",
#"raw_dataset/Saint Basil's Cathedral",
#"raw_dataset/Santorini",
#"raw_dataset/Schönbrunn Palace",
#"raw_dataset/Sheikh Zayed Mosque",
#"raw_dataset/Sistine Chapel",
#"raw_dataset/Space Needle",
#"raw_dataset/St. Paul's Cathedral",
#"raw_dataset/St. Peter's Basilica",
#"raw_dataset/Statue of Liberty",
#"raw_dataset/Stonehenge",
#"raw_dataset/Sydney Opera House",
#"raw_dataset/Taj Mahal",
#"raw_dataset/Tenerife",
#"raw_dataset/Terracotta Army",
#"raw_dataset/The Louvre",
#"raw_dataset/Tower Bridge",
#"raw_dataset/Tower of London",
#"raw_dataset/Trevi Fountain",
#"raw_dataset/Uluru",
#"raw_dataset/Vatican Museums",
#"raw_dataset/Venice Canals",
#"raw_dataset/Versailles Palace",
#"raw_dataset/Victoria Falls",
#"raw_dataset/Wawel Castle",
#"raw_dataset/Westminster Abbey",
#"raw_dataset/Yellowstone National Park",
#"raw_dataset/Edinburgh Castle",
#"raw_dataset/Pena Palace",
#"raw_dataset/Roman Forum"

]

class PILWarningParser:
    def __init__(self):
        self.warning_files = set()
        self.original_showwarning = warnings.showwarning

    def custom_showwarning(self, message, category, filename, lineno, file=None, line=None):
        if "PIL.Image" in str(filename) and "Palette images with Transparency" in str(message):
            frame = warnings._getframe(4)  # Поднимаемся по стеку вызовов
            if 'filename' in frame.f_locals:
                self.warning_files.add(frame.f_locals['filename'])
        else:
            self.original_showwarning(message, category, filename, lineno, file, line)

    def __enter__(self):
        warnings.showwarning = self.custom_showwarning
        return self

    def __exit__(self, *args):
        warnings.showwarning = self.original_showwarning

def process_image(file_path):
    try:
        with Image.open(file_path) as img:
            # Конвертируем проблемные изображения
            if img.mode == 'P' and 'transparency' in img.info:
                print(f"🖼️ Конвертируем палитровое изображение: {file_path}")
                img = img.convert('RGBA')
                img.save(file_path)
    except Exception as e:
        print(f"⛔ Ошибка обработки {file_path}: {str(e)}")

def check_and_clean_folders(folders):
    for folder in folders:
        try:
            if not os.path.exists(folder):
                print(f"⚠️ Папка {folder} не существует, пропускаем")
                continue
                
            if not os.path.isdir(folder):
                print(f"⚠️ {folder} не является папкой, пропускаем")
                continue

            print(f"\n🔍 Анализируем папку: {folder}")
            
            # Шаг 1: Предварительная обработка изображений
            for root, _, files in os.walk(folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    process_image(file_path)
            
            # Шаг 2: Анализ с CleanVision
            with PILWarningParser() as parser:
                imagelab = Imagelab(data_path=folder)
                imagelab.find_issues()
                
                warning_files = parser.warning_files

            # Выводим информацию о файлах с предупреждениями
            if warning_files:
                print("\n⚠️ Файлы с предупреждениями PIL:")
                for wf in warning_files:
                    print(f"• {wf}")
            else:
                print("✅ Нет файлов с предупреждениями PIL")

            # Остальная логика обработки...
            
        except Exception as e:
            print(f"\n🔥 Критическая ошибка при обработке папки: {folder}")
            print(f"⛔ Тип ошибки: {type(e).__name__}")
            print(f"📄 Сообщение об ошибке: {str(e)}")
            print("🔍 Трассировка ошибки:")
            traceback.print_exc()  # Вывод полной трассировки
            print("\n🛑 Возможно, проблема связана с одним из файлов в этой папке. Проверьте последний обработанный файл или файлы, упомянутые в трассировке.")

if __name__ == "__main__":
    # Настраиваем фильтры предупреждений
    warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")
    
    check_and_clean_folders(FOLDERS_TO_CHECK)
    print("\n✅ Обработка завершена")