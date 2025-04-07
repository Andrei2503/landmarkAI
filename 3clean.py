from cleanvision import Imagelab
import os

# Список папок для проверки
FOLDERS_TO_CHECK = [
    
#"raw_dataset/Acropolis of Athens",
#"raw_dataset/Alcázar of Seville",
#"raw_dataset/Alhambra",
#"raw_dataset/Amalfi Coast",
#"raw_dataset/Angkor Wat",
#"raw_dataset/Auschwitz-Birkenau",
#"raw_dataset/Banff National Park",
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
"raw_dataset/Edinburgh Castle",
"raw_dataset/Pena Palace",
"raw_dataset/Roman Forum"
]

def check_and_clean_folders(folders):
    for folder in folders:
        try:
            # Проверяем существование папки
            if not os.path.exists(folder):
                print(f"⚠️ Папка {folder} не существует, пропускаем")
                continue
                
            if not os.path.isdir(folder):
                print(f"⚠️ {folder} не является папкой, пропускаем")
                continue

            print(f"\n🔍 Анализируем папку: {folder}")
            
            # Инициализируем Imagelab
            imagelab = Imagelab(data_path=folder)
            
            # Ищем проблемы
            imagelab.find_issues()
            
            # Получаем список проблем
            issues = imagelab.issues
            
            # Проверяем наличие изображений
            if issues.empty:
                print("✅ Нет проблемных изображений")
                continue
                
            # Фильтруем проблемы
            dark_images = []
            blurry_images = []
            
            if 'is_dark_issue' in issues.columns:
                dark_images = issues[issues["is_dark_issue"]].index.tolist()
                for img in dark_images:
                    print(f"🌑 Темное изображение: {os.path.join(folder, img)}")
            
            if 'is_blurry_issue' in issues.columns:
                blurry_images = issues[issues["is_blurry_issue"]].index.tolist()
                for img in blurry_images:
                    print(f"📸 Размытое изображение: {os.path.join(folder, img)}")
            
            # Удаляем проблемные файлы
            for img in dark_images + blurry_images:
                file_path = os.path.join(folder, img)
                try:
                    os.remove(file_path)
                    print(f"🗑️ Удалено: {file_path}")
                except Exception as e:
                    print(f"❌ Ошибка при удалении {file_path}: {str(e)}")
                    
        except Exception as e:
            print(f"🔥 Критическая ошибка при обработке {folder}: {str(e)}")

if __name__ == "__main__":
    check_and_clean_folders(FOLDERS_TO_CHECK)
    print("\n✅ Обработка завершена")