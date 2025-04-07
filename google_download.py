from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import os
import requests
from tqdm import tqdm
import time
import random

# Папка для сохранения
output_dir = "selenium_dataset"
os.makedirs(output_dir, exist_ok=True)

# Настройки браузера
chrome_options = Options()
chrome_options.add_argument("--headless")  # Режим без отображения
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Прокси (если нужно)
# chrome_options.add_argument("--proxy-server=socks5://127.0.0.1:9050")  # Для Tor

# Автоматическая установка ChromeDriver
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=chrome_options
)

def download_images(query, num_images=50):
    # Создаем папку для запроса
    query_dir = os.path.join(output_dir, query.replace(" ", "_"))
    os.makedirs(query_dir, exist_ok=True)

    # Поиск в Google Images
    driver.get(f"https://www.google.com/search?q={query}&tbm=isch")
    
    # Прокрутка страницы для загрузки изображений
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # Сбор ссылок на изображения
    thumbnails = driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")
    img_urls = []
    
    for img in thumbnails[:num_images]:
        try:
            img.click()
            time.sleep(1)
            # Получаем URL полноразмерного изображения
            actual_img = driver.find_element(By.CSS_SELECTOR, "img.sFlh5c")
            src = actual_img.get_attribute("src")
            if src.startswith("http"):
                img_urls.append(src)
        except:
            continue

    # Скачивание изображений
    for i, url in enumerate(tqdm(img_urls, desc=f"Скачивание {query}")):
        try:
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                with open(os.path.join(query_dir, f"{query}_{i+1}.jpg"), "wb") as f:
                    f.write(response.content)
            time.sleep(random.uniform(0.5, 2))  # Случайная задержка
        except:
            continue

# Пример использования
download_images("Machu Picchu", num_images=20)
download_images("Great Wall of China", num_images=20)

# Закрываем браузер
driver.quit()