from bing_image_downloader import downloader


# Параметры
landmarks = [
    #"Eiffel Tower",
    #"Taj Mahal",
    #"Colosseum",
    #"Statue of Liberty",
    #"Great Wall of China",
    #"Machu Picchu",
    #"Sydney Opera House",
    #"Pyramids of Giza",
    #"Saint Basil's Cathedral",
    #"Big Ben",
    #"Christ the Redeemer",
    #"Petra",
    #"Chichen Itza",
    #"Mount Rushmore",
    #"Angkor Wat",
    #"Hagia Sophia",
    #"Acropolis of Athens",
    #"Stonehenge",
    #"Leaning Tower of Pisa",
    #"Sagrada Familia",
    #"Notre-Dame Cathedral",
    #"Brandenburg Gate",
    #"Burj Khalifa",
    #"Golden Gate Bridge",
    #"Hollywood Sign",
    #"Empire State Building",
    #"Niagara Falls",
    #"Grand Canyon",
    #"Yellowstone National Park",
    #"Mount Fuji",
    #"The Louvre",
    #"Buckingham Palace",
    #"Neuschwanstein Castle",
    #"Matterhorn",
    #"Venice Canals",
    #"Alhambra",
    #"Forbidden City",
    #"Terracotta Army",
    #"Victoria Falls",
    #"Uluru",
    #"Santorini",
    #"Meteora",
    #"Prague Castle",
    #"Charles Bridge",
    #"Dubrovnik Old Town",
    #"Plitvice Lakes",
    #"Mont Saint-Michel",
    #"Versailles Palace",
    #"Louvre Abu Dhabi",
    #"Sheikh Zayed Mosque",
    #"Burj Al Arab",
    #"Palm Jumeirah",
    #"Space Needle",
    #"CN Tower",
    #"Banff National Park",
    #"Château de Chambord",
    #"Mesa Verde",
    #"Red Square",
    #"Hermitage Museum",
    #"Kremlin",
    #"St. Peter's Basilica",
    #"Vatican Museums",
    #"Sistine Chapel",
    #"Trevi Fountain",
    #"Pantheon",
    #"Roman Forum",
    #"Pompeii",
    #"Amalfi Coast",
    #"Capri",
    #"Cinque Terre",
    #"Milan Cathedral",
    #"Florence Cathedral",
    #"Rialto Bridge",
    #"Basilica di San Marco",
    #"Tower Bridge",
    #"Westminster Abbey",
    #"London Eye",
    #"Edinburgh Castle",
    #"Giant's Causeway",
    #"St. Paul's Cathedral",
    #"Tower of London",
    #"Kraków Old Town",
    #"Wawel Castle",
    #"Auschwitz-Birkenau",
    #"Schönbrunn Palace",
    #"Hallstatt",
    #"Plaza de España",
    #"La Sagrada Familia",
    #"Park Güell",
    #"Alcázar of Seville",
    #"Mezquita of Córdoba",
    #"Guggenheim Museum Bilbao",
    #"Ibiza",
    #"Tenerife",
    #"Mount Teide",
    #"Lisbon Cathedral",
    #"Belém Tower",
    #"Pena Palace",
    #"Douro Valley",
    #"Cliffs of Moher"
]
images_per_class = 300
output_dir = "raw_dataset"

# Скачивание
for landmark in landmarks:
    downloader.download(
        query=landmark,
        limit=images_per_class,
        output_dir=output_dir,
        adult_filter_off=False,
        force_replace=False,
        timeout=30
    )

print("Скачивание завершено!")