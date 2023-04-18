import requests
from bs4 import BeautifulSoup

class WebScraper:
    def __init__(self, url):
        self.url = url
    
    def get_data(self):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, 'html.parser')
        data = []
        for tag in soup.find_all('tag'):
            data.append(tag.text)
        return data
    
    def preprocess_data(self, data):
        # Clean and preprocess the collected data
        processed_data = []
        for item in data:
            # Remove irrelevant or redundant information
            if 'irrelevant' not in item and 'redundant' not in item:
                processed_data.append(item)
        return processed_data
