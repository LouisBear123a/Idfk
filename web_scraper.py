import scrapy
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup


class WebScraper:
    def __init__(self, url1, url2):
        self.url1 = url1
        self.url2 = url2
    
    def scrape_data(self):
        # Step 1: Scrape data from the first URL using Scrapy
        process1 = CrawlerProcess()
        process1.crawl(WebScraperSpider, url=self.url1)
        process1.start()

        # Step 2: Scrape data from the second URL using BeautifulSoup
        response2 = scrapy.Request(url=self.url2)
        soup = BeautifulSoup(response2.text, 'html.parser')
        data2 = soup.find_all('code')

        # Combine the data from both sources
        data = process1.spider.data + [d.text for d in data2]
        
        return data


class WebScraperSpider(scrapy.Spider):
    name = 'WebScraperSpider'
    data = []

    def __init__(self, url, **kwargs):
        super().__init__(**kwargs)
        self.start_urls = [url]

    def parse(self, response):
        # Extract data from the web page
        data = response.css('td.views-field-title a::text').getall()

        # Store the data in the spider
        self.data = data
