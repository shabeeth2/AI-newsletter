import requests
from langchain.tools import tool
import json

class ScraperTool():
    @tool("Scraper Tool")
    def scrape(url: str):
        "Useful tool to scrap a website content, use to learn more about a given url."
        
        api_url = f'https://r.jina.ai/{url}'
        
        response = requests.get(api_url)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.text
            return data
        else:
            return f"Failed to retrieve data from {url}"