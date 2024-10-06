from firecrawl import FirecrawlApp
from typing import List, Optional
from pydantic import BaseModel, Field
import json
import os
from pathlib import Path

import dotenv; dotenv.load_dotenv()
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

class WebsiteSchema(BaseModel):
    title: str = Field(description="descriptive title of the site")
    site_description: str = Field(description="the kind of site this is")
    content_description: str = Field(description="description of the site's content")
    user_analysis: str = Field(description="brief reflection and analysis of what visiting the site may suggest about the user")


class FirecrawlWrapper:
    def __init__(self, api_key):
        self.app = FirecrawlApp(api_key=api_key)
        self.artifacts_dir = Path("personality/artifacts")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def scrape_url_with_schema(self, url):
        """
        Scrape a URL with the custom schema and save as markdown.
        
        :param url: The URL to scrape
        :return: Extracted data according to the schema
        """
        data = self.app.scrape_url(url, {
            'formats': ['extract', 'markdown'],
            'extract': {
                'schema': WebsiteSchema.model_json_schema()
            }
        })
        
        extracted_data = data["extract"]
        markdown_content = data.get("markdown", "")
        
        # Create a markdown file
        file_name = url.split("//")[-1].replace("/", "_") + ".md"
        file_path = self.artifacts_dir / file_name
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {extracted_data['title']}\n\n")
            f.write(f"## Site Description\n{extracted_data['site_description']}\n\n")
            f.write(f"## Content Description\n{extracted_data['content_description']}\n\n")
            # f.write(f"## Site Content\n{extracted_data['site_content']}\n\n")
            f.write(f"## User Analysis\n{extracted_data['user_analysis']}\n\n")
            # f.write(f"## Metadata\n```json\n{json.dumps(extracted_data['metadata'], indent=2)}\n```\n\n")
            f.write("## Full Content\n")
            f.write(markdown_content)
        
        print(f"Saved markdown file: {file_path}")
        return extracted_data

    def scrape_multiple_urls(self, urls):
        """
        Scrape multiple URLs and save as markdown files.
        
        :param urls: List of URLs to scrape
        :return: List of extracted data for each URL
        """
        results = []
        for url in urls:
            print(f"Scraping: {url}")
            result = self.scrape_url_with_schema(url)
            results.append(result)
        return results

# Example usage:
if __name__ == "__main__":
    firecrawl = FirecrawlWrapper(api_key=FIRECRAWL_API_KEY)
    
    urls_to_scrape = [
        "https://www.lesswrong.com/s/NHXY86jBahi968uW4/p/aMHq4mA2PHSM2TMoH",
        "https://shoshin.blog/",
        "https://agency42.co/about.html"
    ]
    
    print('Scraping websites and saving as markdown')
    print('*' * 100)
    extracted_data = firecrawl.scrape_multiple_urls(urls_to_scrape)
    print("Scraping completed. Check the personality/artifacts directory for the markdown files.")
    print('*' * 100)
