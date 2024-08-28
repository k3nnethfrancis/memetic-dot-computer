# 

import os
import json
import argparse
from typing import List, Dict
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BingSearchAPI:
    def __init__(self):
        # Initialize with API key and endpoint from environment variables
        self.subscription_key = os.environ['BING_SUBSCRIPTION_KEY']
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"

    def raw_search(self, query: str, count: int = 10) -> Dict:
        """
        Perform a raw search without any parsing.
        This function can be used in the future if needed.
        """
        headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
        params = {"q": query, "count": count, "textDecorations": True, "textFormat": "HTML"}
        
        response = requests.get(self.endpoint, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def search(self, query: str, count: int = 10) -> List[Dict]:
        """
        Perform a search and return parsed results.
        """
        raw_results = self.raw_search(query, count)
        parsed_results = []

        if 'webPages' in raw_results and 'value' in raw_results['webPages']:
            for result in raw_results['webPages']['value']:
                parsed_result = {
                    'title': result.get('name', ''),
                    'content': result.get('snippet', ''),
                    'url': result.get('url', ''),
                    'date_published': result.get('datePublished', ''),
                    'date_last_crawled': result.get('dateLastCrawled', '')
                }
                parsed_results.append(parsed_result)

        return parsed_results

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Perform a Bing search")
    parser.add_argument("query", help="The search query")
    parser.add_argument("--count", type=int, default=10, help="Number of results to return")
    args = parser.parse_args()

    # Initialize the BingSearchAPI
    bing_search = BingSearchAPI()

    # Perform the search
    results = bing_search.search(args.query, args.count)

    # Print the results
    for result in results:
        print(f"Title: {result['title']}")
        print(f"Content: {result['content']}")
        print(f"URL: {result['url']}")
        print(f"Date Published: {result['date_published']}")
        print(f"Date Last Crawled: {result['date_last_crawled']}")
        print("---")

if __name__ == "__main__":
    main()