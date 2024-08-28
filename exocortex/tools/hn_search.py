import os
import json
import argparse
from typing import List, Dict
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class HackerNewsAPI:
    def __init__(self):
        # Initialize with the Algolia API endpoint for Hacker News
        self.endpoint = "http://hn.algolia.com/api/v1/search"

    def raw_search(self, query: str, count: int = 10) -> Dict:
        """
        Perform a raw search without any parsing.
        This function can be used in the future if needed.
        """
        params = {"query": query, "hitsPerPage": count}
        
        response = requests.get(self.endpoint, params=params)
        response.raise_for_status()
        return response.json()

    def search(self, query: str, count: int = 10) -> List[Dict]:
        """
        Perform a search and return parsed results.
        """
        raw_results = self.raw_search(query, count)
        parsed_results = []

        if 'hits' in raw_results:
            for result in raw_results['hits']:
                parsed_result = {
                    'title': result.get('title', ''),
                    'content': result.get('story_text', ''),
                    'url': result.get('url', ''),
                    'author': result.get('author', ''),
                    'points': result.get('points', 0),
                    'num_comments': result.get('num_comments', 0),
                    'created_at': result.get('created_at', ''),
                    'hn_url': f"https://news.ycombinator.com/item?id={result.get('objectID', '')}"
                }
                parsed_results.append(parsed_result)

        return parsed_results

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Perform a Hacker News search")
    parser.add_argument("query", help="The search query")
    parser.add_argument("--count", type=int, default=10, help="Number of results to return")
    args = parser.parse_args()

    # Initialize the HackerNewsAPI
    hn_search = HackerNewsAPI()

    # Perform the search
    results = hn_search.search(args.query, args.count)

    # Print the results
    for result in results:
        print(f"Title: {result['title']}")
        print(f"Content: {result['content']}")
        print(f"URL: {result['url']}")
        print(f"Author: {result['author']}")
        print(f"Points: {result['points']}")
        print(f"Number of Comments: {result['num_comments']}")
        print(f"Created At: {result['created_at']}")
        print(f"Hacker News URL: {result['hn_url']}")
        print("---")

if __name__ == "__main__":
    main()