import os
import json
import argparse
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SubstackSearch:
    def __init__(self):
        self.search_url = "https://substack.com/search/{query}?utm_source=global-search&searching=all_posts"

    def search(self, query: str, count: int = 10) -> List[Dict]:
        url = self.search_url.format(query=query)
        print(f"Searching URL: {url}")
        
        response = requests.get(url)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        print(f"Page title: {soup.title.string if soup.title else 'No title found'}")
        
        posts = soup.find_all('div', class_='_linkRow_214uo_28')
        print(f"Found {len(posts)} posts")
        
        results = []
        for post in posts[:count]:
            link_elem = post.find('a', class_='_linkRowA_214uo_50')
            title_elem = post.find('div', class_='reader2-post-title')
            author_elem = post.find('div', class_='pub-name')
            excerpt_elem = post.find('div', class_='reader2-paragraph')
            
            if link_elem and title_elem:
                result = {
                    'title': title_elem.text.strip(),
                    'author': author_elem.text.strip() if author_elem else 'No author found',
                    'excerpt': excerpt_elem.text.strip() if excerpt_elem else 'No excerpt found',
                    'url': link_elem['href'],
                }
                results.append(result)
                print(f"Found post: {result['title']}")
            else:
                print(f"Skipping a post due to missing elements")
                if link_elem:
                    print(f"Link found: {link_elem['href']}")
                if title_elem:
                    print(f"Title found: {title_elem.text.strip()}")
        
        return results

    def fetch_post_content(self, url: str) -> str:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for the main content container
        content = soup.find('div', class_='body markup')
        
        if not content:
            content = soup.find('div', class_='post-content')  # Try an alternative class
        
        if content:
            # Remove any script or style tags
            for script in content(["script", "style"]):
                script.decompose()
            return content.get_text(separator='\n', strip=True)
        else:
            return "Could not extract content"

def main():
    parser = argparse.ArgumentParser(description="Search Substack posts")
    parser.add_argument("query", help="The search query")
    parser.add_argument("--count", type=int, default=10, help="Number of results to return")
    parser.add_argument("--fetch-content", action="store_true", help="Fetch full content of posts")
    args = parser.parse_args()

    substack_search = SubstackSearch()

    try:
        results = substack_search.search(args.query, args.count)

        if not results:
            print("No results found.")
        else:
            for result in results:
                print(f"Title: {result['title']}")
                print(f"Author: {result['author']}")
                print(f"Excerpt: {result['excerpt']}")
                print(f"URL: {result['url']}")
                
                if args.fetch_content:
                    print("Content:")
                    content = substack_search.fetch_post_content(result['url'])
                    print(content[:500] + "..." if len(content) > 500 else content)
                
                print("---")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()