import os
import json
import argparse
from typing import List, Dict
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RedditAPI:
    def __init__(self):
        # Initialize with Reddit API credentials
        self.client_id = os.environ['REDDIT_CLIENT_ID']
        self.client_secret = os.environ['REDDIT_CLIENT_SECRET']
        self.user_agent = os.environ['REDDIT_USER_AGENT']
        self.auth_endpoint = "https://www.reddit.com/api/v1/access_token"
        self.search_endpoint = "https://oauth.reddit.com/search"
        self.access_token = self._get_access_token()

    def _get_access_token(self) -> str:
        """
        Get an access token from Reddit API.
        """
        auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
        data = {"grant_type": "client_credentials"}
        headers = {"User-Agent": self.user_agent}
        response = requests.post(self.auth_endpoint, auth=auth, data=data, headers=headers)
        response.raise_for_status()
        return response.json()['access_token']

    def raw_search(self, query: str, count: int = 10) -> Dict:
        """
        Perform a raw search without any parsing.
        This function can be used in the future if needed.
        """
        headers = {
            "Authorization": f"bearer {self.access_token}",
            "User-Agent": self.user_agent
        }
        params = {
            "q": query,
            "limit": count,
            "sort": "relevance",
            "type": "link"  # This ensures we're searching for posts, not subreddits
        }
        
        response = requests.get(self.search_endpoint, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def search(self, query: str, count: int = 10) -> List[Dict]:
        """
        Perform a search and return parsed results.
        """
        raw_results = self.raw_search(query, count)
        parsed_results = []

        if 'data' in raw_results and 'children' in raw_results['data']:
            for post in raw_results['data']['children']:
                data = post['data']
                parsed_result = {
                    'title': data.get('title', ''),
                    'content': data.get('selftext', ''),
                    'url': f"https://www.reddit.com{data.get('permalink', '')}",
                    'author': data.get('author', ''),
                    'score': data.get('score', 0),
                    'num_comments': data.get('num_comments', 0),
                    'created_utc': data.get('created_utc', ''),
                    'subreddit': data.get('subreddit', '')
                }
                parsed_results.append(parsed_result)

        return parsed_results

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Perform a Reddit search for posts")
    parser.add_argument("query", help="The search query")
    parser.add_argument("--count", type=int, default=10, help="Number of results to return")
    args = parser.parse_args()

    # Initialize the RedditAPI
    reddit_search = RedditAPI()

    # Perform the search
    results = reddit_search.search(args.query, args.count)

    # Print the results
    for result in results:
        print(f"Title: {result['title']}")
        print(f"Content: {result['content'][:200]}...")  # Truncate long content
        print(f"URL: {result['url']}")
        print(f"Author: {result['author']}")
        print(f"Score: {result['score']}")
        print(f"Number of Comments: {result['num_comments']}")
        print(f"Created At (UTC): {result['created_utc']}")
        print(f"Subreddit: {result['subreddit']}")
        print("---")

if __name__ == "__main__":
    main()