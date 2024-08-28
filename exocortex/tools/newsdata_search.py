import os  # For accessing environment variables
import argparse  # For parsing command-line arguments
from typing import List, Dict  # For type hinting
import requests  # For making HTTP requests
from dotenv import load_dotenv  # For loading environment variables from .env file

# Load environment variables from .env file
load_dotenv()

def build_query(
        terms: List[str], 
        exact: bool = False, 
        and_op: bool = False, 
        or_op: bool = False, 
        not_terms: List[str] = None) -> str:
    """
    Build a query string based on user input and search options.
    """
    # Check if both AND and OR operators are used simultaneously (which is not allowed)
    if and_op and or_op:
        raise ValueError("Cannot use both AND and OR operators simultaneously.")

    # Build the query based on the search options
    if exact:
        query = f'"{" ".join(terms)}"'  # Wrap terms in quotes for exact match
    elif and_op:
        query = " AND ".join(terms)  # Join terms with AND operator
    elif or_op:
        query = " OR ".join(terms)  # Join terms with OR operator
    else:
        query = " ".join(terms)  # Simple space-separated query

    # Add NOT terms if provided
    if not_terms:
        not_query = " AND ".join(f"NOT {term}" for term in not_terms)
        query = f"({query}) {not_query}"

    return query

def search_news(query: str, num_results: int = 10) -> List[Dict]:
    """
    Search for news articles using the NewsData.io API.
    """
    # API endpoint URL
    url = "https://newsdata.io/api/1/news"
    # Get API key from environment variables
    api_key = os.getenv('NEWSDATA_API_KEY')
    
    # Check if API key is available
    if not api_key:
        raise ValueError("NEWSDATA_API_KEY not found in environment variables")

    # Set up parameters for the API request
    params = {
        'apikey': api_key,
        'q': query,
        'language': 'en',
        'size': min(num_results, 50)  # Ensure we don't exceed the API limit
    }

    # Make the API request
    response = requests.get(url, params=params)
    
    # Process the response
    if response.status_code == 200:
        data = response.json()
        return data.get('results', [])
    else:
        print(f"Error: Unable to fetch news. Status code: {response.status_code}")
        return []

def display_results(articles: List[Dict]):
    """Display the search results."""
    for i, article in enumerate(articles, 1):
        # Print each article's details
        print(f"\n{i}. {article.get('title', 'No title')}")
        print(f"   Source: {article.get('source_id', 'Unknown source')}")
        print(f"   Published: {article.get('pubDate', 'Unknown date')}")
        description = article.get('description', 'No description')
        if description:
            print(f"   Description: {description[:100]}...")
        else:
            print("   Description: No description available")

def main():
    """Main function to run the NewsData search from command line."""
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Search news articles using NewsData.io API.")
    parser.add_argument("terms", nargs='+', help="Search terms")
    parser.add_argument("-k", "--top_k", type=int, default=10, help="Number of top results to return (max 50)")
    parser.add_argument("--exact", action="store_true", help="Search for exact phrase")
    parser.add_argument("--and", action="store_true", dest="and_op", help="Use AND operator between terms")
    parser.add_argument("--or", action="store_true", dest="or_op", help="Use OR operator between terms")
    parser.add_argument("--not", nargs='+', dest="not_terms", help="Terms to exclude from search")
    args = parser.parse_args()

    # Build the query string based on command-line arguments
    query = build_query(args.terms, args.exact, args.and_op, args.or_op, args.not_terms)
    print(f"Searching for: {query}")
    
    # Perform the search
    articles = search_news(query, args.top_k)

    # Display the results
    if articles:
        print(f"Found {len(articles)} articles:")
        display_results(articles)
    else:
        print("No articles found.")

if __name__ == "__main__":
    main()

"""
Helper: How to use different search approaches

1. Basic search:
   python newsdata_search.py apple

2. Exact phrase search:
   python newsdata_search.py "artificial intelligence" --exact

3. Search with AND operator:
   python newsdata_search.py laptop fire --and

4. Search with OR operator:
   python newsdata_search.py fruits vegetables --or

5. Exclude terms:
   python newsdata_search.py apple --not pears

6. Complex search:
   python newsdata_search.py social pizza --or --not pasta

7. Specify number of results:
   python newsdata_search.py technology -k 20

Note: 
- The --and and --or flags cannot be used together.
- The --not flag can be used with any other search type.
- The -k or --top_k flag specifies the number of results (max 50).
- Use quotes for multi-word terms when using --exact, --and, or --or.
"""