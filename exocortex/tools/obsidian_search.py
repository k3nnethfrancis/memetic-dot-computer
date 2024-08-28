"""
Module for searching Obsidian notes using BM25 algorithm.
"""

import os
import argparse
from typing import List, Tuple
import bm25s
import markdown
from dotenv import load_dotenv
import Stemmer
import numpy as np

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(script_dir)), '.env')
# Load the .env file
load_dotenv(env_path)


def read_markdown_files(directory: str) -> List[str]:
    """
    Read all markdown files from the given directory.

    Args:
        directory (str): Path to the directory containing markdown files.

    Returns:
        List[str]: List of document contents.
    """
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    text = markdown.markdown(content)
                    text = text.replace('<p>', '').replace('</p>', '\n')
                    documents.append(text)
    return documents


class ObsidianSearch:
    """Class for searching Obsidian notes using BM25 algorithm."""

    def __init__(self, obsidian_path: str):
        """
        Initialize the ObsidianSearch object.

        Args:
            obsidian_path (str): Path to the Obsidian vault.
        """
        self.docs = read_markdown_files(obsidian_path)
        self.stemmer = Stemmer.Stemmer("english")
        self.corpus_tokens = bm25s.tokenize(self.docs, stopwords="en", stemmer=self.stemmer)
        self.retriever = bm25s.BM25()
        self.retriever.index(self.corpus_tokens)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search the Obsidian notes.

        Args:
            query (str): Search query.
            top_k (int): Number of top results to return.

        Returns:
            List[Tuple[str, float]]: List of (document content, score) tuples.
        """
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        results, scores = self.retriever.retrieve(query_tokens, corpus=self.docs, k=top_k)
        
        search_results = []
        for i in range(results.shape[1]):
            doc_content, score = results[0, i], scores[0, i]
            if isinstance(doc_content, np.str_):
                doc_content = str(doc_content)
            search_results.append((doc_content, score))
        
        return search_results


def main():
    """Main function to run the Obsidian search from command line."""
    parser = argparse.ArgumentParser(description="Search Obsidian notes using BM25.")
    parser.add_argument("query", nargs='+', help="Search query")
    parser.add_argument("-k", "--top_k", type=int, default=5, help="Number of top results to return")
    args = parser.parse_args()

    query = ' '.join(args.query)  # Join all query words into a single string

    obsidian_path = os.getenv('OBSIDIAN_PATH')
    if not obsidian_path:
        raise ValueError("OBSIDIAN_PATH not set in .env file")

    searcher = ObsidianSearch(obsidian_path)
    results = searcher.search(query, args.top_k)

    for i, (content, score) in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Score: {score:.4f}")
        # print(f"Content: {content[:200]}...")  # Print first 200 characters
        print(f"Content: {content}...")  # Print first 200 characters
        print()


if __name__ == "__main__":
    main()