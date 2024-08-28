"""
Module for searching creator data using BM25 algorithm.
"""

import os
import argparse
from typing import List, Tuple
import bm25s
import json
from dotenv import load_dotenv
import Stemmer
import numpy as np
import itertools
import math
from collections import Counter, defaultdict

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(script_dir)), '.env')
# Load the .env file
load_dotenv(env_path)


def read_json_file(file_path: str) -> List[dict]:
    """
    Read creator data from the given JSON file.

    Args:
        file_path (str): Path to the JSON file containing creator data.

    Returns:
        List[dict]: List of creator data dictionaries.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return list(data['docstore/data'].values())

import itertools

class CreatorSearch:
    """Class for searching creators using BM25 algorithm."""

    def __init__(self, json_path: str):
        """
        Initialize the CreatorSearch object.

        Args:
            json_path (str): Path to the JSON file containing creator data.
        """
        self.creators = read_json_file(json_path)
        self.stemmer = Stemmer.Stemmer("english")
        self.corpus_tokens, self.doc_freqs, self.doc_lengths = self._prepare_corpus()
        self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        self.k1 = 1.5
        self.b = 0.75

    def _prepare_corpus(self):
        """Prepare the corpus for indexing."""
        corpus_tokens = []
        doc_freqs = defaultdict(int)
        doc_lengths = {}
        
        for i, creator in enumerate(self.creators):
            text = f"{creator['__data__']['id_']} {creator['__data__']['text']}"
            tokens = bm25s.tokenize(text, stopwords="en", stemmer=self.stemmer)
            # Flatten the tokens and ensure they are strings
            flat_tokens = list(itertools.chain.from_iterable(tokens))
            flat_tokens = [str(token) for token in flat_tokens]  # Ensure all tokens are strings
            corpus_tokens.append(flat_tokens)
            
            doc_lengths[i] = len(flat_tokens)
            term_freq = Counter(flat_tokens)
            for term in term_freq:
                doc_freqs[term] += 1
        
        return corpus_tokens, doc_freqs, doc_lengths

    def _score_document(self, query_tokens, doc_tokens, doc_id):
        score = 0
        doc_length = self.doc_lengths[doc_id]
        for term in query_tokens:
            if term in doc_tokens:
                tf = doc_tokens.count(term)
                idf = math.log((len(self.creators) - self.doc_freqs[term] + 0.5) / (self.doc_freqs[term] + 0.5) + 1)
                score += idf * ((tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)))
        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[dict, float]]:
        """
        Search for creators.

        Args:
            query (str): Search query.
            top_k (int): Number of top results to return.

        Returns:
            List[Tuple[dict, float]]: List of (creator data, score) tuples.
        """
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer)
        query_tokens = list(itertools.chain.from_iterable(query_tokens))
        query_tokens = [str(token) for token in query_tokens]  # Ensure all tokens are strings
        print(f"Query tokens: {query_tokens}")  # Debug statement
        
        scores = []
        for i, doc_tokens in enumerate(self.corpus_tokens):
            score = self._score_document(query_tokens, doc_tokens, i)
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:top_k]
        
        search_results = []
        for creator_index, score in top_results:
            creator_data = self.creators[creator_index]
            search_results.append((creator_data, score))
        
        return search_results

def main():
    """Main function to run the creator search from command line."""
    parser = argparse.ArgumentParser(description="Search creators using BM25.")
    parser.add_argument("query", nargs='+', help="Search query")
    parser.add_argument("-k", "--top_k", type=int, default=5, help="Number of top results to return")
    args = parser.parse_args()

    query = ' '.join(args.query)  # Join all query words into a single string

    # Update this path to point directly to the JSON file
    json_path = os.path.join('/Users/kenneth/Desktop/lab/memetic.computer/noo', 'test_store.json')

    searcher = CreatorSearch(json_path)
    results = searcher.search(query, args.top_k)

    for i, (creator, score) in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Score: {score:.4f}")
        print(f"Creator: {creator['__data__']['id_']}")
        print(f"Metadata: {creator['__data__']['metadata']}")
        print()

if __name__ == "__main__":
    main()