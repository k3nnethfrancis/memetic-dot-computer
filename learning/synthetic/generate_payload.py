"""
Generate Training Payload

This script processes a corpus of markdown files containing blog posts and generates
training data for a language model. It uses various strategies to create diverse
training examples, including Q&A pairs, blog post generation, summarization,
interesting questions and answers, style analysis, point verification, and
passage completion.

The script uses the Ollama model for text generation tasks and implements
error handling, input/output validation, and logging to ensure robustness.

Usage:
    python3 -m learning.generate_payload

The script will run tests before processing the corpus. If all tests pass,
it will proceed to generate training data and save it in multiple formats:
1. Single JSON file
2. JSONL file
3. Train, test, and validation JSONL files
"""

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import json
import markdown
from bs4 import BeautifulSoup
import random
import re
import concurrent.futures
from tqdm import tqdm
import argparse
import unittest
from unittest.mock import Mock, patch

from cognition.llms.ollama import OllamaModel
from cognition.models.chat_models import ChatMessage

# Set up logging
def setup_logging():
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # File handler for general logs
    file_handler = RotatingFileHandler(
        log_dir / f"generate_payload_{timestamp}.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)

    # File handler for error logs
    error_file_handler = RotatingFileHandler(
        log_dir / f"generate_payload_errors_{timestamp}.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_file_handler.setLevel(logging.ERROR)

    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    error_file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_file_handler)

    return logger

# Initialize logger
logger = setup_logging()

# Add this constant at the top of the file
MAX_TOKENS = None  # No limit on tokens


def read_markdown_file(file_path: Path) -> str:
    """Read and parse a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    html = markdown.markdown(content)
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()


def validate_input(content: str, title: str) -> bool:
    """Validate input content and title."""
    if not content or not title:
        return False
    if len(content) < 50:  # Arbitrary minimum length
        return False
    return True


def validate_completion(completion: str) -> bool:
    """Validate that a completion is not empty and meets minimum length requirements."""
    return completion and isinstance(completion, str) and len(completion.strip()) >= 20  # Arbitrary minimum length


def generate_multiple_qa_pairs(
    ollama: OllamaModel,
    content: str,
    title: str,
    num_pairs: int = 3
) -> List[Dict[str, str]]:
    """Generate multiple Q&A pairs based on the blog post content."""
    logger.info(f"Generating multiple Q&A pairs for '{title}'")
    prompt = f"""Generate {num_pairs} diverse question and answer pairs based on this blog post about '{title}'. 
    Output as JSON with the following structure:
    {{
        "qa_pairs": [
            {{"question": "Q1", "answer": "A1"}},
            {{"question": "Q2", "answer": "A2"}},
            {{"question": "Q3", "answer": "A3"}}
        ]
    }}
    Blog post content:
    {content}"""
    
    try:
        response = ollama.generate_json(prompt)
        qa_pairs = []
        for pair in response.get("qa_pairs", []):
            if validate_completion(pair.get("answer")):
                qa_pairs.append({"prompt": pair["question"], "completion": pair["answer"]})
            else:
                logger.warning(f"Skipping QA pair due to invalid completion: {pair.get('question', 'Unknown question')}")
        
        if not qa_pairs:
            logger.warning(f"No valid QA pairs generated for '{title}'")
        
        return qa_pairs
    except Exception as e:
        logger.error(f"Error generating QA pairs for '{title}': {str(e)}")
        return []


def truncate_to_token_limit(ollama: OllamaModel, text: str, max_tokens: int = None) -> str:
    """Truncate text to a specified maximum number of tokens while maintaining coherence."""
    if max_tokens is None or ollama.count_tokens(text) <= max_tokens:
        return text

    sentences = re.split(r'(?<=[.!?])\s+', text)
    truncated_text = ""
    
    for sentence in sentences:
        if ollama.count_tokens(truncated_text + sentence) > max_tokens:
            break
        truncated_text += sentence + " "
    
    return truncated_text.strip()


def chunk_text(ollama: OllamaModel, text: str, max_tokens: int = None) -> List[str]:
    """Break down long text into smaller, coherent chunks based on token count."""
    if max_tokens is None:
        return [text]  # Return the entire text as a single chunk

    chunks = []
    current_chunk = ""
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        if ollama.count_tokens(current_chunk + sentence) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
            else:
                chunks.append(ollama.truncate_to_token_limit(sentence, max_tokens))
        else:
            current_chunk += sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def generate_blog_prompt(ollama: OllamaModel, title: str, content: str) -> Dict[str, str]:
    """Generate a prompt for blog post generation."""
    logger.info(f"Generating blog prompt for '{title}'")
    messages = [
        ChatMessage(role="system", content="Generate a brief description for a blog post. Respond with only the description, without any additional text."),
        ChatMessage(role="user", content=f"Generate a description for a blog post titled '{title}'.")
    ]
    description = ollama.generate(messages)
    prompt = f"Generate a blog post about '{title}'. Description: {description}"
    completion = content  # No truncation
    return {"prompt": prompt, "completion": completion}


def summarize_and_expand(ollama: OllamaModel, content: str, title: str) -> Dict[str, str]:
    """Summarize the blog post and create an expansion prompt."""
    logger.info(f"Summarizing and creating expansion prompt for '{title}'")
    truncated_content = content  # No truncation
    messages = [
        ChatMessage(role="system", content="Summarize the given blog post in a single paragraph. Respond with only the summary, without any additional text."),
        ChatMessage(role="user", content=f"Summarize this blog post about '{title}':\n\n{truncated_content}")
    ]
    summary = ollama.generate(messages)
    return {"prompt": f"Expand on this summary of '{title}': {summary}", "completion": content}


def generate_interesting_questions(
    ollama: OllamaModel,
    content: str,
    title: str,
    num_questions: int = 3
) -> List[str]:
    """Generate multiple interesting questions based on the blog post content."""
    logger.info(f"Generating interesting questions for '{title}'")
    prompt = f"""Generate {num_questions} interesting and thought-provoking questions based on this blog post about '{title}'. 
    Output as JSON with the following structure:
    {{
        "questions": [
            "Question 1",
            "Question 2",
            "Question 3"
        ]
    }}
    Blog post content:
    {content}"""
    
    response = ollama.generate_json(prompt)
    return response.get("questions", [])


def answer_interesting_question(
    ollama: OllamaModel,
    question: str,
    content: str,
    title: str
) -> Dict[str, str]:
    """Generate an answer to the interesting question based on the blog post content."""
    logger.info(f"Answering interesting question for '{title}'")
    truncated_content = content  # No truncation
    messages = [
        ChatMessage(role="system", content="Answer the given question based on the context provided from the blog post. Respond with only the answer, without any additional text."),
        ChatMessage(role="user", content=f"Question: {question}\n\nContext from the blog post '{title}':\n\n{truncated_content}")
    ]
    answer = ollama.generate(messages)
    return {"prompt": question, "completion": answer}


def analyze_writing_style(ollama: OllamaModel, content: str, title: str) -> str:
    """Analyze and describe the writing style of the blog post."""
    logger.info(f"Analyzing writing style for '{title}'")
    truncated_content = content  # No truncation
    messages = [
        ChatMessage(role="system", content="Analyze the writing style of the given blog post. Describe the style concisely without any additional text."),
        ChatMessage(role="user", content=f"Analyze the writing style of this blog post about '{title}':\n\n{truncated_content}")
    ]
    return ollama.generate(messages)


def extract_key_points(
    ollama: OllamaModel,
    content: str,
    title: str,
    num_points: int = 5
) -> List[Dict[str, str]]:
    """Extract key points from the blog post."""
    logger.info(f"Extracting key points from '{title}'")
    truncated_content = content  # No truncation
    prompt = f"""Extract {num_points} key points from this blog post about '{title}' and provide a brief explanation for each. 
    Output as JSON with the following structure:
    {{
        "key_points": [
            {{"point": "Point 1", "explanation": "Brief explanation 1"}},
            {{"point": "Point 2", "explanation": "Brief explanation 2"}},
            // ... more points ...
        ]
    }}
    Blog post content:
    {truncated_content}"""
    
    response = ollama.generate_json(prompt)
    return response.get("key_points", [])


def generate_passage_completions(ollama: OllamaModel, content: str) -> List[Dict[str, str]]:
    """Generate passage completion examples based on content length."""
    chunks = chunk_text(ollama, content)  # No token limit
    completions = []
    
    for i in range(len(chunks) - 1):
        completions.append({
            "prompt": f"Complete the passage: {chunks[i]}",
            "completion": chunks[i+1]
        })
    
    return completions


def generate_summary(ollama: OllamaModel, content: str, title: str) -> Dict[str, str]:
    """Generate a summary of the blog post."""
    logger.info(f"Generating summary for '{title}'")
    prompt = f"Summarize this blog post about '{title}' in 3-5 sentences:"
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant that summarizes blog posts."),
        ChatMessage(role="user", content=f"{prompt}\n\n{content}")
    ]
    try:
        summary = ollama.generate(messages)
        return {"prompt": prompt, "completion": summary}
    except Exception as e:
        logger.error(f"Error generating summary for '{title}': {str(e)}")
        return {"prompt": prompt, "completion": "Error generating summary."}


def generate_style_transfer(ollama: OllamaModel, content: str, title: str) -> Dict[str, str]:
    """Generate a style transfer task."""
    styles = ["formal", "casual", "poetic", "technical", "humorous"]
    style = random.choice(styles)
    prompt = f"Rewrite this blog post about '{title}' in a {style} style:"
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant that can rewrite text in different styles."),
        ChatMessage(role="user", content=f"{prompt}\n\n{content}")
    ]
    rewrite = ollama.generate(messages)
    return {"prompt": prompt, "completion": rewrite}


def generate_domain_specific_question(ollama: OllamaModel, content: str, title: str) -> Dict[str, str]:
    """Generate a domain-specific question based on the blog post content."""
    question_prompt = f"Generate a domain-specific question that requires deep understanding of the topic '{title}', based on this content:"
    question_messages = [
        ChatMessage(role="system", content="You are a helpful assistant that generates domain-specific questions."),
        ChatMessage(role="user", content=f"{question_prompt}\n\n{content}")
    ]
    question = ollama.generate(question_messages)
    
    answer_prompt = f"Answer the following question based on the blog post about '{title}':\n{question}"
    answer_messages = [
        ChatMessage(role="system", content="You are a helpful assistant that answers questions based on given content."),
        ChatMessage(role="user", content=f"{answer_prompt}\n\nContent:\n{content}")
    ]
    answer = ollama.generate(answer_messages)
    
    return {"prompt": question, "completion": answer}


def generate_logical_reasoning_task(ollama: OllamaModel, content: str, title: str) -> Dict[str, str]:
    """Generate a logical reasoning task based on the blog post content."""
    question_prompt = f"Create a logical reasoning question that requires analysis of the information in the blog post about '{title}':"
    question_messages = [
        ChatMessage(role="system", content="You are a helpful assistant that creates logical reasoning questions."),
        ChatMessage(role="user", content=f"{question_prompt}\n\n{content}")
    ]
    question = ollama.generate(question_messages)
    
    answer_prompt = f"Provide a step-by-step solution to this logical reasoning question:\n{question}\n\nBased on the blog post about '{title}':"
    answer_messages = [
        ChatMessage(role="system", content="You are a helpful assistant that provides step-by-step solutions to logical reasoning questions."),
        ChatMessage(role="user", content=f"{answer_prompt}\n\n{content}")
    ]
    answer = ollama.generate(answer_messages)
    
    return {"prompt": question, "completion": answer}


def process_file(file_path: Path, ollama: OllamaModel) -> List[Dict[str, str]]:
    """Process a single markdown file."""
    content = read_markdown_file(file_path)
    title = file_path.stem.replace('-', ' ').title()
    file_training_data = []

    if not validate_input(content, title):
        logger.warning(f"Skipping invalid input file: {file_path}")
        return file_training_data

    try:
        file_training_data.extend(generate_multiple_qa_pairs(ollama, content, title))
        blog_prompt = generate_blog_prompt(ollama, title, content)
        for chunk in chunk_text(ollama, blog_prompt["completion"]):
            file_training_data.append({
                "prompt": blog_prompt["prompt"],
                "completion": chunk
            })
        file_training_data.append(summarize_and_expand(ollama, content, title))
        
        questions = generate_interesting_questions(ollama, content, title)
        for question in questions:
            file_training_data.append(answer_interesting_question(ollama, question, content, title))
        
        style_analysis = analyze_writing_style(ollama, content, title)
        file_training_data.append({
            "prompt": f"Write a blog post in the following style: {style_analysis}",
            "completion": content  # No truncation
        })
        
        key_points = extract_key_points(ollama, content, title)
        for point in key_points:
            file_training_data.append({
                "prompt": f"Explain this point from the blog post '{title}': {point['point']}",
                "completion": point['explanation']
            })
        
        file_training_data.extend(generate_passage_completions(ollama, content))
        summary_data = generate_summary(ollama, content, title)
        if summary_data["completion"] != "Error generating summary.":
            file_training_data.append(summary_data)
        file_training_data.append(generate_style_transfer(ollama, content, title))
        file_training_data.append(generate_domain_specific_question(ollama, content, title))
        file_training_data.append(generate_logical_reasoning_task(ollama, content, title))

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)

    return file_training_data


def process_corpus(corpus_path: str, ollama: OllamaModel, max_workers: int = 4) -> List[Dict[str, str]]:
    """Process all markdown files in the corpus directory using parallel processing."""
    corpus_path = Path(corpus_path)
    all_files = list(corpus_path.glob('*.md'))
    training_data = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, file, ollama): file for file in all_files}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(all_files), desc="Processing files"):
            file = future_to_file[future]
            try:
                file_training_data = future.result()
                training_data.extend(file_training_data)
            except Exception as e:
                logger.error(f"Error processing file {file}: {str(e)}")

    return training_data


def calculate_splits(train_split: float, test_split: float, val_split: float) -> Dict[str, float]:
    """
    Calculate and validate the dataset splits.

    Args:
        train_split (float): Proportion of data for training.
        test_split (float): Proportion of data for testing.
        val_split (float): Proportion of data for validation.

    Returns:
        Dict[str, float]: Validated splits.

    Raises:
        ValueError: If splits don't sum to 1 or are invalid.
    """
    total = train_split + test_split + val_split
    if not (0.99 <= total <= 1.01):  # Allow for small floating-point errors
        raise ValueError(f"Splits must sum to 1. Current sum: {total}")
    
    if any(split < 0 for split in [train_split, test_split, val_split]):
        raise ValueError("Splits must be non-negative")

    return {
        "train": train_split,
        "test": test_split,
        "val": val_split
    }


def save_training_data(training_data: List[Dict[str, str]], output_dir: str, train_split: float = 0.8, test_split: float = 0.1, val_split: float = 0.1):
    """
    Save the generated training data in multiple formats:
    1. Single JSON file
    2. JSONL file
    3. Train, test, and validation JSONL files (based on provided splits)

    Args:
        training_data (List[Dict[str, str]]): The generated training data.
        output_dir (str): Directory to save output files.
        train_split (float): Proportion of data for training (default: 0.8).
        test_split (float): Proportion of data for testing (default: 0.1).
        val_split (float): Proportion of data for validation (default: 0.1).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save as single JSON file
    json_output_file = output_dir / f"training_data_{timestamp}.json"
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Training data saved to {json_output_file}")

    # 2. Save as JSONL file
    jsonl_output_file = output_dir / f"training_data_{timestamp}.jsonl"
    with open(jsonl_output_file, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Training data saved to {jsonl_output_file}")

    # 3. Save train, test, and validation JSONL files
    splits = calculate_splits(train_split, test_split, val_split)
    random.shuffle(training_data)
    total = len(training_data)

    start_idx = 0
    for split_name, split_ratio in splits.items():
        if split_ratio > 0:
            end_idx = start_idx + int(split_ratio * total)
            split_data = training_data[start_idx:end_idx]
            split_file = output_dir / f"{split_name}_data_{timestamp}.jsonl"
            with open(split_file, 'w', encoding='utf-8') as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"{split_name.capitalize()} data saved to {split_file}")
            start_idx = end_idx

    logger.info(f"Generated {len(training_data)} examples with splits: {splits}")


class TestGeneratePayload(unittest.TestCase):
    def test_validate_completion(self):
        self.assertTrue(validate_completion("This is a valid completion with more than 20 characters."))
        self.assertFalse(validate_completion("Short answer"))
        self.assertFalse(validate_completion(""))
        self.assertFalse(validate_completion(None))

    @patch('cognition.llms.ollama.OllamaModel')
    def test_generate_multiple_qa_pairs(self, mock_ollama):
        mock_ollama.generate_json.return_value = {
            "qa_pairs": [
                {"question": "Q1", "answer": "This is a valid answer for Q1 with more than 20 characters."},
                {"question": "Q2", "answer": "Short"},
                {"question": "Q3", "answer": "This is another valid answer for Q3 with sufficient length."}
            ]
        }
        
        result = generate_multiple_qa_pairs(mock_ollama, "Sample content", "Sample Title")
        self.assertEqual(len(result), 2)  # Only two valid pairs should be returned
        self.assertEqual(result[0]["prompt"], "Q1")
        self.assertEqual(result[1]["prompt"], "Q3")

    @patch('cognition.llms.ollama.OllamaModel')
    def test_generate_multiple_qa_pairs_error(self, mock_ollama):
        mock_ollama.generate_json.side_effect = Exception("API Error")
        
        result = generate_multiple_qa_pairs(mock_ollama, "Sample content", "Sample Title")
        self.assertEqual(result, [])  # Should return an empty list on error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training payload from markdown files.")
    parser.add_argument("--corpus_path", default="exocortex/corpus/posts", help="Path to the corpus directory")
    parser.add_argument("--output_dir", default="learning/payloads", help="Directory to save output files")
    parser.add_argument("--model", default="llama3.1", help="Ollama model to use")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker threads")
    parser.add_argument("--train_split", type=float, default=0.8, help="Proportion of data for training")
    parser.add_argument("--test_split", type=float, default=0.1, help="Proportion of data for testing")
    parser.add_argument("--val_split", type=float, default=0.1, help="Proportion of data for validation")
    args = parser.parse_args()

    # Run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('learning', pattern='test_*.py')
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)

    if test_result.wasSuccessful():
        logger.info("All tests passed. Proceeding with corpus processing.")
        
        ollama = OllamaModel(model=args.model)
        training_data = process_corpus(args.corpus_path, ollama, args.max_workers)
        save_training_data(training_data, args.output_dir, args.train_split, args.test_split, args.val_split)
        logger.info(f"Generated {len(training_data)} training examples.")
    else:
        logger.error("Tests failed. Please fix the issues before running the script.")