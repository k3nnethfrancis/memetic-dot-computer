import asyncio
import logging
import json
from typing import List, AsyncGenerator
from llama_index.llms.ollama import Ollama
from cognition.models.chat_models import ChatMessage
from cognition.llms.base_llm import BaseLLM
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaModel(BaseLLM):
    def __init__(self, model: str = "llama3.1", request_timeout: float = 120.0):
        self.model = model
        self.request_timeout = request_timeout
        self.llm = Ollama(model=model, request_timeout=request_timeout)
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Use a default tokenizer

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        return len(self.tokenizer.encode(text))

    def truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to a specified maximum number of tokens."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[:max_tokens])

    async def generate(self, messages: List[ChatMessage]) -> str:
        """Generate a response to a list of chat messages."""
        logger.info(f"Generating response for messages: {messages}")
        try:
            serialized_messages = self._serialize_messages(messages)
            response = await self.llm.acomplete(serialized_messages)
            logger.info(f"Generated response: {response}")
            return str(response)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    async def stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        """Stream a response to a list of chat messages."""
        logger.info(f"Streaming response for messages: {messages}")
        try:
            serialized_messages = self._serialize_messages(messages)
            stream_response = await self.llm.astream_complete(serialized_messages)
            async for token in self._async_iterate(stream_response):
                yield token.delta
        except Exception as e:
            logger.error(f"Error streaming response: {str(e)}")
            raise

    async def _async_iterate(self, async_iterable):
        while True:
            try:
                yield await async_iterable.__anext__()
            except StopAsyncIteration:
                break

    async def function_call(self, function_name: str, function_args: dict) -> str:
        """Implement function calling if supported by Ollama, otherwise raise NotImplementedError."""
        raise NotImplementedError("Function calling is not implemented for Ollama models.")

    async def generate_json(self, prompt: str) -> dict:
        """Generate a JSON response to a prompt."""
        logger.info(f"Generating JSON response for prompt: {prompt}")
        try:
            json_llm = Ollama(model=self.model, request_timeout=self.request_timeout, json_mode=True)
            response = await json_llm.acomplete(prompt)
            json_response = json.loads(str(response))
            logger.info(f"Generated JSON response: {json_response}")
            return json_response
        except Exception as e:
            logger.error(f"Error generating JSON response: {str(e)}")
            raise

    def _serialize_messages(self, messages: List[ChatMessage]) -> str:
        """Serialize ChatMessage objects into a string format."""
        return "\n".join([f"{msg.role}: {msg.content}" for msg in messages])

# Example usage
async def main():
    ollama = OllamaModel(model="llama3.1")

    messages = [
        ChatMessage(role="system", content="You are a pirate with a colorful personality"),
        ChatMessage(role="user", content="What is your name?"),
    ]

    # Generate
    print("Generate:")
    response = await ollama.generate(messages)
    print(response)

    # Stream
    print("\nStream:")
    async for token in ollama.stream(messages):
        print(token, end="", flush=True)
    print()

    # Generate JSON
    print("\nGenerate JSON:")
    json_response = await ollama.generate_json("Who is Paul Graham? Output as a structured JSON object.")
    print(json.dumps(json_response, indent=2))

if __name__ == "__main__":
    asyncio.run(main())