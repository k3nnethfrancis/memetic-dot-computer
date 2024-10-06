import asyncio
import os
from typing import List, AsyncGenerator
from openai import AsyncOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from cognition.llms.base_llm import BaseLLM
from cognition.models.chat_models import ChatMessage
from typing import List, AsyncGenerator
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt

import dotenv; dotenv.load_dotenv()


class OpenAIModel(BaseLLM):
    def __init__(self, model: str = 'gpt-3.5-turbo', tools=None):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.tools = tools

    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
    async def generate(self, messages: List[ChatMessage]) -> str:
        json_data = {"model": self.model, "messages": self._format_messages(messages)}
        if self.tools:
            json_data["tools"] = self.tools
            json_data["tool_choice"] = "auto"

        try:
            response = await self.client.chat.completions.create(**json_data)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Unable to generate ChatCompletion response: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
    async def stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        json_data = {"model": self.model, "messages": self._format_messages(messages), "stream": True}
        if self.tools:
            json_data["tools"] = self.tools
            json_data["tool_choice"] = "auto"

        try:
            stream = await self.client.chat.completions.create(**json_data)
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Unable to generate streaming ChatCompletion response: {e}")
            raise

    async def function_call(self, function_name: str, function_args: dict) -> str:
        # Implement function calling logic here
        pass

    def _format_messages(self, messages: List[ChatMessage]) -> List[dict]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

# Example usage
async def main():
    model = OpenAIModel(model="gpt-3.5-turbo")
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is the capital of France?")
    ]
    response = await model.generate(messages)
    print("Generated response:", response)

    print("\nStreaming response:")
    async for token in model.stream(messages):
        print(token, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())

