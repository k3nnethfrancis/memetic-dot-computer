import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, AsyncGenerator
from cognition.llms.base_llm import BaseLLM
from cognition.models.chat_models import ChatMessage

class HuggingFaceModel(BaseLLM):
    def __init__(self, model_path: str = "././weights/meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="mps" if torch.backends.mps.is_available() else "auto",
        )
        self.model.eval()
        self.has_chat_template = hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None

    async def generate(self, messages: List[ChatMessage]) -> str:
        if self.has_chat_template:
            return await self._generate_with_chat_template(messages)
        else:
            return await self._generate_without_chat_template(messages)

    async def stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        # Implement streaming logic
        full_response = await self.generate(messages)
        for token in full_response.split():
            yield token + " "

    async def function_call(self, function_name: str, function_args: dict) -> str:
        # Implement function calling logic here (if applicable)
        raise NotImplementedError("Function calling is not implemented for this model.")

    async def _generate_with_chat_template(self, messages: List[ChatMessage]) -> str:
        serialized_messages = self._serialize_messages(messages)
        inputs = self.tokenizer.apply_chat_template(
            serialized_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=True)
        full_response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract only the assistant's response
        assistant_response = full_response.split("assistant")[-1].strip()
        return assistant_response

    async def _generate_without_chat_template(self, messages: List[ChatMessage]) -> str:
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        prompt += "\nassistant:"  # Add a prompt for the model to continue

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        assistant_response = generated_text.split("assistant:")[-1].strip()
        return assistant_response

    def _serialize_messages(self, messages: List[ChatMessage]) -> List[dict]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

# Example usage
async def main():
    model = HuggingFaceModel()
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is the capital of France?")
    ]
    
    print("Generating response...")
    response = await model.generate(messages)
    print("Generated response:", response)

    print("\nStreaming response:")
    async for token in model.stream(messages):
        print(token, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())