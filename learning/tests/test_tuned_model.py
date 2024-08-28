import time
from cognition.llms.huggingface import LocalModel
from cognition.models.chat_models import ChatMessage
from learning.finetune import OUTPUT_DIR

llm = LocalModel(model_path=OUTPUT_DIR)


base_model_system_prompt = """
You are a large language model developed by Kenneth Francis to serve as his digital twin.
You will respond to user prompts in a conversational manner. Your attitude is friendly and fun to talk to.
You are highly intelligent but a bit snarky.
Always stay in character and respond appropriately to the user's questions.
"""

messages = [
    ChatMessage(role="system", content=base_model_system_prompt),
    ChatMessage(role="user", content="How are you?"),
]

def test_generate(messages):
    start_time = time.time()
    response = llm.generate(messages)
    end_time = time.time()
    print(f"Response time: {end_time - start_time:.2f} seconds")
    print("Assistant:", response)
    print()
    return response

# Test 1
response = test_generate(messages)
messages.append(ChatMessage(role="assistant", content=response))

# Test 2
messages.append(ChatMessage(role="user", content="What's your purpose?"))
response = test_generate(messages)
messages.append(ChatMessage(role="assistant", content=response))

# Test 3
messages.append(ChatMessage(role="user", content="Tell me a joke."))
response = test_generate(messages)
messages.append(ChatMessage(role="assistant", content=response))

# Test 4
messages.append(ChatMessage(role="user", content="Tell me how you would govern mars."))
response = test_generate(messages)
messages.append(ChatMessage(role="assistant", content=response))

# Test 5
messages.append(ChatMessage(role="user", content="What is your philosophy on life?"))
response = test_generate(messages)
messages.append(ChatMessage(role="assistant", content=response))


# Display full conversation
print("Full conversation:")
print('-' * 20)
for msg in messages:
    print(f"{msg.role}: {msg.content}")
    print()
print('-' * 20)