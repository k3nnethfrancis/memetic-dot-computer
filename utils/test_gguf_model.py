from llama_cpp import Llama

model_path = "/Users/kenneth/Desktop/lab/memetic.computer/weights/gguf/Meta-Llama-3.1-8B-Instruct_finetune.gguf"

llm = Llama(
    model_path=model_path,
    # n_gpu_layers=-1, # Uncomment to use GPU acceleration
    # seed=1337, # Uncomment to set a specific seed
    # n_ctx=2048, # Uncomment to increase the context window
)

output = llm(
    "Q: Name the planets in the solar system? A: ", # Prompt
    max_tokens=32, # Generate up to 32 tokens
    stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
    echo=True # Echo the prompt back in the output
)

print(output)