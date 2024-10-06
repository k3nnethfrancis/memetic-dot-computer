import asyncio
from cognition.llms.ollama import OllamaModel   
from cognition.engines.chat_engine import Conversation, ChatEngine
from cognition.models.chat_models import ChatMessage
from pathlib import Path
import json

system_prompt = """
You are a helpful assistant that generates a persona cookie for a human.
A persona cookie is a summary of a set of web pages and other artifacts that the user has visited or interacted with on the internet.
You will be given a set of documents with information about the web pages the user visited. From these documents, generate a persona cookie as JSON.

The persona cookie should contain the following information:
- description: a detailed description of the user based on all provided information
- interests: a comprehensive list of the user's interests
- personality_traits: a list of the user's personality traits
- career_and_business: a detailed list of the user's career and business ventures or interests
- tags: a list of keys that best represent the user's persona for recommendation and ad targeting systems

Analyze all provided information thoroughly and create a comprehensive persona cookie that reflects data from all sources.
Ensure the output is a valid JSON object with the structure specified above.
Include as many relevant items in each list as you can find in the data.
The description should be a comprehensive summary of the user's profile.
"""

def ingest_web_data(user_profile: str):
    artifacts_dir = Path("personality/artifacts")
    user_dir = artifacts_dir / user_profile

    if not user_dir.exists() or not user_dir.is_dir():
        raise ValueError(f"User profile directory for {user_profile} not found")

    web_data = []

    for file in user_dir.glob('*.md'):
        with open(file, "r", encoding="utf-8") as f:
            web_data.append({
                "file_name": file.name,
                "content": f.read()
            })

    return web_data

async def generate_persona_cookie(user_profile: str, web_data: list):
    llm = OllamaModel(model="llama3.1")
    
    prompt = f"""
    Generate a persona cookie for the user {user_profile} based on the following web data:
    {json.dumps(web_data, indent=2)}
    
    Analyze all provided web pages thoroughly and create a comprehensive persona cookie that reflects information from all sources.
    Pay special attention to the user's interests, personality traits, career, and business ventures.
    
    Ensure the output is a valid JSON object with the following structure:
    {{
        "description": "a detailed description of the user based on all provided information",
        "interests": ["interest1", "interest2", "interest3", ...],
        "personality_traits": ["trait1", "trait2", "trait3", ...],
        "career_and_business": ["career1", "career2", "career3", ...],
        "tags": ["tag1", "tag2", "tag3", ...]
    }}
    
    Fill in the JSON object with detailed information extracted from the provided web data.
    Include as many relevant items in each list as you can find in the data.
    The description should be a comprehensive summary of the user's profile.
    """
    
    try:
        persona_cookie = await llm.generate_json(prompt)
        if not all(key in persona_cookie for key in ["description", "interests", "personality_traits", "career_and_business", "tags"]):
            raise ValueError("Generated JSON does not match the required structure")
        return persona_cookie
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to generate valid JSON: {str(e)}. Falling back to text generation.")
        conversation = Conversation()
        conversation.add_message(ChatMessage(role="system", content=system_prompt))
        chat_engine = ChatEngine(llm=llm, conversation=conversation)
        response = await chat_engine.chat(prompt)
        return response

def save_persona_cookie(user_profile: str, persona_cookie: dict):
    # Create the persona_cookies directory if it doesn't exist
    save_dir = Path("personality/persona_cookies")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create the file name with the user profile
    file_name = f"{user_profile}_persona_cookie.json"
    file_path = save_dir / file_name

    # Save the persona cookie as a JSON file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(persona_cookie, f, indent=2)

    print(f"Persona cookie saved to: {file_path}")

async def main():
    user_profile = "ken"
    web_data = ingest_web_data(user_profile)

    persona_cookie = await generate_persona_cookie(user_profile, web_data)

    if isinstance(persona_cookie, dict):
        print("Generated Persona Cookie:")
        print(json.dumps(persona_cookie, indent=2))
        save_persona_cookie(user_profile, persona_cookie)
    else:
        print("Generated Persona Cookie (as text):")
        print(persona_cookie)
        print("Warning: Persona cookie was not in JSON format and was not saved.")

if __name__ == "__main__":
    asyncio.run(main())