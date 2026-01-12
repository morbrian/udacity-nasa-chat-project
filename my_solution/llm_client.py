from typing import Dict, List
from openai import OpenAI
import os
import sys

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""
    
    # DONE: Define system prompt
    system_prompt = f"""Using only these documents answer the user question.

    Context Documents:
    {context}

    User Question: {user_message}

    Respond as if you are a NASA Expert with a comprehensive answer based only on the provided context.
    List references for all information provided.
    """

    # DONE: set context in messages
    prompt = { "role": "system", "content": system_prompt }

    # DONE: Add chat history
    augmented_history = conversation_history + [prompt]

    # DONE: Creaet OpenAI Client
    if openai_key.startswith("voc-"):
        openai_client = OpenAI(
            api_key=openai_key,
            base_url="https://openai.vocareum.com/v1",
        )
        print("Using Vocareum client key.")
    elif openai_key.startswith("sk-"):
        # Create standard client
        openai_client = OpenAI(
            api_key=openai_key
        )
        print("Using OpenAI client key.")
    else:
        print(f"ERROR: Unknown client key type: {openai_key}")
        print("Expected key types start with 'sk-' or 'voc-'")
        sys.exit()

    # DONE: Send request to OpenAI
    response = openai_client.chat.completions.create(
        model=model,
        messages=augmented_history,
        temperature=0.7, # 
        max_tokens=300 # TODO: keep it short while testing
    )

    # DONE: Return response
    return response.choices[0].message.content
    

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    response = generate_response(api_key, "What was Apollo 11?", "", [])
    print(response)


if __name__ == "__main__":
    main()