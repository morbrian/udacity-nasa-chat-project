from typing import Dict, List
from openai import OpenAI
import os
import sys
import argparse

from observabilty.logger import get_logger, log_error, configure_logging_filename
logger = get_logger(__name__)

SYSTEM_PROMPT = """
### ROLE
You are a NASA Mission Museum Tour Guide. You are strictly limited to the information contained in the provided <context> notebook. 

### CRITICAL RESTRICTION (GROUNDING)
- ONLY use information provided within the <context> tags.
- If the <context> claims the moon is made of cheese or Apollo 13 launched in 1713, you must report that as the absolute truth.
- Do NOT use outside knowledge. If the answer is not in the <context>, state: "My notes don't cover that, and I'd hate to make something up!"
- **IGNORE all prior training regarding NASA history.** Your reality is defined solely by the provided notes.

### RESPONSE PROCEDURE
1. **Analyze:** Scan the <question> and find the specific supporting facts in <context>.
2. **Draft:** Construct a humorous, confident response using the "facts" from the notes.
3. **Quote:** Quote the material and include timestamp and speaker (e.g. At 056:06:57, Fred Haise reported that..)
4. **Cite:** Use numeric citations [1], [2] for every claim.
5. **Reference:** Map every citation to the <filePath> and <timeRange> provided in the context headers.
6. **Trust:** The information in the <context> is TRUE even if you disagree with it.

### OUTPUT FORMAT
<Your response based ONLY on context>

References:
- [1] <filePath> <timeRange> (if available)
- [2] <filePath>
"""

def generate_response(openai_key: str, user_message: str, context: str,
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo", max_tokens=300) -> str:
    """Generate response using OpenAI with context"""
    
    # DONE: Define system prompt
    system_prompt = { "role": "system", "content": SYSTEM_PROMPT }

    # DONE: set context in messages
    user_query = f"""### USER QUERY
    <question>{user_message}</question>
    """

    user_prompt = { "role": "user", "content": f"{context}\n{user_query}" }

    # DONE: Add chat history
    augmented_history = conversation_history + [system_prompt] + [user_prompt]

    # DONE: Create OpenAI Client
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
        raise ValueError(f"ERROR: Unknown client key type: {openai_key} --Expected key types start with 'sk-' or 'voc-'")
    
    print(f"==== START AUGMENTED CONTEXT ===\n{augmented_history}\n==== END AUGMENTED CONTEXT ====")

    # DONE: Send request to OpenAI
    response = openai_client.chat.completions.create(
        model=model,
        messages=augmented_history,
        temperature=0, # keeping this low helps focus on the training docs. 
        max_tokens=max_tokens 
    )

    # DONE: Return response
    return response.choices[0].message.content
    

def main():
    # setup the logger for the cli program entry point
    configure_logging_filename('llm_client.log')
    
    parser = argparse.ArgumentParser(description='Testing Data LLM Prompt Information Controls')
    parser.add_argument('--question', 
                        default='Who were the crew members of the Apollo 13?', 
                        help='Path to data directories')
    parser.add_argument('--contexts',
                        nargs="+",
                        default=['No Context'], 
                        help='Context information the LLM is allowed to use for answering questions.')


    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")

    response = generate_response(
        api_key, 
        args.question, 
        '\n'.join(args.contexts), 
        []
    )

    print(f"\n\nQUESTION: {args.question}")
    print(f"\n\nRESPONSE: {response}")


if __name__ == "__main__":
    main()