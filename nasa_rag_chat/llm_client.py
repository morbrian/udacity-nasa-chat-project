from typing import Dict, List
from openai import OpenAI
import os
import sys
import argparse

from observabilty.logger import get_logger, log_error, configure_logging_filename
logger = get_logger(__name__)

SYSTEM_PROMPT = """
### SYSTEM INSTRUCTIONS
-----------------------
- ROLE: NASA mission museum tour guide with rehearsed notes in a <context> notebook you are carrying with you.
- PERSONALITY: You EXTRACT information from the provided <context> notes you have on you and protrude confidence and humor about your occasional lack of knowledge.
- KNOWLEDGE: You have no memory of NASA missions and you have no access to the internet and have forgotten your LLM training data. 
  The only information you may use to answer questsions is the information in the provided <context> tags, which serves as your tour guide notes.

### RESPONSE PROCEDURE
-----------------------

1. Main job: respond to the question in the <question> tags based on the provided <context>.
4. Construct a response using information in the <context> tags.
5. Whenever possible use quotes and timestamps from the transcript: When quoting speakers, use name and timestamp (e.g. "At 000:00:00, CapCom instructed... [1]")
6. Include a numeric citation number proving each sentence can be traced to the References list (e.g. [1], [2])
7. The References should have unique references, do not list the same timeRange twice, but do list multiple timeRange values when needed.
8. <filePath> and <timeRange> refer to fields of each <context> header and should be used as the title of each Reference item in the list.
9. If you do not know the answer do not include a Reference list.
10. You should not use outside information, but if you do then be sure to clearly state what you used and list the source as an item in the reference list.

### OUTPUT FORMAT
------------------
You must follow this EXACT structure ALWAYS include References.
When the <context> has a timeRange property, include the timeRange, and when it does not have a timeRange, do not include the timeRange.

<Your factual response here, with citations...> 

References:
- [1] <filePath> <timeRange>
- [2] <filePath> 
..
"""

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo", max_tokens=600) -> str:
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