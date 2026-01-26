from typing import Dict, List
from openai import OpenAI
import os
import sys
import argparse

SYSTEM_PROMPT = """### SYSTEM INSTRUCTIONS
-----------------------
- ROLE: NASA mission expert.
- PERSONALITY: professional, dry, boring and on point, you EXTRACT information but you do NOT explain the information with any depth.
- KNOWLEDGE: You have no memory of NASA missions and you have no access to the internet. The only information you have available is the information provided between the user's <context> tags.

### RESPONSE PROCEDURE
-----------------------
Use this step by step process to answer the user's question found between the <question> tags.
NON-NEGOTIABLE: YOU MUST NOT INCLUDE ANY INFORMATION THAT YOU DID NOT FIND DIRECTLY IN THE PROVIDED CONTEXT AND YOU MUST ACKNOWLEDGE THE ABSENCE OF INFORMATION IF THE QUESTION CANNOT BE ANSWERED

1. Identify the question betwen the <question> tags.
2. Read the information between the <context> tags and identify each facts and <timestamps> related to the question.
3. Use the <acronym> section to improve your understanding of the <context>.
4. Associate each fact and <timestamp> with the title of the context section it was found in (e.g. <title> <timestamp>)
5. If a speaker is identified in the text, attribute the quote using the format 'Speaker: [Quote]'."
6. Build a numbered Refernces list of each identified fact with timestamp (eg - [1] <title> <timestamp>)
7. Every sentence in the response must use ONLY the facts found in the <context>
8. Every sentence in the response must include a <timestamp> and quoted reference statement when availble, (e.g. "At 000:00:00, CapCom instructed... [1]")
9. Every sentence MUST end with a citation associated to the References list (e.g. [1], [2]).
10. Construct the References: list AFTER the response paragraph with ONLY facts that were explicitly used and referenced by the full response, and leave out others.

ALWAYS include timestamps and full speaker name for ALL quoted text.

### OUTPUT FORMAT
------------------
You must follow this EXACT structure ALWAYS include References:
<Your factual response here, with citations...> 

References:
- [1] <title/id> <timestamp>
- [2] <title/id> <timestamp>
- [3] provided context
"""

# some notes on prompt choices
# forcing the LLM to never use its training data for something as well known as the NASA data was a little tricky,
# and even now there is no guarantee that it won't sometimes respond with information not provided in the context.
# 
# A) When I included the following instruction the LLM was often unable to answer questions even when the answer was clearly in the context:
#    * If the answer is not found in the <context>, you must state: "I'm sorry, the provided context does not contain information regarding this request."
# 

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
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
        max_tokens=300 # TODO: keep it short while testing
    )

    # DONE: Return response
    return response.choices[0].message.content
    

def main():
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
        '\n'.join(args.context), 
        []
    )

    print(f"\n\nQUESTION: {args.question}")
    print(f"\n\nRESPONSE: {response}")


if __name__ == "__main__":
    main()