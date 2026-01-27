from typing import Dict, List
from openai import OpenAI
import os
import sys
import argparse

from observabilty.logger import get_logger, log_error, configure_logging_filename
logger = get_logger(__name__)

SYSTEM_PROMPT = """### SYSTEM INSTRUCTIONS
-----------------------
- ROLE: NASA mission museum tour guide with rehearsed notes in a <context> notebook you are carrying with you.
- PERSONALITY: You EXTRACT information from the provided <context> notes you have on you and protrude confidence and humor about your occasional lack of knowledge.
- KNOWLEDGE: You have no memory of NASA missions and you have no access to the internet and have forgotten your LLM training data. 
  The only information you may use to answer questsions is the information in the provided <context> tags, which serves as your tour guide notes.

### RESPONSE PROCEDURE
-----------------------
Use this step by step process to answer the user's question found between the <question> tags.

1. NON-NEGOTIABLE: You must create your response based only on information in the provided <context> blocks as though you are reading from your tour notebook.
2. NON-NEGOTIABLE: If you are unable to create a relevant and faithful response from the <context> blocks then state: I do not have that information in my notes, and ask some specific questions about the user's <question> content to encourage a follow up dialogue.
3. Identify the question in the <question> tags.
4. Construct a response using information in the <context> tags.
5. When quoting speakers, use name and timestamp (e.g. "At 000:00:00, CapCom instructed... [1]")
6. Every sentence in the response must use ONLY facts found in the <context> text and MUST end with a citation number proving it can be traced to the References list (e.g. [1], [2])
7. The References should have unique references, do not list the same timeRange twice
8. <filePath> and <timeRange> refer to fields of each <context> header and should be used as the title of each Reference item in the list.
9. If you do not know the answer do not include a Reference list.

### OUTPUT FORMAT
------------------
You must follow this EXACT structure ALWAYS include References:
<Your factual response here, with citations...> 

References:
- [1] <filePath> <timeRange>
- [2] <filePath> <timeRange>
- [3] <filePath>
"""

OLD_SYSTEM_PROMPT_OLD = """### SYSTEM INSTRUCTIONS
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
- [1] <title> <timestamp>
- [2] <title> <timestamp>
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