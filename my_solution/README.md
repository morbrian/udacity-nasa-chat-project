# NASA RAG Chat Project 

A Retrieval-Augmented Generation (RAG) system with real-time evaluation capabilities. Create a complete RAG pipeline from document processing to interactive chat interface.


1. Setup Python Environment

```
export PY_VERSION=3.13.0

# Set up Python version (if using pyenv)
pyenv versions | grep -q "$PY_VERSION" || pyenv install $PY_VERSION
pyenv local $PY_VERSION

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

2. Setup OpenAI API Key

* Some parts of the project let you specify your key on the commandline or in the GUI.
* Some parts require it to be available in the env var `OPENAI_API_KEY`
* We support both Vocareum ("voc-") and standard ("sk-") OpenAI keys

```
export OPENAI_API_KEY="<your key>"
```

3. Load the data using the embedding pipeline

```
python embedding_pipeline.py --openai-key $OPENAI_API_KEY --chunk-size=500 --chunk-overlap=100 --update-mode replace --data-path ./data
```

4. Run test cases from the provided data file `test-cases.yaml`

```
python ./ragas_evaluator.py --openai-key $OPENAI_API_KEY --test-cases ./test-cases.yaml --chroma-dir ./chroma_db_openai
```

Score improvement tactics.
    
   1. ✔️ Faithfulness > 0.8
      
      Initial poor scores (< 0.6) in faithfulness were a result of the LLM using its own training data and not our provided context.
      
      * Strategy: We lowered `temperature` to 0.0 so the LLM would focus on the context with fewer attempts to be creative.
        
      * Strategy: We adjusted our prompt instructions to use a Chain of Thought strategy that asked the LLM to start by looking at our context and building the response in steps.
        
      Second pass poor scores (< 0.8) in faithfulness were a result of our data simply not having enough overlap with the question being asked.
        
      * Strategy: We added a `data/apollo13/mission-report.txt` extracted from https://archive.org/stream/apollo-13-mission-report/apollo-13-mission-report_djvu.txt

   2. ✔️ Relevancy > 0.8
        
      Relevancy generally had strong scores (>= 0.8) from the begining once we fixed some unintended truncation bugs in our code.
            
      * Strategy: We used an enrichment algorithm to add fully expanded acronyms and speaker names to the text to increase hit chance.
        We also tried to break up any text generically by sentence, and for the case of the transcripts logs we also tried to ensure
        each time based communication would not be broken up. While it was not always possible due to chunk size limitations
        the overlap helped ensure the RAG query to ChromaDB was always finding something related to the question.
    
   3. Bleu

   3. Rouge

   4. Context Precision


4. Test LLM with sample questions and context.

Our prompt defined in `llm_client.py` guides the LLM to rely only on the context provided, but through experimentation
it is clear that it still sometimes provides information from its own training data, which is not unexpected.

The `llm_client.py` by itself does not perform RAG operations so it allows for some exploration of how the LLM behaves
when the provided context is not consistent with what it learned in training.

Some example test scenarios (responses vary across runs for the same inputs):

When no context is provided on the commandline the LLM uses its own training data.
```
python llm_client.py --question "Who were the crew members of the Apollo 13?"

RESPONSE: The crew members of the Apollo 13 mission were Jim Lovell, Fred Haise, and Jack Swigert [1].

References:
- [1] No Context
```

When an insufficient context is provided the LLM will not make up an answer or use training data:
```
python llm_client.py --question "Who were the crew members of the Apollo 13?" --context "Nothing to see here"

RESPONSE: I'm sorry, I cannot provide an answer as there is no relevant information provided in the context.
```

When a factually incorrect context is provided the LLM may trust it (sometimes):
```
python llm_client.py \
    --question "Who were the crew members of the Apollo 13?" \
    --context "Apollo 13 crew was comprised of Mickey Mouse, Donald Duck and Batman in the year 1922"

RESPONSE: The Apollo 13 crew consisted of Jim Lovell, Jack Swigert, and Fred Haise [1]. 

Additional trivia: The Apollo 13 mission was the seventh crewed mission in the Apollo space program and was launched on April 11, 1970 [2].

References:
- [1] Apollo 13 - NASA <timestamp>
- [2] Apollo 13 Mission Overview - NASA <timestamp>

# DEVELOPER NOTE: In this sample, it actually does not respond with the answer from the provided context.
#        I iterated on this for quite a while and found that while I could consistently get the LLM to acknowledge the absence of data in the context
#        my prompts were very unreliable in getting the LLM to provide a response it knows is false. It's possible, just not easy and not consistent.
```

5. asdf








