# NASA RAG Chat Project 

A Retrieval-Augmented Generation (RAG) system with real-time evaluation capabilities. Create a complete RAG pipeline from document processing to interactive chat interface.


## Environment Setup 

1. Setup Python Environment

   ```bash
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

   ```bash
   export OPENAI_API_KEY="<your key>"
   ```

## Load Data and Smoke Test System

1. Load the data using the embedding pipeline

   ```bash
   python nasa_rag_chat/embedding_pipeline.py --openai-key $OPENAI_API_KEY --update-mode replace --data-path ./data
   ```

2. **Test LLM Client**

The LLM client tests our prompt without the benefit of our ChromaDB context. The prompt is designed to constrain
the LLM from using its own training knowledge over the context we provide. We forced this demand onto the LLM
because we found that the NASA data was so well known that the LLM would sometimes know the answer to questions
about the missions even when we failed to provide the answer in the context.

  **Example-1:** Asking questions with no context should always cause the LLM to admit a lack of knowldge.
   ```bash
   python nasa_rag_chat/llm_client.py --question 'Who were the crew members of the Apollo 13?'
    
    ...<also prints context and question as output>
    RESPONSE: My notes don't cover that, and I'd hate to make something up!
   ```

  **Example-2:** Asking a question related to a provided context should provide a reasonable response ONLY from the context even if its fictional.
  * This test demonstrates the LLM can pull facts from the context.
   ```bash
   python nasa_rag_chat/llm_client.py --question 'Who were the crew members of the Apollo 13?' \
      --contexts "<context file_path='commandline'> The Apollo 13 Mission is a historic achievment in space travel that took place in 1713. The Lead Pilot of the space craft was Mickey Mouse, supported by Donald Duck as Number Two and Walt Disney in the role of Medicine Man</context>"

      ...<also prints context and question as output>
      RESPONSE: The crew members of the Apollo 13 mission were Lead Pilot Mickey Mouse, Number Two Donald Duck, and Medicine Man Walt Disney[1].

      References:
      - [1] commandline
   ```

  **Example-3:** This next test demonstrates the LLM is able to recognize when a somewhat related question does not have an answer in the context.
   ```bash
   python nasa_rag_chat/llm_client.py --question 'Who did NASA select as the first person to walk on the moon and what year did the Apollo 3 mission take place?' \
      --contexts "<context file_path='commandline'>The Apollo 13 Mission is a historic achievment in space travel that took place in 1713. The Lead Pilot of the space craft was Mickey Mouse, supported by Donald Duck as Number Two and Walt Disney in the role of Medicine Man</context>"

      ...<also prints context and question as output>
      RESPONSE: My notes don't cover that, and I'd hate to make something up!
   ```

3. **Test RAG Client**

   Either of the following commands verify the `rag_client` has access to ChromaDB, and provide similar stats printouts.

   rag_client
   ```
   python nasa_rag_chat/rag_client.py
   ```

   embedding_pipeline
   ```bash
   python nasa_rag_chat/embedding_pipeline.py --openai-key $OPENAI_API_KEY --stats-only
   ```

4. **Test Evaluation**

   This is a short smoke test of the RAGAS implementation, a deeper dive on a larger dataset follows in the next section of this README.md
* This example only tests the evaluate_response_quality(..) function with the static text inputs.
* The more general use case for the ragas_evaluator is to run a number of tests defined in a file like `test-cases-apollo-13.yaml`
   ```bash
   python nasa_rag_chat/ragas_evaluator.py \
       --openai-key $OPENAI_API_KEY
       --question "What is a common color for grass?" \
       --answer "Grass is commonly green" \
       --contexts "The most comon color of grass is green."
   ```

## **Integration Testing With Browser**

1. **Run the complete pipeline**:

   This will run the chatbot server at http://localhost:8501

   ```bash
   # Process documents
   python nasa_rag_chat/embedding_pipeline.py --openai-key $OPENAI_API_KEY --data-path ./data
   
   # Launch chat interface
   streamlit run nasa_rag_chat/chat.py
   ```

### **Extra Features**

* **Max Tokens:** Used to tell LLM how verbose it can make the response.

* **History Control:** Used to reduce or expand how much of your chat history is remembered and sent to the LLM with the context.

* **Select Mission:** Used to specify which mission the chat dicussion refers to by using a mission_filter with ChromaDB, or select 'all' to include all data.

* **Include Adjacent Documents**: After retrieving the related documents, this tells the system to also get the document before and after the initially found documents, Useful for increasing the information in the context while keeping the original chunk sizes small and focused.


## Evaluator Deep Dive

1. Run test cases from the provided data file `test-cases.yaml`

   ```bash
   python nasa_rag_chat/ragas_evaluator.py --openai-key $OPENAI_API_KEY --mission apollo_13 --test-cases ./nasa_rag_chat/test-cases-apollo-13.yaml
   ```

Score improvement tactics.
    
   1. Faithfulness
      
      Initial poor scores (< 0.6) in faithfulness were a result of the LLM using its own training data and not our provided context.
      
      * Strategy: We lowered `temperature` to 0.0 so the LLM would focus on the context with fewer attempts to be creative.
        
      * Strategy: We adjusted our prompt instructions to use a Chain of Thought strategy that asked the LLM to start by looking at our context and building the response in steps.

      * Strategy: We added several very specific instructions telling the LLM to trust the context and not rely on training data.
        
      Second pass poor scores (< 0.8) in faithfulness were a result of our data simply not having enough overlap with the question being asked.
        
      * Strategy: We added a `data/apollo13/mission-report.txt` extracted from https://archive.org/stream/apollo-13-mission-report/apollo-13-mission-report_djvu.txt

        * Having the extra document actually made our scores drop again on our prepared test cases but seemed to provide more useful answers.

   2. Relevancy
        
      Relevancy generally had stronger scores (>= 0.7) from the begining once we fixed some unintended truncation bugs in our code.
            
      * Strategy: We kept chunk sizes to 250 to help the vector search focus on the questions asked by the user.

      * Strategy: We have a special processor for the apollo 13 transcripts to identify time stamps and note them in the meta data, and also use the timestamp pattern as a preferred chunk start indicator.
    
   3. Bleu

      We include Bleu but did not tailor a matching dataset to use with it.

   3. Rouge

      We include Rouge but did not tailor matching dataset to use with it.

   4. Context Precision

      We include Precision but we believe our implementation is incorrect as we only ever see 0.0 valued scores.



