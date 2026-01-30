# Implementation Report

The development process for the RAG project focused on the following major components.

1. Embedding Pipeling: This processes file data and loads it into ChromaDB.
2. Data Retrieval: Strategies for find the right data in ChromaDB to proved context to user queries.
3. Prompt Engineering: This block of text tells the LLM how to use the data to generate a response.
4. Result Evaluation: We used RAGAS and sample question/answer pairs to gather metrics on how effective the solution was.
5. Testing: Using the chat program we could quickly enter sample questions and observe the result with larger chat history contetxts.

## Embedding Pipeline

chunk size effect on data retrieval
acronym enrichment
summarizing title page
timestamp extraction

and finally, after trying the above and many other clever ideas
 the fact that chunk size had by far the greates impact on query relevance


## Data Retrieval

Mission filtered queries
number of docs
use of content "adjacency" to pull in more context surrounding the best semantic match
allowing smaller focused chunks but still providing access to larger blocks of context

## Prompt Engineering

difficulting blocking the llm from leaking training data into the response
discuss how we extract and add to metadata and in <context> headers so the LLM can use in Reference List

## Result Evaluation

Discuss scores, and how they are not always consistent


## Testing

