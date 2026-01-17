from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
import os
import asyncio

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str], reference: str = None) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    base_url = None if openai_api_key.startswith("sk-") else "https://openai.vocareum.com/v1"

    # DONE: Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=openai_api_key,
        base_url=base_url
    ))

    # DONE: Create evaluator_embeddings with model text-embedding-3-small
    langchain_openai_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key,
        openai_api_base=base_url
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(langchain_openai_embeddings)

    # DONE: Define an instance for each metric to evaluate

    # 1. Traditional NLP Metrics (Non-LLM)
    # BleuScore and RougeScore expects a reference in latest api, but the chat program is not ready to provide one
    bleu = BleuScore() 
    rouge = RougeScore(rouge_type="rougeL")

    # 2. Retrieval Metrics (Context Focused)
    context_precision = NonLLMContextPrecisionWithReference()

    # 3. Generation Metrics (LLM-Based)
    faithfulness = Faithfulness(llm=evaluator_llm)

    # ResponseRelevancy requires embeddings to measure similarity 
    # between the generated answer and the original question
    response_relevancy = ResponseRelevancy(
        llm=evaluator_llm, 
        embeddings=evaluator_embeddings
    )


    # DONE: Evaluate the response using the metrics
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference_contexts=reference
    )

    results = {
        "faithfulness": asyncio.run(faithfulness.single_turn_ascore(sample)),
        "relevancy": asyncio.run(response_relevancy.single_turn_ascore(sample))
    }
    
    if (reference is not None):
        results = results | { 
            "bleu": asyncio.run(bleu.single_turn_ascore(sample)),
            "rouge": asyncio.run(rouge.single_turn_ascore(sample)),
            "context_precision": asyncio.run(context_precision.single_turn_ascore(sample))
        }

    # DONE: Return the evaluation results
    return results
