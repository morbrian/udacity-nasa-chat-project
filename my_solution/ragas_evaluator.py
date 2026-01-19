from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional, Any
import os
import sys
import asyncio
import argparse
import yaml
from pathlib import Path

import rag_client
import llm_client

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

LOG_PREFIX = "[RAGAS_EVALUATOR]"

def evaluate_response_quality(question: str, answer: str, contexts: List[str], ground_truth: List[str] = None) -> Dict[str, float]:
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
        reference_contexts=ground_truth,
        reference=ground_truth[0] if ground_truth is not None else None
    )

    results = {
        "faithfulness": asyncio.run(faithfulness.single_turn_ascore(sample)),
        "relevancy": asyncio.run(response_relevancy.single_turn_ascore(sample))
    }
    
    if (ground_truth is not None):
         results = results | { 
            "bleu": asyncio.run(bleu.single_turn_ascore(sample)),
            "rouge": asyncio.run(rouge.single_turn_ascore(sample)),
            "context_precision": asyncio.run(context_precision.single_turn_ascore(sample))
        }

    # DONE: Return the evaluation results
    return results


def evaluate_test_case(openai_api_key: str, question: str, ground_truth: str, collection, n_docs=3) -> Dict[str, Any]:
    """
    Sends the question to the LLM and evalutes the response against ground_truth using a ragas evaluator.
    
    :param question: General test question a user might ask.
    :type question: str
    :param ground_truth: Semantic variant of an expected correct answer.
    :type ground_truth: str
    :return: ragas results dictionary
    :rtype: Dict[str, float]
    """
    
    print(f"{LOG_PREFIX} Retrieve documents for question({question})")
    try:
        docs_result = rag_client.retrieve_documents(
            collection, 
            question, 
            n_docs
        )
    except:
        raise Exception(f"Failed to retrieve documents: {e}")
    
    context = ""
    contexts_list = []
    if docs_result and docs_result.get("documents"):
        contexts_list = docs_result["documents"][0]
        context = rag_client.format_context(documents=contexts_list, metadatas=docs_result["metadatas"][0])
    
    print(f"{LOG_PREFIX} Query LLM for question ({question}) and formatted RAG context.")
    try:
        answer = llm_client.generate_response(
            openai_key=openai_api_key, 
            user_message=question,
            context=context, 
            conversation_history=[]
        )
    except Exception as e:
        raise Exception(f"Failed to query LLM with question {question}: {e}")

    print(f"{LOG_PREFIX} Evaluate quality of answer {answer}")
    try:
        scores = evaluate_response_quality(
            question=question,
            answer=answer,
            contexts=contexts_list,
            ground_truth=[ground_truth]
        )
    except Exception as e:
        raise Exception(f"Failed to complete evaluation of test case for {question}: {e}")

    return {
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
        "scores": scores
    }

def evaluate_test_case_bundle(openai_api_key: str, test_cases_file: str, collection):
    """
    Reads test case data defined in the yaml formatted file specified by {test_cases_file}

    File input example:
    ```
    test_cases:
        - question: "Who were the crew members of the Apollo 13 mission?"
          ground_truth: |
            The crew members of the Apollo 13 mission were Jim Lovell, Fred Haise, and Jack Swigert
    ```
    
    :param test_cases_file: yaml formatted file of test cases
    :type test_cases_file: str
    """

    with open(test_cases_file, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    results_bundle = []
    for i, case in enumerate(data['test_cases']):
        question = case.get('question', None)
        ground_truth = case.get('ground_truth', None)
        if question is None:
            print(f"{LOG_PREFIX} Test Case {i} is missing question data")
            return
        if ground_truth is None:
            print(f"{LOG_PREFIX} Test Case {i} is missing ground_truth data")
            return
        results = evaluate_test_case(openai_api_key=openai_api_key, question=question, ground_truth=ground_truth, collection=collection)
        results_bundle.append(results)
    
    return results_bundle

def display_evaluation_metrics(scores: Dict[str, float]):
    """Display evaluation metrics in text format"""
    if "error" in scores:
        print(f"{LOG_PREFIX} Evaluation Error: {scores['error']}")
        return
    
    print(f"{LOG_PREFIX} 📊 Response Quality")
    
    for metric_name, score in scores.items():
        if isinstance(score, (int, float)):
            # Color code based on score
            if score >= 0.8:
                icon = "✔️"
            elif score >= 0.6:
                icon = "🟰"
            else:
                icon = "✖️"
            
            label = metric_name.replace('_', ' ').title()
            
            print(f"{LOG_PREFIX}     {icon}  {label}: {score}")

def valid_file(path_str):
    """Validates the file exists and is actually a file (not a directory)."""
    path = Path(path_str)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"The file '{path_str}' does not exist.")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"'{path_str}' is a directory, not a file.")
    return path

def valid_directory(path_str):
    """Validates a folder exists at path_str."""
    path = Path(path_str)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"The folder '{path_str}' does not exist.")
    if path.is_file():
        raise argparse.ArgumentTypeError(f"'{path_str}' is NOT a directory, it is a file.")
    return path

def valid_openai_api_key(openai_api_key):
    if openai_api_key.startswith("sk-") or openai_api_key.startswith("voc-"):
        return openai_api_key
    else:
        raise argparse.ArgumentTypeError(f"{openai_api_key} does not match match the supported key types, keys must start with 'voc-' or 'sk-'")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='RAG System Evaluator')
    parser.add_argument('--test-cases', type=valid_file, default='./test-cases.json', help='Path to test cases file')
    parser.add_argument('--openai-key', type=valid_openai_api_key, required=True, help='OpenAI API key')
    parser.add_argument('--chroma-dir', type=valid_directory, default='./chroma_db_openai', help='ChromaDB persist directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Name of existing populated collection in local ChromaDB')

    args = parser.parse_args()

    # put the api key as state in the environment as some apis look for it there
    os.environ["OPENAI_API_KEY"] = args.openai_key
    os.environ["CHROMA_OPENAI_API_KEY"] = args.openai_key

    print(f"{LOG_PREFIX} Initialize RAG system...")
    collection = None
    try:
       collection, success, error = rag_client.initialize_rag_system(args.chroma_dir, args.collection_name)
       if not success:
           raise Exception(f"RAG system did not throw an exception but reports an initialization failure {error}")
    except Exception as e:
        print(f"{LOG_PREFIX} Failed to initialize RAG system: {e}")
        sys.exit()
    
    results_bundle = evaluate_test_case_bundle(test_cases_file=args.test_cases, collection=collection, openai_api_key=args.openai_key)

    averages = {}
    for record in results_bundle:
        print(f"\n{LOG_PREFIX} 💼 === START Test Case ====")
        print(f"{LOG_PREFIX} ❔ Question: {record.get('question', '')}")
        print(f"{LOG_PREFIX} 💬 Answer: {record.get('answer', '')}")
        print(f"{LOG_PREFIX} 💯 Ground Truth: {record.get('ground_truth', '')}")
        display_evaluation_metrics(record.get('scores', []))
        print(f"\n{LOG_PREFIX} 💼 === END Test Case ====")

    print(f"\n\n{LOG_PREFIX} === TOTAL AVERAGES OF {len(results_bundle)} TEST CASES ===") 
    averages = {
        key: sum(item['scores'][key] for item in results_bundle) / len(results_bundle) 
        for key in results_bundle[0]['scores']
    }
    display_evaluation_metrics(averages)

if __name__ == "__main__":
    main()    



