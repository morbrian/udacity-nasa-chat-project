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
import time

import rag_client
import llm_client

from observabilty.logger import get_logger, log_error, configure_logging_filename
logger = get_logger(__name__)

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

DEFAULT_MISSION = 'apollo_13'

def evaluate_response_quality(question: str, answer: str, contexts: List[str], ground_truth: List[str] = None, model='gpt-3.5-turbo') -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    base_url = None if openai_api_key.startswith("sk-") else "https://openai.vocareum.com/v1"

    # DONE: Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
        model=model,
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


def evaluate_test_case(
        openai_api_key: str, 
        question: str, 
        ground_truth: str, 
        collection,
        mission=DEFAULT_MISSION,
        n_docs=3, 
        include_adjacent=False, 
        test_id="", 
        gen_model='gpt-3.5-turbo',
        eval_model='gpt-3.5-turbo',
        max_tokens=300
    ) -> Dict[str, Any]:
    """
    Sends the question to the LLM and evalutes the response against ground_truth using a ragas evaluator.
    
    :param question: General test question a user might ask.
    :type question: str
    :param ground_truth: Semantic variant of an expected correct answer.
    :type ground_truth: str
    :return: ragas results dictionary
    :rtype: Dict[str, float]
    """
    
    logger.info(f"{LOG_PREFIX} {test_id} Retrieve documents for question({question})")
    try:
        docs_result = rag_client.retrieve_documents(
            collection=collection, 
            query=question, 
            n_results=n_docs,
            include_adjacent=include_adjacent,
            mission_filter=mission
        )
        contexts_list = docs_result["documents"][0]
        metadatas = docs_result["metadatas"][0]
    except Exception as e:
        raise Exception(f"{test_id} Failed to retrieve documents: {e}")

    context = rag_client.format_context(documents=contexts_list, metadatas=metadatas)
    
    logger.info(f"{LOG_PREFIX} {test_id} Query LLM for question ({question}) and formatted RAG context.")
    try:
        answer = llm_client.generate_response(
            openai_key=openai_api_key, 
            user_message=question,
            context=context, 
            conversation_history=[],
            model=gen_model,
            max_tokens=max_tokens
        )
    except Exception as e:
        raise Exception(f"{test_id} Failed to query LLM with question {question}: {e}")

    try:
        scores = evaluate_response_quality(
            question=question,
            answer=answer,
            contexts=[context],
            ground_truth=[ground_truth],
            model=eval_model
        )
    except Exception as e:
        raise Exception(f"{test_id} Failed to complete evaluation of test case for {question}: {e}")

    record =  {
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
        "scores": scores
    }

    return record

def evaluate_test_case_bundle(
        openai_api_key: str, 
        test_cases_file: str, 
        collection, 
        mission=DEFAULT_MISSION,
        n_docs=3, 
        include_adjacent=False,
        gen_model='gpt-3.5-turbo',
        eval_model='gpt-3.5-turbo',
        max_tokens=300
    ):
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
        logger.info(f"\n{LOG_PREFIX} [{i}] 💼 === START Test Case ====")
        question = case.get('question', None)
        ground_truth = case.get('ground_truth', None)
        if question is None:
            logger.info(f"{LOG_PREFIX} [{i}] Test Case {i} is missing question data")
            return
        if ground_truth is None:
            logger.info(f"{LOG_PREFIX} [{i}] Test Case {i} is missing ground_truth data")
            return
        logger.info(f"{LOG_PREFIX} [{i}] ...... test phase in progress ......\n")
        results = evaluate_test_case(
            openai_api_key=openai_api_key, 
            question=question, 
            ground_truth=ground_truth, 
            collection=collection,
            mission=mission,
            n_docs=n_docs, 
            include_adjacent=include_adjacent, 
            test_id=f"[{i}]",
            gen_model=gen_model,
            eval_model=eval_model,
            max_tokens=max_tokens
        )
        logger.info(f"\n{LOG_PREFIX} [{i}] ...... test phase complete ......")
        results_bundle.append(results)
        logger.info(f"{LOG_PREFIX} [{i}] ❔ Question: {results.get('question', '')}")
        logger.info(f"{LOG_PREFIX} [{i}] 💬 Answer: {results.get('answer', '')}")
        logger.info(f"{LOG_PREFIX} [{i}] 💯 Ground Truth: {results.get('ground_truth', '')}\n")
        display_evaluation_metrics(results.get('scores', []), test_id=f"[{i}]")
        logger.info(f"\n{LOG_PREFIX} [{i}] 💼 === END Test Case ====")

    return results_bundle

def display_evaluation_metrics(scores: Dict[str, float], test_id=""):
    """Display evaluation metrics in text format"""
    if "error" in scores:
        logger.warning(f"{LOG_PREFIX} {test_id} Evaluation Error: {scores['error']}")
        return
    
    logger.info(f"{LOG_PREFIX} {test_id} 📊 Response Quality")
    
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
            
            logger.info(f"{LOG_PREFIX} {test_id}     {icon}  {label}: {score}")

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

def print_averages(results_bundle):
    logger.info(f"\n\n{LOG_PREFIX} === TOTAL AVERAGES OF {len(results_bundle)} TEST CASES ===") 
    averages = {
        key: sum(item['scores'][key] for item in results_bundle) / len(results_bundle) 
        for key in results_bundle[0]['scores']
    }
    display_evaluation_metrics(averages)

def main():
    """Main function"""
    # setup the logger for the embedding_pipeline process
    configure_logging_filename('ragas_evaluator.log')

    parser = argparse.ArgumentParser(description='RAG System Evaluator')
    parser.add_argument('--test-cases', type=valid_file, help='Path to test cases file to run a bundle of test cases, mutually exclusive to the single test run of question/answer/contexts parameters.')
    parser.add_argument('--question', default='What is a common color for grass?', help='Question query for the LLM to answer')
    parser.add_argument('--answer', default='Grass is commonly green.', help='Answer we would expect from the LLM')
    parser.add_argument('--contexts', 
                        nargs="+", 
                        default=['The most common color of grass is green.'], 
                        help='Question query for the LLM to answer' )
    parser.add_argument('--mission', choices=["all", "apollo_11", "apollo_13"], default=DEFAULT_MISSION, help="Specify the mission to focus on when retrieving documents.")
    
    parser.add_argument('--gen-model', choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"], default='gpt-3.5-turbo', help="Model used for response generation from LLM.")
    parser.add_argument('--eval-model', choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"], default='gpt-3.5-turbo', help="Model used to evaluate response from LLM.")
    parser.add_argument('--max-tokens', type=int, default=300, help="Max tokens of the response returned from LLM")
    parser.add_argument('--n-docs', type=int, default=3, help="Number of retrieved document chunks to include with prompt sent to LLM.")
    parser.add_argument('--include-adjacent', type=bool, default=False, help="When true, RAG will include the previous and next document chunks for each of the retrieved docs, increasing the context data by 3x.")

    parser.add_argument('--openai-key', type=valid_openai_api_key, required=True, help='OpenAI API key')
    parser.add_argument('--chroma-dir', type=valid_directory, default='./chroma_db_openai', help='ChromaDB persist directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Name of existing populated collection in local ChromaDB')

    args = parser.parse_args()

    # put the api key as state in the environment as some apis look for it there
    os.environ["OPENAI_API_KEY"] = args.openai_key
    os.environ["CHROMA_OPENAI_API_KEY"] = args.openai_key

    logger.info(f"{LOG_PREFIX} Initialize RAG system...")
    collection = None
    try:
       collection, success, error = rag_client.initialize_rag_system(args.chroma_dir, args.collection_name)
       if not success:
           raise Exception(f"RAG system did not throw an exception but reports an initialization failure {error}")
    except Exception as e:
        log_error(f"{LOG_PREFIX} Failed to initialize RAG system: {e}")
        sys.exit()
    
    try:
        start_time = time.time()
        if args.test_cases:
            results_bundle = evaluate_test_case_bundle(
                test_cases_file=args.test_cases, 
                collection=collection, 
                openai_api_key=args.openai_key,
                mission=args.mission,
                n_docs=args.n_docs,
                include_adjacent=args.include_adjacent,
                gen_model=args.gen_model,
                eval_model=args.eval_model
            )
            print_averages(results_bundle)
        else:
            results_bundle = evaluate_response_quality(
                question=args.question, 
                answer=args.answer, 
                contexts=args.contexts,
                model=args.eval_model
            )
            display_evaluation_metrics(results_bundle)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Evaluation completed in {duration:.2f} seconds")
    except Exception as e:
        log_error(logger, f"Failed to complete test cycle: {e}", e)

if __name__ == "__main__":
    main()    



