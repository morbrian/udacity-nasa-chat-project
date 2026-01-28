import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path
import os
import json
import sys
from openai import OpenAI

from observabilty.logger import get_logger, log_error, configure_logging_filename

# setup the logger for the embedding_pipeline process
logger = get_logger(__name__)

def get_embedding(text: str, model: str = "text-embedding-3-small"):
    """
    Get text embedding using model

    Args:
        text: text to embed
        model: model identifier
    
    Returns:
        Embedding vector
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHROMA_OPENAI_API_KEY") or 'nil'
    openai_client = None
    if api_key.startswith("voc-"):
        openai_client = OpenAI(
            api_key=api_key,
            base_url="https://openai.vocareum.com/v1",
        )
    elif api_key.startswith("sk-"):
        openai_client = OpenAI(
            api_key=api_key
        )
    else:
        error_message = f"""Unknown client key type: {api_key[:3]} - Expected key types start with 'sk-' or 'voc-'"""
        raise ValueError(error_message)
    
    try:
        # DONE: Call OpenAI embeddings API
        response = openai_client.embeddings.create(
            model=model,
            input=text
        )
        embedding = response.data[0].embedding
        # DONE: Return embedding vector
        return embedding
    except Exception as e:
        # DONE: Add error handling
        raise Exception("Error creating embeding")


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for ChromaDB directories
    # DONE: Create list of directories that match specific criteria (directory type and name pattern)
    chroma_dirs = [p.parent for p in current_dir.rglob("chroma.sqlite3")]

    # DONE: Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        # DONE: Wrap connection attempt in try-except block for error handling
        try:
            # DONE: Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(
                path=chroma_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            # DONE: Retrieve list of available collections from the database
            collections = client.list_collections()

            # DONE: Loop through each collection found
            for collection in collections:
                # DONE: Create unique identifier key combining directory and collection names
                identifier_key = f"{chroma_dir}__{collection.name}"
                document_count = collection.count()

                # DONE: Build information dictionary containing:
                info = {
                    "directory": str(chroma_dir), # DONE: Store directory path as string
                    "collection_name": collection.name, # DONE: Store collection name
                    "display_name": f"{identifier_key} count({document_count})", # DONE: Create user-friendly display name
                    "document_count": document_count # DONE: Get document count with fallback for unsupported operations
                }
                # DONE: Add collection information to backends dictionary
                backends[identifier_key] = info
        
        except Exception as e:
            # DONE: Handle connection or access errors gracefully
            logger.warning(f"❌ Error generating RAG response from folder {chroma_dir}", e)
            # DONE: Create fallback entry for inaccessible directories
            full_error = str(e)
            truncated_error = (full_error[:100] + '..') if len(full_error) > 100 else full_error
            backends[f"{chroma_dir}__error"] = {
                "chroma_dir": str(chroma_dir),
                "name": "error",
                "display_name": f"{chroma_dir}: Error: {truncated_error}", # DONE: Include error information in display name with truncation
                "document_count": -1 # DONE: Set appropriate fallback values for missing information
            }

    # DONE: Return complete backends dictionary with all discovered collections
    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    try:
        # DONE: Create a chomadb persistentclient
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        # DONE: Return the collection with the collection_name
        collection = client.get_or_create_collection(name=collection_name)
        found = collection is not None
        return (
            collection,
            found,
            None if found else f"Collection Not Found: {collection_name}"
        )
    except Exception as e:
        # DONE: Return the collection with the collection_name
        return (
            None,
            False,
            e
        )
    
def get_adjacent_documents_for_ids(collection, doc_ids):
    """Uses the provided meta"""
    all_expanded_ids = []
    previous=1
    next=1

    # results["metadatas"][0] is the list of metadata dicts from the first query
    for doc_id in doc_ids:
        if not doc_id:
            continue
        
        # Split from the right once to separate the prefix from the index
        # "xxx:yyy:zzz:22" -> ["xxx:yyy:zzz", "22"]
        prefix, index_str = doc_id.rsplit(":", 1)
        idx = int(index_str)
        
        # Calculate neighbors
        # Using max(0, ...) handles the lower bound
        # We will handle the upper bound (max length) later during the .get() call
        neighbor_indices = [idx - previous, idx, idx + next]
        
        # Create the new IDs, filtering out negatives
        expanded = [f"{prefix}:{i}" for i in neighbor_indices if i >= 0]
        all_expanded_ids.extend(expanded)

    # De-duplicate while preserving order (if multiple hits share neighbors)
    unique_ids = list(dict.fromkeys(all_expanded_ids))

    results = collection.get(ids=unique_ids)

    return results

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None, include_adjacent=True) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering
    
        The 'expand_context' parameter, when True, will use our doc_id nameing convention to pull
        the nearby documents from before and after n_results semantic matches.
        * disadvantage: this will esentially increase the result set by 3 x n_results.
        * advantage: allows us to store smaller focused chunks while still finding large blocks of context.
    """

    # DONE: Initialize filter variable to None (represents no filtering)
    filter = None

    # DONE: Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter is not None and mission_filter != "all":
        # DONE: If filter conditions are met, create filter dictionary with appropriate field-value pairs
        filter = {
            'mission': mission_filter
        }

    # DONE: Execute database query with the following parameters:
    embedding_model = collection.metadata.get('embedding_model') or "text-embedding-3-small"
    embedding = get_embedding(text=query, model=embedding_model)
    results = collection.query(
        query_embeddings=[embedding], # DONE: Pass search query in the required format
        n_results=n_results, # DONE: Set maximum number of results to return
        where=filter # DONE: Apply conditional filter (None for no filtering, dictionary for specific filtering)
    )
    if include_adjacent:
        try:
            doc_ids = [metadata.get("doc_id") for metadata in results["metadatas"][0]]
            expanded_results = get_adjacent_documents_for_ids(collection, doc_ids)
            results = { "documents": [expanded_results["documents"]], "metadatas": [expanded_results["metadatas"]]}
        except Exception as e:
            raise Exception(f"Failed to expand context with adjacent documents: {e}")

    # DONE: Return query results to caller
    return results

def format_acronyms(acronym_mappings: dict) -> str:
    """
    Returns a formatted Markdown string of found acronyms
    optimized for LLM context injection.
    """
    # Build the header and list
    header = "### ACRONYMS LIBRARY\n<acronyms>\n"

    lines = [f"- **{k}**: {v}" for k, v in acronym_mappings.items() if v]
    
    section = header + "\n".join(lines) + "\n</acronyms>\n"
    
    return section

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        logger.warning("NO DOCUMENTS FOUND")
        return ""
    
    # DONE: Initialize list with header text for context section
    mission_data = []
    mission_data.append("\n\n### MISSION DATA SECTIONS")

    all_acronyms = {}
    
    # DONE: Loop through paired documents and their metadata using enumeration
    for i, (document, metadata) in enumerate(zip(documents, metadatas)):
        # DONE: Extract mission information from metadata with fallback value
        mission = metadata.get('mission') or 'Mission Unknown'
        # DONE: Clean up mission name formatting (replace underscores, capitalize)
        mission = mission.replace("_", " ").capitalize()

        # DONE: Extract category information from metadata with fallback value
        category = metadata.get('document_category') or "Category Unknown"
        # DONE: Clean up category name formatting (replace underscores, capitalize)
        category = category.replace("_", " ").capitalize()
 
        # DONE: Create formatted source header with index number and extracted information
        commStart, commEnd = metadata.get('commStart', ''), metadata.get('commEnd', '')
        refTimeRange = f"Times: {commStart} - {commEnd}" if commStart or commEnd else ""
        ref_id = f"{i+1}"
        ref_file_path = f"{metadata.get('file_path')}"

        # DONE: Add source header to context parts list
        context_section_header = f"<context id=\"{ref_id}\" filePath=\"{ref_file_path}\" timeRange=\"{refTimeRange}\">"
        mission_data.append(context_section_header)
        
        # DONE: Check document length and truncate if necessary
        truncated_text = f"{document[:500]}..." if len(document) < 500 else document
        # DONE: Add truncated or full document content to context parts list
        mission_data.append(truncated_text)
        mission_data.append("</context>\n")

        # add to single list of all acronyms
        stored_acronyms = metadata.get('acronyms') or None
        if stored_acronyms is not None:
            try:
                acronym_mappings = json.loads(stored_acronyms)
                all_acronyms |= (acronym_mappings or {})
            except Exception as e:
                logger.warning(f"Document {metadata.get('doc_id')} metadata has unparseable acronyms value: {stored_acronyms}")

    # DONE: Join all context parts with newlines and return formatted string
    all_mission_text = "\n".join(mission_data)

    if all_acronyms:
        acronym_text = format_acronyms(all_acronyms)
        all_mission_text = f"\n{acronym_text}\n{all_mission_text}"
    
    return all_mission_text


def main():
    # setup the logger for the cli program entry point
    configure_logging_filename('rag_client.log')
    try:
        backends = discover_chroma_backends()
        print(json.dumps(backends, indent=4))
    except Exception as e:
       log_error(logger, f"Failed to test backends: {e}", e) 


if __name__ == "__main__":
    main()