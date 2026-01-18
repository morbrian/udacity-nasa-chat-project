import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path
import os
import json
import sys
from openai import OpenAI

def get_embedding(text: str, model: str):
    """
    Get text embedding using model

    Args:
        text: text to embed
        model: model identifier
    
    Returns:
        Embedding vector
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv["CHROMA_OPENAI_API_KEY"]
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
        print(f"""ERROR: Unknown client key type: {api_key} - Expected key types start with 'sk-' or 'voc-'""")
        sys.exit()
    
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
        print(f"Error creating embedding: {e}")
        raise


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
            full_error = str(e)
            truncated_error = (full_error[:100] + '..') if len(full_error) > 100 else full_error
            # DONE: Handle connection or access errors gracefully
            print(f"❌ Error generating RAG response from folder {chroma_dir}: {truncated_error}")
            # DONE: Create fallback entry for inaccessible directories
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

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # DONE: Initialize filter variable to None (represents no filtering)
    filter = None

    # DONE: Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter is not None and mission_filter != "all":
        # DONE: If filter conditions are met, create filter dictionary with appropriate field-value pairs
        filter = {
            'mission': mission_filter['mission'],
            'data_type': mission_filter['data_type'],
            'doc_category': mission_filter['doc_category'],
            'file_type': mission_filter['file_type']
        }

    # DONE: Execute database query with the following parameters:
    embedding_model = collection.metadata.get('embedding_model') or "text-embedding-3-small"
    embedding = get_embedding(text=query, model=embedding_model)
    results = collection.query(
        query_embeddings=[embedding], # DONE: Pass search query in the required format
        n_results=n_results, # DONE: Set maximum number of results to return
        where=filter # DONE: Apply conditional filter (None for no filtering, dictionary for specific filtering)
    )

    # TODO: Return query results to caller
    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        print("NO DOCUMENTS FOUND")
        return ""
    
    # DONE: Initialize list with header text for context section
    context =["Reference Information"]

    # DONE: Loop through paired documents and their metadata using enumeration
    for i, (document, metadata) in enumerate(zip(documents, metadatas)):
        # DONE: Extract mission information from metadata with fallback value
        mission = metadata.get('mission') or 'Mission Unknown'
        # DONE: Clean up mission name formatting (replace underscores, capitalize)
        mission = mission.replace("_", " ").capitalize()

        # DONE: Extract source information from metadata with fallback value
        source = metadata.get('source') or 'Source Unknown'

        # DONE: Extract category information from metadata with fallback value
        category = metadata.get('document_category') or "Category Unknown"
        # DONE: Clean up category name formatting (replace underscores, capitalize)
        category = category.replace("_", " ").capitalize()

        # extract section
        section = metadata.get('section') or 'Section Unknown'

        
        # DONE: Create formatted source header with index number and extracted information
        source_header = f"\nReference Title: {i+1}: {source}:{mission}:{category}:{section}\n"
        # DONE: Add source header to context parts list
        context.append(source_header)
        
        # DONE: Check document length and truncate if necessary
        truncated_document = f"{document[:100]}..." if len(document) < 100 else document
        # DONE: Add truncated or full document content to context parts list
        context.append("Reference Context:")
        context.append(truncated_document)
        

    # DONE: Join all context parts with newlines and return formatted string
    formatted_context = "\n".join(context)

    return formatted_context


def main():
   backends = discover_chroma_backends()
   print(json.dumps(backends, indent=4))


if __name__ == "__main__":
    main()