import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path
import os
import json

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
                    "chroma_dir": str(chroma_dir), # DONE: Store directory path as string
                    "name": collection.name, # DONE: Store collection name
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

    # DONE: Create a chomadb persistentclient
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    # DONE: Return the collection with the collection_name
    return client.get_or_create(name=collection_name)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # TODO: Initialize filter variable to None (represents no filtering)
    filter = None

    # TODO: Check if filter parameter exists and is not set to "all" or equivalent
    
    # TODO: If filter conditions are met, create filter dictionary with appropriate field-value pairs

    # TODO: Execute database query with the following parameters:
    results = collection.query(
        query_texts=[query], # TODO: Pass search query in the required format
        n_results=n_results, # TODO: Set maximum number of results to return
        # TODO: Apply conditional filter (None for no filtering, dictionary for specific filtering)
    )

    # TODO: Return query results to caller
    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    # TODO: Initialize list with header text for context section

    # TODO: Loop through paired documents and their metadata using enumeration
        # TODO: Extract mission information from metadata with fallback value
        # TODO: Clean up mission name formatting (replace underscores, capitalize)
        # TODO: Extract source information from metadata with fallback value  
        # TODO: Extract category information from metadata with fallback value
        # TODO: Clean up category name formatting (replace underscores, capitalize)
        
        # TODO: Create formatted source header with index number and extracted information
        # TODO: Add source header to context parts list
        
        # TODO: Check document length and truncate if necessary
        # TODO: Add truncated or full document content to context parts list

    # TODO: Join all context parts with newlines and return formatted string


def main():
   backends = discover_chroma_backends()
   print(json.dumps(backends, indent=4))


if __name__ == "__main__":
    main()