#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only

This script reads parsed text data from various NASA space mission folders and creates
a permanent ChromaDB collection with OpenAI embeddings for RAG applications.
Optimized to process only text files to avoid duplication with JSON versions.

Supported data sources:
- Apollo 11 extracted data (text files only)
- Apollo 13 extracted data (text files only)
- Apollo 11 Textract extracted data (text files only)
- Challenger transcribed audio data (text files only)
"""


import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
import hashlib
import time
from datetime import datetime
import argparse
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

import tiktoken
import re
from typing import Generator, Dict, Any
from pprint import pprint

from processors.text_generic import generic_chunk_text, get_comm_bounds
from services.openai import get_openai_client

from observabilty.logger import get_logger, log_error, configure_logging_filename
logger = get_logger(__name__)

class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings - Text files only"""
    
    def __init__(self, 
                 openai_api_key: str,
                 chroma_persist_directory: str = "./chroma_db",
                 collection_name: str = "nasa_space_missions_text",
                 embedding_model: str = "text-embedding-3-small",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 batch_size: int = 500):
        """
        Initialize the embedding pipeline
        
        Args:
            openai_api_key: OpenAI API key
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
        """
        if not openai_api_key:
            raise ValueError(f"Must specify valid OpenAI key")
        
        # DONE: Initialize OpenAI client
        self.openai_client = get_openai_client(openai_api_key)

        # DONE: Store configuration parameters
        self.parameters = {
            "chroma_persist_directory": chroma_persist_directory,
            "collection_name": collection_name,
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "batch_size": batch_size
        }

        # DONE: Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)

        # create a compatible encoder to use when chunking documents
        self.local_encoding = tiktoken.encoding_for_model(embedding_model)
        
        # DONE: Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={
                "embedding_model": embedding_model
            }
        )
    
    def get_metadata_log_prefix(self, metadata) -> str:
        prefix = f"[doc_id({metadata.get('doc_id', 'unknown')})]"
        return prefix

    def summarize_text(self, text: str, size: int, model: str = "gpt-3.5-turbo") -> List[Dict[str, Any]]:
        """
        Summarize the text into a single summary statement fitting into {size} tokens.
        """

        # get non-empty lines
        clean_lines = [line.strip() for line in text.splitlines() if line.strip()]

        system_prompt = {
            "role": "system", 
            "content": f"You are a specialized technical librarian. Summarize the following content into a dense, search-optimized description under {size} tokens."
        }

        user_prompt = {
             "role": "user",
             "content": "\n".join(clean_lines)
        }

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[system_prompt, user_prompt],
                temperature=0, # we want predictable consistent summaries
                max_tokens=size
            )

            summary_text = response.choices[0].message.content
            token_check = self.local_encoding.encode(summary_text)
            if (len(token_check) > size):
                logger.warning(f'Content summary created by OpenAI exceeds the requested {size} tokens at {len(token_check)} tokens instead')

            return summary_text
        except Exception as e:
            log_error(logger, f'Failed to summarize text: {e}', e)
            raise


    # def parse_transcript_document(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    #     """
    #     Preprocess text with data_type=transcript and return dictionary of prepared data.

    #     A transcript document is structured as follows:
    #     1. Title page
    #     2. INTRODUCTION
    #     3. ACRONYM LIST
    #     4. Transcript log
    #       We have read that all NASA transcripts are formatted like the Apollo 13 example transcript we have.
    #       time, speaker, and text. 
    #       The time column consists of four two-digit pairs for days, hours, minutes, and seconds (e.g., 04 22 45 12). 
    #       The speaker column indicates the source of a transmission; 
    #       the text column contains the verbatim transcript of the communications.
        
    #     Args:
    #         text: Text to process
    #         metadata: Base metadata for the text
            
    #     Returns:
    #         List of dictionaries with entries:
    #           text: enriched text sized for use as embedding
    #           metadata:
    #             "token_count": len(chunk_tokens),
    #             "position": numeric index into section of document,
    #             "section": name of document section where the content came from
    #     """
    #     logger_prefix = self.get_metadata_log_prefix(metadata)
    #     logger.info(f"{logger_prefix} Begin processing transcript")
    #     chunk_size = self.parameters['chunk_size']
    #     position_tracker = 0

    #     # the transcript processor will help us produce enriched sentence strings with abbreviations filled in to increase search relevance
    #     transcript_processor = TranscriptProcessor(acronym_lookups)

    #     # Split text into sections of the transcript type of document.
    #     title_text, intro_text, acronym_text, log_text = transcript_processor.parse_transcript_sections(text)

    #     # build a dictionary representation of the acronyms we can use to enrich the logs during processing
    #     logger.info(f"{logger_prefix} Build Acronym lookup table")
    #     log_metadata = metadata | { "section": "transcript" }

    #     if intro_text is not None or acronym_text is not None:
    #         acronym_lookups = get_acronym_lookup(f"{intro_text}\n{acronym_text}")
    #         acronym_expander = AcronymExpander(acronym_lookups)

    #     chunks = []

    #     # create title page chunk
    #     title_summary = self.summarize_text(text=title_text, size=chunk_size)
    #     logger.info(f"{logger_prefix} Extract Title: {title_summary}")
    #     chunk = (
    #         title_summary,
    #         metadata | {
    #             "section": "title",
    #             "token_count": len(self.local_encoding.encode(title_summary)),
    #             "position": 0
    #         }
    #     )
    #     chunks.append(chunk)
    #     position_tracker = len(chunks)

    #     # do a generic chunking on the introduction text
    #     logger.info(f"{logger_prefix} Extract Introduction")
    #     intro_metadata = metadata | { "section": "introduction" }
    #     intro_chunks = self.generic_chunk_text(
    #         intro_text, 
    #         intro_metadata, 
    #         position_start=position_tracker,
    #         chunk_size=self.parameters['chunk_size'],
    #         chunk_overlap=self.parameters['chunk_overlap'],
    #     )
    #     chunks.extend(intro_chunks)
    #     position_tracker = len(chunks)

    #     # do a generic chunking on the acronyms section
    #     logger.info(f"{logger_prefix} Extract Acronym text")
    #     acronym_metadata = metadata | { "section": "acronymn" }
    #     acronym_chunks = self.generic_chunk_text(
    #         acronym_text, 
    #         acronym_metadata, 
    #         position_start=position_tracker,
    #         chunk_size=self.parameters['chunk_size'],
    #         chunk_overlap=self.parameters['chunk_overlap'],
    #     )
    #     chunks.extend(acronym_chunks)
    #     position_tracker = len(chunks)

    #     sentences = []
    #     count = 0
    #     for record in transcript_processor.process_transcript(log_text, log_metadata):
    #         sentences.append(record.get("enriched_text"))
    #         count += 1
        
    #     enriched_log_text = "\n".join(sentences)
    #     transcript_chunks = self.generic_chunk_text(enriched_log_text, log_metadata, position_start=position_tracker, acronym_expander=acronym_expander)
    #     chunks.extend(transcript_chunks)
    #     position_tracker = len(chunks)

    #     logger.info(f"{logger_prefix} Finished Processing Total of ({count}) Transcript Records")

    #     return chunks
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Text to chunk
            metadata: Base metadata for the text
            
        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        chunk_size = self.parameters['chunk_size']
        chunk_overlap = self.parameters['chunk_overlap']

        # transcript_processor = TranscriptProcessor()
        
        # # Split text into sections of the transcript type of document.
        # _title_text, intro_text, acronym_text, _log_text = transcript_processor.parse_transcript_sections(text)

        # if intro_text is not None or acronym_text is not None:
        #     acronym_lookups = None
        #     acronym_lookups = get_acronym_lookup(f"{intro_text}\n{acronym_text}")
        #     acronym_expander = AcronymExpander(acronym_lookups)

        # get the data_type so we can use an the most effective transform for the type
        # data_type = metadata['data_type'] 
        # if data_type == 'transcript':
        #     return self.parse_transcript_document(text, metadata)
        # else:
        #     return self.generic_chunk_text(text=text, metadata=metadata)
        
        # chunks =  generic_chunk_text(text=text, metadata=metadata, acronym_expander=acronym_expander)

        chunks = generic_chunk_text(text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # add position id as a the sequence number of the document
        tuples = [(chunk, metadata | { "position": i }) for i, chunk in enumerate(chunks)]

        # generate doc_id from the metadata for the document
        tuples = [(chunk, metadata | { "doc_id": self.generate_document_id('', metadata)}) for chunk, metadata in tuples]

        # set fields commStart and commEnd if there are any timestamps in the text
        tuples = [
            (
                text, 
                {**metadata, **bounds} if (bounds := get_comm_bounds(text)) else metadata 
            )
            for text, metadata in tuples
        ]

        return tuples


    def check_document_exists(self, doc_id: str) -> bool:
        """
        Check if a document with the given ID already exists in the collection
        
        Args:
            doc_id: Document ID to check
            
        Returns:
            True if document exists, False otherwise
        """
        # DONE: Query collection for document ID
        response = self.collection.get(ids=[doc_id])

        # DONE: Return True if exists, False otherwise
        return response is not None and len(response['documents']) > 0
    
    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Update an existing document in the collection
        
        Args:
            doc_id: Document ID to update
            text: New text content
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get new embedding
            embedding = self.get_embedding(text)
            
            # Update the document
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            logger.debug(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
    
    def delete_documents_by_source(self, source_pattern: str) -> int:
        """
        Delete all documents from a specific source (useful for re-processing files)
        
        Args:
            source_pattern: Pattern to match source names
            
        Returns:
            Number of documents deleted
        """
        try:
            # Get all documents
            all_docs = self.collection.get()
            
            # Find documents matching the source pattern
            ids_to_delete = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if source_pattern in metadata.get('source', ''):
                    ids_to_delete.append(all_docs['ids'][i])
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents matching source pattern: {source_pattern}")
                return len(ids_to_delete)
            else:
                logger.info(f"No documents found matching source pattern: {source_pattern}")
                return 0
                
        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0
    
    def get_file_documents(self, file_path: Path) -> List[str]:
        """
        Get all document IDs for a specific file
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of document IDs for the file
        """
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)
            
            # Get all documents
            all_docs = self.collection.get()
            
            # Find documents from this file
            file_doc_ids = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if (metadata.get('source') == source and 
                    metadata.get('mission') == mission):
                    file_doc_ids.append(all_docs['ids'][i])
            
            return file_doc_ids
            
        except Exception as e:
            logger.error(f"Error getting file documents: {e}")
            return []
    
    def get_embeddings(self, text: List[str]) -> List[List[float]]:
        """
        Get OpenAI embedding for text
        
        Args:
            text: array of Text to embed
            
        Returns:
            Array of Embedding vectors
        """
        try:
            # DONE: Call OpenAI embeddings API
            response = self.openai_client.embeddings.create(
                model=self.parameters['embedding_model'],
                input=text
            )
            embeddings = [item.embedding for item in response.data]
            # DONE: Return embedding vector
            return embeddings
        except Exception as e:
            # DONE: Add error handling
            logger.error(f"Error creating embedding: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """
        Get OpenAI embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # DONE: Call OpenAI embeddings API
            response = self.openai_client.embeddings.create(
                model=self.parameters['embedding_model'],
                input=text
            )
            embedding = response.data[0].embedding
            # DONE: Return embedding vector
            return embedding
        except Exception as e:
            # DONE: Add error handling
            logger.error(f"Error creating embedding: {e}")
            raise

    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """
        Generate stable document ID based on file path and chunk position
        This allows for document updates without changing IDs
        """
        # DONE: Create consistent ID format
        # DONE: Use mission, source, and chunk_index
        mission = metadata.get('mission', '')
        source = metadata.get('source', '')
        data_type = metadata.get('data_type', '')
        category = metadata.get('category', '')
        position = metadata.get('position', '')
        doc_id = f"{mission}:{data_type}:{category}:{source}:{position}"
        # Format: mission_source_chunk_0001
        return doc_id
    
    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process plain text files with enhanced metadata extraction
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of (text, metadata) tuples
        """
        try:
            with open(file_path, 'r', encoding='UTF-8') as f:
                content = f.read()
            
            if not content.strip():
                return []
            
            # Enhanced metadata extraction
            metadata = {
                'source': str(file_path.stem),
                'file_path': str(file_path),
                'file_type': 'text',
                'content_type': 'full_text',
                'mission': self.extract_mission_from_path(file_path),
                'data_type': self.extract_data_type_from_path(file_path),
                'document_category': self.extract_document_category_from_filename(file_path.name),
                'processed_timestamp': datetime.now().isoformat()
            }
            
            return self.chunk_text(content, metadata)
            
        except Exception as e:
            raise e
    
    def extract_mission_from_path(self, file_path: Path) -> str:
        """Extract mission name from file path"""
        path_str = str(file_path).lower()
        if 'apollo11' in path_str or 'apollo_11' in path_str:
            return 'apollo_11'
        elif 'apollo13' in path_str or 'apollo_13' in path_str:
            return 'apollo_13'
        elif 'challenger' in path_str:
            return 'challenger'
        else:
            return 'unknown'
    
    def extract_data_type_from_path(self, file_path: Path) -> str:
        """Extract data type from file path"""
        path_str = str(file_path).lower()
        if 'transcript' in path_str:
            return 'transcript'
        elif 'textract' in path_str:
            return 'textract_extracted'
        elif 'audio' in path_str:
            return 'audio_transcript'
        elif 'flight_plan' in path_str:
            return 'flight_plan'
        else:
            return 'document'
    
    def extract_document_category_from_filename(self, filename: str) -> str:
        """Extract document category from filename for better organization"""
        filename_lower = filename.lower()
        
        # Apollo transcript types
        if 'pao' in filename_lower:
            return 'public_affairs_officer'
        elif 'cm' in filename_lower:
            return 'command_module'
        elif 'tec' in filename_lower:
            return 'technical'
        elif 'flight_plan' in filename_lower:
            return 'flight_plan'
        
        # Challenger audio segments
        elif 'mission_audio' in filename_lower:
            return 'mission_audio'
        
        # NASA archive documents
        elif 'ntrs' in filename_lower:
            return 'nasa_archive'
        elif '19900066485' in filename_lower:
            return 'technical_report'
        elif '19710015566' in filename_lower:
            return 'mission_report'
        elif 'reports' in filename_lower:
            return 'mission_report'
        elif 'biographies' in filename_lower:
            return 'biographies'
        
        # General categories
        elif 'full_text' in filename_lower:
            return 'complete_document'
        else:
            return 'general_document'
    
    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """
        Scan data directories for text files only (avoiding JSON duplicates)
        
        Args:
            base_path: Base directory path
            
        Returns:
            List of text file paths to process
        """
        base_path = Path(base_path)
        files_to_process = []
        
        # Define directories to scan
        data_dirs = [
            'apollo11',
            'apollo13',
            'challenger'
        ]
        
        for data_dir in data_dirs:
            dir_path = base_path / data_dir
            if dir_path.exists():
                logger.info(f"Scanning directory: {dir_path}")
                
                # Find only text files
                text_files = list(dir_path.glob('**/*.txt'))
                files_to_process.extend(text_files)
                logger.info(f"Found {len(text_files)} text files in {data_dir}")
        
        # Filter out unwanted files
        filtered_files = []
        for file_path in files_to_process:
            # Skip system files and summaries
            if (file_path.name.startswith('.') or 
                'summary' in file_path.name.lower() or
                file_path.suffix.lower() != '.txt'):
                continue
            filtered_files.append(file_path)
        
        logger.info(f"Total text files to process: {len(filtered_files)}")
        
        # Log file breakdown by mission
        mission_counts = {}
        for file_path in filtered_files:
            mission = self.extract_mission_from_path(file_path)
            mission_counts[mission] = mission_counts.get(mission, 0) + 1
        
        logger.info("Files by mission:")
        for mission, count in mission_counts.items():
            logger.info(f"  {mission}: {count} files")
        
        return filtered_files
    
    def add_documents_to_collection(self, documents: List[Tuple[str, Dict[str, Any]]], 
                                   file_path: Path, batch_size: int = 50, 
                                   update_mode: str = 'skip') -> Dict[str, int]:
        """
        Add documents to ChromaDB collection in batches with update handling
        
        Args:
            documents: List of (text, metadata) tuples
            file_path: Path to the source file
            batch_size: Number of documents to process in each batch
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents
                        'update' - update existing documents
                        'replace' - delete all existing documents from file and re-add
            
        Returns:
            Dictionary with counts of added, updated, and skipped documents
        """
        stats = {'added': 0, 'updated': 0, 'skipped': 0}

        if not documents:
            return stats
        
        if update_mode == 'replace':
            source = file_path.stem
            removed_count = self.delete_documents_by_source(source)
            logger.debug(f"Removed {removed_count} documents to replace {source}")

        # DONE: Process documents in batches
        batch_tracker = {
            'count': 0,
            'ids': [],
            'documents': [],
            'metadatas': [],
            'embeddings': []
        }

        # DONE: For each document:
        #   - Generate document ID
        #   - Check if exists
        #   - Get embedding
        #   - Add or update in collection
        document_count = len(documents)
        for i, document in enumerate(documents):
            doc_text = document[0]
            doc_metadata = document[1]

            #   - Generate document ID
            doc_id = self.generate_document_id(file_path=file_path, metadata=doc_metadata)
            logger_prefix = self.get_metadata_log_prefix(doc_metadata)
            
            #   - Check if exists
            doc_status = self.check_document_exists(doc_id)
            
            # DONE: Handle different update modes (skip, update, replace)
            if doc_status and update_mode == 'skip' or not doc_text:
                stats['skipped'] += 1
                continue

            #   - Add or update in collection
            if doc_status and update_mode == 'update':
                updated = self.update_document(
                    doc_id=doc_id, 
                    text=doc_text, 
                    metadata=doc_metadata
                )
                if updated:
                    stats['updated'] += 1
                    logger.debug(f"{logger_prefix}  UPDATE {i}-of-{document_count}: modified existing document")
            else:
                #   - Get embedding
                if batch_tracker['count'] == 0:
                    build_batch_start_time = time.time()
                batch_tracker['count'] += 1
                batch_tracker['ids'].append(doc_id)
                batch_tracker['documents'].append(doc_text)
                batch_tracker['metadatas'].append(doc_metadata)
                # logger.debug(f"{logger_prefix} [doc_id({doc_id})] ADD {i}-of-{document_count} TO BATCH: as item-{batch_tracker['count']} in batch")

            # if the current batch is filled or we are on the final document
            # then add the batch to the vector db
            if batch_tracker['count'] == batch_size or i == document_count - 1:
                if (batch_tracker['count'] > 0):
                    build_batch_end_time = time.time()
                    build_batch_duration = build_batch_end_time - build_batch_start_time
                    logger.info(f"Filled batch of size {batch_tracker['count']} in {build_batch_duration:.2f} seconds at document-{i} of total-{document_count}")
                    embeddings_call_start = time.time()
                    embeddings = self.get_embeddings(batch_tracker['documents'])
                    embeddings_call_end = time.time()
                    embeddings_call_duration = embeddings_call_end - embeddings_call_start
                    logger.info(f"Finished creating {len(batch_tracker['documents'])} embeddings in {embeddings_call_duration:.2f} seconds")
                    self.collection.add(
                        ids=batch_tracker['ids'],
                        documents=batch_tracker['documents'],
                        metadatas=batch_tracker['metadatas'],
                        embeddings=embeddings
                    )
                    stats['added'] += batch_tracker['count']
                    batch_tracker = {
                        'count': 0,
                        'ids': [],
                        'documents': [],
                        'metadatas': [],
                        'embeddings': []
                    }
        
        # DONE: Return statistics
        return stats
    
    def process_all_text_data(self, base_path: str, update_mode: str = 'skip', batch_size: int = 50) -> Dict[str, int]:
        """
        Process all text files and add to ChromaDB
        
        Args:
            base_path: Base directory containing data folders
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents (default)
                        'update' - update existing documents
                        'replace' - delete all existing documents from file and re-add
            
        Returns:
            Statistics about processed files
        """
        stats = {
            'files_processed': 0,
            'documents_added': 0,
            'documents_updated': 0,
            'documents_skipped': 0,
            'errors': 0,
            'total_chunks': 0,
            'missions': {}
        }
        
        # DONE: Get files to process
        text_files = self.scan_text_files_only(base_path)
        # DONE: Loop through each file
        for file_path in text_files:
            start = time.time()
            try:
                # DONE: Process file and add to collection
                documents = self.process_text_file(file_path)
                document_stats = self.add_documents_to_collection(
                    documents=documents,
                    file_path=file_path,
                    update_mode=update_mode,
                    batch_size=batch_size
                )

                # DONE: Update statistics
                stats['files_processed'] += 1
                stats['documents_added'] += document_stats['added']
                stats['documents_updated'] += document_stats['updated']
                stats['documents_skipped'] += document_stats['skipped']
            except Exception as e:
                # DONE: Handle errors gracefully
                stats['errors'] += 1
                log_error(logger, f"Error adding documents from path {file_path} to collection: {e}", e)

            end = time.time()
            duration = end - start
            logger.info(f"Finished processing {file_path} in {duration} sections")
        return stats
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection"""
        # DONE: Return collection name, document count, metadata
        collection_name = self.collection.name
        document_count = self.collection.count()
        metadata = self.collection.metadata

        return {
            'collection_name': collection_name,
            'document_count': document_count,
            'metadata': metadata
        }
    
    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the collection for testing
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            
        Returns:
            Query results
        """
        # DONE: Perform test query and return results
        embedding = self.get_embedding(query_text)
        response = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )

        return response

    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the collection"""
        try:
            # Get all documents to analyze
            all_docs = self.collection.get()
            
            if not all_docs['metadatas']:
                return {'error': 'No documents in collection'}
            
            stats = {
                'total_documents': len(all_docs['metadatas']),
                'missions': {},
                'data_types': {},
                'document_categories': {},
                'file_types': {}
            }
            
            # Analyze metadata
            for metadata in all_docs['metadatas']:
                mission = metadata.get('mission', 'unknown')
                data_type = metadata.get('data_type', 'unknown')
                doc_category = metadata.get('document_category', 'unknown')
                file_type = metadata.get('file_type', 'unknown')
                
                # Count by mission
                stats['missions'][mission] = stats['missions'].get(mission, 0) + 1
                
                # Count by data type
                stats['data_types'][data_type] = stats['data_types'].get(data_type, 0) + 1
                
                # Count by document category
                stats['document_categories'][doc_category] = stats['document_categories'].get(doc_category, 0) + 1
                
                # Count by file type
                stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}

def main():
    """Main function"""
    # setup the logger for the embedding_pipeline process
    configure_logging_filename('chroma_embedding_text_only.log')
    
    parser = argparse.ArgumentParser(description='ChromaDB Embedding Pipeline for NASA Data')
    parser.add_argument('--data-path', default='.', help='Path to data directories')
    parser.add_argument('--openai-key', required=True, help='OpenAI API key')
    parser.add_argument('--chroma-dir', default='./chroma_db_openai', help='ChromaDB persist directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Collection name')
    parser.add_argument('--embedding-model', default='text-embedding-3-small', help='OpenAI embedding model')
    parser.add_argument('--chunk-size', type=int, default=250, help='Text chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=50, help='Chunk overlap size')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--update-mode', choices=['skip', 'update', 'replace'], default='skip',
                       help='How to handle existing documents: skip, update, or replace')
    parser.add_argument('--test-query', help='Test query after processing')
    parser.add_argument('--stats-only', action='store_true', help='Only show collection statistics')
    parser.add_argument('--delete-source', help='Delete all documents from a specific source pattern')
    parser.add_argument('--log-level',choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Set Log Level')
    
    args = parser.parse_args()

    # set log level
    
    
    # Initialize pipeline
    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Handle delete source operation
    if args.delete_source:
        deleted_count = pipeline.delete_documents_by_source(args.delete_source)
        logger.info(f"Deleted {deleted_count} documents matching source pattern: {args.delete_source}")
        return
    
    # If stats only, show collection statistics and exit
    if args.stats_only:
        logger.info("Collection Statistics:")
        stats = pipeline.get_collection_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        return
    
    # Process all data
    logger.info(f"Starting text data processing with update mode: {args.update_mode}")
    start_time = time.time()
    
    stats = pipeline.process_all_text_data(args.data_path, update_mode=args.update_mode, batch_size=args.batch_size)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print results
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Documents added to collection: {stats['documents_added']}")
    logger.info(f"Documents updated in collection: {stats['documents_updated']}")
    logger.info(f"Documents skipped (already exist): {stats['documents_skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    
    # Mission breakdown
    logger.info("\nMission breakdown:")
    for mission, mission_stats in stats['missions'].items():
        logger.info(f"  {mission}: {mission_stats['files']} files, {mission_stats['chunks']} chunks")
        logger.info(f"    Added: {mission_stats['added']}, Updated: {mission_stats['updated']}, Skipped: {mission_stats['skipped']}")
    
    # Collection info
    collection_info = pipeline.get_collection_info()
    logger.info(f"\nCollection: {collection_info.get('collection_name', 'N/A')}")
    logger.info(f"Total documents in collection: {collection_info.get('document_count', 'N/A')}")
    
    # Test query if provided
    if args.test_query:
        logger.info(f"\nTesting query: '{args.test_query}'")
        results = pipeline.query_collection(args.test_query)
        if results and 'documents' in results:
            logger.info(f"Found {len(results['documents'][0])} results:")
            for i, doc in enumerate(results['documents'][0][:3]):  # Show top 3
                logger.info(f"Result {i+1}: {doc[:200]}...")
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_error(logger, f"Pipeline failed: {e}", e)

