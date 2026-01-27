
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

def generic_chunk_text(
    text: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 50
) -> list[str]:
    """Chunk generic text, with consideration for sentence termination (e.g. .!?) and timestamp markers as contained blocks of text"""
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap.")
    
    custom_separators = [
        # 1. Structured Transcript Markers (Priority)
        r"\n\d{3}:\d{2}:\d{2}\s+[A-Z]{2,4}\n", 
        
        # 2. Markdown/Structural Headers
        "\n\n", 
        
        # 3. Sentence Boundaries (using regex lookbehind)
        r"(?<=[.!?])\s+", 
        
        # 4. Fallbacks
        "\n", 
        " ", 
        ""
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=custom_separators,
        is_separator_regex=True,
        strip_whitespace=True 
    )
    
    # reduce more than 3 consecutive newlines to max of 2
    clean_text = re.sub(r'\n{3,}', '\n\n', text)
    return splitter.split_text(clean_text)

# Compile ONCE at the module level
TIMESTAMP_PATTERN = re.compile(r"(\d{3}:\d{2}:\d{2})\s+[A-Z]{2,4}")

def get_comm_bounds(text):
    matches = list(TIMESTAMP_PATTERN.finditer(text))
    if not matches:
        return None
    
    # matches[0][1] gets the first capture group of the first match
    # matches[-1][1] gets the first capture group of the last match
    return {
        "commStart": matches[0].group(1), 
        "commEnd": matches[-1].group(1)
    }   

# def generic_chunk_text_CODED(
#     text: str, 
#     chunk_size: int = 500, 
#     chunk_overlap: int = 50
# ) -> List[str]:
#     if chunk_size <= chunk_overlap:
#         raise ValueError("chunk_size must be greater than chunk_overlap.")

#     chunks = []
#     start_idx = 0
#     text_len = len(text)

#     while start_idx < text_len:
#         # 1. Calculate the boundary of our current search window
#         end_idx = min(start_idx + chunk_size, text_len)
        
#         # 2. Find the best split point if we aren't at the end of the text
#         if end_idx < text_len:
#             window = text[start_idx:end_idx]
            
#             # Look for sentence endings (. ! ?)
#             boundary_match = list(re.finditer(r'[.!?](\s|$)', window))
            
#             if boundary_match:
#                 # Use the end of the last complete sentence in this window
#                 end_idx = start_idx + boundary_match[-1].end()
#             else:
#                 # Fallback: Last space for word boundary
#                 last_space = window.rfind(' ')
#                 if last_space != -1:
#                     end_idx = start_idx + last_space + 1
#                 # Final Fallback: Hard split at chunk_size (no boundary found)
        
#         # 3. Extract and store the chunk
#         chunk = text[start_idx:end_idx].strip()
#         if chunk:
#             chunks.append(chunk)
            
#         # 4. Advance the pointer
#         # If we reached the end of the text, break the loop
#         if end_idx >= text_len:
#             break
            
#         # Ensure the next start_idx is always moving forward 
#         # but attempts to respect the overlap
#         new_start_idx = end_idx - chunk_overlap
        
#         # Safety check: if the split point was so early that end_idx - overlap 
#         # puts us behind or at the current start_idx, force movement forward.
#         start_idx = max(new_start_idx, start_idx + 1)

#     return chunks


