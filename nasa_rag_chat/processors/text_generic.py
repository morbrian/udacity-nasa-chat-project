
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


