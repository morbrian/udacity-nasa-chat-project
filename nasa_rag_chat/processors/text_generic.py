
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
        r"\n\d{3}:\d{2}:\d{2}\s+[A-Z]{2,4}\n", # original AS13_TEC.txt
        r"\d{2}\s\d{2}\s\d{2}\s\d{2}\n[A-Z]{2,}", # new transcripts for both apollo 11,13
        r"\[\d{2}:\d{2}\]\s+spk_\d+:", # new transcripts for challenger
        
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
TIMESTAMP_PATTERN = re.compile(
    r"("
    r"\d{3}:\d{2}:\d{2}(?=\s+[A-Z]{2,4})"  # Format 1: 087:13:06
    r"|"
    r"(?<=\[)\d{2}:\d{2}(?=\])"            # Format 2: 03:23 (inside brackets)
    r"|"
    r"\d{2}\s\d{2}\s\d{2}\s\d{2}(?=\n[A-Z]{2,})" # Format 3: 00 00 10 04
    r")"
)

def get_comm_bounds(text):
    """Return the first and last times found in the text block."""
    matches = TIMESTAMP_PATTERN.findall(text)
    
    if not matches:
        return None
    
    return {
        "commStart": matches[0], 
        "commEnd": matches[-1]
    }

