import pytest
from nasa_rag_chat.processors.text_generic import generic_chunk_text

# --- Fixtures ---
@pytest.fixture
def sample_prose():
    """Provides a standard string of sentences for testing."""
    return r"""
This is the first sentence. When the CDR and LMP are in the undocked lunar module or on the lunar surface, their speaker designations will be suffixed by either LM or EVA to indicate their status (e.g., CDR-EVA or LMP-LM). Voice calls during this mission were assigned in accordance with the following station operating procedures: "For all phases when only the CSM is manned, the AS-508 call sign will be Apollo 13. When both vehicles are manned, the call sign will be Odyssey for the CSM and Aquarius for the LM. The call signs for the CDR and LMP during lunar surface operations will be the individual crew's first names."
Transcription of these tapes was managed by David M. Goldenbaum, Test Division, Apollo Spacecraft Program Office, to whom questions regarding this document should be referred.
"""

@pytest.fixture
def sample_transcript():
    """Provides some example transcript communication with timestamp and speaker."""
    return r"""
078:08:23 CC Apollo 13, Houst - Aquarius, Houston.

078:08:31 LMP Go ahead.

078:08:33 CC Fred, after the burn, we'll get the powerdown instructions, or checklist changes to you. At the same time, we'd like to get a consumables status to you. All I'll say right now is that we think you look in reasonably good shape.

078:08:52 CMP Okay. Very good.

"""

# --- Tests ---

def test_generic_chunk_text_respects_max_size(sample_prose):
    """Ensure no chunk ever exceeds the chunk_size limit."""
    # Arrange
    size = 50
    overlap = 10
    
    # Act
    chunks = generic_chunk_text(sample_prose, chunk_size=size, chunk_overlap=overlap)
    
    # Assert
    for chunk in chunks:
        assert len(chunk) <= size, f"Chunk exceeded limit: {len(chunk)} > {size}"

def test_generic_chunk_text_timestamps_respectes_max_size(sample_transcript):
    """Verify timestamp markers start the sentence when available"""
    # Arrange
    size = 50
    overlap = 10
    
    # Act
    chunks = generic_chunk_text(sample_transcript, chunk_size=size, chunk_overlap=overlap)
    
    # Assert
    for chunk in chunks:
        assert len(chunk) <= size, f"Chunk exceeded limit: {len(chunk)} > {size}"

def test_generic_chunk_text_aligns_to_sentences(sample_prose):
    """Ensure chunks attempt to end on sentence boundaries (. ! ?)."""
    # Arrange
    # "This is the first sentence." is 27 chars.
    size = 40 
    
    # Act
    chunks = generic_chunk_text(sample_prose, chunk_size=size, chunk_overlap=0)
    
    # Assert
    # The first chunk should be exactly the first sentence.
    assert chunks[0] == "This is the first sentence."

def test_generic_chunk_text_aligns_to_timestamps(sample_transcript):
    """Ensure chunks attempt to end on sentence boundaries (. ! ?)."""
    # Arrange
    size = 63 # 63 includes the 2nd timestamp, we verify it does not get separated from the related text.
    overlap = 15
    
    # Act
    chunks = generic_chunk_text(sample_transcript, chunk_size=size, chunk_overlap=overlap)
    for chunk in chunks:
        print(f"* {chunk}")
    # Assert
    # The first chunk should be exactly the first communication and not chop into the second communication.
    assert chunks[0] == "078:08:23 CC Apollo 13, Houst - Aquarius, Houston."

def test_generic_chunk_text_terminates_at_end_of_string():
    """Verify that the function does not 'stutter' or duplicate the last chunk."""
    # Arrange
    text = "Short sentence. Another one."
    size = 20
    overlap = 5
    
    # Act
    chunks = generic_chunk_text(text, chunk_size=size, chunk_overlap=overlap)
    
    # Assert
    # With a small string, we shouldn't see "Another one." repeated.
    last_chunk = chunks[-1]
    assert chunks.count(last_chunk) == 1



@pytest.mark.parametrize("size, overlap", [
    (100, 110),
    (50, 50),
    (10, 20),
])
def test_generic_chunk_text_raises_value_error_on_invalid_overlap(size, overlap):
    """Check that we catch invalid size/overlap configurations early."""
    with pytest.raises(ValueError, match="chunk_size must be greater than chunk_overlap"):
        generic_chunk_text("Some text", chunk_size=size, chunk_overlap=overlap)

def test_generic_chunk_text_handles_empty_input():
    """Edge case: ensuring an empty string doesn't break the logic."""
    assert generic_chunk_text("", chunk_size=100) == []