import os
from openai import OpenAI
from observabilty.logger import get_logger, log_error

logger = get_logger(__name__)

def get_openai_client(openai_api_key: str = None):
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")

    if openai_api_key.startswith("voc-"):
        openai_client = OpenAI(
            api_key=openai_api_key,
            base_url="https://openai.vocareum.com/v1",
        )
        logger.info("Using Vocareum client key.")
    elif openai_api_key.startswith("sk-"):
        openai_client = OpenAI(
            api_key=openai_api_key
        )
        logger.info("Using OpenAI client key.")
    else:
        raise ValueError(f"Unknown client key type: Expected key types start with 'sk-' or 'voc-'")

    return openai_client