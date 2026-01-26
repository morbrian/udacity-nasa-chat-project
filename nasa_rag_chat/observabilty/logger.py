
import logging
from rich.logging import RichHandler
import os

import logging

# We define a global variable to hold the filename
_log_file = "app_logging.log"

def configure_logging_filename(filename: str):
    """Call this in the main script before doing anything else."""
    global _log_file
    _log_file = filename

def get_logger(name: str):
    """Configures and returns a logger instance."""
    logger = logging.getLogger(name)
    
    # Only add handlers if they don't exist to avoid duplicate logs
    if not logger.handlers:
        # Default to INFO, but allow override via ENV
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File Handler
        fh = logging.FileHandler(_log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Rich logging expands stack trace information and formatting in console for debug log levels
        handler = RichHandler(rich_tracebacks=True) 
        logger.addHandler(handler)
        
        # Prevent logs from propagating to the root logger twice
        logger.propagate = False

    return logger

def log_error(logger, message, error, include_traceback=False):
    """
    Logs an error. Traceback is included only if explicitly 
    requested or if the logger is set to DEBUG.
    """
    if logger.isEnabledFor(logging.DEBUG) or include_traceback:
        logger.error(f"{message}: {error}", exc_info=True)
    else:
        logger.error(f"{message}: {error}")


















