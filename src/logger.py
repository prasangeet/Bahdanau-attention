import logging
from pathlib import Path


def setup_logger(log_dir, log_name="training.log"):
    """Create a logger that writes both to the console and to a log file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("attention_project")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Reset old handlers so repeated script runs do not duplicate log lines.
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_dir / log_name)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
