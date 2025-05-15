import sys

from loguru import logger

# Configure the default logger:
logger.remove()

# DEBUG, and INFO messages go to stdout
logger.add(
    sys.stdout,
    format="<lvl>{message}</lvl>",
    level="DEBUG",
    filter=lambda record: record["level"].name in ["DEBUG", "INFO"],
)

# WARNING and above go to stderr
logger.add(sys.stderr, format="<lvl>{message}</lvl>", level="WARNING")
