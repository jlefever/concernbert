import logging
import os
import sys

from concernbert.cli import cli


class RelativePathFilter(logging.Filter):
    def filter(self, record):
        record.pathname = os.path.relpath(record.pathname, os.getcwd())
        return True


stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

# file_handler = logging.FileHandler("concernbert.log")
# file_handler.setLevel(logging.DEBUG)
# file_handler.addFilter(RelativePathFilter())

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s][%(levelname)s][%(pathname)s:%(lineno)d %(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # handlers=[stream_handler, file_handler],
    handlers=[stream_handler],
)

try:
    cli()
except Exception:
    logging.exception("An error occurred while running the CLI.")
