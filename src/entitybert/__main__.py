import logging

from entitybert.cli import cli

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
cli()
