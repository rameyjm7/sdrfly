import logging
import logging.handlers
from pathlib import Path

_log_location = Path.home() / "sdrfly"
_log_location.mkdir(parents=True, exist_ok=True)
_rotating_file_handler = logging.handlers.RotatingFileHandler(
    _log_location / "sdrfly.log", maxBytes=10 ^ 6
)
_rotating_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
print(_log_location)
logging.getLogger(__name__).addHandler(_rotating_file_handler)
logging.getLogger(__name__).setLevel(logging.INFO)
