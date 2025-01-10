import logging
import logging.handlers
from pathlib import Path

# Define the log directory and ensure it exists
_log_location = Path.home() / "sdrfly"
_log_location.mkdir(parents=True, exist_ok=True)

# Setup a rotating file handler for SDRFLY
_rotating_file_handler = logging.handlers.RotatingFileHandler(
    _log_location / "sdrfly.log", maxBytes=10**6, backupCount=5  # Corrected maxBytes to 10^6 and added backupCount
)

# Set a formatter for the logs
_rotating_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - SDRFLY - %(name)s - %(levelname)s - %(message)s")
)

# Create a named logger for SDRFLY
sdrfly_logger = logging.getLogger("SDRFLY")
sdrfly_logger.addHandler(_rotating_file_handler)
sdrfly_logger.setLevel(logging.INFO)

# Print log directory for verification
print(f"Log directory: {_log_location}")
