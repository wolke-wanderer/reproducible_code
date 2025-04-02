import logging
import logging.config
import yaml
import os
from datetime import datetime

# Function to load YAML configuration
def setup_logging(default_path='logs/logging_config.yaml', tag=""):
    with open(default_path, 'r') as file:
        config = yaml.safe_load(file)

        # Modify the log file name based on current time and tag
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if len(tag) > 0:
            tag += "_"
        log_filename = f"chip_{tag}{current_time}.log"
        
        # Add an archive dir
        archive_dir = "logs/archive"
        os.makedirs(archive_dir, exist_ok=True)

        # Update the file handler's filename
        config['handlers']['chiplog']['filename'] = f"{archive_dir}/{log_filename}"

        # Configure logging
        logging.config.dictConfig(config)
