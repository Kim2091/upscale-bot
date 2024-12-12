import configparser
import os

_config = None  # Private variable to store singleton instance

def read_config():
    global _config
    if _config is None:
        _config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.ini')
        _config.read(config_path)
    return _config

# Usage in your scripts
config = read_config()
