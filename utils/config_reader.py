import configparser
import os

def read_config():
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.ini')
    config.read(config_path)
    return config

# Usage in your scripts
config = read_config()