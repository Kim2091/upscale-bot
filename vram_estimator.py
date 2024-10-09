import csv
import os
import torch
import configparser
import math

# Read configuration
config = configparser.ConfigParser()
config.read('config.ini')

VRAM_SAFETY_MULTIPLIER = float(config['Processing'].get('VRAMSafetyMultiplier', '1.2'))
AVAILABLE_VRAM_USAGE_FRACTION = float(config['Processing'].get('AvailableVRAMUsageFraction', '0.8'))
DEFAULT_TILE_SIZE = int(config['Processing'].get('DefaultTileSize', '384'))
MAX_TILE_SIZE = int(config['Processing'].get('MaxTileSize', '512'))

def load_vram_data(csv_file):
    vram_data = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['Name'].strip().lower()
            vram = float(row['VRAM'])
            fps = float(row['FPS'])
            sec_per_img = float(row['sec/img'])
            params = int(row['Params'])
            vram_data[name] = {'name': name, 'vram': vram, 'fps': fps, 'sec_per_img': sec_per_img, 'params': params}
    return vram_data

def get_free_vram():
    if torch.cuda.is_available():
        free_vram, total_vram = torch.cuda.mem_get_info()
        return free_vram / (1024 ** 3)  # Convert to GB
    return 0  # Return 0 if CUDA is not available

def estimate_vram_and_tile_size(model, input_size, vram_data):
    model_name = model.architecture.name.lower()
    matched_data = next((data for name, data in vram_data.items() if name in model_name or model_name in name), None)
    
    free_vram = get_free_vram()
    
    if matched_data is None:
        print(f"Warning: No matching data for {model_name}. Using default values.")
        return free_vram * AVAILABLE_VRAM_USAGE_FRACTION, DEFAULT_TILE_SIZE

    base_size = 640 * 480
    size_factor = (input_size[0] * input_size[1]) / base_size
    
    # Model-specific adjustments
    if 'atd' in model_name:
        vram_scale_factor = 1.4  # More conservative for ATD
        size_exponent = 0.95  # Closer to linear scaling
    elif 'dat' in model_name:
        vram_scale_factor = 1.2  # DAT2 needs a bit more conservative estimate
        size_exponent = 0.9
    elif any(name in model_name for name in ['hat', 'omnisr', 'swinir']):
        vram_scale_factor = 1.2  # More conservative for complex models
        size_exponent = 0.92
    else:
        vram_scale_factor = 1.1  # Default slightly more conservative
        size_exponent = 0.85  # Default sub-linear scaling

    estimated_vram = matched_data['vram'] * (size_factor ** size_exponent) * vram_scale_factor
    
    # Dynamic safety factor based on model complexity
    base_safety_factor = VRAM_SAFETY_MULTIPLIER
    complexity_factor = 1 + (matched_data['vram'] / 20)  # Increases with base VRAM usage
    safety_factor = base_safety_factor * complexity_factor
    estimated_vram *= safety_factor

    safe_vram = free_vram * AVAILABLE_VRAM_USAGE_FRACTION
    if estimated_vram <= safe_vram:
        tile_size = MAX_TILE_SIZE
    else:
        vram_ratio = (safe_vram / estimated_vram) ** 0.75
        tile_size = int(MAX_TILE_SIZE * vram_ratio)
    
    # Ensure tile size is even and within bounds
    tile_size = max(64, min(tile_size - (tile_size % 64), MAX_TILE_SIZE))

    return estimated_vram, tile_size

# Load VRAM data at module level
csv_file = os.path.join(os.path.dirname(__file__), 'vram_data.csv')
vram_data = load_vram_data(csv_file)

# Example usage and testing (can be commented out in production)
if __name__ == "__main__":
    class DummyModel:
        def __init__(self, name):
            self.architecture = type('obj', (object,), {'name': name})

    # Example models and input sizes
    models = [
        (DummyModel("realcugan"), (1920, 1080)),
        (DummyModel("esrgan"), (3840, 2160)),
        (DummyModel("atd"), (2560, 1440)),
        (DummyModel("dat2"), (1920, 1080)),
        (DummyModel("hat"), (3840, 2160)),
        (DummyModel("unknown_model"), (1280, 720))
    ]

    print(f"Current free VRAM: {get_free_vram():.2f} GB")

    for model, input_size in models:
        estimated_vram, tile_size = estimate_vram_and_tile_size(model, input_size, vram_data)
        print(f"\nModel: {model.architecture.name}")
        print(f"Input size: {input_size}")
        print(f"Estimated VRAM usage: {estimated_vram:.2f} GB")
        print(f"Calculated tile size: {tile_size}")