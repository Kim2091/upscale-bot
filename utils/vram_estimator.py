import csv
import os
import torch
import configparser
import math
from .config_reader import read_config
import subprocess

# Read configuration
config = read_config()

VRAM_SAFETY_MULTIPLIER = float(config['Processing'].get('VRAMSafetyMultiplier', '1.2'))
AVAILABLE_VRAM_USAGE_FRACTION = float(config['Processing'].get('AvailableVRAMUsageFraction', '0.8'))
DEFAULT_TILE_SIZE = int(config['Processing'].get('DefaultTileSize', '384'))
MAX_TILE_SIZE = int(config['Processing'].get('MaxTileSize', '512'))
PRECISION = config['Processing'].get('Precision', 'auto').lower()

def load_vram_data(markdown_file):
    vram_data = {}
    current_scale = None

    with open(markdown_file, 'r') as f:
        content = f.read()

    for line in content.split('\n'):
        line = line.strip()
        
        if line.startswith("###"):
            current_scale = int(line.split()[1].rstrip("x"))
            continue
        
        if line.startswith("|Name|") or line.startswith("|:-") or not line:
            continue
        
        parts = line.split("|")
        if len(parts) < 5:
            continue
        
        model_name = parts[1].strip().lower()  # Convert to lowercase for case-insensitive matching
        vram_str = parts[4].strip().split()[0]
        
        try:
            vram = float(vram_str) if vram_str != '-' else None
        except ValueError:
            print(f"Warning: Invalid VRAM value '{vram_str}' for model '{model_name}'. Using None.")
            vram = None
        
        if model_name not in vram_data:
            vram_data[model_name] = {}
        
        vram_data[model_name][current_scale] = {
            "full_name": model_name,
            "vram": vram
        }

    return vram_data

def get_vram_data(precision, model):
    if precision == "auto":
        if model.supports_bfloat16:
            filename = "vram_data_bffp16.md"
        elif model.supports_half:
            filename = "vram_data_bffp16.md"
        else:
            filename = "vram_data_fp32.md"
    elif precision in ["fp16", "bf16"]:
        filename = "vram_data_bffp16.md"
    else:  # fp32
        filename = "vram_data_fp32.md"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)
    return load_vram_data(file_path)

def get_free_vram():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], 
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout.strip()
        free_vram = int(output) / 1024  # Convert MiB to GiB
        print(f"Available VRAM: {free_vram:.2f} GB")
        return free_vram
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return 0
    except ValueError as e:
        print(f"Error parsing nvidia-smi output: {e}")
        return 0

def estimate_vram_and_tile_size(model, input_size):
    model_name = model.architecture.name.lower()
    
    # Load the appropriate VRAM data based on precision
    vram_data_source = get_vram_data(PRECISION, model)
    
    scale = model.scale
    
    matched_data = next((data for name, data in vram_data_source.items() if name in model_name or model_name in name), None)
    
    free_vram = get_free_vram()
    
    if matched_data is None or scale not in matched_data or matched_data[scale]["vram"] is None:
        print(f"Warning: No matching VRAM data for {model_name} at scale {scale}. Using default values.")
        return free_vram * AVAILABLE_VRAM_USAGE_FRACTION, DEFAULT_TILE_SIZE

    vram = matched_data[scale]["vram"]

    base_size = 640 * 480
    size_factor = (input_size[0] * input_size[1]) / base_size
    
    # Model-specific adjustments
    if 'esrgan' in model_name:
        vram_scale_factor = 1
        size_exponent = 1.0
    elif 'atd' in model_name:
        vram_scale_factor = 1
        size_exponent = 0.92
    elif 'dat' in model_name:
        vram_scale_factor = 1
        size_exponent = 0.9
    elif any(name in model_name for name in ['hat', 'omnisr', 'swinir']):
        vram_scale_factor = 1
        size_exponent = 0.92
    else:
        vram_scale_factor = 1.1
        size_exponent = 0.85

    estimated_vram = vram * (size_factor ** size_exponent) * vram_scale_factor
    
    # Dynamic safety factor based on model complexity
    base_safety_factor = VRAM_SAFETY_MULTIPLIER
    complexity_factor = 1 + (vram / 20)
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

    print(f"Model: {model_name}")
    print(f"Scale: {scale}x")
    print(f"Precision: {PRECISION}")
    print(f"Estimated VRAM usage: {estimated_vram:.2f} GB")
    print(f"Free VRAM: {free_vram:.2f} GB")
    print(f"Safe VRAM usage: {safe_vram:.2f} GB")
    print(f"Calculated tile size: {tile_size}")

    return estimated_vram, tile_size
