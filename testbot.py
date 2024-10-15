# WIP Test script. Requires upscale-bot.py to be renamed to upscale_bot.py. Not fully functional yet, model loading is broken & info command is missing

import asyncio
import pytest
import sys
import os
from unittest.mock import AsyncMock, Mock, patch
from PIL import Image
import io
import traceback
import aiohttp

# Set a fixed path to the model folder
FIXED_MODEL_PATH = r"H:\test"  # Update this path to your actual model folder path

# Override the MODEL_PATH in upscale_bot
import upscale_bot
upscale_bot.MODEL_PATH = FIXED_MODEL_PATH

from upscale_bot import bot, load_model, list_available_models, upscale_image, handle_alpha, resize_image, download_image, estimate_vram_and_tile_size, get_free_vram

print(f"Using fixed model path: {FIXED_MODEL_PATH}")

# List contents of the directory
print("Contents of the model directory:")
for file in os.listdir(FIXED_MODEL_PATH):
    print(f"  {file}")

# Set up pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

# Update the asyncio marker to use loop_scope
pytestmark = pytest.mark.asyncio(loop_scope="function")

# Debug function to print available models
def print_available_models():
    models = list_available_models()
    print("Available models:")
    for model in models:
        print(f"  {model}")
    return models

# Mock classes and helper functions
class MockContext:
    def __init__(self):
        self.message = Mock()
        self.send = AsyncMock()
        self.author = Mock()
        self.author.id = '123456789'  # Mocked user ID

class MockAttachment:
    def __init__(self, filename="test_image.png"):
        self.filename = filename
        self.url = "http://fake-url.com/image.png"

    async def read(self):
        return b"fake_image_data"

def create_mock_image(size=(100, 100), mode='RGB'):
    return Image.new(mode, size, color='red')

# Tests
@pytest.mark.parametrize("model_type", [".pth", ".safetensors"])
async def test_load_model(model_type):
    models = print_available_models()
    models_of_type = [m for m in models if m.endswith(model_type)]
    if not models_of_type:
        pytest.skip(f"No {model_type} models available to test")
    model_name = models_of_type[0]
    model = load_model(model_name)
    assert model is not None
    print(f"Successfully loaded {model_type} model: {model_name}")

@pytest.mark.parametrize("scale_algorithm", ['nearest', 'box', 'linear', 'hermite', 'hamming', 'hann', 'lanczos', 'catrom', 'mitchell', 'bspline', 'lagrange', 'gauss'])
async def test_resize_image(scale_algorithm):
    test_image = create_mock_image()
    result = resize_image(test_image, 2, scale_algorithm)
    assert isinstance(result, Image.Image)
    assert result.size[0] == test_image.size[0] * 2 and result.size[1] == test_image.size[1] * 2
    print(f"Successfully resized image using {scale_algorithm} algorithm")

async def test_handle_alpha():
    test_image = create_mock_image(mode='RGBA')
    
    def mock_upscale(img):
        return img.resize((200, 200))
    
    for method in ['upscale', 'resize', 'discard']:
        result = handle_alpha(test_image, mock_upscale, method, False)
        assert isinstance(result, Image.Image)
        assert result.size == (200, 200)
        if method != 'discard':
            assert result.mode == 'RGBA'
        else:
            assert result.mode == 'RGB'
        print(f"Successfully handled alpha channel with method: {method}")

@pytest.mark.parametrize("input_type", ["attachment", "link"])
async def test_upscale_command(input_type):
    ctx = MockContext()
    if input_type == "attachment":
        ctx.message.attachments = [MockAttachment()]
    else:
        ctx.message.attachments = []
    
    with patch('upscale_bot.Image.open', return_value=create_mock_image()), \
         patch.object(upscale_bot, 'upscale_image', return_value=create_mock_image(size=(200, 200))), \
         patch('upscale_bot.io.BytesIO', return_value=io.BytesIO(b"fake_upscaled_image")), \
         patch('upscale_bot.download_image', return_value=(create_mock_image(), None)), \
         patch('upscale_bot.estimate_vram_and_tile_size', return_value=(1.0, 512)), \
         patch('upscale_bot.get_free_vram', return_value=8.0), \
         patch('upscale_bot.find_closest_models', return_value=[('test_model', 100, 'exact')]):
        
        if input_type == "attachment":
            await bot.get_command('upscale')(ctx, "test_model")
        else:
            await bot.get_command('upscale')(ctx, "test_model", "http://fake-url.com/image.png")
    
    ctx.send.assert_called()
    print(f"Upscale command test passed for {input_type} input")

async def test_models_command():
    ctx = MockContext()
    
    with patch('upscale_bot.list_available_models', return_value=['model1.pth', 'model2.safetensors']):
        await bot.get_command('models')(ctx)
    
    ctx.send.assert_called()
    print("Models command test passed")

async def test_resize_command():
    ctx = MockContext()
    ctx.message.attachments = [MockAttachment()]
    
    with patch('upscale_bot.Image.open', return_value=create_mock_image()), \
         patch.object(upscale_bot, 'resize_image', return_value=create_mock_image(size=(200, 200))), \
         patch('upscale_bot.io.BytesIO', return_value=io.BytesIO(b"fake_resized_image")):
        
        await bot.get_command('resize')(ctx, "2", "bicubic")
    
    ctx.send.assert_called()
    print("Resize command test passed")

async def test_download_image():
    test_image_url = "https://github.com/Kim2091/PBRify_Upscaler/raw/main/Tutorial.png"
    
    try:
        image, error = await download_image(test_image_url)
        
        if image is None:
            print(f"Error returned: {error}")
            assert False, f"download_image failed: {error}"
        
        assert error is None, f"Unexpected error: {error}"
        assert isinstance(image, Image.Image), f"Expected PIL.Image.Image, got {type(image)}"
        
        print(f"Successfully downloaded image from {test_image_url}")
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
        print(f"Image format: {image.format}")
        
        print("Download image test passed")
    except Exception as e:
        print(f"Unexpected exception in test_download_image: {str(e)}")
        traceback.print_exc()
        assert False, f"Test failed due to unexpected exception: {str(e)}"

async def test_download_image_http_error():
    non_existent_url = "https://example.com/non_existent_image.jpg"
    try:
        image, error = await download_image(non_existent_url)

        assert image is None, f"Expected None, but got image: {image}"
        assert error == "Failed to download the image.", f"Expected 'Failed to download the image.', but got: {error}"
        print("Download image HTTP error test passed")
    except Exception as e:
        print(f"Unexpected exception in test_download_image_http_error: {str(e)}")
        traceback.print_exc()
        assert False, f"Test failed due to unexpected exception: {str(e)}"

async def test_download_image_invalid_format():
    invalid_format_url = "https://example.com/robots.txt"
    try:
        image, error = await download_image(invalid_format_url)

        assert image is None, f"Expected None, but got image: {image}"
        assert "Failed to download the image." in error, f"Expected error about failed download, but got: {error}"
        print("Download image invalid format test passed")
    except Exception as e:
        print(f"Unexpected exception in test_download_image_invalid_format: {str(e)}")
        traceback.print_exc()
        assert False, f"Test failed due to unexpected exception: {str(e)}"

async def test_estimate_vram_and_tile_size():
    mock_model = Mock()
    mock_model.scale = 2
    input_size = (1000, 1000)
    
    estimated_vram, tile_size = estimate_vram_and_tile_size(mock_model, input_size, {})
    
    assert isinstance(estimated_vram, float)
    assert isinstance(tile_size, int)
    print(f"Estimated VRAM: {estimated_vram} GB, Tile size: {tile_size}")
    print("VRAM and tile size estimation test passed")

async def test_get_free_vram():
    free_vram = get_free_vram()
    assert isinstance(free_vram, float)
    print(f"Free VRAM: {free_vram} GB")
    print("Get free VRAM test passed")

async def diagnose_download_image(url):
    print(f"Diagnosing download_image for URL: {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                print(f"HTTP Status: {resp.status}")
                content_type = resp.headers.get('Content-Type', 'Unknown')
                print(f"Content-Type: {content_type}")
                content_length = resp.headers.get('Content-Length', 'Unknown')
                print(f"Content-Length: {content_length}")
                
                if resp.status == 200:
                    data = await resp.read()
                    print(f"Successfully read {len(data)} bytes")
                    try:
                        image = Image.open(io.BytesIO(data))
                        print(f"Successfully opened image: format={image.format}, size={image.size}, mode={image.mode}")
                    except Exception as e:
                        print(f"Failed to open image: {str(e)}")
                else:
                    print(f"Failed to download image: HTTP {resp.status}")
    except Exception as e:
        print(f"Exception during diagnosis: {str(e)}")
        traceback.print_exc()

async def test_diagnose_download():
    await diagnose_download_image("https://example.com/robots.txt")

# Main test runner
if __name__ == "__main__":
    pytest.main([__file__])