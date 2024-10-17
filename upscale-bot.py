# Standard library imports
import os
import io
import gc
import re
import time
import asyncio
import traceback
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import threading
import concurrent.futures
from functools import partial

# Third-party library imports
import torch
import numpy as np
import aiohttp
import discord
from discord.ext import commands
import configparser
from PIL import Image, UnidentifiedImageError
import spandrel
import spandrel_extra_arches

# Local module imports
from utils.vram_estimator import estimate_vram_and_tile_size, get_free_vram, vram_data
from utils.fuzzy_model_matcher import find_closest_models, search_models
from utils.alpha_handler import handle_alpha
from utils.resize_module import resize_image, resize_command
from utils.image_info import get_image_info, format_image_info

# Install extra architectures
spandrel_extra_arches.install()

# Configuration
config = configparser.ConfigParser()
config.read('config.ini')

TOKEN = config['Discord']['Token']
ADMIN_ID = config['Discord']['AdminId']
MODEL_PATH = config['Paths']['ModelPath']
MAX_TILE_SIZE = int(config['Processing']['MaxTileSize'])
PRECISION = config['Processing'].get('Precision', 'auto').lower()
MAX_OUTPUT_TOTAL_PIXELS = int(config['Processing']['MaxOutputTotalPixels'])
UPSCALE_TIMEOUT = int(config['Processing'].get('UpscaleTimeout', 60))
OTHER_STEP_TIMEOUT = int(config['Processing'].get('OtherStepTimeout', 30))
THREAD_POOL_WORKERS = int(config['Processing'].get('ThreadPoolWorkers', 1))
MAX_CONCURRENT_UPSCALES = int(config['Processing'].get('MaxConcurrentUpscales', 1))
DEFAULT_ALPHA_HANDLING = config['Processing'].get('DefaultAlphaHandling', 'resize').lower()
GAMMA_CORRECTION = config['Processing'].getboolean('GammaCorrection', False)

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='--', intents=intents)

# Thread pool and queue setup
thread_pool = ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS)
upscale_semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPSCALES)
upscale_queue = asyncio.Queue()

# Model management
models = {}
last_cleanup_time = time.time()
CLEANUP_INTERVAL = 3 * 60 * 60  # 3 hours in seconds

# Utility classes
class StepLogger:
    def __init__(self):
        self.current_step = ""
        self.step_start_time = 0

    def log_step(self, step_description):
        self.current_step = step_description
        self.step_start_time = time.time()
        print(f"Step: {self.current_step}")

    async def monitor_progress(self):
        while True:
            await asyncio.sleep(10)
            if self.current_step and self.current_step != "Idle":
                elapsed_time = time.time() - self.step_start_time
                print(f"Still working on: {self.current_step} (Elapsed time: {elapsed_time:.2f}s)")

step_logger = StepLogger()

# Sanitize error messages
def sanitize_error_message(error_message):
    # Convert to string if it's not already
    error_message = str(error_message)
    
    # Remove file paths
    sanitized = re.sub(r'File ".*?"', 'File "[PATH REMOVED]"', error_message)
    
    return sanitized

def load_model(model_name):
    if model_name in models:
        return models[model_name]
    
    # Check for both .pth and .safetensors files
    pth_path = os.path.join(MODEL_PATH, f"{model_name}.pth")
    safetensors_path = os.path.join(MODEL_PATH, f"{model_name}.safetensors")
    
    if os.path.exists(pth_path):
        model_path = pth_path
    elif os.path.exists(safetensors_path):
        model_path = safetensors_path
    else:
        raise ValueError(f"Model file not found: {model_name}")
    
    try:
        model = spandrel.ModelLoader().load_from_file(model_path)
        if isinstance(model, spandrel.ImageModelDescriptor):
            models[model_name] = model.cuda().eval()
            print(f"Loaded model: {model_name}")
            return models[model_name]
        else:
            raise ValueError(f"Invalid model type for {model_name}")
    except Exception as e:
        print(f"Failed to load model {model_name}: {str(e)}")
        raise

def list_available_models():
    return [os.path.splitext(f)[0] for f in os.listdir(MODEL_PATH) if f.endswith(('.pth', '.safetensors'))]

# Image processing functions
async def download_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                return None, "Failed to download the image."
            data = await resp.read()
            
            try:
                image = Image.open(BytesIO(data))
                
                if image.format.lower() not in ['jpeg', 'png', 'gif', 'webp']:
                    return None, "The URL does not point to a supported image format. Supported formats are JPEG, PNG, GIF, and WebP."
                
                return image, None
            except UnidentifiedImageError:
                return None, "The URL does not point to a valid image file."
            except Exception as e:
                return None, f"Error processing the image: {str(e)}"

def upscale_image(image, model, tile_size, alpha_handling, has_alpha, precision, check_cancelled):
    def upscale_func(img):
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).cuda()
        _, _, h, w = img_tensor.shape
        output_h, output_w = h * model.scale, w * model.scale
        step_logger.log_step("Processing image in tiles")
        
        # Determine the output dtype and inference mode based on model capabilities and PRECISION setting
        if model.supports_bfloat16 and PRECISION in ['auto', 'bf16']:
            output_dtype = torch.bfloat16
            autocast_dtype = torch.bfloat16
        elif model.supports_half and PRECISION in ['auto', 'fp16']:
            output_dtype = torch.float16
            autocast_dtype = torch.float16
        else:
            output_dtype = torch.float32
            autocast_dtype = None

        print(f"Using precision mode: {autocast_dtype}")

        output_tensor = torch.zeros((1, img_tensor.shape[1], output_h, output_w), dtype=output_dtype, device='cuda')
        
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                step_logger.log_step(f"Processing tile at ({x}, {y})")
                if check_cancelled():
                    raise asyncio.CancelledError("Upscale operation was cancelled")
               
                tile = img_tensor[:, :, y:min(y+tile_size, h), x:min(x+tile_size, w)]
                with torch.inference_mode():
                    if autocast_dtype:
                        with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                            upscaled_tile = model(tile)
                    else:
                        upscaled_tile = model(tile)
                output_tensor[:, :, y*model.scale:min((y+tile_size)*model.scale, output_h),
                              x*model.scale:min((x+tile_size)*model.scale, output_w)].copy_(upscaled_tile)
        
        return Image.fromarray((output_tensor[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8))

    # Use the alpha_handler to process the image if it has an alpha channel
    if has_alpha:
        return handle_alpha(image, upscale_func, alpha_handling, GAMMA_CORRECTION)
    else:
        return upscale_func(image)
# Bot event handlers
@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    bot.loop.create_task(process_upscale_queue())
    bot.loop.create_task(cleanup_models())  # Start the periodic cleanup task

# Bot commands
@bot.command()
async def upscale(ctx, *args):
    status_msg = None
    selection_msg = None
    try:
        step_logger.log_step("Initializing upscale command")
        
        help_text = """ To use the upscale command:
1. Attach an image and type: 
   `--upscale <model_name> [alpha_handling]`
2. Provide an image URL: 
   `--upscale <model_name> [alpha_handling] <image_url>`

Example:
`--upscale RealESRGAN_x4plus resize https://example.com/image.jpg`

Alpha handling options: `upscale`, `resize`, `discard`
If not specified, the default from the config will be used.

Available commands:
`--upscale <model_name> [alpha_handling] [image_url]` - Upscale an image using the specified model
`--models` - List all available upscaling models
`--resize <scale_factor> <method>` - Allows you to resize images up or down using normal scaling methods (e.g. bicubic, lanczos)

Use `--models` to see available models. """

        # Parse arguments
        model_name = None
        image_url = None
        alpha_handling = None

        if args and args[0].lower() == 'downscale':
            await downscale_command(ctx, args[1:], download_image, GAMMA_CORRECTION)
            return

        if len(args) >= 1:
            model_name = args[0]
        if len(args) >= 2:
            if args[1] in ['upscale', 'resize', 'discard']:
                alpha_handling = args[1]
                if len(args) >= 3:
                    image_url = args[2]
            elif args[1].startswith('http'):
                image_url = args[1]
            else:
                await ctx.send(f"Invalid alpha handling option or image URL: {args[1]}. Using default alpha handling.")
        if len(args) >= 3 and not image_url:
            image_url = args[2]
        
        if model_name is None:
            await ctx.send(help_text)
            return

        alpha_handling = alpha_handling if alpha_handling else DEFAULT_ALPHA_HANDLING
        if alpha_handling not in ['upscale', 'resize', 'discard']:
            await ctx.send(f"Invalid alpha handling option: {alpha_handling}. Using default: {DEFAULT_ALPHA_HANDLING}")
            alpha_handling = DEFAULT_ALPHA_HANDLING

        # Model selection and validation
        available_models = list_available_models()
        if model_name not in available_models:
            closest_matches = find_closest_models(model_name, available_models)
            if closest_matches:
                if len(closest_matches) == 1 or (closest_matches[0][1] - closest_matches[1][1] > 5):
                    best_match, similarity, match_type = closest_matches[0]
                    model_name = best_match
                    await ctx.send(f"Using model: {model_name} (similarity: {similarity}%, match type: {match_type})")
                else:
                    match_message = f"Model '{model_name}' not found. Multiple close matches found.\n\nPlease select a number:"
                    for i, (match, similarity, match_type) in enumerate(closest_matches, 1):
                        match_message += f"\n{i}. {match} (similarity: {similarity}%, match type: {match_type})"
                    match_message += "\n\nOr type 'cancel' to abort."
                    
                    selection_msg = await ctx.send(match_message)
                    
                    def check(m):
                        return m.author == ctx.author and m.channel == ctx.channel and (m.content.isdigit() or m.content.lower() == 'cancel')
                    
                    try:
                        reply = await bot.wait_for('message', check=check, timeout=30.0)
                        if reply.content.lower() == 'cancel':
                            await ctx.send("Upscale operation cancelled.")
                            return
                        selection = int(reply.content)
                        if 1 <= selection <= len(closest_matches):
                            model_name = closest_matches[selection-1][0]
                            await ctx.send(f"Selected model: {model_name}")
                        else:
                            await ctx.send("Invalid selection. Upscale operation cancelled.")
                            return
                    except asyncio.TimeoutError:
                        await ctx.send("Selection timed out. Upscale operation cancelled.")
                        return
            else:
                await ctx.send(f"Model '{model_name}' not found and no close matches. Use --models to see available models.")
                return

        # Load the model descriptor
        model_descriptor = load_model(model_name)

        # Image acquisition (from attachment or URL)
        if len(ctx.message.attachments) > 0:
            attachment = ctx.message.attachments[0]
            if not attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                await ctx.send("Please upload a valid image file (PNG, JPG, JPEG, or WebP).")
                return
            step_logger.log_step("Reading attached image")
            try:
                async with asyncio.timeout(OTHER_STEP_TIMEOUT):
                    image_data = await attachment.read()
                    image = Image.open(BytesIO(image_data))
            except asyncio.TimeoutError:
                await ctx.send("Error: Image reading took too long and was cancelled.")
                return
        elif image_url:
            step_logger.log_step("Downloading image from URL")
            try:
                async with asyncio.timeout(OTHER_STEP_TIMEOUT):
                    image, error_message = await download_image(image_url)
                if image is None:
                    await ctx.send(f"Error: {error_message} Please try uploading the image directly to Discord.")
                    return
            except asyncio.TimeoutError:
                await ctx.send("Error: Image download took too long and was cancelled.")
                return
        else:
            await ctx.send("Please either attach an image or provide a valid image URL.")
            return

        # Calculate the output image size
        input_width, input_height = image.size
        scale = model_descriptor.scale
        output_width = input_width * scale
        output_height = input_height * scale
        output_total_pixels = output_width * output_height

        # Check if the output size exceeds the limit
        if output_total_pixels > MAX_OUTPUT_TOTAL_PIXELS:
            max_megapixels = MAX_OUTPUT_TOTAL_PIXELS / (1024 * 1024)
            await ctx.send(f"Error: The output image size ({output_width}x{output_height}, {output_total_pixels / (1024 * 1024):.2f} megapixels) would exceed the maximum allowed total of {max_megapixels:.2f} megapixels ({MAX_OUTPUT_TOTAL_PIXELS:,} pixels).")
            return

        # Check if the image has an alpha channel
        has_alpha = image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info)

        # Queue the upscale operation
        status_msg = await ctx.send("Your upscale request has been queued.")
        
        # Add the task to the queue
        await upscale_queue.put(process_upscale(ctx, model_name, image, status_msg, alpha_handling, has_alpha))

    except Exception as e:
        error_message = f"<@{ADMIN_ID}> Error! {sanitize_error_message(str(e))}"
        await ctx.send(error_message)
        print(f"Error in upscale command:")
        traceback.print_exc()
    finally:
        # Ensure we clean up the selection message if it exists
        if selection_msg:
            await selection_msg.delete()
        
@bot.command(aliases=['scale'])
async def resize(ctx, *args):
    await resize_command(ctx, args, download_image, GAMMA_CORRECTION)

@bot.command(name='models')
async def list_models(ctx, search_term: str = None):
    available_models = list_available_models()
    if not available_models:
        await ctx.send("No models are currently available.")
        return

    if search_term:
        matches = search_models(search_term, available_models)
        if matches:
            match_list = "\n".join(f"{match[0]} (similarity: {match[1]}%)" for match in matches)
            await ctx.send(f"Models matching '{search_term}':\n```\n{match_list}\n```")
        else:
            await ctx.send(f"No models found matching '{search_term}'.")
        return

    # Sort the models alphabetically
    available_models.sort()
    
    # Calculate the maximum number of models per message
    max_models_per_message = 50  # Adjust this number as needed
    
    # Split the models into chunks
    model_chunks = [available_models[i:i + max_models_per_message] 
                    for i in range(0, len(available_models), max_models_per_message)]
    
    for i, chunk in enumerate(model_chunks, 1):
        model_list = "\n".join(chunk)
        message = f"Available models (Page {i}/{len(model_chunks)}):\n```\n{model_list}\n```"
        await ctx.send(message)
    
    # If there are multiple pages, send a summary message
    if len(model_chunks) > 1:
        await ctx.send(f"Total number of available models: {len(available_models)}")

@bot.command()
async def info(ctx, *args):
    try:
        # Check if an image is attached
        if ctx.message.attachments:
            attachment = ctx.message.attachments[0]
            # Download the attachment
            await attachment.save(attachment.filename)
            file_path = attachment.filename
        elif args:
            # If no attachment, check if a URL was provided
            image_url = args[0]
            file_path = 'temp_image'
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status == 200:
                        with open(file_path, 'wb') as f:
                            f.write(await resp.read())
                    else:
                        await ctx.send("Failed to download the image.")
                        return
        else:
            await ctx.send("Please attach an image or provide an image URL.")
            return

        # Get and format the image info
        image_info = get_image_info(file_path)
        formatted_info = format_image_info(image_info)

        # Send the formatted info
        await ctx.send(f"```\n{formatted_info}\n```")

    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")
    finally:
        # Clean up the temporary file if it was created
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)

# Queue processing
async def process_upscale_queue():
    while True:
        upscale_task = await upscale_queue.get()
        try:
            async with upscale_semaphore:
                await upscale_task
        finally:
            upscale_queue.task_done()

async def process_upscale(ctx, model_name, image, status_msg, alpha_handling, has_alpha):
    try:
        await status_msg.edit(content="Processing your image. This may take a while...")

        step_logger.log_step("Preparing model and estimating VRAM usage")
        model = load_model(model_name)

        input_size = (image.width, image.height)
        estimated_vram, adjusted_tile_size = estimate_vram_and_tile_size(model, input_size, vram_data)

        # Get the original filename
        if ctx.message.attachments:
            original_filename = ctx.message.attachments[0].filename
            image_source = f"attachment: {original_filename}"
        else:
            original_filename = "image.png"
            image_source = "provided URL"

        filename_parts = os.path.splitext(original_filename)

        status_content = (
            f"Processing image from {image_source}\n"
            f"Model: {model_name}\n"
            f"Architecture: {model.architecture.name}"
        )
        if has_alpha:
            status_content += f"\nAlpha handling: {alpha_handling}"
        
        await status_msg.edit(content=status_content)

        print(f"Starting upscale of image from {image_source}")
        print(f"Model: {model_name}")
        print(f"Architecture: {model.architecture.name}")
        print(f"Input size: {input_size}")
        print(f"Estimated VRAM usage: {estimated_vram:.2f} GB")
        print(f"Adjusted tile size: {adjusted_tile_size}")
        if has_alpha:
            print(f"Alpha handling: {alpha_handling}")

        start_time = time.time()

        step_logger.log_step("Upscaling image")
        await status_msg.edit(content=f"{status_content}\nUpscaling...")

        # Create a cancellation event
        cancel_event = asyncio.Event()

        # Create the upscale function
        def upscale_func():
            def check_cancelled():
                return cancel_event.is_set()
            
            if check_cancelled():
                raise asyncio.CancelledError("Upscale cancelled")
            return upscale_image(image, model, adjusted_tile_size, alpha_handling, has_alpha, PRECISION, check_cancelled)


        # Run the upscale function in a separate thread
        upscale_task = asyncio.create_task(asyncio.to_thread(upscale_func))

        # Wait for the upscale task to complete or timeout
        try:
            result = await asyncio.wait_for(upscale_task, timeout=UPSCALE_TIMEOUT)
        except asyncio.TimeoutError:
            cancel_event.set()
            print("Upscale operation timed out and was cancelled.")
            await status_msg.edit(content="Error: Image processing took too long and was cancelled.")
            try:
                await upscale_task
            except asyncio.CancelledError:
                print("Upscale task was successfully cancelled after timeout.")
            return
        except asyncio.CancelledError:
            print("Upscale operation was cancelled.")
            await status_msg.edit(content="Error: Image processing was cancelled.")
            return

        upscale_time = time.time() - start_time
        await status_msg.edit(content=f"{status_content}\nUpscale completed in {upscale_time:.2f} seconds")
        print(f"Upscale completed in {upscale_time:.2f} seconds")

        # Add downscaling option within upscale result
        if result.width * result.height > MAX_OUTPUT_TOTAL_PIXELS:
            scale_factor = (MAX_OUTPUT_TOTAL_PIXELS / (result.width * result.height)) ** 0.5
            await ctx.send(f"The upscaled image exceeds the maximum allowed size. Automatically downscaling with factor {scale_factor:.2f}.")
            await process_downscale(ctx, args, result, scale_factor, 'lanczos', GAMMA_CORRECTION)
            return

        def estimate_file_size(img, format, **params):
            temp_buffer = io.BytesIO()
            img.save(temp_buffer, format=format, **params)
            return temp_buffer.tell()
        def find_webp_quality(img, max_size):
            low, high = 1, 100
            while low <= high:
                mid = (low + high) // 2
                size = estimate_file_size(img, 'WEBP', quality=mid)
                if size < max_size:
                    low = mid + 1
                else:
                    high = mid - 1
            return high

        step_logger.log_step("Saving upscaled image")
        await status_msg.edit(content=f"{status_content}\nUpscale completed in {upscale_time:.2f} seconds\nCompressing...")
        output_buffer = io.BytesIO()
        save_format = 'WEBP (lossless)'
        new_filename = f"{filename_parts[0]}_upscaled.webp"
        max_file_size = 10 * 1024 * 1024  # 10 MB in bytes
        compression_info = None

        compression_start_time = time.time()
        try:
            async with asyncio.timeout(OTHER_STEP_TIMEOUT):
                # Try PNG first
                await asyncio.to_thread(lambda: result.save(output_buffer, 'PNG'))
                if output_buffer.tell() > max_file_size:
                    raise Exception("PNG file size too large")
                save_format = 'PNG'
                new_filename = f"{filename_parts[0]}_upscaled.png"
        except Exception as e:
            print(f"PNG save failed: {str(e)}")
            output_buffer.seek(0)
            output_buffer.truncate(0)
            
            try:
                # Try WebP lossless
                await asyncio.to_thread(lambda: result.save(output_buffer, 'WEBP', lossless=True))
                if output_buffer.tell() > max_file_size:
                    raise Exception("WebP lossless file size too large")
                save_format = 'WEBP (lossless)'
                new_filename = f"{filename_parts[0]}_upscaled.webp"
            except Exception as e:
                print(f"WebP lossless save failed: {str(e)}")
                output_buffer.seek(0)
                output_buffer.truncate(0)
                
                try:
                    # Try WebP lossy
                    webp_quality = await asyncio.to_thread(find_webp_quality, result, max_file_size)
                    await asyncio.to_thread(lambda: result.save(output_buffer, 'WEBP', quality=webp_quality))
                    save_format = 'WEBP'
                    new_filename = f"{filename_parts[0]}_upscaled.webp"
                    compression_info = f"lossy (quality {webp_quality})"
                except Exception as e:
                    print(f"WebP lossy save failed: {str(e)}")
                    raise Exception("Unable to compress image to under 10 MB")

        compression_time = time.time() - compression_start_time
        file_size = output_buffer.tell() / (1024 * 1024)  # Convert to MB
        log_message = f"Image saved in {compression_time:.2f} seconds as {save_format}"
        if compression_info:
            log_message += f" with {compression_info}"
        log_message += f", size: {file_size:.2f} MB"
        print(log_message)

        await status_msg.edit(content=f"{status_content}\nUpscale completed in {upscale_time:.2f} seconds\nCompressing completed in {compression_time:.2f} seconds\nUploading...")

        output_buffer.seek(0)

        step_logger.log_step("Uploading upscaled image")
        try:
            async with asyncio.timeout(OTHER_STEP_TIMEOUT):
                message = f"<@{ctx.author.id}> Here's your image upscaled with `{model_name}`"
                if has_alpha:
                    message += f" and alpha method `{alpha_handling}`"
                if compression_info:
                    message += f"\nNote: The image was saved as {save_format} with {compression_info} due to size limitations."
                await ctx.send(message, file=discord.File(fp=output_buffer, filename=new_filename))
                
                final_status = f"{status_content}\nUpscale completed in {upscale_time:.2f} seconds\nCompressing completed in {compression_time:.2f} seconds\nUpload successful!"
                await status_msg.edit(content=final_status)
                
                await asyncio.sleep(5)
                await status_msg.delete()
        except discord.errors.HTTPException as e:
            if e.code == 40005:  # Payload Too Large
                await status_msg.edit(content=f"<@{ctx.author.id}> The upscaled image is too large to upload, even after compression. You may need to upscale a smaller image or use a model with a lower upscaling factor.")
            else:
                raise
        except asyncio.TimeoutError:
            await status_msg.edit(content="Error: Image upload took too long and was cancelled.")
            return

        upload_time = time.time() - start_time - upscale_time - compression_time
        total_time = time.time() - start_time
        print(f"Image uploaded in {upload_time:.2f} seconds")
        print(f"Total processing time: {total_time:.2f} seconds")

    except torch.cuda.OutOfMemoryError as e:
        error_message = f"CUDA out of memory error occurred. Please try a smaller image or a different model."
        print(f"CUDA out of memory error in upscale processing: {str(e)}")
        await status_msg.edit(content=error_message)
    except Exception as e:
        error_message = f"<@{ADMIN_ID}> Error processing upscale! {sanitize_error_message(str(e))}"
        print(f"Error in upscale processing: {str(e)}")
        await status_msg.edit(content=error_message)
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()
        gc.collect()
        print("Upscale cleanup completed, returned to idle state.")

# Cleanup task
async def cleanup_models():
    global models, last_cleanup_time
    while True:
        await asyncio.sleep(60)  # Check every minute
        current_time = time.time()
        if current_time - last_cleanup_time >= CLEANUP_INTERVAL:
            print("Performing periodic cleanup of unused models...")
            models.clear()
            torch.cuda.empty_cache()
            gc.collect()
            last_cleanup_time = current_time
            print("Cache cleanup completed. All models unloaded and memory freed.")

# Main execution
if __name__ == "__main__":
    bot.run(TOKEN)
