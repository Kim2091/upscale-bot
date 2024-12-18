# Standard library imports
import asyncio
import configparser
import gc
import io
import sys
import os
import time
import traceback
from io import BytesIO
import logging
from logging.handlers import RotatingFileHandler
import psutil
import subprocess

# Third-party library imports
import aiohttp
import discord
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from discord.ext import commands
import spandrel
import spandrel_extra_arches

# Local module imports
from utils.alpha_handler import handle_alpha
from utils.fuzzy_model_matcher import find_closest_models, search_models
from utils.image_info import get_image_info, format_image_info
from utils.resize_module import resize_command
from utils.vram_estimator import estimate_vram_and_tile_size

# Setup logging
log_formatter = logging.Formatter('\033[38;2;118;118;118m%(asctime)s\033[0m - \033[38;2;59;120;255m%(levelname)s\033[0m - %(message)s')
log_handler = RotatingFileHandler('upscale_bot.log', maxBytes=10*1024*1024, backupCount=5)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))  # Plain format for file

logger = logging.getLogger('UpscaleBot')
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

# Add colored console output
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# Install extra architectures
spandrel_extra_arches.install()

# Configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Constants
TOKEN = config['Discord']['Token']
ADMIN_ID = config['Discord']['AdminId']
MODEL_PATH = config['Paths']['ModelPath']
MAX_TILE_SIZE = int(config['Processing']['MaxTileSize'])
PRECISION = config['Processing'].get('Precision', 'auto').lower()
MAX_OUTPUT_TOTAL_PIXELS = int(config['Processing']['MaxOutputTotalPixels'])
UPSCALE_TIMEOUT = int(config['Processing'].get('UpscaleTimeout', 60))
OTHER_STEP_TIMEOUT = int(config['Processing'].get('OtherStepTimeout', 30))
MAX_CONCURRENT_UPSCALES = int(config['Processing'].get('MaxConcurrentUpscales', 1))
DEFAULT_ALPHA_HANDLING = config['Processing'].get('DefaultAlphaHandling', 'resize').lower()
GAMMA_CORRECTION = config['Processing'].getboolean('GammaCorrection', False)
CLEANUP_INTERVAL = int(config['Processing'].get('CleanupInterval', 3)) * 60 * 60  # Convert hours to seconds

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True

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
`--info` - Allows you to view the details of a given image. Useful for DDS images to view the compression type

Use `--models` to see available models. """

# Utility classes
class StepLogger:
    def __init__(self):
        self.current_step = ""
        self.step_start_time = 0
        self.running = True

    def log_step(self, step_description):
        self.current_step = step_description
        self.step_start_time = time.time()
        logger.info(f"Step: {self.current_step}")

    def clear_step(self):
        self.current_step = ""
        self.step_start_time = 0

    async def monitor_progress(self):
        while self.running:
            try:
                await asyncio.sleep(10)
                if self.current_step:
                    elapsed_time = time.time() - self.step_start_time
                    logger.info(f"Still working on: {self.current_step} (Elapsed time: {elapsed_time:.2f}s)")
            except asyncio.CancelledError:
                self.running = False
                break

class UpscaleBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running = True
        self.tasks = []
        self.progress_logger = StepLogger()
        self.upscale_queue = asyncio.Queue()
        self.upscale_semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPSCALES)
        self.models = {}
        self.last_cleanup_time = time.time()

    async def setup_hook(self):
        """Called when the bot is starting up"""
        self.tasks.extend([
            self.loop.create_task(self.process_upscale_queue()),
            self.loop.create_task(self.cleanup_models()),
            self.loop.create_task(self.progress_logger.monitor_progress())
        ])

    async def close(self):
        """Called when the bot is shutting down"""
        self.running = False
        
        # Cancel tasks one by one with a limit on recursion
        for task in self.tasks:
            try:
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=2.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
            except Exception as e:
                logger.error(f"Error cancelling task: {e}")
        
        # Clear the task list
        self.tasks.clear()
        
        # Ensure parent close method is called
        try:
            await super().close()
        except Exception as e:
            logger.error(f"Error in parent close: {e}")

    async def process_upscale_queue(self):
        while self.running:
            try:
                upscale_task = await asyncio.wait_for(self.upscale_queue.get(), timeout=1.0)
                try:
                    async with self.upscale_semaphore:
                        await upscale_task
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("CUDA cache cleared and garbage collected after upscale task.")
                    self.upscale_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def cleanup_models(self):
        while self.running:
            try:
                await asyncio.sleep(60)
                current_time = time.time()
                if current_time - self.last_cleanup_time >= CLEANUP_INTERVAL:
                    logger.info("Performing periodic cleanup of unused models...")
                    self.models.clear()
                    torch.cuda.empty_cache()
                    gc.collect()
                    self.last_cleanup_time = current_time
                    logger.info("Cache cleanup completed. All models unloaded and memory freed.")
            except asyncio.CancelledError:
                break

# Initialize the bot
bot = UpscaleBot(command_prefix='--', intents=intents)
bot.remove_command('help')  # Remove the default help command

# Model management
def load_model(model_name):
    if model_name in bot.models:
        return bot.models[model_name]
    
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
            bot.models[model_name] = model.cuda().eval()
            logger.info(f"Loaded model: {model_name}")
            return bot.models[model_name]
        else:
            raise ValueError(f"Invalid model type for {model_name}")
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
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
        bot.progress_logger.log_step("Processing image in tiles")
        
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

        logger.info(f"Using precision mode: {autocast_dtype}")

        output_tensor = torch.zeros((1, img_tensor.shape[1], output_h, output_w), dtype=output_dtype, device='cuda')
        
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                bot.progress_logger.log_step(f"Processing tile at ({x}, {y})")
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
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info("Note: This bot is configured to work only in servers, not in DMs.")

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("Command not found. Use --upscale, --models, --resize, or --info")
    # Let other errors propagate up
    else:
        raise error

# Bot commands
@bot.command()
async def help(ctx):
    """Shows the help message"""
    await ctx.send(help_text)

@bot.command()
async def upscale(ctx, *args):
    status_msg = None
    selection_msg = None
    try:
        bot.progress_logger.log_step("Initializing upscale command")
        
        # Check if any arguments were provided
        if not args:
            await ctx.send(help_text)
            bot.progress_logger.clear_step()
            return
        
        # Parse arguments
        model_name = args[0]
        image_url = None
        alpha_handling = None

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
            bot.progress_logger.clear_step()
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
            bot.progress_logger.log_step("Reading attached image")
            try:
                async with asyncio.timeout(OTHER_STEP_TIMEOUT):
                    image_data = await attachment.read()
                    image = Image.open(BytesIO(image_data))
            except asyncio.TimeoutError:
                await ctx.send("Error: Image reading took too long and was cancelled.")
                return
        elif image_url:
            bot.progress_logger.log_step("Downloading image from URL")
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
        await bot.upscale_queue.put(process_upscale(ctx, model_name, image, status_msg, alpha_handling, has_alpha))

    except Exception as e:
        error_message = f"<@{ADMIN_ID}> Error! {str(e)}"
        await ctx.send(error_message)
        logger.error("Error in upscale command:")
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

async def process_upscale(ctx, model_name, image, status_msg, alpha_handling, has_alpha):
    try:
        start_time = time.time()
        
        # Helper function for safer status updates
        async def update_status(message):
            try:
                await status_msg.edit(content=message)
            except discord.HTTPException as e:
                logger.error(f"Failed to update status: {str(e)}")
            except discord.NotFound:
                logger.error("Status message was deleted")
                return False
            return True

        # Initial status update
        if not await update_status("Processing your image. This may take a while..."):
            return
            
        bot.progress_logger.log_step("Preparing model and estimating VRAM usage")
        
        model = load_model(model_name)

        input_size = (image.width, image.height)
        estimated_vram, adjusted_tile_size = estimate_vram_and_tile_size(model, input_size)

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
        
        if not await update_status(status_content):
            return

        logger.info(f"Starting upscale of image from {image_source}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Architecture: {model.architecture.name}")
        logger.info(f"Input size: {input_size}")
        logger.info(f"Estimated VRAM usage: {estimated_vram:.2f} GB")
        logger.info(f"Adjusted tile size: {adjusted_tile_size}")
        if has_alpha:
            logger.info(f"Alpha handling: {alpha_handling}")

        bot.progress_logger.log_step("Upscaling image")
        await status_msg.edit(content=f"{status_content}\nUpscaling...")

        # Create a cancellation event
        cancel_event = asyncio.Event()

        def upscale_func():
            def check_cancelled():
                return cancel_event.is_set()
            return upscale_image(image, model, adjusted_tile_size, alpha_handling, has_alpha, PRECISION, check_cancelled)

        # Run the upscale function in a separate thread
        upscale_task = asyncio.create_task(asyncio.to_thread(upscale_func))

        try:
            result = await asyncio.wait_for(upscale_task, timeout=UPSCALE_TIMEOUT)
        except asyncio.TimeoutError:
            cancel_event.set()
            bot.progress_logger.clear_step()
            logger.error("Upscale operation timed out and was cancelled.")
            await status_msg.edit(content="Error: Image processing took too long and was cancelled.")
            try:
                await upscale_task
            except asyncio.CancelledError:
                logger.info("Upscale task was successfully cancelled after timeout.")
            return

        upscale_time = time.time() - start_time
        bot.progress_logger.clear_step()
        await status_msg.edit(content=f"{status_content}\nUpscale completed in {upscale_time:.2f} seconds")

        bot.progress_logger.log_step("Saving and compressing image")
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

        bot.progress_logger.log_step("Saving upscaled image")
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
            logger.debug(f"PNG save failed: {str(e)}")
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
                logger.debug(f"WebP lossless save failed: {str(e)}")
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
                    logger.error(f"WebP lossy save failed: {str(e)}")
                    raise Exception("Unable to compress image to under 10 MB")

        compression_time = time.time() - compression_start_time
        file_size = output_buffer.tell() / (1024 * 1024)  # Convert to MB
        log_message = f"Image saved in {compression_time:.2f} seconds as {save_format}"
        if compression_info:
            log_message += f" with {compression_info}"
        log_message += f", size: {file_size:.2f} MB"
        logger.info(log_message)

        await status_msg.edit(content=f"{status_content}\nUpscale completed in {upscale_time:.2f} seconds\nCompressing completed in {compression_time:.2f} seconds\nUploading...")

        output_buffer.seek(0)

        bot.progress_logger.log_step("Uploading image")
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
        except asyncio.TimeoutError:
            await status_msg.edit(content="Error: Image upload took too long and was cancelled.")
            return

    except (torch.cuda.OutOfMemoryError, torch.cuda.CudaError, RuntimeError) as e:
        bot.progress_logger.clear_step()
        error_message = f"Critical CUDA error occurred. Restarting bot... Error: {str(e)}"
        logger.error(error_message)
        await status_msg.edit(content=error_message)
        
        try:
            # Get the current process
            current_process = psutil.Process()
            
            # Terminate child processes more safely
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # Brief wait for graceful termination
            psutil.wait_procs(children, timeout=3)
            
            # Force kill remaining processes
            for child in children:
                try:
                    if child.is_running():
                        child.kill()
                except psutil.NoSuchProcess:
                    pass
            
            # Clean shutdown of the bot
            try:
                await bot.close()
            except Exception as close_error:
                logger.error(f"Error during bot shutdown: {close_error}")
            
            # Start new process
            python = sys.executable
            script_path = os.path.abspath(sys.argv[0])
            subprocess.Popen([python, script_path])
            
            # Exit current process
            sys.exit(0)  # Use sys.exit instead of os._exit for more graceful shutdown
            
        except Exception as restart_error:
            error_message = f"Failed to restart bot after CUDA error: {restart_error}"
            logger.error(error_message)
            await status_msg.edit(content=f"<@{ADMIN_ID}> Restart failed. Details: {error_message}")

    except Exception as e:
        bot.progress_logger.clear_step()
        error_message = f"<@{ADMIN_ID}> Error! {str(e)}"
        await ctx.send(error_message)
        logger.error("Error in upscale command:", exc_info=True)    

    finally:
        bot.progress_logger.clear_step()
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Upscale cleanup completed, returned to idle state.")
        try:
            await status_msg.delete()
        except discord.HTTPException:
            pass

# Main execution
if __name__ == "__main__":
    bot.run(TOKEN)
