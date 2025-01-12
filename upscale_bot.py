# Standard library imports
import asyncio
import configparser
import gc
import os
import time
from io import BytesIO
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Literal

# Third-party library imports
import aiohttp
import discord
import torch
from PIL import Image, UnidentifiedImageError
from discord.ext import commands
import spandrel
import spandrel_extra_arches
from discord import app_commands
import numpy as np

# Local module imports
from utils.alpha_handler import handle_alpha
from utils.fuzzy_model_matcher import find_closest_models, search_models
from utils.image_info import get_image_info, format_image_info
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

DM_ALLOWED_USERS = set()
if 'Permissions' in config and 'DMAllowedUsers' in config['Permissions']:
    # Split by commas and convert to set of integers
    DM_ALLOWED_USERS = set(int(uid.strip()) for uid in config['Permissions']['DMAllowedUsers'].split(',') if uid.strip())

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True

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

# Modify the UpscaleBot class to include tree support
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
        self.model_path = MODEL_PATH
        self.default_alpha_handling = DEFAULT_ALPHA_HANDLING

    def list_available_models(self, search_term=None):
        """Returns a list of available model names"""
        models = [os.path.splitext(f)[0] for f in os.listdir(self.model_path) 
                 if f.endswith(('.pth', '.safetensors'))]
        if search_term:
            return [m for m in models if search_term.lower() in m.lower()]
        return sorted(models)

    async def setup_hook(self):
        """Called when the bot is starting up"""
        logger.info("Setting up bot...")
        
        # Register slash commands
        logger.info("Registering slash commands...")
        
        try:
            # For global commands
            logger.info("Syncing global commands...")
            try:
                await self.tree.sync()  # Attempt to sync globally
                logger.info("Global commands synced successfully.")
            except Exception as e:
                logger.error(f"Failed to sync global commands: {e}")
            
            # Log all registered commands
            commands = await self.tree.fetch_commands()
            logger.info(f"Registered commands: {[cmd.name for cmd in commands]}")
            
        except Exception as e:
            logger.error(f"Failed to sync command tree: {e}", exc_info=True)

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
            
    async def process_upscale(self, interaction, model_name, image_source, source_type, status_msg, alpha_handling):
        try:
            start_time = time.time()
            
            # Get the image
            if source_type == "attachment":
                image_data = await image_source.read()
                image = Image.open(BytesIO(image_data))
                image_source_desc = f"attachment: {image_source.filename}"
            else:  # url
                image, error = await download_image(image_source)
                if error:
                    await status_msg.edit(content=f"Error: {error}")
                    return
                image_source_desc = "provided URL"

            # Load model and check input channels
            logger.info(f"Loading model: {model_name}")
            model = load_model(model_name)
            if model.input_channels == 4:
                await status_msg.edit(content="4 channel models are not supported, please pick another model.")
                return

            # Calculate output size
            input_width, input_height = image.size
            scale = model.scale
            output_width = input_width * scale
            output_height = input_height * scale
            output_total_pixels = output_width * output_height

            logger.info(f"Input size: {input_width}x{input_height}")
            logger.info(f"Output size: {output_width}x{output_height}")

            # Check if output size is reasonable
            if output_total_pixels > MAX_OUTPUT_TOTAL_PIXELS:
                max_megapixels = MAX_OUTPUT_TOTAL_PIXELS / (1024 * 1024)
                await status_msg.edit(content=f"Error: Output size would exceed maximum allowed total of {max_megapixels:.2f} megapixels.")
                return

            # Check if image has alpha channel
            has_alpha = image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info)

            # Estimate VRAM usage and get tile size
            estimated_vram, adjusted_tile_size = estimate_vram_and_tile_size(model, (input_width, input_height))

            # Initial status message
            status_content = (
                f"Processing image from {image_source_desc}\n"
                f"Model: {model_name}\n"
                f"Architecture: {model.architecture.name}\n"
                f"Input size: {input_width}x{input_height}\n"
                f"Output size: {output_width}x{output_height}\n"
                f"Estimated VRAM usage: {estimated_vram:.2f}GB\n"
                f"Using tile size: {adjusted_tile_size}"
            )
            if has_alpha:
                status_content += f"\nAlpha handling: {alpha_handling}"
            await status_msg.edit(content=status_content + "\nStarting upscale process...")

            # Process the image
            def check_cancelled():
                return False

            self.progress_logger.log_step("Starting upscale process")
            result = self.upscale_image(image, model, adjusted_tile_size, alpha_handling, has_alpha, PRECISION, check_cancelled)
            
            # Calculate upscale time before saving
            upscale_time = time.time() - start_time
            logger.info(f"Upscale completed in {upscale_time:.2f} seconds")
            await status_msg.edit(content=status_content + f"\nUpscale completed in {upscale_time:.2f} seconds\nSaving image...")

            # Save the result
            self.progress_logger.log_step("Saving upscaled image")
            logger.info("Saving upscaled image")
            output_buffer = BytesIO()
            result.save(output_buffer, format='PNG')
            output_buffer.seek(0)
            logger.info("Image saved successfully")
            await status_msg.edit(content=status_content + f"\nUpscale completed in {upscale_time:.2f} seconds\nImage saved, sending to Discord...")

            # Send the result
            self.progress_logger.log_step("Sending upscaled image")
            logger.info("Sending upscaled image")
            message = f"<@{interaction.user.id}> Here's your image upscaled with `{model_name}`"
            if has_alpha:
                message += f" and alpha method `{alpha_handling}`"
            message += f"\nProcessing time: {upscale_time:.2f} seconds"
            
            await interaction.followup.send(message, file=discord.File(fp=output_buffer, filename=f"upscaled_{model_name}.png"))
            logger.info("Image sent successfully")
            await status_msg.edit(content=status_content + f"\nUpscale completed in {upscale_time:.2f} seconds\nImage sent successfully!")

            # Clean up after 5 seconds
            await asyncio.sleep(5)
            await status_msg.delete()
            self.progress_logger.clear_step()
            logger.info("Upscale process completed successfully")

        except Exception as e:
            error_msg = f"Error during upscale: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await status_msg.edit(content=error_msg)

    def upscale_image(self, image, model, tile_size, alpha_handling, has_alpha, precision, check_cancelled):
        def upscale_func(img):
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).cuda()
            _, _, h, w = img_tensor.shape
            output_h, output_w = h * model.scale, w * model.scale
            self.progress_logger.log_step("Processing image in tiles")
            
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
                    self.progress_logger.log_step(f"Processing tile at ({x}, {y})")
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

# Define autocomplete before the command that uses it
async def model_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:
    """Autocomplete for model names"""
    models = bot.list_available_models()
    if current:
        matches = find_closest_models(current, models)
        model_names = [match[0] for match in matches[:25]]
    else:
        model_names = models[:25]  # Discord limits to 25 choices
    return [app_commands.Choice(name=model, value=model) for model in model_names]

# Initialize bot first
bot = UpscaleBot(command_prefix=None, intents=intents)

# Then define sync_global command
@bot.tree.command(description="Syncs your slash commands to the Discord API globally.")
async def sync_global(interaction: discord.Interaction) -> None:
    # Check if the user is the admin
    if interaction.user.id != int(ADMIN_ID):
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)
        return

    await interaction.response.send_message("Syncing global commands...")
    try:
        await bot.tree.sync()
        logger.info("Slash commands synced successfully.")
        await interaction.followup.send("Slash commands synced successfully!")  # Send follow-up message
    except Exception as e:
        logger.error(f"Failed to sync global commands: {e}")
        await interaction.followup.send("Failed to sync global commands. Please check the logs for more details.")

# Define sync command for the current guild
@bot.tree.command(description="Syncs commands to the current guild.")
async def sync(interaction: discord.Interaction) -> None:
    logger.info(f"Sync command called by user: {interaction.user.id}")

    # Check if the user is the admin
    if interaction.user.id != int(ADMIN_ID):
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)
        logger.warning(f"User {interaction.user.id} attempted to use sync command without permission.")
        return

    # Acknowledge the interaction
    await interaction.response.defer(thinking=True)

    guild_id = interaction.guild.id  # Get the current guild ID
    test_guild = discord.Object(id=guild_id)  # Create a guild object for the current guild
    logger.info(f"Clearing commands for guild: {test_guild.id}")

    # Clear existing commands for the guild
    bot.tree.clear_commands(guild=test_guild)

    logger.info(f"Copying global commands to guild: {test_guild.id}")
    
    # Copy global commands to the guild
    bot.tree.copy_global_to(guild=test_guild)
    logger.info("Syncing commands to guild...")
    
    try:
        await bot.tree.sync(guild=test_guild)
        logger.info("Commands synced successfully to guild.")
        await interaction.followup.send("Commands synced successfully to this guild!")  # Send follow-up message
    except Exception as e:
        logger.error(f"Failed to sync commands to guild: {e}")
        await interaction.followup.send("Failed to sync commands to this guild. Please check the logs for more details.")

def can_use_dm(ctx):
    """Check if a user can use the bot in DMs"""
    if ctx.guild is not None:  # If in a guild, always allow
        return True
    return ctx.author.id in DM_ALLOWED_USERS  # In DM, check if user is allowed

@bot.tree.command(name="help", description="Show help information for the bot")
async def help_slash(interaction: discord.Interaction):
    """Send help information for the bot."""
    help_text = (
        "**Available Commands:**\n\n"
        "1. **/upscale**\n"
        "   - **Description**: Upscale an image using a specified model.\n"
        "   - **Parameters**:\n"
        "     - `model`: The upscaling model to use (required).\n"
        "     - `image`: The image file to upscale (optional).\n"
        "     - `url`: URL of the image to upscale (optional).\n"
        "     - `alpha_handling`: How to handle alpha/transparency (options: `resize`, `upscale`, `discard`).\n\n"
        
        "2. **/models**\n"
        "   - **Description**: List available upscaling models.\n"
        "   - **Parameters**:\n"
        "     - `search_term`: Optional term to filter models by name.\n\n"
        
        "3. **/resize**\n"
        "   - **Description**: Resize an image using specified scaling method.\n"
        "   - **Parameters**:\n"
        "     - `scale_factor`: Scale factor for resizing (e.g., `2.0` for 2x) (required).\n"
        "     - `method`: The resizing method to use (e.g., `lanczos`, `bicubic`) (optional, default: `lanczos`).\n"
        "     - `image`: The image file to resize (optional).\n"
        "     - `url`: URL of the image to resize (optional).\n\n"
        
        "4. **/info**\n"
        "   - **Description**: Get information about an image.\n"
        "   - **Parameters**:\n"
        "     - `image`: The image file to analyze (optional).\n"
        "     - `url`: URL of the image to analyze (optional).\n"
    )
    await interaction.response.send_message(help_text)

# Global permissions check for slash commands
@bot.check
async def global_permissions(ctx):
    """Global check for all commands"""
    has_permission = can_use_dm(ctx)
    if not has_permission:
        if ctx.guild is None:  # Check if the context is a DM
            await ctx.send("This bot can only be used in servers. If you need DM access, please contact the bot administrator.")
        return False
    return True

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

# Bot event handlers
@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info("Note: This bot is configured to work only in servers, not in DMs.")
    


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("Command not found. Use --upscale, --models, --resize, or --info")
    elif isinstance(error, commands.CheckFailure):
        # We've already sent a message in the global check, so just silently handle it
        pass
    # Let other errors propagate up
    else:
        raise error

@bot.tree.command(name="resize", description="Resize an image using specified scaling method")
@app_commands.describe(
    scale_factor="Scale factor for resizing (e.g., 2.0 for 2x)",
    method="The resizing method to use",
    image="The image file to resize",
    url="URL of the image to resize"
)
async def resize_slash(
    interaction: discord.Interaction, 
    scale_factor: float,
    method: str = "box",
    image: Optional[discord.Attachment] = None,
    url: Optional[str] = None
):
    """Resize an image using specified scaling method"""
    await interaction.response.defer()
    
    try:
        # Input validation
        if not image and not url:
            await interaction.followup.send("Please provide either an image or a URL.")
            return
        if image and url:
            await interaction.followup.send("Please provide either an image or a URL, not both.")
            return

        # Create status message
        status_msg = await interaction.followup.send("Processing your request...")

        try:
            # Get the image
            if image:
                image_data = await image.read()
                img = Image.open(BytesIO(image_data))
            else:  # url
                img, error = await download_image(url)
                if error:
                    await status_msg.edit(content=f"Error: {error}")
                    return

            # Process the resize using the existing module
            from utils.resize_module import resize_image, get_available_filters

            # Validate method
            available_filters = get_available_filters()
            if method.lower() not in available_filters:
                filter_list = "\n".join(f"• {filter_name}" for filter_name in available_filters)
                await interaction.followup.send(f"Unsupported method: {method}. Available methods are:\n{filter_list}")
                return

            # Perform the resize
            resized_image = resize_image(img, scale_factor, method, GAMMA_CORRECTION)

            # Save and send the result
            output_buffer = BytesIO()
            resized_image.save(output_buffer, format='PNG')
            output_buffer.seek(0)

            # Send the final result
            operation = "upscaled" if scale_factor > 1 else "downscaled"
            message = (
                f"<@{interaction.user.id}> Here's your {operation} image\n"
                f"Scale factor: {scale_factor}\n"
                f"Method: {method}\n"
                f"New size: {resized_image.size[0]}x{resized_image.size[1]}"
            )
            
            await interaction.followup.send(message, file=discord.File(fp=output_buffer, filename=f"resized_{scale_factor}x.png"))

        finally:
            # Cleanup
            await asyncio.sleep(5)
            await status_msg.delete()

    except Exception as e:
        error_msg = f"Error during resize: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if 'status_msg' in locals():
            await status_msg.edit(content=error_msg)
        else:
            await interaction.followup.send(error_msg)

@bot.tree.command(name="models", description="List available upscaling models")
@app_commands.describe(search_term="Optional term to filter models by name")
async def models_slash(interaction: discord.Interaction, search_term: Optional[str] = None):
    """List available upscaling models, optionally filtered by search term"""
    available_models = bot.list_available_models(search_term)
    if not available_models:
        await interaction.response.send_message("No models are currently available.")
        return

    if search_term:
        matches = search_models(search_term, available_models)
        if matches:
            match_list = "\n".join(f"{match[0]} (similarity: {match[1]}%)" for match in matches)
            await interaction.response.send_message(f"Models matching '{search_term}':\n```\n{match_list}\n```")
        else:
            await interaction.response.send_message(f"No models found matching '{search_term}'.")
        return

    # Sort the models alphabetically
    available_models.sort()
    
    # Calculate the maximum number of models per message
    max_models_per_message = 50  # Adjust this number as needed
    
    # Split the models into chunks
    model_chunks = [available_models[i:i + max_models_per_message] 
                    for i in range(0, len(available_models), max_models_per_message)]
    
    # Send the first chunk as the initial response
    first_chunk = model_chunks[0]
    model_list = "\n".join(first_chunk)
    initial_message = f"Available models (Page 1/{len(model_chunks)}):\n```\n{model_list}\n```"
    await interaction.response.send_message(initial_message)
    
    # Send remaining chunks as follow-up messages
    for i, chunk in enumerate(model_chunks[1:], 2):
        model_list = "\n".join(chunk)
        message = f"Available models (Page {i}/{len(model_chunks)}):\n```\n{model_list}\n```"
        await interaction.followup.send(message)
    
    # If there are multiple pages, send a summary message
    if len(model_chunks) > 1:
        await interaction.followup.send(f"Total number of available models: {len(available_models)}")

@bot.tree.command(name="info", description="Get information about an image")
@app_commands.describe(
    image="The image file to analyze",
    url="URL of the image to analyze"
)
async def info_slash(
    interaction: discord.Interaction, 
    image: Optional[discord.Attachment] = None,
    url: Optional[str] = None
):
    """Get information about an image"""
    await interaction.response.defer()
    
    try:
        # Input validation
        if not image and not url:
            await interaction.followup.send("Please provide either an image or a URL.")
            return
        if image and url:
            await interaction.followup.send("Please provide either an image or a URL, not both.")
            return

        # Create a temporary file to store the image
        temp_filename = None
        try:
            if image:
                # Download the attachment
                image_data = await image.read()
                temp_filename = image.filename
                with open(temp_filename, 'wb') as f:
                    f.write(image_data)
            else:
                # Download from URL
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            temp_filename = 'temp_image'
                            content = await resp.read()
                            with open(temp_filename, 'wb') as f:
                                f.write(content)
                        else:
                            await interaction.followup.send("Failed to download the image.")
                            return

            # Get and format the image info
            image_info = get_image_info(temp_filename)
            formatted_info = format_image_info(image_info)

            # Send the formatted info
            await interaction.followup.send(f"```\n{formatted_info}\n```")

        finally:
            # Clean up the temporary file
            if temp_filename and os.path.exists(temp_filename):
                os.remove(temp_filename)

    except Exception as e:
        error_msg = f"Error getting image info: {str(e)}"
        logger.error(error_msg, exc_info=True)
        await interaction.followup.send(error_msg)

@app_commands.autocomplete(model=model_autocomplete)
@bot.tree.command(name="upscale", description="Upscale an image using a specified model")
async def upscale_slash(
    interaction: discord.Interaction, 
    model: str,
    image: Optional[discord.Attachment] = None,
    url: Optional[str] = None,
    alpha_handling: Optional[Literal["resize", "upscale", "discard"]] = None
):
    """Upscale an image using a specified model"""
    await interaction.response.defer()
    
    try:
        # Input validation
        if not image and not url:
            await interaction.followup.send("Please provide either an image or a URL.")
            return
        if image and url:
            await interaction.followup.send("Please provide either an image or a URL, not both.")
            return

        # Create status message and start timing
        start_time = time.time()
        status_msg = await interaction.followup.send("Processing your request...")
        
        # Get the image
        if image:
            image_data = await image.read()
            img = Image.open(BytesIO(image_data))
            image_source_desc = f"attachment: {image.filename}"
        else:  # url
            img, error = await download_image(url)
            if error:
                await status_msg.edit(content=f"Error: {error}")
                return
            image_source_desc = "provided URL"

        # Load model and check input channels
        logger.info(f"Loading model: {model}")
        model_obj = load_model(model)
        if model_obj.input_channels == 4:
            await status_msg.edit(content="4 channel models are not supported, please pick another model.")
            return

        # Calculate output size
        input_width, input_height = img.size
        scale = model_obj.scale
        output_width = input_width * scale
        output_height = input_height * scale
        output_total_pixels = output_width * output_height

        logger.info(f"Input size: {input_width}x{input_height}")
        logger.info(f"Output size: {output_width}x{output_height}")

        # Check if output size is reasonable
        if output_total_pixels > MAX_OUTPUT_TOTAL_PIXELS:
            max_megapixels = MAX_OUTPUT_TOTAL_PIXELS / (1024 * 1024)
            await status_msg.edit(content=f"Error: Output size would exceed maximum allowed total of {max_megapixels:.2f} megapixels.")
            return

        # Check if image has alpha channel
        has_alpha = img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info)
        alpha_mode = alpha_handling or bot.default_alpha_handling

        # Estimate VRAM usage and get tile size
        estimated_vram, adjusted_tile_size = estimate_vram_and_tile_size(model_obj, (input_width, input_height))

        # Update status message with processing details
        status_content = (
            f"Processing image from {image_source_desc}\n"
            f"Model: {model}\n"
            f"Architecture: {model_obj.architecture.name}\n"
            f"Input size: {input_width}x{input_height}\n"
            f"Output size: {output_width}x{output_height}\n"
            f"Estimated VRAM usage: {estimated_vram:.2f}GB\n"
            f"Using tile size: {adjusted_tile_size}"
        )
        if has_alpha:
            status_content += f"\nAlpha handling: {alpha_mode}"
        await status_msg.edit(content=status_content + "\nStarting upscale process...")

        # Process the image
        bot.progress_logger.log_step("Starting upscale process")
        result = bot.upscale_image(img, model_obj, adjusted_tile_size, alpha_mode, has_alpha, PRECISION, lambda: False)
        
        # Calculate processing time
        upscale_time = time.time() - start_time
        logger.info(f"Upscale completed in {upscale_time:.2f} seconds")
        await status_msg.edit(content=status_content + f"\nUpscale completed in {upscale_time:.2f} seconds\nSaving image...")

        # Save and send the result
        bot.progress_logger.log_step("Saving and sending upscaled image")
        output_buffer = BytesIO()
        result.save(output_buffer, format='PNG')
        output_buffer.seek(0)

        # Send the final result
        message = f"<@{interaction.user.id}> Here's your image upscaled with `{model}`"
        if has_alpha:
            message += f" and alpha method `{alpha_mode}`"
        message += f"\nProcessing time: {upscale_time:.2f} seconds"
        
        await interaction.followup.send(message, file=discord.File(fp=output_buffer, filename=f"upscaled_{model}.png"))
        await status_msg.edit(content=status_content + f"\nUpscale completed in {upscale_time:.2f} seconds\nImage sent successfully!")

        # Cleanup
        await asyncio.sleep(5)
        await status_msg.delete()
        bot.progress_logger.clear_step()
        logger.info("Upscale process completed successfully")

    except Exception as e:
        error_msg = f"Error during upscale: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if 'status_msg' in locals():
            await status_msg.edit(content=error_msg)
        else:
            await interaction.followup.send(error_msg)

async def sync_commands():
    """Sync commands to Discord API."""
    try:
        # Clear existing commands
        existing_commands = await bot.tree.fetch_commands()
        for command in existing_commands:
            await bot.tree.delete_command(command.id)

        # Sync new commands
        await bot.tree.sync()
        logger.info("Commands synced successfully.")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")

# Main execution
if __name__ == "__main__":
    bot.run(TOKEN)
