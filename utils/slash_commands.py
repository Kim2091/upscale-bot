import discord
from discord import app_commands
import logging
from typing import Optional, Literal

from utils.alpha_handler import handle_alpha
from utils.resize_module import resize_command
from utils.fuzzy_model_matcher import find_closest_models, search_models

logger = logging.getLogger('UpscaleBot')

# Define choices for alpha handling
ALPHA_CHOICES = Literal["resize", "upscale", "discard"]
# Define choices for resize methods
RESIZE_METHODS = Literal["nearest", "box", "linear", "hermite", "hamming", "hann",
        "lanczos", "catrom", "mitchell", "bspline", "lagrange", "gauss"]

class ImageSource(discord.app_commands.Group):
    """A group to ensure mutual exclusivity between image and URL parameters"""
    def __init__(self):
        super().__init__()

def register_slash_commands(bot):
    """Register all slash commands with the bot"""
    
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

    @bot.tree.command(name="models", description="List available upscaling models")
    async def models_slash(interaction: discord.Interaction, search_term: str = None):
        logger.info(f"Models command called with search term: {search_term}")
        try:
            models_list = bot.list_available_models(search_term)
            if not models_list:
                await interaction.response.send_message("No models found.")
                return
            
            # Discord has a message length limit, so we'll split long lists
            message_chunks = []
            current_chunk = "Available Models:\n"
            for model in models_list:
                if len(current_chunk) + len(model) + 2 > 2000:  # Leave room for formatting
                    message_chunks.append(current_chunk)
                    current_chunk = "Continued:\n"
                current_chunk += f"- {model}\n"
            
            if current_chunk:
                message_chunks.append(current_chunk)
            
            # Send the first chunk and follow up with others if needed
            await interaction.response.send_message(message_chunks[0])
            for chunk in message_chunks[1:]:
                await interaction.followup.send(chunk)

            # Send the total number of models at the end
            total_models = len(models_list)
            await interaction.followup.send(f"Total number of available models: {total_models}")

        except Exception as e:
            logger.error(f"Error in models slash command: {e}")
            await interaction.response.send_message("An error occurred while listing models.")

    async def model_autocomplete(
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        """Autocomplete for model names"""
        models = bot.list_available_models()
        if current:
            matches = find_closest_models(current, models)
            # Extract just the model names from the matches (first element of each tuple)
            model_names = [match[0] for match in matches[:25]]
        else:
            model_names = models[:25]  # Discord limits to 25 choices
            
        return [
            app_commands.Choice(name=model, value=model)
            for model in model_names
        ]

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
        logger.info(f"Info command called with image: {image}, url: {url}")
        if not image and not url:
            await interaction.response.send_message("Please provide either an image or a URL.", ephemeral=True)
            return
        if image and url:
            await interaction.response.send_message("Please provide either an image or a URL, not both.", ephemeral=True)
            return
            
        # Reuse the existing info function logic
        ctx = await bot.get_context(interaction)
        
        # Prepare arguments for the existing info function
        args = []
        if image:
            args.append(image.url)
        if url:
            args.append(url)
        
        # Call the existing info function
        await bot.get_command('info')(ctx, *args)

    @bot.tree.command(name="upscale", description="Upscale an image using a specified model")
    @app_commands.describe(
        model="The upscaling model to use",
        image="The image file to upscale",
        url="URL of the image to upscale",
        alpha_handling="How to handle alpha/transparency"
    )
    @app_commands.autocomplete(model=model_autocomplete)
    async def upscale_slash(
        interaction: discord.Interaction, 
        model: str,
        image: Optional[discord.Attachment] = None,
        url: Optional[str] = None,
        alpha_handling: Optional[ALPHA_CHOICES] = None
    ):
        logger.info(f"Upscale command called with model: {model}, image: {image}, url: {url}, alpha_handling: {alpha_handling}")
        if not image and not url:
            await interaction.response.send_message("Please provide either an image or a URL.", ephemeral=True)
            return
        if image and url:
            await interaction.response.send_message("Please provide either an image or a URL, not both.", ephemeral=True)
            return
            
        # Reuse the existing upscale function logic
        ctx = await bot.get_context(interaction)
        
        # Prepare arguments for the existing upscale function
        args = [model]
        if alpha_handling:
            args.append(alpha_handling)
        if image:
            args.append(image.url)
        if url:
            args.append(url)
        
        # Call the existing upscale function
        await bot.get_command('upscale')(ctx, *args)

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
        method: RESIZE_METHODS = "lanczos",
        image: Optional[discord.Attachment] = None,
        url: Optional[str] = None
    ):
        logger.info(f"Resize command called with scale_factor: {scale_factor}, method: {method}, image: {image}, url: {url}")
        if not image and not url:
            await interaction.response.send_message("Please provide either an image or a URL.", ephemeral=True)
            return
        if image and url:
            await interaction.response.send_message("Please provide either an image or a URL, not both.", ephemeral=True)
            return
            
        # Reuse the existing resize function logic
        ctx = await bot.get_context(interaction)
        
        # Prepare arguments for the existing resize function
        args = [str(scale_factor), method]
        if image:
            args.append(image.url)
        if url:
            args.append(url)
        
        # Call the existing resize function
        await bot.get_command('resize')(ctx, *args)
