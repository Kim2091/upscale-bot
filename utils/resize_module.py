import discord
from discord.ext import commands
import chainner_ext
import numpy as np
from PIL import Image
from io import BytesIO
import configparser
from .config_reader import read_config

# Read configuration
config = read_config()

MIN_SCALE_FACTOR = float(config['Processing']['MinScaleFactor'])
MAX_SCALE_FACTOR = float(config['Processing']['MaxScaleFactor'])

def get_available_filters():
    return [
        'nearest', 'box', 'linear', 'hermite', 'hamming', 'hann',
        'lanczos', 'catrom', 'mitchell', 'bspline', 'lagrange', 'gauss'
    ]

def resize_image(image, scale_factor, method='lanczos', gamma_correction=True, ignore_scale_limits=False):
    """
    Resize an image using chainner_ext and the specified method and scale factor.
    
    :param image: PIL Image object or numpy array
    :param scale_factor: float, scale factor for resizing
    :param method: str, resizing method (see get_available_filters() for options)
    :param gamma_correction: bool, whether to apply gamma correction
    :param ignore_scale_limits: bool, whether to ignore the scale factor limits
    :return: PIL Image object
    """
    interpolation_map = {
        'nearest': chainner_ext.ResizeFilter.Nearest,
        'box': chainner_ext.ResizeFilter.Box,
        'linear': chainner_ext.ResizeFilter.Linear,
        'hermite': chainner_ext.ResizeFilter.Hermite,
        'hamming': chainner_ext.ResizeFilter.Hamming,
        'hann': chainner_ext.ResizeFilter.Hann,
        'lanczos': chainner_ext.ResizeFilter.Lanczos,
        'catrom': chainner_ext.ResizeFilter.CubicCatrom,
        'mitchell': chainner_ext.ResizeFilter.CubicMitchell,
        'bspline': chainner_ext.ResizeFilter.CubicBSpline,
        'lagrange': chainner_ext.ResizeFilter.Lagrange,
        'gauss': chainner_ext.ResizeFilter.Gauss
    }
    
    if method.lower() not in interpolation_map:
        raise ValueError(f"Unsupported resize method: {method}")
    
    if not ignore_scale_limits and (scale_factor < MIN_SCALE_FACTOR or scale_factor > MAX_SCALE_FACTOR):
        raise ValueError(f"Scale factor must be between {MIN_SCALE_FACTOR} and {MAX_SCALE_FACTOR} (inclusive).")
    
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        img_array = np.array(image, dtype=np.float32) / 255.0
        original_mode = image.mode
    else:
        img_array = image.astype(np.float32) / 255.0
        original_mode = 'L' if img_array.ndim == 2 else 'RGB'
    
    # Ensure the array is 3D
    if img_array.ndim == 2:
        img_array = img_array[:, :, np.newaxis]
    
    # Calculate new dimensions
    h, w = img_array.shape[:2]
    new_h = max(1, int(h * scale_factor))
    new_w = max(1, int(w * scale_factor))
    
    print(f"Original size: {w}x{h}")
    print(f"New size: {new_w}x{new_h}")
    
    # Disable gamma correction for nearest neighbor
    if method.lower() == 'nearest':
        gamma_correction = False
    
    # Resize the image using chainner_ext
    resized_array = chainner_ext.resize(
        img_array,
        (new_w, new_h),
        interpolation_map[method.lower()],
        gamma_correction=gamma_correction
    )
    
    # Convert back to 0-255 range and clip values
    resized_array = np.clip(resized_array * 255, 0, 255).astype(np.uint8)
    
    # If the input was 2D, return a 2D output
    if original_mode == 'L':
        resized_array = resized_array.squeeze()
    
    # Convert numpy array back to PIL Image
    return Image.fromarray(resized_array, mode=original_mode)

async def process_resize(ctx, image, scale_factor, method, gamma_correction):
    try:
        available_filters = get_available_filters()
        if method.lower() not in available_filters:
            filter_list = "\n".join(f"• {filter_name}" for filter_name in available_filters)
            await ctx.send(f"Unsupported method: {method}. Available methods are:\n{filter_list}")
            return

        print(f"Original image size: {image.size}")
        print(f"Original image mode: {image.mode}")
        print(f"Scale factor: {scale_factor}")
        print(f"Method: {method}")

        # Perform resizing using chainner_ext via resize_image function
        resized_image = resize_image(image, scale_factor, method, gamma_correction)

        print(f"Resized image size: {resized_image.size}")
        print(f"Resized image mode: {resized_image.mode}")

        # Save and send the resized image
        output_buffer = BytesIO()
        resized_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)

        operation = "upscaled" if scale_factor > 1 else "downscaled"
        await ctx.send(f"Here's your {operation} image (scale factor: {scale_factor}, method: {method}, new size: {resized_image.size[0]}x{resized_image.size[1]}):",
                       file=discord.File(fp=output_buffer, filename="resized_image.png"))

    except Exception as e:
        await ctx.send(f"Error in resize process: {str(e)}")
        print(f"Error in resize process:")
        import traceback
        traceback.print_exc()

async def resize_command(ctx, args, download_image, GAMMA_CORRECTION):
    try:
        available_filters = get_available_filters()
        filter_list = "\n".join(f"• {filter_name}" for filter_name in available_filters)
        
        usage_message = (
            f"Usage: `--resize <scale_factor> <method> [image_url]`\n"
            f"Or attach an image and use: `--resize <scale_factor> <method>`\n"
            f"Scale factor must be between {MIN_SCALE_FACTOR} and {MAX_SCALE_FACTOR} (inclusive).\n"
            f"Available methods:\n{filter_list}"
        )

        if len(args) < 2:
            await ctx.send(usage_message)
            return

        try:
            scale_factor = float(args[0])
            if scale_factor < MIN_SCALE_FACTOR or scale_factor > MAX_SCALE_FACTOR:
                await ctx.send(f"Scale factor must be between {MIN_SCALE_FACTOR} and {MAX_SCALE_FACTOR} (inclusive).")
                return
            method = args[1].lower()
        except ValueError:
            await ctx.send(f"Invalid scale factor. Please ensure your scale factor is between {MIN_SCALE_FACTOR} and {MAX_SCALE_FACTOR}. Also ensure your command is in the valid format: `--resize <scale_factor> <method> [image_url]`")
            return

        if method not in available_filters:
            await ctx.send(f"Unsupported method: {method}. Available methods are:\n{filter_list}")
            return

        image_url = args[2] if len(args) > 2 else None

        if len(ctx.message.attachments) > 0:
            attachment = ctx.message.attachments[0]
            if not attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                await ctx.send("Please upload a valid image file (PNG, JPG, JPEG, or WebP).")
                return
            image_data = await attachment.read()
            image = Image.open(BytesIO(image_data))
        elif image_url:
            image, error_message = await download_image(image_url)
            if image is None:
                await ctx.send(f"Error: {error_message} Please try uploading the image directly to Discord.")
                return
        else:
            await ctx.send("Please either attach an image or provide a valid image URL.")
            return

        await process_resize(ctx, image, scale_factor, method, GAMMA_CORRECTION)

    except Exception as e:
        await ctx.send(f"Error in resize command: {str(e)}")
        print(f"Error in resize command:")
        import traceback
        traceback.print_exc()