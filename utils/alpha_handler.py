import numpy as np
from PIL import Image
from .resize_module import resize_image

def handle_alpha(image, upscale_func, alpha_handling, gamma_correction):
    # Early return for 'discard' option to avoid unnecessary conversions
    if alpha_handling == 'discard':
        return upscale_func(image.convert('RGB'))

    # Validate alpha_handling parameter
    if alpha_handling not in ('resize', 'upscale', 'discard'):
        raise ValueError(f"Invalid alpha_handling mode: {alpha_handling}. Must be 'resize', 'upscale', or 'discard'.")

    # Extract alpha channel based on image mode
    if image.mode == 'LA':
        # LA mode: Luminance + Alpha (2 channels)
        alpha = image.split()[1]
    elif image.mode == 'P' and 'transparency' in image.info:
        # Palette mode with transparency - convert to RGBA first
        alpha = image.convert('RGBA').split()[3]
    elif image.mode == 'RGBA':
        # Standard RGBA mode
        alpha = image.split()[3]
    else:
        # Fallback: try to convert to RGBA and extract alpha
        try:
            alpha = image.convert('RGBA').split()[3]
        except (IndexError, ValueError) as e:
            raise ValueError(f"Unable to extract alpha channel from image mode '{image.mode}': {e}")

    # Convert image to RGB
    rgb_image = image.convert('RGB')

    # Upscale RGB Portion
    upscaled_rgb = upscale_func(rgb_image)

    if alpha_handling == 'upscale':
        # Create a 3-channel image from the alpha channel
        alpha_array = np.array(alpha)
        alpha_3channel = np.stack([alpha_array, alpha_array, alpha_array], axis=2)
        alpha_image = Image.fromarray(alpha_3channel)
        
        # Upscale the 3-channel alpha
        upscaled_alpha_3channel = upscale_func(alpha_image)
        
        # Extract a single channel from the result
        upscaled_alpha = upscaled_alpha_3channel.split()[0]
    elif alpha_handling == 'resize':
        # Calculate the scale factor based on the upscaled RGB image
        scale_factor = upscaled_rgb.width / image.width
        
        # Resize alpha using resize_image function without scale factor limitations
        upscaled_alpha = resize_image(
            alpha,
            scale_factor,
            method='mitchell',
            gamma_correction=gamma_correction,
            ignore_scale_limits=True
        )

    # Merge upscaled RGB and alpha
    upscaled_rgba = upscaled_rgb.convert('RGBA')
    upscaled_rgba.putalpha(upscaled_alpha)
    return upscaled_rgba
