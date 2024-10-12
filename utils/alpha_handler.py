import numpy as np
from PIL import Image
from .resize_module import resize_image

def handle_alpha(image, upscale_func, alpha_handling, gamma_correction):
    # Convert image to RGB
    rgb_image, alpha = image.convert('RGB'), image.split()[3]

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
    elif alpha_handling == 'discard':
        return upscaled_rgb

    # Merge upscaled RGB and alpha
    upscaled_rgba = upscaled_rgb.convert('RGBA')
    upscaled_rgba.putalpha(upscaled_alpha)
    return upscaled_rgba