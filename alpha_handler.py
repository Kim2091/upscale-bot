import numpy as np
from PIL import Image
import chainner_ext

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
        upscaled_alpha = Image.fromarray(np.array(upscaled_alpha_3channel)[:,:,0])
    elif alpha_handling == 'resize':
        # Resize alpha using chainner_ext.resize with CubicMitchell filter
        alpha_np = np.array(alpha, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        alpha_np = alpha_np.reshape(alpha_np.shape[0], alpha_np.shape[1], 1)  # Add channel dimension
        upscaled_alpha_np = chainner_ext.resize(
            alpha_np,
            (upscaled_rgb.width, upscaled_rgb.height),
            chainner_ext.ResizeFilter.CubicMitchell,
            gamma_correction=gamma_correction
        )
        # Convert back to 0-255 range and clip values
        upscaled_alpha_np = np.clip(upscaled_alpha_np * 255, 0, 255)
        upscaled_alpha = Image.fromarray(upscaled_alpha_np.squeeze().astype(np.uint8))
    elif alpha_handling == 'discard':
        return upscaled_rgb

    # Merge upscaled RGB and alpha
    upscaled_rgba = upscaled_rgb.convert('RGBA')
    upscaled_rgba.putalpha(upscaled_alpha)
    return upscaled_rgba