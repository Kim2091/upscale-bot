# utils/image_info.py

import os
from PIL import Image
from PIL.ExifTags import TAGS
import wand.image
import struct

def get_image_info(file_path):
    """
    Get detailed information about an image file.
    Supports a wide variety of formats, including DDS.
    
    :param file_path: Path to the image file
    :return: Dictionary containing image information
    """
    info = {
        "filename": os.path.basename(file_path),
        "file_size": os.path.getsize(file_path),
        "format": None,
        "mode": None,
        "width": None,
        "height": None,
        "color_depth": None,
        "compression": None,
        "exif": {},
    }

    # Try with PIL first
    try:
        with Image.open(file_path) as img:
            info["format"] = img.format
            info["mode"] = img.mode
            info["width"], info["height"] = img.size

            # Get color depth
            if img.mode in ("1", "L", "P"):
                info["color_depth"] = 8
            elif img.mode in ("RGB", "YCbCr", "LAB", "HSV"):
                info["color_depth"] = 24
            elif img.mode in ("RGBA", "CMYK"):
                info["color_depth"] = 32

            # Get compression info if available
            if "compression" in img.info:
                info["compression"] = img.info["compression"]

            # Get EXIF data
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    info["exif"][tag] = value

    except Exception as e:
        # If PIL fails, try with Wand (ImageMagick)
        try:
            with wand.image.Image(filename=file_path) as img:
                info["format"] = img.format
                info["width"], info["height"] = img.size
                info["color_depth"] = img.depth
                info["compression"] = img.compression

        except Exception as wand_error:
            # If both PIL and Wand fail, check if it's a DDS file
            if file_path.lower().endswith('.dds'):
                with open(file_path, 'rb') as f:
                    try:
                        # Read DDS header
                        magic = f.read(4)
                        if magic == b'DDS ':
                            header_size = struct.unpack('<I', f.read(4))[0]
                            if header_size == 124:
                                flags = struct.unpack('<I', f.read(4))[0]
                                height = struct.unpack('<I', f.read(4))[0]
                                width = struct.unpack('<I', f.read(4))[0]
                                
                                info["format"] = "DDS"
                                info["width"] = width
                                info["height"] = height
                    except Exception as dds_error:
                        print(f"Error reading DDS file: {dds_error}")
            else:
                print(f"Error processing image with both PIL and Wand: {e}, {wand_error}")

    return info

def format_image_info(info):
    """
    Format the image information into a readable string.
    
    :param info: Dictionary containing image information
    :return: Formatted string with image details
    """
    formatted = f"File: {info['filename']}\n"
    formatted += f"Size: {info['file_size']} bytes\n"
    formatted += f"Format: {info['format']}\n"
    if info['mode']:
        formatted += f"Mode: {info['mode']}\n"
    formatted += f"Dimensions: {info['width']}x{info['height']}\n"
    if info['color_depth']:
        formatted += f"Color Depth: {info['color_depth']} bits\n"
    if info['compression']:
        formatted += f"Compression: {info['compression']}\n"
    
    if info['exif']:
        formatted += "EXIF Data:\n"
        for tag, value in info['exif'].items():
            formatted += f"  {tag}: {value}\n"
    
    return formatted