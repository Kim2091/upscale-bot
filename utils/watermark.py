import numpy as np
import cv2
from PIL import Image

def add_watermark(image):
    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Ensure we're working with RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL to OpenCV format
    cv_image = np.array(image)
    cv_image = cv_image[:, :, ::-1].copy()  # RGB to BGR
    
    # Get dimensions
    height, width = cv_image.shape[:2]
    text = "UV2"
    
    # Calculate base text size based on image diagonal
    diagonal = (width ** 2 + height ** 2) ** 0.5
    
    # Add randomization to font scale (±5%)
    base_font_scale = diagonal / 100
    font_scale = base_font_scale * np.random.uniform(0.95, 1.05)
    
    # Random thickness variation (±1 from base)
    base_thickness = max(4, int(diagonal / 50))
    thickness = max(3, base_thickness + np.random.randint(-1, 2))
    
    # Random font selection
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_COMPLEX
    ]
    font = np.random.choice(fonts)
    
    # Create overlay
    overlay = cv_image.copy()
    
    # Calculate total width of text for centering
    total_width = 0
    char_sizes = []
    for char in text:
        (char_width, char_height), baseline = cv2.getTextSize(char, font, font_scale, thickness)
        char_sizes.append((char_width, char_height))
        total_width += char_width
    
    # Add random spacing between characters (±10% of average character width)
    avg_char_width = total_width / len(text)
    spacing_variations = [np.random.uniform(0.9, 1.1) * avg_char_width * 0.3 for _ in range(len(text)-1)]
    total_spacing = sum(spacing_variations)
    
    # Calculate starting position (centered with random offset)
    x_offset = int(width * np.random.uniform(-0.03, 0.03))
    y_offset = int(height * np.random.uniform(-0.03, 0.03))
    
    start_x = (width - (total_width + total_spacing)) // 2 + x_offset
    y = (height + char_sizes[0][1]) // 2 + y_offset
    
    # Draw each character with varying spacing
    current_x = start_x
    for i, char in enumerate(text):
        cv2.putText(overlay, char, (int(current_x), y), font, font_scale, (255, 255, 255), 
                    thickness, lineType=cv2.LINE_AA)
        if i < len(text) - 1:
            current_x += char_sizes[i][0] + spacing_variations[i]
    
    # Random opacity (between 15% and 25%)
    alpha = np.random.uniform(0.15, 0.25)
    cv2.addWeighted(overlay, alpha, cv_image, 1 - alpha, 0, cv_image)
    
    # Convert back to PIL
    result = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    return result
