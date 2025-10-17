#!/usr/bin/env python3
"""Create placeholder extension icons"""
from PIL import Image, ImageDraw, ImageFont
import os

# Create icons directory
icons_dir = os.path.join(os.path.dirname(__file__), 'public', 'icons')
os.makedirs(icons_dir, exist_ok=True)

def create_icon(size, filename):
    """Create a simple blue icon with 'MT' text"""
    # Create blue background
    img = Image.new('RGB', (size, size), color='#3b82f6')
    draw = ImageDraw.Draw(img)
    
    # Add white text "MT" (Manga Translator)
    font_size = size // 3
    try:
        # Try to use a default font
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', font_size)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Draw text centered
    text = "MT"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    position = ((size - text_width) // 2, (size - text_height) // 2 - 2)
    draw.text(position, text, fill='white', font=font)
    
    # Save PNG
    filepath = os.path.join(icons_dir, filename)
    img.save(filepath, 'PNG')
    print(f'✓ Created {filename} ({size}x{size})')

# Create all three sizes
create_icon(16, 'icon-16.png')
create_icon(48, 'icon-48.png')
create_icon(128, 'icon-128.png')

print('\n✅ All icons created successfully!')
print(f'Location: {icons_dir}')
