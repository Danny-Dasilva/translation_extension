#!/bin/bash
# Create simple placeholder PNG icons using ImageMagick (if available) or echo fallback

cd "$(dirname "$0")/public/icons"

# Check if ImageMagick is available
if command -v convert &> /dev/null; then
    echo "Creating icons with ImageMagick..."
    
    # 16x16 icon
    convert -size 16x16 xc:#3b82f6 \
        -gravity center \
        -pointsize 10 -fill white -annotate +0+0 "MT" \
        icon-16.png
    
    # 48x48 icon
    convert -size 48x48 xc:#3b82f6 \
        -gravity center \
        -pointsize 24 -fill white -font Arial-Bold -annotate +0+0 "MT" \
        icon-48.png
    
    # 128x128 icon
    convert -size 128x128 xc:#3b82f6 \
        -gravity center \
        -pointsize 64 -fill white -font Arial-Bold -annotate +0+0 "MT" \
        icon-128.png
    
    echo "✓ Icons created successfully!"
else
    echo "ImageMagick not found. Please install it or create icons manually."
    echo ""
    echo "On Ubuntu/Debian: sudo apt-get install imagemagick"
    echo "On macOS: brew install imagemagick"
    echo ""
    echo "Or create 16x16, 48x48, and 128x128 PNG files manually."
fi
