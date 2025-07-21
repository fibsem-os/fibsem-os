import numpy as np
from typing import Tuple, Optional

def draw_character(char: str, size: Tuple[int, int] = (32, 24), thickness: int = 2, image_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Draw any alphanumeric character as a numpy array.
    
    Args:
        char: Character to draw (A-Z, 0-9, case insensitive)
        size: (height, width) of the character itself
        thickness: Line thickness for drawing
        image_shape: (height, width) of the output image. If None, uses size.
        
    Returns:
        np.ndarray: Binary array with the character drawn (1 = foreground, 0 = background)
    """
    char = char.upper()
    if not char.isalnum() or len(char) != 1:
        raise ValueError("Character must be a single alphanumeric character (A-Z, 0-9)")
    
    # If no image_shape specified, use the character size
    if image_shape is None:
        image_shape = size
    
    img_height, img_width = image_shape
    char_height, char_width = size
    
    # Create the full image
    img = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Calculate offset to center the character
    offset_y = (img_height - char_height) // 2
    offset_x = (img_width - char_width) // 2
    
    # Create a temporary image for the character
    char_img = np.zeros((char_height, char_width), dtype=np.uint8)
    
    # Define all character patterns as coordinate lists
    # Each pattern is defined relative to a normalized coordinate system
    patterns = {
        # Numbers 0-9
        '0': [
            # Outer rectangle
            [(0.2, 0.1), (0.8, 0.1), (0.8, 0.9), (0.2, 0.9), (0.2, 0.1)]
        ],
        '1': [
            # Vertical line and top stroke
            [(0.5, 0.1), (0.5, 0.9)],
            [(0.3, 0.2), (0.5, 0.1)]
        ],
        '2': [
            # Top horizontal, right vertical, middle horizontal, left vertical, bottom horizontal
            [(0.2, 0.1), (0.8, 0.1)],
            [(0.8, 0.1), (0.8, 0.5)], 
            [(0.8, 0.5), (0.2, 0.5)],
            [(0.2, 0.5), (0.2, 0.9)],
            [(0.2, 0.9), (0.8, 0.9)]
        ],
        '3': [
            # Top horizontal, right vertical, middle horizontal, right vertical, bottom horizontal
            [(0.2, 0.1), (0.8, 0.1)],
            [(0.8, 0.1), (0.8, 0.5)],
            [(0.5, 0.5), (0.8, 0.5)],
            [(0.8, 0.5), (0.8, 0.9)],
            [(0.2, 0.9), (0.8, 0.9)]
        ],
        '4': [
            # Left vertical (top half), horizontal, right vertical
            [(0.2, 0.1), (0.2, 0.5)],
            [(0.2, 0.5), (0.8, 0.5)],
            [(0.8, 0.1), (0.8, 0.9)]
        ],
        '5': [
            # Top horizontal, left vertical, middle horizontal, right vertical, bottom horizontal
            [(0.2, 0.1), (0.8, 0.1)],
            [(0.2, 0.1), (0.2, 0.5)],
            [(0.2, 0.5), (0.8, 0.5)],
            [(0.8, 0.5), (0.8, 0.9)],
            [(0.2, 0.9), (0.8, 0.9)]
        ],
        '6': [
            # Top horizontal, left vertical, middle horizontal, left vertical (bottom), right vertical (bottom), bottom horizontal
            [(0.2, 0.1), (0.8, 0.1)],
            [(0.2, 0.1), (0.2, 0.9)],
            [(0.2, 0.5), (0.8, 0.5)],
            [(0.8, 0.5), (0.8, 0.9)],
            [(0.2, 0.9), (0.8, 0.9)]
        ],
        '7': [
            # Top horizontal, right vertical
            [(0.2, 0.1), (0.8, 0.1)],
            [(0.8, 0.1), (0.8, 0.9)]
        ],
        '8': [
            # Full rectangle outline with middle horizontal
            [(0.2, 0.1), (0.8, 0.1), (0.8, 0.9), (0.2, 0.9), (0.2, 0.1)],
            [(0.2, 0.5), (0.8, 0.5)]
        ],
        '9': [
            # Top rectangle and right vertical to bottom
            [(0.2, 0.1), (0.8, 0.1), (0.8, 0.5), (0.2, 0.5), (0.2, 0.1)],
            [(0.8, 0.5), (0.8, 0.9)],
            [(0.2, 0.9), (0.8, 0.9)]
        ],
        # Letters A-Z
        'A': [
            # Left vertical, right vertical, top horizontal, middle horizontal
            [(0.2, 0.9), (0.2, 0.3), (0.5, 0.1), (0.8, 0.3), (0.8, 0.9)],
            [(0.3, 0.5), (0.7, 0.5)]
        ],
        'B': [
            # Left vertical, top horizontal, middle horizontal, bottom horizontal, top right vertical, bottom right vertical
            [(0.2, 0.1), (0.2, 0.9)],
            [(0.2, 0.1), (0.7, 0.1), (0.7, 0.5), (0.2, 0.5)],
            [(0.2, 0.5), (0.7, 0.5), (0.7, 0.9), (0.2, 0.9)]
        ],
        'C': [
            # Top horizontal, left vertical, bottom horizontal
            [(0.7, 0.1), (0.3, 0.1), (0.2, 0.2), (0.2, 0.8), (0.3, 0.9), (0.7, 0.9)]
        ],
        'D': [
            # Left vertical, top horizontal with curve, bottom horizontal with curve
            [(0.2, 0.1), (0.2, 0.9)],
            [(0.2, 0.1), (0.6, 0.1), (0.8, 0.3), (0.8, 0.7), (0.6, 0.9), (0.2, 0.9)]
        ],
        'E': [
            # Left vertical, top horizontal, middle horizontal, bottom horizontal
            [(0.2, 0.1), (0.2, 0.9)],
            [(0.2, 0.1), (0.8, 0.1)],
            [(0.2, 0.5), (0.6, 0.5)],
            [(0.2, 0.9), (0.8, 0.9)]
        ],
        'F': [
            # Left vertical, top horizontal, middle horizontal
            [(0.2, 0.1), (0.2, 0.9)],
            [(0.2, 0.1), (0.8, 0.1)],
            [(0.2, 0.5), (0.6, 0.5)]
        ],
        'G': [
            # Like C but with horizontal line at middle right
            [(0.7, 0.1), (0.3, 0.1), (0.2, 0.2), (0.2, 0.8), (0.3, 0.9), (0.7, 0.9)],
            [(0.5, 0.5), (0.8, 0.5), (0.8, 0.9)]
        ],
        'H': [
            # Left vertical, right vertical, middle horizontal
            [(0.2, 0.1), (0.2, 0.9)],
            [(0.8, 0.1), (0.8, 0.9)],
            [(0.2, 0.5), (0.8, 0.5)]
        ],
        'I': [
            # Top horizontal, middle vertical, bottom horizontal
            [(0.3, 0.1), (0.7, 0.1)],
            [(0.5, 0.1), (0.5, 0.9)],
            [(0.3, 0.9), (0.7, 0.9)]
        ],
        'J': [
            # Top horizontal, right vertical, bottom curve
            [(0.3, 0.1), (0.8, 0.1)],
            [(0.7, 0.1), (0.7, 0.7), (0.5, 0.9), (0.3, 0.9), (0.2, 0.8)]
        ],
        'K': [
            # Left vertical, diagonal up, diagonal down
            [(0.2, 0.1), (0.2, 0.9)],
            [(0.2, 0.5), (0.8, 0.1)],
            [(0.2, 0.5), (0.8, 0.9)]
        ],
        'L': [
            # Left vertical, bottom horizontal
            [(0.2, 0.1), (0.2, 0.9)],
            [(0.2, 0.9), (0.8, 0.9)]
        ],
        'M': [
            # Left vertical, right vertical, left diagonal, right diagonal
            [(0.2, 0.1), (0.2, 0.9)],
            [(0.8, 0.1), (0.8, 0.9)],
            [(0.2, 0.1), (0.5, 0.4)],
            [(0.8, 0.1), (0.5, 0.4)]
        ],
        'N': [
            # Left vertical, right vertical, diagonal
            [(0.2, 0.1), (0.2, 0.9)],
            [(0.8, 0.1), (0.8, 0.9)],
            [(0.2, 0.1), (0.8, 0.9)]
        ],
        'O': [
            # Oval/rectangle
            [(0.3, 0.1), (0.7, 0.1), (0.8, 0.2), (0.8, 0.8), (0.7, 0.9), (0.3, 0.9), (0.2, 0.8), (0.2, 0.2), (0.3, 0.1)]
        ],
        'P': [
            # Left vertical, top horizontal, middle horizontal, top right vertical
            [(0.2, 0.1), (0.2, 0.9)],
            [(0.2, 0.1), (0.7, 0.1), (0.8, 0.2), (0.8, 0.4), (0.7, 0.5), (0.2, 0.5)]
        ],
        'Q': [
            # Like O but with diagonal tail
            [(0.3, 0.1), (0.7, 0.1), (0.8, 0.2), (0.8, 0.8), (0.7, 0.9), (0.3, 0.9), (0.2, 0.8), (0.2, 0.2), (0.3, 0.1)],
            [(0.6, 0.6), (0.8, 0.9)]
        ],
        'R': [
            # Like P but with diagonal leg
            [(0.2, 0.1), (0.2, 0.9)],
            [(0.2, 0.1), (0.7, 0.1), (0.8, 0.2), (0.8, 0.4), (0.7, 0.5), (0.2, 0.5)],
            [(0.5, 0.5), (0.8, 0.9)]
        ],
        'S': [
            # S curve
            [(0.8, 0.2), (0.7, 0.1), (0.3, 0.1), (0.2, 0.2), (0.2, 0.4), (0.3, 0.5), (0.7, 0.5), (0.8, 0.6), (0.8, 0.8), (0.7, 0.9), (0.3, 0.9), (0.2, 0.8)]
        ],
        'T': [
            # Top horizontal, middle vertical
            [(0.2, 0.1), (0.8, 0.1)],
            [(0.5, 0.1), (0.5, 0.9)]
        ],
        'U': [
            # Left vertical, bottom horizontal, right vertical
            [(0.2, 0.1), (0.2, 0.8), (0.3, 0.9), (0.7, 0.9), (0.8, 0.8), (0.8, 0.1)]
        ],
        'V': [
            # Two diagonals meeting at bottom
            [(0.2, 0.1), (0.5, 0.9)],
            [(0.8, 0.1), (0.5, 0.9)]
        ],
        'W': [
            # Like inverted V with middle peak
            [(0.2, 0.1), (0.3, 0.9), (0.5, 0.6), (0.7, 0.9), (0.8, 0.1)]
        ],
        'X': [
            # Two diagonals crossing
            [(0.2, 0.1), (0.8, 0.9)],
            [(0.8, 0.1), (0.2, 0.9)]
        ],
        'Y': [
            # Two diagonals meeting, then vertical down
            [(0.2, 0.1), (0.5, 0.5)],
            [(0.8, 0.1), (0.5, 0.5)],
            [(0.5, 0.5), (0.5, 0.9)]
        ],
        'Z': [
            # Top horizontal, diagonal, bottom horizontal
            [(0.2, 0.1), (0.8, 0.1)],
            [(0.8, 0.1), (0.2, 0.9)],
            [(0.2, 0.9), (0.8, 0.9)]
        ]
    }
    
    if char not in patterns:
        raise ValueError(f"Character '{char}' is not supported")
    
    # Draw the pattern on the character image
    for line_segments in patterns[char]:
        if len(line_segments) >= 2:
            for i in range(len(line_segments) - 1):
                x1, y1 = line_segments[i]
                x2, y2 = line_segments[i + 1]
                
                # Convert normalized coordinates to pixel coordinates
                px1, py1 = int(x1 * char_width), int(y1 * char_height)
                px2, py2 = int(x2 * char_width), int(y2 * char_height)
                
                # Draw line on character image
                draw_line(char_img, px1, py1, px2, py2, thickness)
    
    # Place the character in the center of the main image
    end_y = min(offset_y + char_height, img_height)
    end_x = min(offset_x + char_width, img_width)
    char_end_y = end_y - offset_y
    char_end_x = end_x - offset_x
    
    img[offset_y:end_y, offset_x:end_x] = char_img[:char_end_y, :char_end_x]
    
    return img

def draw_number(num: int, size: Tuple[int, int] = (32, 24), thickness: int = 2, image_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Draw a number character as a numpy array.
    
    DEPRECATED: Use draw_character() instead for unified alphanumeric support.
    
    Args:
        num: Number to draw (0-9)
        size: (height, width) of the digit itself
        thickness: Line thickness for drawing
        image_shape: (height, width) of the output image. If None, uses size.
        
    Returns:
        np.ndarray: Binary array with the number drawn (1 = foreground, 0 = background)
    """
    if not 0 <= num <= 9:
        raise ValueError("Number must be between 0 and 9")
    
    return draw_character(str(num), size, thickness, image_shape)

def draw_letter(letter: str, size: Tuple[int, int] = (32, 24), thickness: int = 2, image_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Draw an alphabetical character as a numpy array.
    
    DEPRECATED: Use draw_character() instead for unified alphanumeric support.
    
    Args:
        letter: Letter to draw (A-Z, case insensitive)
        size: (height, width) of the letter itself
        thickness: Line thickness for drawing
        image_shape: (height, width) of the output image. If None, uses size.
        
    Returns:
        np.ndarray: Binary array with the letter drawn (1 = foreground, 0 = background)
    """
    letter = letter.upper()
    if not letter.isalpha() or len(letter) != 1:
        raise ValueError("Letter must be a single alphabetical character")
    
    return draw_character(letter, size, thickness, image_shape)

def draw_line(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, thickness: int = 1):
    """Draw a line on the image using Bresenham's algorithm with thickness."""
    height, width = img.shape
    
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    x, y = x1, y1
    
    while True:
        # Draw thick point
        for tx in range(-thickness//2, thickness//2 + 1):
            for ty in range(-thickness//2, thickness//2 + 1):
                px, py = x + tx, y + ty
                if 0 <= px < width and 0 <= py < height:
                    img[py, px] = 1
        
        if x == x2 and y == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

def draw_multi_digit_number(num: int, size: Tuple[int, int] = (32, 24), spacing: int = 4, thickness: int = 2, image_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Draw a multi-digit number as a numpy array.
    
    Args:
        num: Number to draw (can be multiple digits)
        size: (height, width) of each digit
        spacing: Pixels between digits
        thickness: Line thickness for drawing
        image_shape: (height, width) of the output image. If None, fits digits exactly.
        
    Returns:
        np.ndarray: Binary array with the number drawn
    """
    digits = [int(d) for d in str(abs(num))]
    height, digit_width = size
    
    total_width = len(digits) * digit_width + (len(digits) - 1) * spacing
    
    # If no image_shape specified, use the exact size needed
    if image_shape is None:
        image_shape = (height, total_width)
    
    img_height, img_width = image_shape
    result = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Calculate offset to center the number sequence
    offset_y = (img_height - height) // 2
    offset_x = (img_width - total_width) // 2
    
    for i, digit in enumerate(digits):
        digit_img = draw_number(digit, size, thickness)
        start_col = offset_x + i * (digit_width + spacing)
        end_col = start_col + digit_width
        start_row = offset_y
        end_row = start_row + height
        
        # Ensure we don't go out of bounds
        if start_col >= 0 and end_col <= img_width and start_row >= 0 and end_row <= img_height:
            result[start_row:end_row, start_col:end_col] = digit_img
    
    return result

def draw_text(text: str, size: Tuple[int, int] = (32, 24), spacing: int = 4, thickness: int = 2, image_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Draw a text string (letters and numbers) as a numpy array.
    
    Args:
        text: Text to draw (letters A-Z and numbers 0-9, case insensitive)
        size: (height, width) of each character
        spacing: Pixels between characters
        thickness: Line thickness for drawing
        image_shape: (height, width) of the output image. If None, fits text exactly.
        
    Returns:
        np.ndarray: Binary array with the text drawn
    """
    # Convert to uppercase and filter valid characters
    text = text.upper()
    valid_chars = [char for char in text if char.isalnum()]
    
    if not valid_chars:
        raise ValueError("Text must contain at least one alphanumeric character")
    
    height, char_width = size
    total_width = len(valid_chars) * char_width + (len(valid_chars) - 1) * spacing
    
    # If no image_shape specified, use the exact size needed
    if image_shape is None:
        image_shape = (height, total_width)
    
    img_height, img_width = image_shape
    result = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Calculate offset to center the text sequence
    offset_y = (img_height - height) // 2
    offset_x = (img_width - total_width) // 2
    
    for i, char in enumerate(valid_chars):
        # Draw any alphanumeric character
        char_img = draw_character(char, size, thickness)
        
        start_col = offset_x + i * (char_width + spacing)
        end_col = start_col + char_width
        start_row = offset_y
        end_row = start_row + height
        
        # Ensure we don't go out of bounds
        if start_col >= 0 and end_col <= img_width and start_row >= 0 and end_row <= img_height:
            result[start_row:end_row, start_col:end_col] = char_img
    
    return result

# Example usage and test function
def test_draw_numbers():
    """Test the number drawing functions."""
    import matplotlib.pyplot as plt
    
    # Test single digits
    _, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        row = i // 5
        col = i % 5
        img = draw_number(i, size=(32, 24), thickness=2)
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'Number {i}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Test centering with larger image
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Normal size
    img1 = draw_number(7, size=(32, 24), thickness=2)
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title('Normal size (32x24)')
    axes[0].axis('off')
    
    # Centered in larger image
    
    img2 = draw_number(7, size=(256, 256), thickness=32, image_shape=(1024, 1024))
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title('Centered in 64x80 image')
    axes[1].axis('off')
    
    # Multi-digit centered
    img3 = draw_multi_digit_number(123, size=(24, 18), spacing=3, thickness=2, image_shape=(64, 100))
    axes[2].imshow(img3, cmap='gray')
    axes[2].set_title('Multi-digit centered in 64x100')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Test multi-digit number
    multi_img = draw_multi_digit_number(12345, size=(32, 24), spacing=4, thickness=2)
    plt.figure(figsize=(12, 4))
    plt.imshow(multi_img, cmap='gray')
    plt.title('Multi-digit number: 12345')
    plt.axis('off')
    plt.show()
    
    return multi_img

def test_draw_letters():
    """Test the letter drawing functions."""
    import matplotlib.pyplot as plt
    
    # Test all letters A-Z
    _, axes = plt.subplots(4, 7, figsize=(21, 12))
    axes = axes.flatten()
    
    for i, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        img = draw_letter(letter, size=(32, 24), thickness=2)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Letter {letter}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(26, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Test text drawing with mixed letters and numbers
    _, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Simple word
    img1 = draw_text("HELLO", size=(32, 24), spacing=4, thickness=2)
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title('Text: HELLO')
    axes[0].axis('off')
    
    # Mixed letters and numbers
    img2 = draw_text("ABC123", size=(32, 24), spacing=4, thickness=2)
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title('Mixed: ABC123')
    axes[1].axis('off')
    
    # Centered in larger image
    img3 = draw_text("FIBSEM", size=(24, 18), spacing=3, thickness=2, image_shape=(64, 150))
    axes[2].imshow(img3, cmap='gray')
    axes[2].set_title('Centered: FIBSEM')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return img1, img2, img3

if __name__ == "__main__":
    # Test the functions
    test_draw_numbers()
    test_draw_letters()