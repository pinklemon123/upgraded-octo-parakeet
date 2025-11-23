import fitz
from PIL import Image, ImageDraw, ImageFont
import io

# Use a font that supports both Chinese and English
# Simsun is a common choice on Windows. For other systems, you might need to change this.
# On Debian/Ubuntu: sudo apt-get install fonts-noto-cjk
# On macOS, PingFang SC is a good choice.
try:
    FONT = ImageFont.truetype("simsun.ttc", 14)
except IOError:
    try:
        FONT = ImageFont.truetype("msyh.ttc", 14) # Microsoft YaHei
    except IOError:
        try:
            FONT = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 14)
        except IOError:
            print("警告：找不到中文字体（如宋体、微软雅黑、苹方）。将使用默认字体，中文可能无法显示。")
            FONT = ImageFont.load_default()


def draw_text_with_wrap(draw, text, position, max_width, font, fill="black"):
    """
    Draws text on an image, wrapping it to fit within a specified width.
    """
    x, y = position
    lines = []
    
    # Simple wrapping logic
    if font.getlength(text) <= max_width:
        lines.append(text)
    else:
        current_line = ""
        for char in text:
            if font.getlength(current_line + char) > max_width:
                lines.append(current_line)
                current_line = char
            else:
                current_line += char
        lines.append(current_line)

    line_height = font.getbbox("A")[3] + 2  # Estimate line height
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height


def build_dual_pdf_page(
    original_page_img: Image.Image,
    translated_layout: dict,
    direction: str
):
    """
    Creates a side-by-side image for a single page.
    This version aggregates all translated text and draws it in a single block.
    """
    w, h = original_page_img.size
    # Create a new canvas twice the width of the original page
    canvas = Image.new("RGB", (w * 2, h), "white")
    
    # Paste original page on the left
    canvas.paste(original_page_img, (0, 0))
    
    # Create a drawing context for the right side
    draw = ImageDraw.Draw(canvas)
    
    # Determine which key holds the translated text
    tgt_key = "text_translated" if direction == "en2zh" else "text"

    # Aggregate all translated text blocks into a single string
    # Blocks are sorted by their vertical position to maintain reading order.
    sorted_blocks = sorted(translated_layout.get("blocks", []), key=lambda b: b['bbox'][1])
    full_translated_text = "\n\n".join(
        b.get(tgt_key, "") for b in sorted_blocks if b.get(tgt_key, "").strip()
    )

    # Define the rectangle for the translated text on the right side
    # Add some padding (e.g., 20px)
    padding = 20
    right_rect_x_start = w + padding
    right_rect_y_start = padding
    right_rect_width = w - (2 * padding)
    
    # Draw the aggregated text with wrapping
    if full_translated_text:
        draw_text_with_wrap(
            draw,
            full_translated_text,
            (right_rect_x_start, right_rect_y_start),
            right_rect_width,
            FONT,
            fill="black" # Use black for better readability
        )
            
    return canvas


def build_dual_pdf(
    pdf_path_str: str,
    get_layout_func,
    translate_func,
    direction: str,
    output_pdf_path: str
):
    """
    Main function to generate the dual-language PDF.
    """
    pdf_doc = fitz.open(pdf_path_str)
    page_images = []

    for i, page in enumerate(pdf_doc):
        print(f"Processing page {i+1}/{len(pdf_doc)}...")
        
        # 1. Get original page as image
        pix = page.get_pixmap(dpi=200)
        original_img = Image.open(io.BytesIO(pix.tobytes()))

        # 2. Get layout
        layout = get_layout_func(pdf_path_str, i + 1)
        
        # 3. Translate layout
        translated_layout = translate_func(layout, direction)
        
        # 4. Build the side-by-side page image
        combined_img = build_dual_pdf_page(original_img, translated_layout, direction)
        page_images.append(combined_img)

    if not page_images:
        raise Exception("No pages were processed.")

    # 5. Save all combined images into a new PDF
    page_images[0].save(
        output_pdf_path,
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=page_images[1:]
    )
    print(f"Dual language PDF saved to: {output_pdf_path}")
