import argparse
import os
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from pytesseract import Output
# 硬指定 tesseract.exe 路径（请根据你的实际安装路径修改）
pytesseract.pytesseract.tesseract_cmd = r"E:\enngering\tesseract.exe"


def render_pdf_to_images(pdf_path: Path, images_dir: Path, dpi: int = 144):
    images_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    page_infos = []

    for page_index, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img_path = images_dir / f"page_{page_index + 1}.png"
        pix.save(str(img_path))
        page_infos.append(
            {
                "index": page_index + 1,
                "image_path": img_path,
                "width": pix.width,
                "height": pix.height,
            }
        )
    return page_infos


def ocr_image(image_path: Path, lang: str = "eng"):
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, lang=lang, output_type=Output.DICT)
    texts = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if not text:
            continue
        if conf < 50:  # 置信度太低的忽略
            continue
        left = data["left"][i]
        top = data["top"][i]
        width = data["width"][i]
        height = data["height"][i]
        texts.append(
            {
                "text": text,
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            }
        )
    return texts


def generate_html_per_page(pdf_name: str, page_infos, ocr_results, output_dir: Path):
        """
        为每一页生成一个独立 HTML 文件
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = page_infos[0]["image_path"].parent
        rel_images_dir = os.path.relpath(images_dir, output_dir)

        head = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"UTF-8\">
<title>OCR View - {pdf_name} - Page {page_num}</title>
<style>
body {{
    background: #f0f0f0;
    font-family: Arial, sans-serif;
}}
.container {{
    max-width: 1200px;
    margin: 20px auto;
}}
.page {{
    position: relative;
    margin: 20px auto;
    box-shadow: 0 0 8px rgba(0,0,0,0.2);
    background: #fff;
}}
.page-bg {{
    display: block;
    width: 100%;
    height: auto;
}}
.ocr-layer {{
    position: absolute;
    left: 0;
    top: 0;
}}
.ocr-text {{
    position: absolute;
    white-space: pre;
    color: transparent;
    text-shadow: 0 0 0 #000;
    font-size: 12px;
}}
.page-num {{
    text-align: center;
    margin-top: 4px;
    color: #666;
    font-size: 12px;
}}
</style>
</head>
<body>
<div class=\"container\">
<h2>OCR View - {pdf_name} - Page {page_num}</h2>
<p>提示：在浏览器中打开本页面后，可以使用浏览器自带的「翻译此页」功能，将文字翻译为中文。</p>
"""

        tail = """
</div>
</body>
</html>
"""

        for info in page_infos:
                idx = info["index"]
                img_path = info["image_path"]
                width = info["width"]
                height = info["height"]
                rel_img = os.path.join(rel_images_dir, img_path.name).replace("\\", "/")

                page_html = [
                        f'<div class="page" style="width:{width}px;">',
                        f'  <img class="page-bg" src="{rel_img}" alt="page {idx}">',
                        f'  <div class="ocr-layer" style="width:{width}px; height:{height}px;">',
                ]

                for item in ocr_results.get(idx, []):
                        text = item["text"]
                        left = item["left"]
                        top = item["top"]
                        w = item["width"]
                        h = item["height"]
                        font_size = max(8, int(h * 0.8))
                        span = (
                                f'    <span class="ocr-text" '
                                f'style="left:{left}px; top:{top}px; width:{w}px; height:{h}px; '
                                f'font-size:{font_size}px;">{text}</span>'
                        )
                        page_html.append(span)

                page_html.append("  </div>")
                page_html.append("</div>")
                page_html.append(f'<div class="page-num">Page {idx}</div>')

                html_content = head.format(pdf_name=pdf_name, page_num=idx) + "\n".join(page_html) + tail
                html_file = output_dir / f"{Path(pdf_name).stem}_page_{idx}.html"
                html_file.write_text(html_content, encoding="utf-8")
                print(f"[OK] HTML saved to: {html_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to per-page HTML with OCR text layer (for browser translation)."
    )
    parser.add_argument("pdf", help="Path to input PDF file")
    parser.add_argument(
        "--outdir",
        help="Output HTML directory (default: same folder, _html_pages)",
        default=None,
    )
    parser.add_argument(
        "--ocr-lang",
        help="Tesseract OCR language code (default: eng)",
        default="eng",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        return

    if args.outdir:
        output_dir = Path(args.outdir).expanduser().resolve()
    else:
        output_dir = Path(pdf_path.parent) / "web"

    images_dir = pdf_path.parent / (pdf_path.stem + "_pages")

    print("[INFO] Rendering PDF pages to images...")
    page_infos = render_pdf_to_images(pdf_path, images_dir)

    print(f"[INFO] Running OCR on each page (lang={args.ocr_lang})...")
    ocr_results = {}
    for info in page_infos:
        idx = info["index"]
        img_path = info["image_path"]
        print(f"  - Page {idx}: {img_path.name}")
        texts = ocr_image(img_path, lang=args.ocr_lang)
        ocr_results[idx] = texts

    print("[INFO] Generating per-page HTML files...")
    generate_html_per_page(pdf_path.name, page_infos, ocr_results, output_dir)


if __name__ == "__main__":
    main()
