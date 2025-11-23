import argparse
import json
from pathlib import Path

import cv2
import fitz
from rapidocr_onnxruntime import RapidOCR

def render_pdf_to_images(pdf_path: Path, images_dir: Path, dpi: int = 300):
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

def ocr_page(img_path: Path, ocr_engine: RapidOCR, score_thresh=0.5):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    result, _ = ocr_engine(img)
    lines = []
    if not result:
        return lines

    for box, text, score in result:
        if score < score_thresh:
            continue
        text = text.strip()
        if not text:
            continue

        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        left, top = min(xs), min(ys)

        lines.append(
            {
                "text": text,
                "left": float(left),
                "top": float(top),
            }
        )

    # 排序：从上到下、从左到右
    lines.sort(key=lambda x: (x["top"], x["left"]))
    return lines

def main():
    parser = argparse.ArgumentParser(
        description="Extract text blocks from PDF using RapidOCR."
    )
    parser.add_argument("pdf", help="Path to input PDF file")
    parser.add_argument(
        "--out",
        help="Output JSON file (default: same as pdf with .blocks.json)",
        default=None,
    )
    parser.add_argument(
        "--score-thresh", type=float, default=0.5,
        help="Min confidence for OCR lines (default 0.5)"
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        return

    if args.out:
        out_json = Path(args.out).expanduser().resolve()
    else:
        out_json = pdf_path.with_suffix(".blocks.json")

    images_dir = out_json.parent / (pdf_path.stem + "_pages_rapid_blocks")

    print("[INFO] Rendering pages...")
    page_infos = render_pdf_to_images(pdf_path, images_dir)

    print("[INFO] Initializing RapidOCR...")
    ocr_engine = RapidOCR()

    blocks = []
    block_id = 0

    print("[INFO] OCR pages and collect text blocks...")
    for info in page_infos:
        page_idx = info["index"]
        img_path = info["image_path"]
        print(f"  - Page {page_idx}: {img_path.name}")
        lines = ocr_page(img_path, ocr_engine, score_thresh=args.score_thresh)
        # 这里先把每一行当成一个 block
        for line in lines:
            text = line["text"]
            if not text.strip():
                continue
            blocks.append(
                {
                    "id": block_id,
                    "page": page_idx,
                    "text": text,
                }
            )
            block_id += 1

    print(f"[INFO] Total blocks: {len(blocks)}")
    out_json.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Blocks saved to: {out_json}")

if __name__ == "__main__":
    main()
