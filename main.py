from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
from pathlib import Path
import fitz
import cv2
import numpy as np
import onnxruntime as ort
from rapidocr_onnxruntime import RapidOCR
import os
from openai import OpenAI

app = FastAPI()

# 根路径欢迎信息
@app.get("/")
def read_root():
    return {"message": "FastAPI PDF 后端已启动，API 路径请访问 /api/*"}


# Serve frontend files directly for convenience
@app.get("/page_json_translate.html")
def serve_page():
    path = BASE_DIR / "page_json_translate.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="前端页面不存在")
    return FileResponse(path)


@app.get("/page_json_translate.css")
def serve_css():
    path = BASE_DIR / "page_json_translate.css"
    if not path.exists():
        raise HTTPException(status_code=404, detail="样式文件不存在")
    return FileResponse(path)
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "pdfs"
PDF_DIR.mkdir(exist_ok=True)
MODEL_PATH = BASE_DIR / "models/doclayout_yolo_docstructbench_imgsz1024.onnx"

app.mount("/pdfs", StaticFiles(directory=str(PDF_DIR)), name="pdfs")

class PageLayoutReq(BaseModel):
    file_id: str
    page: int

class TranslateLayoutReq(BaseModel):
    direction: str
    layout: dict

# DocLayout-YOLO wrapper
class DocLayoutYolo:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.img_size = 1024
        # detection score threshold (can lower if missing boxes)
        self.score_thresh = 0.25
        self.id2label = {
            0: "text", 1: "title", 2: "figure", 3: "table", 4: "caption",
            5: "header", 6: "footer", 7: "reference", 8: "equation"
        }

    def preprocess(self, img):
        h, w = img.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 114
        canvas[:nh, :nw, :] = resized
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None, ...]
        return blob, scale, (h, w)

    def detect(self, img):
        blob, scale, (orig_h, orig_w) = self.preprocess(img)
        # Run model and normalize output to 2D numpy array
        raw_out = self.session.run(None, {self.session.get_inputs()[0].name: blob})[0]
        outputs = np.array(raw_out)
        if outputs.ndim == 1:
            outputs = outputs.reshape(1, -1)
        boxes = []
        for row in outputs:
            if len(row) < 6:
                continue
            # Be flexible: accept rows with >=6 cols. Take first 6 as x1,y1,x2,y2,score,cls
            x1, y1, x2, y2, score, cls = row[:6]
            try:
                score = float(score)
            except Exception:
                continue
            if score < self.score_thresh:
                continue
            try:
                cls_i = int(cls)
            except Exception:
                # if class is one-hot or array, try to pick argmax
                try:
                    cls_i = int(np.argmax(row[5:]))
                except Exception:
                    cls_i = 0
            label = self.id2label.get(int(cls_i), "other")
            boxes.append({
                "bbox": [
                    int(x1 / scale),
                    int(y1 / scale),
                    int(x2 / scale),
                    int(y2 / scale),
                ],
                "score": float(score),
                "label": label
            })
        return boxes

# OCR + line merge
def merge_ocr_lines(ocr_result):
    words = []
    for box, text, score in ocr_result:
        if score < 0.5:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        left, top = min(xs), min(ys)
        words.append({"text": text, "left": left, "top": top})
    if not words:
        return []
    words.sort(key=lambda x: (x["top"], x["left"]))
    lines = []
    cur = [words[0]]
    last_top = words[0]["top"]
    for w in words[1:]:
        if abs(w["top"] - last_top) <= 15:
            cur.append(w)
        else:
            lines.append(cur)
            cur = [w]
        last_top = w["top"]
    lines.append(cur)
    merged = []
    for ln in lines:
        ln = sorted(ln, key=lambda x: x["left"])
        merged.append(" ".join([w["text"] for w in ln]))
    return merged

# 后端接口
@app.get("/api/ping")
def ping():
    return {"ok": True, "msg": "backend alive"}

@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")
    file_id = str(uuid.uuid4())
    pdf_path = PDF_DIR / f"{file_id}.pdf"
    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)
    # 用 PyMuPDF 获取页数
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    return {
        "file_id": file_id,
        "total_pages": total_pages,
        "pdf_url": f"/pdfs/{file_id}.pdf"
    }

@app.post("/api/page_layout")
async def page_layout(req: PageLayoutReq):
    pdf_path = PDF_DIR / f"{req.file_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF 文件不存在")
    doc = fitz.open(str(pdf_path))
    if req.page < 1 or req.page > len(doc):
        raise HTTPException(status_code=400, detail="页码超出范围")
    page = doc[req.page - 1]
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    layout = DocLayoutYolo(MODEL_PATH)
    ocr = RapidOCR()
    boxes = layout.detect(img)
    # debug info: print number of raw detections
    print(f"[page_layout] raw detections: {len(boxes)}")
    # broaden accepted labels (DocLayout label set may vary)
    valid_labels = {"text", "title", "caption", "reference", "header", "footer", "figure", "table"}
    regions = [b for b in boxes if b.get("label") in valid_labels]
    print(f"[page_layout] filtered regions: {len(regions)}")
    blocks = []
    for rid, reg in enumerate(regions):
        x1, y1, x2, y2 = reg["bbox"]
        # clip bbox to image bounds
        h, w = img.shape[:2]
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            print(f"[page_layout] skip invalid bbox: {reg['bbox']}")
            continue
        crop = img[y1:y2, x1:x2]
        try:
            ocr_result, _ = ocr(crop)
            lines = merge_ocr_lines(ocr_result)
            block_text = "\n".join(lines)
        except Exception as e:
            print(f"[page_layout] OCR failed for region {rid}: {e}")
            block_text = ""
        blocks.append({
            "id": rid,
            "category": reg.get("label", "unknown"),
            "bbox": [x1, y1, x2, y2],
            "text": block_text
        })
    # Fallback: if no regions found, return whole-page OCR as a single block
    if len(blocks) == 0:
        print("[page_layout] no blocks found, running full-page OCR fallback")
        try:
            ocr_result, _ = ocr(img)
            lines = merge_ocr_lines(ocr_result)
            full_text = "\n".join(lines)
        except Exception as e:
            print(f"[page_layout] full-page OCR failed: {e}")
            full_text = ""
        blocks.append({
            "id": 0,
            "category": "ocr_full",
            "bbox": [0, 0, img.shape[1], img.shape[0]],
            "text": full_text
        })
    return {
        "width": img.shape[1],
        "height": img.shape[0],
        "blocks": blocks
    }

@app.post("/api/translate_layout")
async def translate_layout(req: TranslateLayoutReq):
    direction = req.direction
    layout = req.layout
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(api_key=api_key, **client_kwargs)
    if direction == "en2zh":
        system_prompt = "你是一个专业的英→中科技论文翻译助手。要求：忠实原文，术语准确，风格正式。"
    else:
        system_prompt = "You are a professional assistant for translating Chinese scientific papers to English. Be accurate, formal, and keep technical terms."
    for blk in layout.get("blocks", []):
        src = blk.get("text", "")
        if not src.strip():
            blk["text_translated"] = ""
            continue
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": src},
            ],
            temperature=0.1,
        )
        blk["text_translated"] = resp.choices[0].message.content.strip()
    return layout