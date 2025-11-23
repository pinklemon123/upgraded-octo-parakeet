from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
from pathlib import Path
import os
import random

app = FastAPI()

BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "pdfs"
PDF_DIR.mkdir(exist_ok=True)

app.mount("/pdfs", StaticFiles(directory=str(PDF_DIR)), name="pdfs")

class PageLayoutReq(BaseModel):
    file_id: str
    page: int

class TranslateLayoutReq(BaseModel):
    direction: str  # "en2zh" or "zh2en"
    layout: dict

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
    # 假设 3 页
    total_pages = 3
    return {
        "file_id": file_id,
        "total_pages": total_pages,
        "pdf_url": f"/pdfs/{file_id}.pdf"
    }

@app.post("/api/page_layout")
async def page_layout(req: PageLayoutReq):
    # 返回假 block 数据，方便前端测试
    width, height = 1024, 1448
    blocks = []
    for i in range(1, 6):
        blocks.append({
            "id": i,
            "category": random.choice(["title", "paragraph", "caption"]),
            "bbox": [100 + i*30, 80 + i*100, 900 - i*20, 160 + i*200],
            "text": f"Block {i} text for page {req.page}."
        })
    return {"width": width, "height": height, "blocks": blocks}

@app.post("/api/translate_layout")
async def translate_layout(req: TranslateLayoutReq):
    # 遍历 blocks，假装 AI 翻译
    layout = req.layout
    direction = req.direction
    for blk in layout.get("blocks", []):
        src = blk.get("text", "")
        if direction == "en2zh":
            blk["text_translated"] = f"[中] {src}"
        else:
            blk["text_translated"] = f"[EN] {src}"
    return layout
