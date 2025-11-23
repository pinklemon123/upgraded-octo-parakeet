import argparse
import json
import os
from pathlib import Path
from time import sleep
from openai import OpenAI

SYSTEM_PROMPT = """你是一个专业的英→中科技论文翻译助手。
要求：
1. 严格忠实原文，不要编造内容，不要省略信息。
2. 风格偏正式学术论文风格，术语准确，避免口语。
3. 保持句子结构清晰，可以适度调整语序以符合中文习惯。
4. 不要翻译人名、机构名中的专有名词（如：MIT、NVIDIA）。
5. 如果遇到公式、符号，保持原样输出。"""

def translate_text(client: OpenAI, model: str, text: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()

def main():
    parser = argparse.ArgumentParser(
        description="Translate DocLayout blocks using OpenAI."
    )
    parser.add_argument("blocks_json", help="Path to blocks.json")
    parser.add_argument(
        "--out",
        help="Output HTML file (default: reader.html)",
        default=None,
    )
    parser.add_argument(
        "--model",
        help="OpenAI model name (default: gpt-4o-mini)",
        default="gpt-4o-mini",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Only translate first N blocks (for testing).",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=None,
        help="Only translate blocks from the specified page (for DocLayout blocks).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set in environment.")
        return

    client_kwargs = {}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(api_key=api_key, **client_kwargs)

    blocks_path = Path(args.blocks_json).expanduser().resolve()
    blocks = json.loads(blocks_path.read_text(encoding="utf-8"))

    # 只翻译指定页
    if args.page is not None:
        blocks = [b for b in blocks if b.get("page") == args.page]

    if args.max_blocks is not None:
        blocks = blocks[: args.max_blocks]

    print(f"[INFO] Total blocks to translate: {len(blocks)}")

    translated_blocks = []
    for i, blk in enumerate(blocks):
        text = blk["text"]
        if not text.strip():
            zh = ""
        else:
            while True:
                try:
                    zh = translate_text(client, args.model, text)
                    break
                except Exception as e:
                    print(f"[WARN] error on block {blk.get('region_id', i)}: {e}")
                    print("  retry in 2s...")
                    sleep(2)

        new_blk = dict(blk)
        new_blk["text_zh"] = zh
        translated_blocks.append(new_blk)
        if (i + 1) % 10 == 0:
            print(f"  translated {i+1}/{len(blocks)} blocks")

    # 保存翻译后的 JSON
    out_json = blocks_path.with_suffix(".translated.json")
    out_json.write_text(json.dumps(translated_blocks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Translated blocks json saved to: {out_json}")

    # 生成中英对照 HTML
    if args.out:
        out_html = Path(args.out).expanduser().resolve()
    else:
        out_html = blocks_path.parent / "reader.html"

    # 按页分组
    by_page = {}
    for blk in translated_blocks:
        by_page.setdefault(blk["page"], []).append(blk)

    parts = [
        "<!DOCTYPE html>",
        "<html lang='zh-CN'>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<title>PDF中英对照阅读器</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#f5f5f5; }",
        ".page { margin: 20px auto; max-width: 900px; background:#fff; padding:16px 24px; box-shadow:0 0 8px rgba(0,0,0,0.15); }",
        ".block { display:flex; gap:32px; margin-bottom: 18px; }",
        ".en { color:#444; font-size:14px; width:48%; word-break:break-all; }",
        ".zh { color:#000; font-size:15px; width:48%; word-break:break-all; }",
        ".label { color:#888; font-size:12px; margin-bottom:4px; }",
        ".page-num { text-align:center; color:#888; font-size:12px; margin-bottom:10px; }",
        "</style>",
        "</head>",
        "<body>",
        "<h2 style='text-align:center;'>PDF 中英对照阅读器（按页面 & 段落）</h2>",
    ]

    for page in sorted(by_page.keys()):
        parts.append("<div class='page'>")
        parts.append(f"<div class='page-num'>Page {page}</div>")
        for blk in by_page[page]:
            parts.append("<div class='block'>")
            parts.append(f"<div class='en'><div class='label'>{blk['label']}</div>{blk['text']}</div>")
            parts.append(f"<div class='zh'><div class='label'>译文</div>{blk['text_zh']}</div>")
            parts.append("</div>")
        parts.append("</div>")

    parts.append("</body></html>")

    out_html.write_text("\n".join(parts), encoding="utf-8")
    print(f"[OK] HTML saved to: {out_html}")

if __name__ == "__main__":
    main()
