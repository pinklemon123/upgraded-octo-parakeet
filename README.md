# 📘 PDF Page Translator | PDF 单页/整页对照翻译工具

一个支持 **PDF 结构提取 → JSON 编辑 → AI 翻译 → 对照 PDF 生成** 的完整翻译工作流工具。
能够让 PDF 文档的原文与译文以 **左右/上下对照** 的方式生成美观的 PDF，
并且 **无需重复调用 AI**（可直接使用右侧 JSON 中已有的翻译内容）。

---

## ✨ 功能特点

### 🔍 1. PDF 页面布局自动识别
- 内置版面分析（OCR fallback）
- 自动提取段落 → 生成可编辑的 JSON

### 📝 2. JSON 可视化编辑
- 右下角提供可直接编辑的 JSON 树
- 支持人工修订翻译结果
- 如果 JSON 中已有 `text_translated`，将 **直接使用，不重复调用 API**

### 🤖 3. AI 翻译（OpenAI）
- 针对每个 block 做分段翻译
- chunk 级容错（自动重试）
- 支持 **英 → 中**、**中 → 英**

### 🧩 4. 对照 PDF 自动生成（核心）
支持两种模式：

1. **PyMuPDF 版（弃用）**  
   - 在 Windows 环境下会出现中文字体渲染问题，已替换为 HTML 渲染方案。

### 2.1 自定义中文字体（推荐）

为确保中文渲染稳定，建议设置一个本地中文 TTF/TTC 字体供渲染使用，例如“华文中宋”：

```powershell
# 设置当前会话的中文字体路径（示例）
$env:DUAL_PDF_FONT_PATH = "F:\\creat\\tranla8964\\华文中宋.ttf"

# （可选）持久化到用户环境变量
setx DUAL_PDF_FONT_PATH "F:\\creat\\tranla8964\\华文中宋.ttf"
```

说明：后端会在渲染 HTML → PDF 时通过 `@font-face` 引入该字体，wkhtmltopdf 将使用该字体进行中文渲染。

2. ⭐ **HTML + wkhtmltopdf 高质量渲染（当前方案）**  
   - 左侧：原始 PDF 页面截图  
   - 右侧：整页译文  
   - 完全解决中文乱码 / 字体缺失问题  
   - 支持自定义中文字体（如华文中宋、宋体、思源黑体等）

### 💾 5. 不浪费 API 次数（智能翻译跳过）
- 后端只会翻译 **没有 text_translated 的段落**  
- 手工修改过的译文 **不会被覆盖**

---

## 🖼 界面展示（示例）

> 左：原文 PDF  
> 右：生成后的整页译文 JSON（可编辑）  
> 下：一键生成对照 PDF

（你可以替换成自己的截图）

---

## 📄 生成的对照 PDF（示例）

- 上方：原文页面截图  
- 下方：对应整页 AI/人工译文  
- 字体渲染稳定（推荐中文字体：华文中宋、思源黑体、微软雅黑）
---

## 🏗 技术架构

```
------------------------+
| 前端 Frontend         |
| ---------------------- |
| 上传 PDF               |
| 查看 & 编辑 JSON       |
| 调用 /translate        |
| 调用 /export PDF       |

---

## 📦 依赖（requirements）

提供一个最小依赖清单，便于快速安装：

```
fastapi
uvicorn
PyMuPDF
onnxruntime
rapidocr-onnxruntime
openai
python-dotenv
```

安装命令：

```powershell
pip install -r requirements.txt
```
| +-----------+--------+ |
```
        |
        v
```
------------------------+
|       后端 Backend     |
| FastAPI + Python       |
| ---------------------- |
| PyMuPDF: 页面截图       |
| OCR + Block Layout     |
| AI 段落翻译（OpenAI）   |
| JSON 翻译缓存机制      |
| HTML 模板排版          |
| wkhtmltopdf → PDF      |
| +-----------+--------+ |
```
        |
        v
```
------------------------+
|     输出翻译对照 PDF    |
------------------------+
```

---

## 🚀 本地运行

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

### 2. 安装 wkhtmltopdf（必须）

下载地址（Windows/macOS/Linux）：
https://wkhtmltopdf.org/

安装后设置环境变量（或在 `dual_pdf_html.py` 中写死绝对路径）：

```powershell
setx WKHTMLTOPDF_PATH "C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
```

### 3. 配置 OpenAI API Key

在项目根目录创建并编辑 `.env`：

```
OPENAI_API_KEY=你的key
OPENAI_BASE_URL=https://api.openai.com/v1   # 可选
```

### 4. 启动服务

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000
```

打开浏览器：

```
http://127.0.0.1:8000/page_json_translate.html
```

开始工作流：

1. 上传 PDF
2. 生成布局 JSON
3. AI 翻译（或手动修改 JSON）
4. 生成对照 PDF

---

## 🔧 目录结构（示例）

```
├── main.py                  # FastAPI 主入口
├── dual_pdf_html.py         # HTML → PDF 渲染核心
├── dual_pdf.py              #（旧版）PyMuPDF 渲染，对中文不稳定
├── ocr_site/                # OCR / 布局分析模块
├── pdfs/                    # 输出 PDF 存放目录
├── page_json_translate.html # 前端页面
└── README.md
```

---

## 📌 TODO（未来可拓展）

- 添加“左右对照”专业版排版
- 支持整本书籍翻译缓存
- 支持 PDF 批量导入
- 支持自动分页、多列排版
- 添加用户上传自定义字体功能
- 生成 EPUB / HTML Web 阅读版
- PDF 内嵌高亮（标注差异）

---

## 🎉 致谢 / Thanks

本项目用于探索 **AI + 文档编辑工作流** 的可能性：

- 大幅减少手动处理 PDF 的成本
- 保留可编辑 JSON 的高可控性
- 自动生成高质量对照 PDF

欢迎 Issue / PR！# upgraded-octo-parakeet

本项目把「单页 PDF → 布局检测（DocLayout-YOLO）→ OCR（RapidOCR / Tesseract）→ 结构化 JSON → AI 翻译」的流程串成一个本地前后端工具。

主要内容：
- 后端：`main.py`（FastAPI），提供 `/api/page_layout`、`/api/translate_layout` 等接口。
- 前端：`page_json_translate.html` + `page_json_translate.css`，用于上传 PDF、生成单页 JSON、可视化布局并调用翻译。
- 辅助脚本：`pdf_to_blocks.py`、`convert_pdf_to_html.py`、`translate_blocks_openai.py`、`translate_blocks_doclayout.py`。
- 模型/测试数据：放在 `models/` 与 `pdfs/`（大文件建议使用 Git LFS 或外部下载）。

快速开始（Windows / PowerShell）

```powershell
git clone git@github.com:pinklemon123/upgraded-octo-parakeet.git
cd upgraded-octo-parakeet
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
# 打开页面： http://127.0.0.1:8000/page_json_translate.html
```

检查后端：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/ping -Method Get
# 期望返回: {"ok": true, "msg": "backend alive"}
```

外部依赖

- Tesseract OCR（Windows）：建议使用 UB-Mannheim 构建，安装后确保 `tesseract.exe` 在 PATH 中。
- ONNX 运行时：`onnxruntime`（CPU）或 `onnxruntime-gpu`（GPU）根据你的硬件选择。

关于大文件

- 如果项目需要大型模型或测试 PDF（例如 `models/*.onnx`、`pdfs/*.pdf`），建议使用 Git LFS 或把文件上传到 Release/云盘并在 `INSTALL.md` 中提供下载链接。

更多详细的安装与配置步骤请参阅 `INSTALL.md`。

如果你需要我：
- 把大文件迁移到 Git LFS，或
- 把大文件从仓库移除并把下载脚本/链接写入 README/INSTALL

直接告诉我你的偏好，我会继续操作。
# PDF → HTML + OCR 文字层

## 1. 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install pymupdf pytesseract pillow
```

安装 Tesseract OCR（Windows）：
- 访问 https://github.com/UB-Mannheim/tesseract/wiki 下载并安装。
- 安装后，确保 tesseract.exe 已加入 PATH。
- 测试：
```powershell
tesseract --version
```

## 2. 使用方法

1. 将 PDF 文件放到本目录。
2. 激活虚拟环境：
```powershell
.\.venv\Scripts\activate
```
3. 运行脚本：
```powershell
python convert_pdf_to_html.py "你的PDF文件.pdf"
```

生成的 HTML 文件和图片文件夹会在同目录下。

## 3. 浏览器翻译

用 Chrome/Edge 打开生成的 .ocr.html 文件，使用浏览器自带翻译功能即可。

---

如遇问题请贴报错，我会帮你解决。
