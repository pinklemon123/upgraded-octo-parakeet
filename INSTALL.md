# 安装与配置（快速指南）

下面是给想要下载并在本地运行该项目的人的一步步说明（Windows / PowerShell）。包含虚拟环境、依赖、外部程序、以及如何处理大型模型/PDF 的建议。

## 1. 克隆仓库

```powershell
git clone git@github.com:pinklemon123/upgraded-octo-parakeet.git
cd upgraded-octo-parakeet
```

## 2. （可选）Git LFS（推荐用于大文件）

如果仓库使用 Git LFS 管理了模型或 PDF，请安装并拉取：

```powershell
# 安装： https://git-lfs.github.com/
git lfs install
git lfs pull
```

## 3. 创建虚拟环境并激活

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 4. 安装 Python 依赖

```powershell
pip install -r requirements.txt
```

## 5. 安装外部工具

- Tesseract OCR（Windows）：
  - 推荐下载链接（UB-Mannheim）：https://github.com/UB-Mannheim/tesseract/wiki
  - 安装后确保 `tesseract.exe` 在 PATH 中，或在脚本中设置 `pytesseract.pytesseract.tesseract_cmd`。

## 6. 环境变量（翻译功能）

如果要使用 OpenAI 翻译功能，请设置：

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_BASE_URL = "https://your-openai-proxy.example.com" # 如需
```

## 7. 启动后端（开发）

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# 打开页面： http://127.0.0.1:8000/page_json_translate.html
```

## 8. 常见命令示例

- Ping 后端：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/ping -Method Get
```

- 请求单页布局 JSON（示例）：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/page_layout -Method Post -Body (@{ file_id='YOUR_FILE_ID'; page=1 } | ConvertTo-Json) -ContentType 'application/json'
```

## 9. 关于大文件和模型

- 如果你不希望把模型或大 PDF 放到 Git 仓库：
  - 将模型上传到 GitHub Release、私有云盘或对象存储；在仓库 README/INSTALL 中放下载链接与校验信息。
  - 在本地把模型放到 `models/` 目录，PDF 放到 `pdfs/` 目录。

## 10. 我可以帮你做的事

- 将 `models/*.onnx` 与 `pdfs/*.pdf` 迁移到 Git LFS 并修正远端历史；或
- 将大文件从仓库移出并在 README 中添加下载脚本/链接。

请告诉我你希望我执行哪一步，我将继续操作。
