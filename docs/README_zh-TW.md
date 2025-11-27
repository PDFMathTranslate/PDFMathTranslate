<div align="center">

[English](../README.md) | [简体中文](README_zh-CN.md) | 繁體中文 | [日本語](README_ja-JP.md)

<img src="./images/banner.png" width="320px"  alt="PDF2ZH"/>  

<h2 id="title">PDFMathTranslate</h2>

<p>
  <!-- PyPI -->
  <a href="https://pypi.org/project/pdf2zh/">
    <img src="https://img.shields.io/pypi/v/pdf2zh"/></a>
  <a href="https://pepy.tech/projects/pdf2zh">
    <img src="https://static.pepy.tech/badge/pdf2zh"></a>
  <a href="https://hub.docker.com/repository/docker/byaidu/pdf2zh">
    <img src="https://img.shields.io/docker/pulls/byaidu/pdf2zh"></a>
  <!-- License -->
  <a href="./LICENSE">
    <img src="https://img.shields.io/github/license/Byaidu/PDFMathTranslate"/></a>
  <a href="https://huggingface.co/spaces/reycn/PDFMathTranslate-Docker">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97-Online%20Demo-FF9E0D"/></a>
  <a href="https://www.modelscope.cn/studios/AI-ModelScope/PDFMathTranslate">
    <img src="https://img.shields.io/badge/ModelScope-Demo-blue"></a>
  <a href="https://github.com/Byaidu/PDFMathTranslate/pulls">
    <img src="https://img.shields.io/badge/contributions-welcome-green"/></a>
  <a href="https://gitcode.com/Byaidu/PDFMathTranslate/overview">
    <img src="https://gitcode.com/Byaidu/PDFMathTranslate/star/badge.svg"></a>
  <a href="https://t.me/+Z9_SgnxmsmA5NzBl">
    <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white"/></a>
</p>

<a href="https://trendshift.io/repositories/12424" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12424" alt="Byaidu%2FPDFMathTranslate | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

</div>

科學 PDF 文件翻譯及雙語對照工具

- 📊 保留公式、圖表、目錄和註釋 *([預覽效果](#preview))*
- 🌐 支援 [多種語言](#language) 和 [諸多翻譯服務](#services)
- 🤖 提供 [命令列工具](#usage)、[圖形使用者介面](#gui)，以及 [容器化部署](#docker)

歡迎在 [GitHub Issues](https://github.com/Byaidu/PDFMathTranslate/issues) 或 [Telegram 使用者群](https://t.me/+Z9_SgnxmsmA5NzBl)(https://qm.qq.com/q/DixZCxQej0) 中提出回饋

如需瞭解如何貢獻的詳細資訊，請查閱 [貢獻指南](https://github.com/Byaidu/PDFMathTranslate/wiki/Contribution-Guide---%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97)

<h2 id="updates">近期更新</h2>

- [Dec. 24 2024] 翻譯功能支援接入由 [Xinference](https://github.com/xorbitsai/inference) 執行的本機 LLM _(by [@imClumsyPanda](https://github.com/imClumsyPanda))_
- [Nov. 26 2024] CLI 現在已支援（多個）線上 PDF 檔 *(by [@reycn](https://github.com/reycn))*  
- [Nov. 24 2024] 為了降低依賴大小，提供 [ONNX](https://github.com/onnx/onnx) 支援 *(by [@Wybxc](https://github.com/Wybxc))*  
- [Nov. 23 2024] 🌟 [免費公共服務](#demo) 上線！ *(by [@Byaidu](https://github.com/Byaidu))*  
- [Nov. 23 2024] 新增防止網頁爬蟲的防火牆 *(by [@Byaidu](https://github.com/Byaidu))*  
- [Nov. 22 2024] 圖形使用者介面現已支援義大利語並進行了一些更新 *(by [@Byaidu](https://github.com/Byaidu), [@reycn](https://github.com/reycn))*  
- [Nov. 22 2024] 現在你可以將自己部署的服務分享給朋友 *(by [@Zxis233](https://github.com/Zxis233))*  
- [Nov. 22 2024] 支援騰訊翻譯 *(by [@hellofinch](https://github.com/hellofinch))*  
- [Nov. 21 2024] 圖形使用者介面現在支援下載雙語文件 *(by [@reycn](https://github.com/reycn))*  
- [Nov. 20 2024] 🌟 提供了 [線上示範](#demo)！ *(by [@reycn](https://github.com/reycn))*  

<h2 id="preview">效果預覽</h2>

<div align="center">
<img src="./images/preview.gif" width="80%"/>
</div>

<h2 id="demo">線上示範 🌟</h2>

### 免費服務 (<https://pdf2zh.com/>)

你可以立即嘗試 [免費公共服務](https://pdf2zh.com/) 而無需安裝

### 線上示範

你可以直接在 [HuggingFace 上的線上示範](https://huggingface.co/spaces/reycn/PDFMathTranslate-Docker)和[魔搭的線上示範](https://www.modelscope.cn/studios/AI-ModelScope/PDFMathTranslate)進行嘗試，無需安裝。
請注意，示範使用的運算資源有限，請勿濫用。

<h2 id="install">安裝與使用</h2>

我們提供了四種使用此專案的方法：[命令列工具](#cmd)、[便攜式安裝](#portable)、[圖形使用者介面](#gui) 與 [容器化部署](#docker)。

pdf2zh 在執行時需要額外下載模型（`wybxc/DocLayout-YOLO-DocStructBench-onnx`），該模型也可在魔搭（ModelScope）上取得。如果在啟動時下載該模型時遇到問題，請使用如下環境變數：
```shell
set HF_ENDPOINT=https://hf-mirror.com
```

<h3 id="cmd">方法一、命令列工具</h3>

1. 確保已安裝 Python 版本大於 3.10 且小於 3.12  
2. 安裝此程式：

   ```bash
   pip install pdf2zh
   ```

3. 執行翻譯，生成檔案位於 [目前工作目錄](https://chatgpt.com/share/6745ed36-9acc-800e-8a90-59204bd13444)：

   ```bash
   pdf2zh document.pdf
   ```

<h3 id="portable">方法二、便攜式安裝</h3>

無需預先安裝 Python 環境

下載 [setup.bat](https://raw.githubusercontent.com/Byaidu/PDFMathTranslate/refs/heads/main/script/setup.bat) 並直接雙擊執行

<h3 id="gui">方法三、圖形使用者介面</h3>

1. 確保已安裝 Python 版本大於 3.10 且小於 3.12  
2. 安裝此程式：

   ```bash
   pip install pdf2zh
   ```

3. 在瀏覽器中啟動使用：

   ```bash
   pdf2zh -i
   ```

4. 如果您的瀏覽器沒有自動開啟並跳轉，請手動在瀏覽器開啟：

   ```bash
   http://localhost:7860/
   ```

   <img src="./images/gui.gif" width="500"/>

查看 [documentation for GUI](/README_GUI.md) 以獲取詳細說明

<h3 id="docker">方法四、容器化部署</h3>

1. 拉取 Docker 映像檔並執行：

   ```bash
   docker pull byaidu/pdf2zh
   docker run -d -p 7860:7860 byaidu/pdf2zh
   ```

2. 透過瀏覽器開啟：

   ```
   http://localhost:7860/
   ```

用於在雲服務上部署容器映像檔：

<div>
<a href="https://www.heroku.com/deploy?template=https://github.com/Byaidu/PDFMathTranslate">
  <img src="https://www.herokucdn.com/deploy/button.svg" alt="Deploy" height="26"></a>
<a href="https://render.com/deploy">
  <img src="https://render.com/images/deploy-to-render-button.svg" alt="Deploy to Koyeb" height="26"></a>
<a href="https://zeabur.com/templates/5FQIGX?referralCode=reycn">
  <img src="https://zeabur.com/button.svg" alt="Deploy on Zeabur" height="26"></a>
<a href="https://app.koyeb.com/deploy?type=git&builder=buildpack&repository=github.com/Byaidu/PDFMathTranslate&branch=main&name=pdf-math-translate">
  <img src="https://www.koyeb.com/static/images/deploy/button.svg" alt="Deploy to Koyeb" height="26"></a>
</div>

<h2 id="usage">高級選項</h2>

在命令列中執行翻譯指令，並在目前工作目錄下生成譯文檔案 `example-mono.pdf` 和雙語對照檔案 `example-dual.pdf`。預設使用 Google 翻譯服務。

<img src="./images/cmd.explained.png" width="580px"  alt="cmd"/>  

以下表格列出了所有高級選項，供參考：

| Option    | 功能 | 範例 |
| -------- | ------- |------- |
| files | 本機檔案 |  `pdf2zh ~/local.pdf` |
| links | 線上檔案 |  `pdf2zh http://arxiv.org/paper.pdf` |
| `-i`  | [進入圖形介面](#gui) |  `pdf2zh -i` |
| `-p`  | [僅翻譯部分文件](#partial) |  `pdf2zh example.pdf -p 1` |
| `-li` | [原文語言](#language) |  `pdf2zh example.pdf -li en` |
| `-lo` | [目標語言](#language) |  `pdf2zh example.pdf -lo zh` |
| `-s`  | [指定翻譯服務](#services) |  `pdf2zh example.pdf -s deepl` |
| `-t`  | [多執行緒](#threads) | `pdf2zh example.pdf -t 1` |
| `-o`  | 輸出目錄 | `pdf2zh example.pdf -o output` |
| `-f`, `-c` | [例外規則](#exceptions) | `pdf2zh example.pdf -f "(MS.*)"` |
| `--share` | [獲取 gradio 公開連結] | `pdf2zh -i --share` |
| `--authorized` | [[添加網頁認證及自訂認證頁面](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.)] | `pdf2zh -i --authorized users.txt [auth.html]` |
| `--prompt` | [使用自訂的大模型 Prompt] | `pdf2zh --prompt [prompt.txt]` |
| `--onnx` | [使用自訂的 DocLayout-YOLO ONNX 模型] | `pdf2zh --onnx [onnx/model/path]` |
| `--serverport` | [自訂 WebUI 埠號] | `pdf2zh --serverport 7860` |
| `--dir` | [資料夾翻譯] | `pdf2zh --dir /path/to/translate/` |

<h3 id="partial">全文或部分文件翻譯</h3>

- **全文翻譯**

```bash
pdf2zh example.pdf
```

- **部分翻譯**

```bash
pdf2zh example.pdf -p 1-3,5
```

<h3 id="language">指定原文語言與目標語言</h3>

可參考 [Google 語言代碼](https://developers.google.com/admin-sdk/directory/v1/languages)、[DeepL 語言代碼](https://developers.deepl.com/docs/resources/supported-languages)

```bash
pdf2zh example.pdf -li en -lo ja
```

<h3 id="services">使用不同的翻譯服務</h3>

下表列出了每個翻譯服務所需的 [環境變數](https://chatgpt.com/share/6734a83d-9d48-800e-8a46-f57ca6e8bcb4)。在使用前，請先確保已設定好對應的變數。

|**Translator**|**Service**|**Environment Variables**|**Default Values**|**Notes**|
|-|-|-|-|-|
|**Google (Default)**|`google`|無|N/A|無|
|**Bing**|`bing`|無|N/A|無|
|**DeepL**|`deepl`|`DEEPL_AUTH_KEY`|`[Your Key]`|參閱 [DeepL](https://support.deepl.com/hc/en-us/articles/360020695820-API-Key-for-DeepL-s-API)|
|**DeepLX**|`deeplx`|`DEEPLX_ENDPOINT`|`https://api.deepl.com/translate`|參閱 [DeepLX](https://github.com/OwO-Network/DeepLX)|
|**Ollama**|`ollama`|`OLLAMA_HOST`, `OLLAMA_MODEL`|`http://127.0.0.1:11434`, `gemma2`|參閱 [Ollama](https://github.com/ollama/ollama)|
|**OpenAI**|`openai`|`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_STOP_TOKENS`, `OPENAI_MAX_TOKENS`|`https://api.openai.com/v1`, `[Your Key]`, `gpt-4o-mini`, ` `, `-1`|參閱 [OpenAI](https://platform.openai.com/docs/overview)|
|**AzureOpenAI**|`azure-openai`|`AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_MODEL`|`[Your Endpoint]`, `[Your Key]`, `gpt-4o-mini`|參閱 [Azure OpenAI](https://learn.microsoft.com/zh-cn/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line%2Cjavascript-keyless%2Ctypescript-keyless%2Cpython&pivots=programming-language-python)|
|**Zhipu**|`zhipu`|`ZHIPU_API_KEY`, `ZHIPU_MODEL`|`[Your Key]`, `glm-4-flash`|參閱 [Zhipu](https://open.bigmodel.cn/dev/api/thirdparty-frame/openai-sdk)|
| **ModelScope**       | `modelscope`   |`MODELSCOPE_API_KEY`, `MODELSCOPE_MODEL`|`[Your Key]`, `Qwen/Qwen2.5-Coder-32B-Instruct`| 參閱 [ModelScope](https://www.modelscope.cn/docs/model-service/API-Inference/intro)|
|**Silicon**|`silicon`|`SILICON_API_KEY`, `SILICON_MODEL`|`[Your Key]`, `Qwen/Qwen2.5-7B-Instruct`|參閱 [SiliconCloud](https://docs.siliconflow.cn/quickstart)|
|**Gemini**|`gemini`|`GEMINI_API_KEY`, `GEMINI_MODEL`|`[Your Key]`, `gemini-1.5-flash`|參閱 [Gemini](https://ai.google.dev/gemini-api/docs/openai)|
|**Azure**|`azure`|`AZURE_ENDPOINT`, `AZURE_API_KEY`|`https://api.translator.azure.cn`, `[Your Key]`|參閱 [Azure](https://docs.azure.cn/en-us/ai-services/translator/text-translation-overview)|
|**Tencent**|`tencent`|`TENCENTCLOUD_SECRET_ID`, `TENCENTCLOUD_SECRET_KEY`|`[Your ID]`, `[Your Key]`|參閱 [Tencent](https://www.tencentcloud.com/products/tmt?from_qcintl=122110104)|
|**Dify**|`dify`|`DIFY_API_URL`, `DIFY_API_KEY`|`[Your DIFY URL]`, `[Your Key]`|參閱 [Dify](https://github.com/langgenius/dify)，需要在 Dify 的工作流程輸入中定義三個變數：lang_out、lang_in、text。|
|**AnythingLLM**|`anythingllm`|`AnythingLLM_URL`, `AnythingLLM_APIKEY`|`[Your AnythingLLM URL]`, `[Your Key]`|參閱 [anything-llm](https://github.com/Mintplex-Labs/anything-llm)|
|**Argos Translate**|`argos`| | |參閱 [argos-translate](https://github.com/argosopentech/argos-translate)|
|**Grok**|`grok`| `GORK_API_KEY`, `GORK_MODEL` | `[Your GORK_API_KEY]`, `grok-2-1212` |參閱 [Grok](https://docs.x.ai/docs/overview)|
|**DeepSeek**|`deepseek`| `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` | `[Your DEEPSEEK_API_KEY]`, `deepseek-chat` |參閱 [DeepSeek](https://www.deepseek.com/)|
|**OpenAI-Liked**|`openailiked`| `OPENAILIKED_BASE_URL`, `OPENAILIKED_API_KEY`, `OPENAILIKED_MODEL`, `OPENAILIKED_STOP_TOKENS`, `OPENAILIKED_MAX_TOKENS` | `url`, `[Your Key]`, `model name`, ` `, `-1` | 無 |

對於不在上述表格中，但兼容 OpenAI API 的大語言模型，可以使用與 OpenAI 相同的方式設定環境變數。

使用 `-s service` 或 `-s service:model` 指定翻譯服務：

```bash
pdf2zh example.pdf -s openai:gpt-4o-mini
```

或使用環境變數指定模型：

```bash
set OPENAI_MODEL=gpt-4o-mini
pdf2zh example.pdf -s openai
```

<h3 id="exceptions">指定例外規則</h3>

使用正則表達式指定需要保留的公式字體與字元：

```bash
pdf2zh example.pdf -f "(CM[^RT].*|MS.*|.*Ital)" -c "(\(|\||\)|\+|=|\d|[\u0080-\ufaff])"
```

預設保留 `Latex`, `Mono`, `Code`, `Italic`, `Symbol` 以及 `Math` 字體：

```bash
pdf2zh example.pdf -f "(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Ital|.*Sym|.*Math)"
```

<h3 id="threads">指定執行緒數量</h3>

使用 `-t` 參數指定翻譯使用的執行緒數量：

```bash
pdf2zh example.pdf -t 1
```

<h3 id="prompt">自訂大模型 Prompt</h3>

使用 `--prompt` 指定在使用大模型翻譯時所採用的 Prompt 檔案。

```bash
pdf2zh example.pdf -pr prompt.txt
```

範例 `prompt.txt` 檔案內容：

```
[
    {
        "role": "system",
        "content": "You are a professional,authentic machine translation engine.",
    },
    {
        "role": "user",
        "content": "Translate the following markdown source text to ${lang_out}. Keep the formula notation {{v*}} unchanged. Output translation directly without any additional text.\nSource Text: ${text}\nTranslated Text:",
    },
]
```

在自訂 Prompt 檔案中，可以使用以下三個內建變數來傳遞參數：
|**變數名稱**|**說明**|
|-|-|
|`lang_in`|輸入語言|
|`lang_out`|輸出語言|
|`text`|需要翻譯的文本|

<h2 id="todo">API</h2>

### Python

```python
from pdf2zh import translate, translate_stream

params = {"lang_in": "en", "lang_out": "zh", "service": "google", "thread": 4}
file_mono, file_dual = translate(files=["example.pdf"], **params)[0]
with open("example.pdf", "rb") as f:
    stream_mono, stream_dual = translate_stream(stream=f.read(), **params)
```

### HTTP

```bash
pip install pdf2zh[backend]
pdf2zh --flask
pdf2zh --celery worker
```

```bash
curl http://localhost:11008/v1/translate -F "file=@example.pdf" -F "data={\"lang_in\":\"en\",\"lang_out\":\"zh\",\"service\":\"google\",\"thread\":4}"
{"id":"d9894125-2f4e-45ea-9d93-1a9068d2045a"}

curl http://localhost:11008/v1/translate/d9894125-2f4e-45ea-9d93-1a9068d2045a
{"info":{"n":13,"total":506},"state":"PROGRESS"}

curl http://localhost:11008/v1/translate/d9894125-2f4e-45ea-9d93-1a9068d2045a
{"state":"SUCCESS"}

curl http://localhost:11008/v1/translate/d9894125-2f4e-45ea-9d93-1a9068d2045a/mono --output example-mono.pdf

curl http://localhost:11008/v1/translate/d9894125-2f4e-45ea-9d93-1a9068d2045a/dual --output example-dual.pdf

curl http://localhost:11008/v1/translate/d9894125-2f4e-45ea-9d93-1a9068d2045a -X DELETE
```

<h2 id="acknowledgement">致謝</h2>

- 文件合併：[PyMuPDF](https://github.com/pymupdf/PyMuPDF)
- 文件解析：[Pdfminer.six](https://github.com/pdfminer/pdfminer.six)
- 文件提取：[MinerU](https://github.com/opendatalab/MinerU)
- 文件預覽：[Gradio PDF](https://github.com/freddyaboulton/gradio-pdf)
- 多執行緒翻譯：[MathTranslate](https://github.com/SUSYUSTC/MathTranslate)
- 版面解析：[DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
- PDF 標準：[PDF Explained](https://zxyle.github.io/PDF-Explained/)、[PDF Cheat Sheets](https://pdfa.org/resource/pdf-cheat-sheets/)
- 多語言字型：[Go Noto Universal](https://github.com/satbyy/go-noto-universal)

<h2 id="contrib">貢獻者</h2>

<a href="https://github.com/Byaidu/PDFMathTranslate/graphs/contributors">
  <img src="https://opencollective.com/PDFMathTranslate/contributors.svg?width=890&button=false" />
</a>

![Alt](https://repobeats.axiom.co/api/embed/dfa7583da5332a11468d686fbd29b92320a6a869.svg "Repobeats analytics image")

<h2 id="star_hist">星標歷史</h2>

<a href="https://star-history.com/#Byaidu/PDFMathTranslate&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Byaidu/PDFMathTranslate&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Byaidu/PDFMathTranslate&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Byaidu/PDFMathTranslate&type=Date"/>
 </picture>
</a>