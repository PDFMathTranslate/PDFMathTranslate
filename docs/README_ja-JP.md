<div align="center">

[English](../README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md) | 日本語

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

科学 PDF 文書の翻訳およびバイリンガル比較ツール

- 📊 数式、チャート、目次、注釈を保持 *([プレビュー](#preview))*
- 🌐 [複数の言語](#language) と [多様な翻訳サービス](#services) をサポート
- 🤖 [コマンドラインツール](#usage)、[インタラクティブユーザーインターフェース](#gui)、および [Docker](#docker) を提供

フィードバックは [GitHub Issues](https://github.com/Byaidu/PDFMathTranslate/issues)、[Telegram グループ](https://t.me/+Z9_SgnxmsmA5NzBl)

<h2 id="updates">最近の更新</h2>

- [2024年11月26日] CLIがオンラインファイルをサポートするようになりました *(by [@reycn](https://github.com/reycn))*  
- [2024年11月24日] 依存関係のサイズを削減するために [ONNX](https://github.com/onnx/onnx) サポートを追加しました *(by [@Wybxc](https://github.com/Wybxc))*  
- [2024年11月23日] 🌟 [公共サービス](#demo) がオンラインになりました! *(by [@Byaidu](https://github.com/Byaidu))*  
- [2024年11月23日] ウェブボットを防ぐためのファイアウォールを追加しました *(by [@Byaidu](https://github.com/Byaidu))*  
- [2024年11月22日] GUIがイタリア語をサポートし、改善されました *(by [@Byaidu](https://github.com/Byaidu), [@reycn](https://github.com/reycn))*  
- [2024年11月22日] デプロイされたサービスを他の人と共有できるようになりました *(by [@Zxis233](https://github.com/Zxis233))*  
- [2024年11月22日] Tencent翻訳をサポートしました *(by [@hellofinch](https://github.com/hellofinch))*  
- [2024年11月21日] GUIがバイリンガルドキュメントのダウンロードをサポートするようになりました *(by [@reycn](https://github.com/reycn))*  
- [2024年11月20日] 🌟 [デモ](#demo) がオンラインになりました! *(by [@reycn](https://github.com/reycn))*  

<h2 id="preview">プレビュー</h2>

<div align="center">
<img src="./images/preview.gif" width="80%"/>
</div>

<h2 id="demo">公共サービス 🌟</h2>

### 無料サービス (<https://pdf2zh.com/>)

インストールなしで [公共サービス](https://pdf2zh.com/) をオンラインで試すことができます。  

### デモ

インストールなしで [HuggingFace上のデモ](https://huggingface.co/spaces/reycn/PDFMathTranslate-Docker), [ModelScope上のデモ](https://www.modelscope.cn/studios/AI-ModelScope/PDFMathTranslate) を試すことができます。
デモの計算リソースは限られているため、乱用しないようにしてください。

<h2 id="install">インストールと使用方法</h2>

このプロジェクトを使用するための4つの方法を提供しています：[コマンドライン](#cmd)、[ポータブル](#portable)、[GUI](#gui)、および [Docker](#docker)。

pdf2zhの実行には追加モデル（`wybxc/DocLayout-YOLO-DocStructBench-onnx`）が必要です。このモデルはModelScopeでも見つけることができます。起動時にこのモデルのダウンロードに問題がある場合は、以下の環境変数を使用してください：

```shell
set HF_ENDPOINT=https://hf-mirror.com
```

For PowerShell user:
```shell
$env:HF_ENDPOINT = https://hf-mirror.com
```

<h3 id="cmd">方法1. コマンドライン</h3>

  1. Pythonがインストールされていること (バージョン3.11 <= バージョン <= 3.12)
  2. パッケージをインストールします：

      ```bash
      pip install pdf2zh
      ```

  3. 翻訳を実行し、[現在の作業ディレクトリ](https://chatgpt.com/share/6745ed36-9acc-800e-8a90-59204bd13444) にファイルを生成します：

      ```bash
      pdf2zh document.pdf
      ```

<h3 id="portable">方法2. ポータブル</h3>

Python環境を事前にインストールする必要はありません

[setup.bat](https://raw.githubusercontent.com/Byaidu/PDFMathTranslate/refs/heads/main/script/setup.bat) をダウンロードしてダブルクリックして実行します

<h3 id="gui">方法3. GUI</h3>

1. Pythonがインストールされていること (バージョン3.11 <= バージョン <= 3.12)
2. パッケージをインストールします：

      ```bash
      pip install pdf2zh
      ```

3. ブラウザで使用を開始します：

      ```bash
      pdf2zh -i
      ```

4. ブラウザが自動的に起動しない場合は、次のURLを開きます：

    ```bash
    http://localhost:7860/
    ```

    <img src="./images/gui.gif" width="500"/>

詳細については、[GUIのドキュメント](./README_GUI.md) を参照してください。

<h3 id="docker">方法4. Docker</h3>

1. プルして実行します：

    ```bash
    docker pull byaidu/pdf2zh
    docker run -d -p 7860:7860 byaidu/pdf2zh
    ```

2. ブラウザで開きます：

    ```
    http://localhost:7860/
    ```

クラウドサービスでのDockerデプロイメント用：

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

<h2 id="usage">高度なオプション</h2>

コマンドラインで翻訳コマンドを実行し、現在の作業ディレクトリに翻訳されたドキュメント `example-mono.pdf` とバイリンガルドキュメント `example-dual.pdf` を生成します。デフォルトではGoogle翻訳サービスを使用します。More support translation services can find [HERE](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#services).


<img src="./images/cmd.explained.png" width="580px"  alt="cmd"/>  

以下の表に、参考のためにすべての高度なオプションをリストしました：

| オプション    | 機能 | 例 |
| -------- | ------- |------- |
| files | ローカルファイル |  `pdf2zh ~/local.pdf` |
| links | オンラインファイル |  `pdf2zh http://arxiv.org/paper.pdf` |
| `-i`  | [GUIに入る](#gui) |  `pdf2zh -i` |
| `-p`  | [部分的なドキュメント翻訳](#partial) |  `pdf2zh example.pdf -p 1` |
| `-li` | [ソース言語](#languages) |  `pdf2zh example.pdf -li en` |
| `-lo` | [ターゲット言語](#languages) |  `pdf2zh example.pdf -lo zh` |
| `-s`  | [翻訳サービス](#services) |  `pdf2zh example.pdf -s deepl` |
| `-t`  | [マルチスレッド](#threads) | `pdf2zh example.pdf -t 1` |
| `-o`  | 出力ディレクトリ | `pdf2zh example.pdf -o output` |
| `-f`, `-c` | [例外](#exceptions) | `pdf2zh example.pdf -f "(MS.*)"` |
| `--share` | [gradio公開リンクを取得] | `pdf2zh -i --share` |
| `--authorized` | [[ウェブ認証とカスタム認証ページの追加](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.)] | `pdf2zh -i --authorized users.txt [auth.html]` |
| `--prompt` | [カスタムビッグモデルのプロンプトを使用する] | `pdf2zh --prompt [prompt.txt]` |
| `--onnx` | [カスタムDocLayout-YOLO ONNXモデルの使用] | `pdf2zh --onnx [onnx/model/path]` |
| `--serverport` | [カスタムWebUIポートを使用する] | `pdf2zh --serverport 7860` |
| `--dir` | [batch translate] | `pdf2zh --dir /path/to/translate/` |
| `--config` | [configuration file](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#cofig) | `pdf2zh --config /path/to/config/config.json` |
| `--serverport` | [custom gradio server port] | `pdf2zh --serverport 7860` |

<h3 id="partial">全文または部分的なドキュメント翻訳</h3>

- **全文翻訳**

```bash
pdf2zh example.pdf
```

- **部分翻訳**

```bash
pdf2zh example.pdf -p 1-3,5
```

<h3 id="language">ソース言語とターゲット言語を指定</h3>

[Google Languages Codes](https://developers.google.com/admin-sdk/directory/v1/languages)、[DeepL Languages Codes](https://developers.deepl.com/docs/resources/supported-languages) を参照してください

```bash
pdf2zh example.pdf -li en -lo ja
```

<h3 id="services">異なるサービスで翻訳</h3>

以下の表は、各翻訳サービスに必要な [環境変数](https://chatgpt.com/share/6734a83d-9d48-800e-8a46-f57ca6e8bcb4) を示しています。各サービスを使用する前に、これらの変数を設定してください。

|**Translator**|**Service**|**Environment Variables**|**Default Values**|**Notes**|
|-|-|-|-|-|
|**Google (Default)**|`google`|None|N/A|None|
|**Bing**|`bing`|None|N/A|None|
|**DeepL**|`deepl`|`DEEPL_AUTH_KEY`|`[Your Key]`|See [DeepL](https://support.deepl.com/hc/en-us/articles/360020695820-API-Key-for-DeepL-s-API)|
|**DeepLX**|`deeplx`|`DEEPLX_ENDPOINT`|`https://api.deepl.com/translate`|See [DeepLX](https://github.com/OwO-Network/DeepLX)|
|**Ollama**|`ollama`|`OLLAMA_HOST`, `OLLAMA_MODEL`|`http://127.0.0.1:11434`, `gemma2`|See [Ollama](https://github.com/ollama/ollama)|
|**OpenAI**|`openai`|`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_STOP_TOKENS`, `OPENAI_MAX_TOKENS` |`https://api.openai.com/v1`, `[Your Key]`, `gpt-4o-mini`, ` `, `-1`|See [OpenAI](https://platform.openai.com/docs/overview)|
|**AzureOpenAI**|`azure-openai`|`AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_MODEL`|`[Your Endpoint]`, `[Your Key]`, `gpt-4o-mini`|See [Azure OpenAI](https://learn.microsoft.com/zh-cn/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line%2Cjavascript-keyless%2Ctypescript-keyless%2Cpython&pivots=programming-language-python)|
|**Zhipu**|`zhipu`|`ZHIPU_API_KEY`, `ZHIPU_MODEL`|`[Your Key]`, `glm-4-flash`|See [Zhipu](https://open.bigmodel.cn/dev/api/thirdparty-frame/openai-sdk)|
| **ModelScope**       | `modelscope`   |`MODELSCOPE_API_KEY`, `MODELSCOPE_MODEL`|`[Your Key]`, `Qwen/Qwen2.5-Coder-32B-Instruct`| See [ModelScope](https://www.modelscope.cn/docs/model-service/API-Inference/intro)|
|**Silicon**|`silicon`|`SILICON_API_KEY`, `SILICON_MODEL`|`[Your Key]`, `Qwen/Qwen2.5-7B-Instruct`|See [SiliconCloud](https://docs.siliconflow.cn/quickstart)|
|**Gemini**|`gemini`|`GEMINI_API_KEY`, `GEMINI_MODEL`|`[Your Key]`, `gemini-1.5-flash`|See [Gemini](https://ai.google.dev/gemini-api/docs/openai)|
|**Azure**|`azure`|`AZURE_ENDPOINT`, `AZURE_API_KEY`|`https://api.translator.azure.cn`, `[Your Key]`|See [Azure](https://docs.azure.cn/en-us/ai-services/translator/text-translation-overview)|
|**Tencent**|`tencent`|`TENCENTCLOUD_SECRET_ID`, `TENCENTCLOUD_SECRET_KEY`|`[Your ID]`, `[Your Key]`|See [Tencent](https://www.tencentcloud.com/products/tmt?from_qcintl=122110104)|
|**Dify**|`dify`|`DIFY_API_URL`, `DIFY_API_KEY`|`[Your DIFY URL]`, `[Your Key]`|See [Dify](https://github.com/langgenius/dify),Three variables, lang_out, lang_in, and text, need to be defined in Dify's workflow input.|
|**AnythingLLM**|`anythingllm`|`AnythingLLM_URL`, `AnythingLLM_APIKEY`|`[Your AnythingLLM URL]`, `[Your Key]`|See [anything-llm](https://github.com/Mintplex-Labs/anything-llm)|
|**Argos Translate**|`argos`| | |See [argos-translate](https://github.com/argosopentech/argos-translate)|
|**Grok**|`grok`| `GORK_API_KEY`, `GORK_MODEL` | `[Your GORK_API_KEY]`, `grok-2-1212` |See [Grok](https://docs.x.ai/docs/overview)|
|**DeepSeek**|`deepseek`| `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` | `[Your DEEPSEEK_API_KEY]`, `deepseek-chat` |See [DeepSeek](https://www.deepseek.com/)|
|**MiniMax**|`minimax`| `MINIMAX_API_KEY`, `MINIMAX_MODEL`, `MINIMAX_BASE_URL` (optional) | `[Your MINIMAX_API_KEY]`, `MiniMax-M3`, `https://api.minimax.io/v1` |See [MiniMax](https://platform.minimaxi.com/). Defaults to `MiniMax-M3`; set `MINIMAX_MODEL=MiniMax-M2.7` for the previous model. Set `MINIMAX_BASE_URL=https://api.minimaxi.com/v1` for the China endpoint.|
|**OpenAI-Liked**|`openailiked`| `OPENAILIKED_BASE_URL`, `OPENAILIKED_API_KEY`, `OPENAILIKED_MODEL` | `url`, `[Your Key]`, `model name` | None |
|**OpenAI-Liked**|`openailiked`| `OPENAILIKED_BASE_URL`, `OPENAILIKED_API_KEY`, `OPENAILIKED_MODEL`, `OPENAILIKED_STOP_TOKENS`, `OPENAILIKED_MAX_TOKENS` | `url`, `[Your Key]`, `model name`, `model name`, ` `, `-1` | None |

(need Japenese translation)
For large language models that are compatible with the OpenAI API but not listed in the table above, you can set environment variables using the same method outlined for OpenAI in the table.

`-s service` または `-s service:model` を使用してサービスを指定します：

```bash
pdf2zh example.pdf -s openai:gpt-4o-mini
```

または環境変数でモデルを指定します：

```bash
set OPENAI_MODEL=gpt-4o-mini
pdf2zh example.pdf -s openai
```

For PowerShell user:
```shell
$env:OPENAI_MODEL = gpt-4o-mini
pdf2zh example.pdf -s openai
```

<h3 id="exceptions">例外を指定して翻訳</h3>

正規表現を使用して保持する必要がある数式フォントと文字を指定します：

```bash
pdf2zh example.pdf -f "(CM[^RT].*|MS.*|.*Ital)" -c "(\(|\||\)|\+|=|\d|[\u0080-\ufaff])"
```

デフォルトで `Latex`、`Mono`、`Code`、`Italic`、`Symbol` および `Math` フォントを保持します：

```bash
pdf2zh example.pdf -f "(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Ital|.*Sym|.*Math)"
```

<h3 id="threads">スレッド数を指定</h3>

`-t` を使用して翻訳に使用するスレッド数を指定します：

```bash
pdf2zh example.pdf -t 1
```

<h3 id="prompt">カスタム プロンプト</h3>

`--prompt`を使用して、LLMで使用するプロンプトを指定します：

```bash
pdf2zh example.pdf -pr prompt.txt
```


`prompt.txt`の例：

```txt
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


カスタムプロンプトファイルでは、以下の3つの変数が使用できます。

|**変数**|**内容**|
|-|-|
|`lang_in`|ソース言語|
|`lang_out`|ターゲット言語|
|`text`|翻訳するテキスト|

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

<h2 id="acknowledgement">謝辞</h2>

- ドキュメントのマージ：[PyMuPDF](https://github.com/pymupdf/PyMuPDF)

- ドキュメントの解析：[Pdfminer.six](https://github.com/pdfminer/pdfminer.six)

- ドキュメントの抽出：[MinerU](https://github.com/opendatalab/MinerU)

- ドキュメントプレビュー：[Gradio PDF](https://github.com/freddyaboulton/gradio-pdf)

- マルチスレッド翻訳：[MathTranslate](https://github.com/SUSYUSTC/MathTranslate)

- レイアウト解析：[DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)

- ドキュメント標準：[PDF Explained](https://zxyle.github.io/PDF-Explained/)、[PDF Cheat Sheets](https://pdfa.org/resource/pdf-cheat-sheets/)

- 多言語フォント：[Go Noto Universal](https://github.com/satbyy/go-noto-universal)

<h2 id="contrib">貢献者</h2>

<a href="https://github.com/Byaidu/PDFMathTranslate/graphs/contributors">
  <img src="https://opencollective.com/PDFMathTranslate/contributors.svg?width=890&button=false" />
</a>

![Alt](https://repobeats.axiom.co/api/embed/dfa7583da5332a11468d686fbd29b92320a6a869.svg "Repobeats analytics image")

<h2 id="star_hist">スター履歴</h2>

<a href="https://star-history.com/#Byaidu/PDFMathTranslate&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Byaidu/PDFMathTranslate&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Byaidu/PDFMathTranslate&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Byaidu/PDFMathTranslate&type=Date"/>
 </picture>
</a>
