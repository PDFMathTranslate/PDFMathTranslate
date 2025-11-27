[**Documentation**](https://github.com/Byaidu/PDFMathTranslate) > **Advanced Usage** _(current)_

---

<h3 id="toc">Table of Contents</h3>

- [Full / partial translation](#partial)
- [Specify source and target languages](#language)
- [Translate with different services](#services)
- [Translate wih exceptions](#exceptions)
- [Multi-threads](#threads)
- [Custom prompt](#prompt)
- [Authorization](#auth)
- [Custom configuration file](#cofig)
- [Fonts Subseting](#fonts-subset)
- [Translation cache](#cache)

---

<h3 id="partial">Full / partial translation</h3>

- Entire document

  ```bash
  pdf2zh example.pdf
  ```

- Part of the document

  ```bash
  pdf2zh example.pdf -p 1-3,5
  ```

[⬆️ Back to top](#toc)

---

<h3 id="language">Specify source and target languages</h3>

See [Google Languages Codes](https://developers.google.com/admin-sdk/directory/v1/languages), [DeepL Languages Codes](https://developers.deepl.com/docs/resources/supported-languages)

```bash
pdf2zh example.pdf -li en -lo ja
```

[⬆️ Back to top](#toc)

---

<h3 id="services">Translate with different services</h3>

We've provided a detailed table on the required [environment variables](https://chatgpt.com/share/6734a83d-9d48-800e-8a46-f57ca6e8bcb4) for each translation service. Make sure to set them before using the respective service.

| **Translator**       | **Service**    | **Environment Variables**                                             | **Default Values**                                       | **Notes**                                                                                                                                                                                                 |
|----------------------|----------------|-----------------------------------------------------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Google (Default)** | `google`       | None                                                                  | N/A                                                      | None                                                                                                                                                                                                      |
| **Bing**             | `bing`         | None                                                                  | N/A                                                      | None                                                                                                                                                                                                      |
| **302.AI**           | `302ai`       | `X302AI_API_KEY`, `X302AI_MODEL`                                         | `[Your Key]`, `Gemma-7B` | See [302.AI](https://share.302.ai/tqTWfD)                                                                                                                                                   |
| **OpenAI**           | `openai`       | `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_STOP_TOKENS`, `OPENAI_MAX_TOKENS` | `https://api.openai.com/v1`, `[Your Key]`, `gpt-4o-mini`, ` `, `-1` | See [OpenAI](https://platform.openai.com/docs/overview)                                                                                                                                                   |
| **DeepL**            | `deepl`        | `DEEPL_AUTH_KEY`                                                      | `[Your Key]`                                             | See [DeepL](https://support.deepl.com/hc/en-us/articles/360020695820-API-Key-for-DeepL-s-API)                                                                                                             |
| **DeepLX**           | `deeplx`       | `DEEPLX_ENDPOINT`                                                     | `https://api.deepl.com/translate`                        | See [DeepLX](https://github.com/OwO-Network/DeepLX)                                                                                                                                                       |
| **Ollama**           | `ollama`       | `OLLAMA_HOST`, `OLLAMA_MODEL`                                         | `http://127.0.0.1:11434`, `gemma2`                       | See [Ollama](https://github.com/ollama/ollama)                                                                                                                                                            |
| **Xinference**       | `xinference`   | `XINFERENCE_HOST`, `XINFERENCE_MODEL`                                 | `http://127.0.0.1:9997`, `gemma-2-it`                    | See [Xinference](https://github.com/xorbitsai/inference)                                                                                                                                                                                        |
| **AzureOpenAI**      | `azure-openai` | `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_MODEL` | `[Your Endpoint]`, `[Your Key]`, `gpt-4o-mini`           | See [Azure OpenAI](https://learn.microsoft.com/zh-cn/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line%2Cjavascript-keyless%2Ctypescript-keyless%2Cpython&pivots=programming-language-python) |
| **Zhipu**            | `zhipu`        | `ZHIPU_API_KEY`, `ZHIPU_MODEL`                                        | `[Your Key]`, `glm-4-flash`                              | See [Zhipu](https://open.bigmodel.cn/dev/api/thirdparty-frame/openai-sdk)                                                                                                                                 |
| **ModelScope**       | `modelscope`   | `MODELSCOPE_API_KEY`, `MODELSCOPE_MODEL`                              | `[Your Key]`, `Qwen/Qwen2.5-Coder-32B-Instruct`          | See [ModelScope](https://www.modelscope.cn/docs/model-service/API-Inference/intro)                                                                                                                        |
| **Silicon**          | `silicon`      | `SILICON_API_KEY`, `SILICON_MODEL`                                    | `[Your Key]`, `Qwen/Qwen2.5-7B-Instruct`                 | See [SiliconCloud](https://docs.siliconflow.cn/quickstart)                                                                                                                                                |
| **Gemini**           | `gemini`       | `GEMINI_API_KEY`, `GEMINI_MODEL`                                      | `[Your Key]`, `gemini-1.5-flash`                         | See [Gemini](https://ai.google.dev/gemini-api/docs/openai)                                                                                                                                                |
| **Azure**            | `azure`        | `AZURE_ENDPOINT`, `AZURE_API_KEY`                                     | `https://api.translator.azure.cn`, `[Your Key]`          | See [Azure](https://docs.azure.cn/en-us/ai-services/translator/text-translation-overview)                                                                                                                 |
| **Tencent**          | `tencent`      | `TENCENTCLOUD_SECRET_ID`, `TENCENTCLOUD_SECRET_KEY`                   | `[Your ID]`, `[Your Key]`                                | See [Tencent](https://www.tencentcloud.com/products/tmt?from_qcintl=122110104)                                                                                                                            |
| **Dify**             | `dify`         | `DIFY_API_URL`, `DIFY_API_KEY`                                        | `[Your DIFY URL]`, `[Your Key]`                          | See [Dify](https://github.com/langgenius/dify),Three variables, lang_out, lang_in, and text, need to be defined in Dify's workflow input.                                                                 |
| **AnythingLLM**      | `anythingllm`  | `AnythingLLM_URL`, `AnythingLLM_APIKEY`                               | `[Your AnythingLLM URL]`, `[Your Key]`                   | See [anything-llm](https://github.com/Mintplex-Labs/anything-llm)                                                                                                                                         |
|**Argos Translate**|`argos`| | |See [argos-translate](https://github.com/argosopentech/argos-translate)|
|**Grok**|`grok`| `GORK_API_KEY`, `GORK_MODEL` | `[Your GORK_API_KEY]`, `grok-2-1212` |See [Grok](https://docs.x.ai/docs/overview)|
|**Groq**|`groq`| `GROQ_API_KEY`, `GROQ_MODEL` | `[Your GROQ_API_KEY]`, `llama-3-3-70b-versatile` |See [Groq](https://console.groq.com/docs/models)|
|**DeepSeek**|`deepseek`| `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` | `[Your DEEPSEEK_API_KEY]`, `deepseek-chat` |See [DeepSeek](https://www.deepseek.com/)|
|**OpenAI-Liked**|`openailiked`| `OPENAILIKED_BASE_URL`, `OPENAILIKED_API_KEY`, `OPENAILIKED_MODEL`, `OPENAILIKED_STOP_TOKENS`, `OPENAILIKED_MAX_TOKENS` | `url`, `[Your Key]`, `model name`, ` `, `-1` | None |
|**Ali Qwen Translation**|`qwen-mt`| `ALI_MODEL`, `ALI_API_KEY`, `ALI_DOMAINS` | `qwen-mt-turbo`, `[Your Key]`, `scientific paper` | Tranditional Chinese are not yet supported, it will be translated into Simplified Chinese. More see [Qwen MT](https://bailian.console.aliyun.com/?spm=5176.28197581.0.0.72e329a4HRxe99#/model-market/detail/qwen-mt-turbo) |

For large language models that are compatible with the OpenAI API but not listed in the table above, you can set environment variables using the same method outlined for OpenAI in the table.

Use `-s service` or `-s service:model` to specify service:

```bash
pdf2zh example.pdf -s openai:gpt-4o-mini
```

Or specify model with environment variables:

```bash
set OPENAI_MODEL=gpt-4o-mini
pdf2zh example.pdf -s openai
```

For PowerShell user:

```shell
$env:OPENAI_MODEL = gpt-4o-mini
pdf2zh example.pdf -s openai
```

[⬆️ Back to top](#toc)

---

<h3 id="exceptions">Translate wih exceptions</h3>

Use regex to specify formula fonts and characters that need to be preserved:

```bash
pdf2zh example.pdf -f "(CM[^RT].*|MS.*|.*Ital)" -c "(\(|\||\)|\+|=|\d|[\u0080-\ufaff])"
```

Preserve `Latex`, `Mono`, `Code`, `Italic`, `Symbol` and `Math` fonts by default:

```bash
pdf2zh example.pdf -f "(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Ital|.*Sym|.*Math)"
```

[⬆️ Back to top](#toc)

---

<h3 id="threads">Multi-threads</h3>

Use `-t` to specify how many threads to use in translation:

```bash
pdf2zh example.pdf -t 1
```

[⬆️ Back to top](#toc)

---

<h3 id="prompt">Custom prompt</h3>

Note: System prompt is currently not supported. See [this change](https://github.com/Byaidu/PDFMathTranslate/pull/637).

Use `--prompt` to specify which prompt to use in llm:

```bash
pdf2zh example.pdf --prompt prompt.txt
```

For example:

```txt
You are a professional, authentic machine translation engine. Only Output the translated text, do not include any other text.

Translate the following markdown source text to ${lang_out}. Keep the formula notation {v*} unchanged. Output translation directly without any additional text.

Source Text: ${text}

Translated Text:
```

In custom prompt file, there are three variables can be used.

|**variables**|**comment**|
|-|-|
|`lang_in`|input language|
|`lang_out`|output language|
|`text`|text need to be translated|

[⬆️ Back to top](#toc)

---

<h3 id="auth">Authorization</h3>

Use `--authorized` to specify which user to use Web UI and custom the login page:

```bash
pdf2zh example.pdf --authorized users.txt auth.html
```

example users.txt
Each line contains two elements, username, and password, separated by a comma.

```
admin,123456
user1,password1
user2,abc123
guest,guest123
test,test123
```

example auth.html

```html
<!DOCTYPE html>
<html>
<head>
    <title>Simple HTML</title>
</head>
<body>
    <h1>Hello, World!</h1>
    <p>Welcome to my simple HTML page.</p>
</body>
</html>
```

[⬆️ Back to top](#toc)

---

<h3 id="cofig">Custom configuration file</h3>

Use `--config` to specify which file to configure the PDFMathTranslate:

```bash
pdf2zh example.pdf --config config.json
```

```bash
pdf2zh -i --config config.json
```

example config.json

```json
{
    "USE_MODELSCOPE": "0",
    "PDF2ZH_LANG_FROM": "English",
    "PDF2ZH_LANG_TO": "Simplified Chinese",
    "NOTO_FONT_PATH": "/app/SourceHanSerifCN-Regular.ttf",
    "translators": [
        {
            "name": "deeplx",
            "envs": {
                "DEEPLX_ENDPOINT": "http://localhost:1188/translate/",
                "DEEPLX_ACCESS_TOKEN": null
            }
        },
        {
            "name": "ollama",
            "envs": {
                "OLLAMA_HOST": "http://127.0.0.1:11434",
                "OLLAMA_MODEL": "gemma2"
            }
        }
    ]
}
```

By default, the config file is saved in the `~/.config/PDFMathTranslate/config.json`. The program will start by reading the contents of config.json, and after that it will read the contents of the environment variables. When an environment variable is available, the contents of the environment variable are used first and the file is updated.

[⬆️ Back to top](#toc)

---

<h3 id="font-subset">Fonts subsetting</h3>

By default, PDFMathTranslate uses fonts subsetting to decrease sizes of output files. You can use `--skip-subset-fonts` option to disable fonts subsetting when encoutering compatibility issues.

```bash
pdf2zh example.pdf --skip-subset-fonts
```

[⬆️ Back to top](#toc)

---

<h3 id="cache">Translation cache</h3>

PDFMathTranslate caches translated texts to increase speed and avoid unnecessary API calls for same contents. You can use `--ignore-cache` option to ignore translation cache and force retranslation.

```bash
pdf2zh example.pdf --ignore-cache
```

[⬆️ Back to top](#toc)

---

<h3 id="public-services">Deployment as a public services</h3>

PDFMathTranslate has added the features of **enabling partial services** and **hiding Backend information** in 
the configuration file. You can enable these by setting `ENABLED_SERVICES` and `HIDDEN_GRADIO_DETAILS` in the 
configuration file. Among them:

- `ENABLED_SERVICES` allows you to choose to enable only certain options, limiting the number of available services.
- `HIDDEN_GRADIO_DETAILS` will hide the real API_KEY on the web, preventing users from obtaining server-side keys.

A usable configuration is as follows:

```json
{
    "USE_MODELSCOPE": "0",
    "translators": [
        {
            "name": "grok",
            "envs": {
                "GORK_API_KEY": null,
                "GORK_MODEL": "grok-2-1212"
            }
        },
        {
            "name": "openai",
            "envs": {
                "OPENAI_BASE_URL": "https://api.openai.com/v1",
                "OPENAI_API_KEY": "sk-xxxx",
                "OPENAI_MODEL": "gpt-4o-mini"
            }
        }
    ],
    "ENABLED_SERVICES": [
        "OpenAI",
        "Grok"
    ],
    "HIDDEN_GRADIO_DETAILS": true,
    "PDF2ZH_LANG_FROM": "English",
    "PDF2ZH_LANG_TO": "Simplified Chinese",
    "NOTO_FONT_PATH": "/app/SourceHanSerifCN-Regular.ttf"
}
```

[⬆️ Back to top](#toc)


---

<h3 id="mcp">MCP</h3>

PDFMathTranslate can run as MCP server. To use this, you need to run `uv pip install pdf2zh`, and config `claude_desktop_config.json`, an example config is as follows:

``` json
{
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/path/to/Document"
            ]
        },
        "translate_pdf": {
            "command": "uv",
            "args": [
                "run",
                "pdf2zh",
                "--mcp"
            ]
        }
    }
}
```

[filesystem](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem) is a reuqired mcp server to find pdf file, and `translate_pdf` is our mcp server.

To test if the mcp server works, you can open claude desktop and tell

```
find the `test.pdf` in my Document folder and translate it to Chinese
```
