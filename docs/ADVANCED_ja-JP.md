[**ドキュメント**](https://github.com/Byaidu/PDFMathTranslate) > **高度な使い方** _(現在のページ)_

---

<h3 id="toc">目次</h3>

- [全ページ／指定ページの翻訳](#partial)
- [入出力言語の指定](#language)
- [翻訳サービスの切り替え](#services)
- [例外のある翻訳](#exceptions)
- [マルチスレッド処理](#threads)
- [カスタムプロンプト](#prompt)
- [認証](#auth)
- [独自の設定ファイル](#cofig)
- [フォントサブセット化](#fonts-subset)
- [翻訳キャッシュ](#cache)
- [Riva サーバーの準備](#riva-hosting)
- [公開サービスとしての運用](#public-services)
- [MCP 連携](#mcp)

---

<h3 id="partial">全ページ／指定ページの翻訳</h3>

- 全ページを翻訳する場合

  ```bash
  pdf2zh example.pdf
  ```

- ページを指定する場合

  ```bash
  pdf2zh example.pdf -p 1-3,5
  ```

[⬆️ トップへ戻る](#toc)

---

<h3 id="language">入出力言語の指定</h3>

[Google 言語コード](https://developers.google.com/admin-sdk/directory/v1/languages)、[DeepL 言語コード](https://developers.deepl.com/docs/resources/supported-languages)を参照してください。

```bash
pdf2zh example.pdf -li en -lo ja
```

[⬆️ トップへ戻る](#toc)

---

<h3 id="services">翻訳サービスの切り替え</h3>

各翻訳サービスで必要となる[環境変数の一覧表](https://chatgpt.com/share/6734a83d-9d48-800e-8a46-f57ca6e8bcb4)を用意しています。利用前に該当する環境変数を設定してください。

| **翻訳器**             | **サービス指定値** | **環境変数**                                                           | **既定値**                                                | **備考**                                                                                                                                                                                                  |
|------------------------|--------------------|------------------------------------------------------------------------|-----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Google (既定)**      | `google`           | なし                                                                   | N/A                                                       | 設定不要                                                                                                                                                                                                  |
| **Bing**               | `bing`             | なし                                                                   | N/A                                                       | 同上                                                                                                                                                                                                      |
| **302.AI**             | `302ai`            | `X302AI_API_KEY`, `X302AI_MODEL`                                       | `[Your Key]`, `Gemma-7B`                                  | [302.AI](https://share.302.ai/tqTWfD) を参照                                                                                                                                                              |
| **OpenAI**             | `openai`           | `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`                    | `https://api.openai.com/v1`, `[Your Key]`, `gpt-4o-mini`  | [OpenAI](https://platform.openai.com/docs/overview) を参照                                                                                                                                               |
| **DeepL**              | `deepl`            | `DEEPL_AUTH_KEY`                                                       | `[Your Key]`                                              | [DeepL](https://support.deepl.com/hc/en-us/articles/360020695820-API-Key-for-DeepL-s-API) を参照                                                                                                         |
| **DeepLX**             | `deeplx`           | `DEEPLX_ENDPOINT`                                                      | `https://api.deepl.com/translate`                         | [DeepLX](https://github.com/OwO-Network/DeepLX) を参照                                                                                                                                                    |
| **Ollama**             | `ollama`           | `OLLAMA_HOST`, `OLLAMA_MODEL`                                          | `http://127.0.0.1:11434`, `gemma2`                        | [Ollama](https://github.com/ollama/ollama) を参照                                                                                                                                                        |
| **Xinference**         | `xinference`       | `XINFERENCE_HOST`, `XINFERENCE_MODEL`                                  | `http://127.0.0.1:9997`, `gemma-2-it`                     | [Xinference](https://github.com/xorbitsai/inference) を参照                                                                                                                                              |
| **AzureOpenAI**        | `azure-openai`     | `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_MODEL`  | `[Your Endpoint]`, `[Your Key]`, `gpt-4o-mini`            | [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/) を参照                                                                                                                       |
| **Azure AI**           | `azure`            | `AZURE_REGION`, `AZURE_KEY`                                            | `[Your Region]`, `[Your Key]`                             | [Azure Translator](https://learn.microsoft.com/azure/ai-services/translator/) を参照                                                                                                                     |
| **Claude**             | `claude`           | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`                                 | `[Your Key]`, `claude-3-5-sonnet-20241022`                | [Anthropic](https://www.anthropic.com/api) を参照                                                                                                                                                        |
| **Gemini**             | `gemini`           | `GEMINI_API_KEY`, `GEMINI_MODEL`                                       | `[Your Key]`, `gemini-1.5-pro`                            | [Google AI Studio](https://ai.google.dev/) を参照                                                                                                                                                         |
| **Together AI**        | `together`         | `TOGETHER_API_KEY`, `TOGETHER_MODEL`                                   | `[Your Key]`, `meta-llama/Llama-3.1-8B-Instruct-Turbo`    | [Together AI](https://docs.together.ai/) を参照                                                                                                                                                           |
| **Cohere**             | `cohere`           | `COHERE_API_KEY`, `COHERE_MODEL`                                       | `[Your Key]`, `command-r7b-12-2024`                       | [Cohere](https://docs.cohere.com/) を参照                                                                                                                                                                 |
| **Grok**               | `grok`             | `GROK_API_KEY`, `GROK_MODEL`                                           | `[Your Key]`, `grok-beta`                                 | [xAI](https://docs.x.ai/docs) を参照                                                                                                                                                                      |
| **BabelDOC**           | `babeldoc`         | `BABELDOC_BASE_URL`, `BABELDOC_API_KEY`                                | `http://127.0.0.1:2222`, `[Your Key]`                     | [BabelDOC](https://github.com/funstory-ai/BabelDOC) を参照                                                                                                                                                |
| **Volcengine**         | `volcengine`       | `VOLCENGINE_SECRET_ID`, `VOLCENGINE_SECRET_KEY`, `VOLCENGINE_REGION`   | `[Your Secret ID]`, `[Your Secret Key]`, `cn-north-1`     | [Volcengine](https://www.volcengine.com/docs) を参照                                                                                                                                                     |
| **Tencent Cloud TMT**  | `tencentcloud`     | `TENCENTCLOUD_SECRET_ID`, `TENCENTCLOUD_SECRET_KEY`, `TENCENTCLOUD_REGION` | `[Your Secret ID]`, `[Your Secret Key]`, `ap-beijing`     | [Tencent Cloud](https://www.tencentcloud.com/document/product/551) を参照                                                                                                                                |
| **AILS**               | `ails`             | `AILS_API_KEY`, `AILS_MODEL`                                           | `[Your Key]`, `gpt-4o-mini`                               | [AILS](https://ails.tokyo/) を参照                                                                                                                                                                       |
| **LingvaNex**          | `lingvanex`        | `LINGVANEX_KEY`, `LINGVANEX_HOST`, `LINGVANEX_PROJECT_ID`              | `[Your Key]`, `https://api.lingvanex.com`, `[Project ID]` | [LingvaNex](https://lingvanex.com/) を参照                                                                                                                                                               |
| **Argos Translate**    | `argostranslate`   | なし（`pdf2zh[argostranslate]` のインストールが必要）                   | N/A                                                       | ローカル翻訳エンジン。初回は追加言語パックのダウンロードが必要です。                                                                                                                                    |
| **NVIDIA Riva**        | `riva`             | `RIVA_ENDPOINT`, `RIVA_MODEL`, `RIVA_USE_SSL`, `RIVA_SSL_ROOT_CERT`, `RIVA_SSL_CLIENT_CERT`, `RIVA_SSL_CLIENT_KEY` | `localhost:50051`, `riva_nmt_en_ja_24.10`, `0`, `None`, `None`, `None` | [Riva Quick Start](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide/nmt.html) で立ち上げた翻訳サーバーが必要です。 |

[⬆️ トップへ戻る](#toc)

---

<h3 id="exceptions">例外のある翻訳</h3>

`--except` オプションを使うと、特定範囲を翻訳対象から除外できます。

```bash
pdf2zh example.pdf --except 1-3,5
```

[⬆️ トップへ戻る](#toc)

---

<h3 id="riva-hosting">Riva サーバーの準備</h3>

1. `pip install "pdf2zh[local]"` で Python 依存を追加し、その上で NVIDIA Container Toolkit・Docker・NGC CLI をインストールして `ngc config set` で API キーを登録します。
2. Quick Start パッケージを取得し、翻訳モデルを展開します。
   ```bash
   export RIVA_VERSION=2.23.0
   ngc registry resource download-version "nvidia/riva/riva_quickstart:${RIVA_VERSION}" --dest ~/riva_quickstart
   cd ~/riva_quickstart
   ./riva_init.sh --models "nmt_en_ja,nmt_ja_en" --deploy-type gpu --accept-eula
   ```
3. `./riva_start.sh` でサーバーを起動（既定は `localhost:50051`）。別ホストで動かす場合はそのアドレスを `RIVA_ENDPOINT` に指定します。
4. PDFMathTranslate では `--service riva` あるいは `-s riva:riva_nmt_en_ja_24.10` を指定し、`RIVA_MODEL` で利用する RMIR 名を切り替えます。
5. `ListSupportedLanguagePairs` API で対応言語ペアを確認しておくと、`lang_in` / `lang_out` のミスマッチによる失敗を防げます。

[⬆️ トップへ戻る](#toc)

---

<h3 id="threads">マルチスレッド処理</h3>

`-t` でスレッド数を指定できます。大きな PDF を高速に処理したい場合に有効です。

```bash
pdf2zh example.pdf -t 4
```

[⬆️ トップへ戻る](#toc)

---

<h3 id="prompt">カスタムプロンプト</h3>

現時点でシステムプロンプトには非対応です（[変更内容](https://github.com/Byaidu/PDFMathTranslate/pull/637)を参照）。`--prompt` で LLM に渡すプロンプトを切り替えられます。

```bash
pdf2zh example.pdf --prompt prompt.txt
```

例:

```txt
You are a professional, authentic machine translation engine. Only Output the translated text, do not include any other text.

Translate the following markdown source text to ${lang_out}. Keep the formula notation {v*} unchanged. Output translation directly without any additional text.

Source Text: ${text}

Translated Text:
```

プロンプト内では次の変数が利用できます。

| **変数** | **説明** |
|----------|-----------|
| `${lang_in}` | 入力言語 |
| `${lang_out}` | 出力言語 |
| `${text}` | 翻訳対象テキスト |

[⬆️ トップへ戻る](#toc)

---

<h3 id="auth">認証</h3>

API キーなどの秘密情報は環境変数か設定ファイルに登録してください。Azure や OpenAI など、サービスによって必要な項目は上記の表を参照します。

[⬆️ トップへ戻る](#toc)

---

<h3 id="cofig">独自の設定ファイル</h3>

`pdf2zh/config.py` はユーザー設定を `~/.config/PDFMathTranslate/config.json` に保存します。ファイルの例:

```json
{
    "USE_MODELSCOPE": "0",
    "PDF2ZH_LANG_FROM": "en",
    "PDF2ZH_LANG_TO": "zh-CN",
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

プログラムはまず config.json を読み込み、その後に環境変数を参照します。環境変数が設定されていればその値が優先され、ファイル内容も更新されます。

[⬆️ トップへ戻る](#toc)

---

<h3 id="fonts-subset">フォントサブセット化</h3>

PDFMathTranslate は出力 PDF を軽量化するため、既定でフォントサブセット化を行います。互換性の問題がある場合は `--skip-subset-fonts` で無効化してください。

```bash
pdf2zh example.pdf --skip-subset-fonts
```

[⬆️ トップへ戻る](#toc)

---

<h3 id="cache">翻訳キャッシュ</h3>

同じテキストの再翻訳を避けるため、PDFMathTranslate は翻訳結果をキャッシュします。`--ignore-cache` でキャッシュを無視して再翻訳できます。

```bash
pdf2zh example.pdf --ignore-cache
```

[⬆️ トップへ戻る](#toc)

---

<h3 id="public-services">公開サービスとしての運用</h3>

設定ファイルでは **利用可能なサービスの制限** と **バックエンド情報の非表示** を設定できます。`ENABLED_SERVICES` と `HIDDEN_GRADIO_DETAILS` を指定すると、提供する機能を制御できます。

- `ENABLED_SERVICES`: 利用を許可するサービスのみを列挙します。
- `HIDDEN_GRADIO_DETAILS`: Web 画面に実際の API キーを表示しないようにします。

設定例:

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

[⬆️ トップへ戻る](#toc)

---

<h3 id="mcp">MCP 連携</h3>

PDFMathTranslate は MCP サーバーとして動作できます。以下のように `uv pip install pdf2zh` を行い、`claude_desktop_config.json` を設定してください。

```json
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

[filesystem](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem) は PDF を探索するために必須の MCP サーバーで、`translate_pdf` が PDFMathTranslate の MCP サーバーです。

動作確認には、Claude Desktop を開いて次のように依頼します。

```
find the `test.pdf` in my Document folder and translate it to Chinese
```

[⬆️ トップへ戻る](#toc)
