<div align="center">
	<a href="https://go.warp.dev/PDFMathTranslate" target="_blank">
		<sup>Special thanks to:</sup>
		<br>
		<img alt="Warp sponsorship" width="400" src="https://github.com/warpdotdev/brand-assets/blob/main/Github/Sponsor/Warp-Github-LG-02.png">
		<br>
		<h>Warp, built for coding with multiple AI agents</b>
		<br>
		<sup>Available for macOS, Linux and Windows</sup>
	</a>
</div>

<br>

<div align="center">

English | [日本語](docs/README_ja-JP.md)

<img src="./docs/images/banner.png" width="320px"  alt="PDF2ZH"/>

<h2 id="title">PDFMathTranslate</h2>

<p>
  <!-- PyPI -->
  <a href="https://pypi.org/project/pdf2zh/">
    <img src="https://img.shields.io/pypi/v/pdf2zh"></a>
  <a href="https://pepy.tech/projects/pdf2zh">
    <img src="https://static.pepy.tech/badge/pdf2zh"></a>
  <a href="https://hub.docker.com/r/byaidu/pdf2zh">
    <img src="https://img.shields.io/docker/pulls/byaidu/pdf2zh"></a>
  <a href="https://hellogithub.com/repository/8ec2cfd3ef744762bf531232fa32bc47" target="_blank"><img src="https://api.hellogithub.com/v1/widgets/recommend.svg?rid=8ec2cfd3ef744762bf531232fa32bc47&claim_uid=JQ0yfeBNjaTuqDU&theme=small" alt="Featured｜HelloGitHub" /></a>
  <a href="https://gitcode.com/Byaidu/PDFMathTranslate/overview">
    <img src="https://gitcode.com/Byaidu/PDFMathTranslate/star/badge.svg"></a>
  <a href="https://huggingface.co/spaces/reycn/PDFMathTranslate-Docker">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97-Online%20Demo-FF9E0D"></a>
  <a href="https://www.modelscope.cn/studios/AI-ModelScope/PDFMathTranslate">
    <img src="https://img.shields.io/badge/ModelScope-Demo-blue"></a>
  <a href="https://github.com/Byaidu/PDFMathTranslate/pulls">
    <img src="https://img.shields.io/badge/contributions-welcome-green"></a>
  <a href="https://t.me/+Z9_SgnxmsmA5NzBl">
    <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white"></a>
  <!-- Licenses -->
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-AGPL--3.0-blue.svg" alt="AGPL-3.0 License"></a>
  <a href="./THIRD_PARTY_NOTICES.md">
    <img src="https://img.shields.io/badge/Third%20Party-Licenses-brightgreen.svg" alt="Third-party notices"></a>
</p>

<a href="https://trendshift.io/repositories/12424" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12424" alt="Byaidu%2FPDFMathTranslate | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

</div>

<h2 id="updates">1. What does this do?</h2>

Scientific PDF document translation preserving layouts.

- 📊 Preserve formulas, charts, table of contents, and annotations.
- 🌐 Support [multiple languages](#usage), and diverse [translation services](#usage).
- 🤖 Provides [commandline tool](#usage), [interactive user interface](#install), and [Docker](#install)

<div align="center">
<img src="./docs/images/preview.gif" width="80%"/>
</div>

<h2 id="updates">2. Recent Updates</h2>

- [May 9, 2025] pdf2zh 2.0 Preview Version [#586](https://github.com/Byaidu/PDFMathTranslate/issues/586): The Windows ZIP file and Docker image are now available.

  > [!NOTE]
  >
  > 2.0 Moved to a new repository under the organization: [PDFMathTranslate/PDFMathTranslate-next](https://github.com/PDFMathTranslate/PDFMathTranslate-next)
  > 
  > Version 2.0 official release has been published.

- [Mar. 3, 2025] Experimental support for the new backend [BabelDOC](https://github.com/funstory-ai/BabelDOC) WebUI added as an experimental option (by [@awwaawwa](https://github.com/awwaawwa))
- [Feb. 22 2025] Better release CI and well-packaged windows-amd64 exe (by [@awwaawwa](https://github.com/awwaawwa))


<h2 id="use-section">3. Use 🌟</h2>
<h3 id="demo">3.1 Online Service 🌟</h3>

You can try our application out using either of the following demos:

- [Public free service](https://pdf2zh.com/) online without installation _(recommended)_.
- [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) Free usage quota is available; please refer to the FAQ section on the page for details. _(recommended)_
- [Demo hosted on HuggingFace](https://huggingface.co/spaces/reycn/PDFMathTranslate-Docker)
- [Demo hosted on ModelScope](https://www.modelscope.cn/studios/AI-ModelScope/PDFMathTranslate) without installation.

Note that the computing resources of the demo are limited, so please avoid abusing them.

<h3 id="install">3.2 Local Installation</h3>

For different use cases, we provide distinct methods to use our program:

<details open>
  <summary>3.2.1 Python: Install using uv</summary>

1. Python installed (3.10 <= version <= 3.12)

2. Install our package:

   ```bash
   pip install uv
   uv tool install --python 3.12 pdf2zh
   ```

3. Execute translation, files generated in [current working directory](https://chatgpt.com/share/6745ed36-9acc-800e-8a90-59204bd13444):

   ```bash
   pdf2zh document.pdf
   ```

</details>
<details>
  <summary>3.2.2 Python: Install using pip</summary>

1. Python installed (3.10 <= version <= 3.12)
2. Install our package:

   ```bash
   pip install pdf2zh
   ```

3. Execute translation, files generated in [current working directory](https://chatgpt.com/share/6745ed36-9acc-800e-8a90-59204bd13444):

   ```bash
   pdf2zh document.pdf
   ```

</details>
<details>
  <summary>3.3.3 Python: Graphic user interface</summary>

1. Python installed (3.10 <= version <= 3.12)

2. Install our package:

  ```bash
  pip install pdf2zh
  ```

3. Start using in browser:

   ```bash
   pdf2zh -i
   ```

4. If your browser has not been started automatically, goto

   ```bash
   http://localhost:7860/
   ```

   <img src="./docs/images/gui.gif" width="500"/>

See [documentation for GUI](./docs/README_GUI.md) for more details.

</details>

<details>
  <summary>3.2.4 Application: On Windows</summary>

1. Download pdf2zh-version-win64.zip from [release page](https://github.com/Byaidu/PDFMathTranslate/releases)

2. Unzip and double-click `pdf2zh.exe` to run.


  > [!TIP]
  >
  > - If you're using Windows and cannot open the file after downloading, please install [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) and try again.
  > 
</details>


<details>

<summary>3.2.5 Reference manager: Zotero Plugin</summary>


See [Zotero PDF2zh](https://github.com/guaguastandup/zotero-pdf2zh) for more details.

</details>


<details>
  <summary>3.2.6 Docker: Containerized Deployment</summary>

1. Pull and run:

   ```bash
   docker pull byaidu/pdf2zh
   docker run -d -p 7860:7860 byaidu/pdf2zh
   ```

2. Open in browser:

   ```
   http://localhost:7860/
   ```

For docker deployment on cloud service:

<div>
<a href="https://www.heroku.com/deploy?template=https://github.com/Byaidu/PDFMathTranslate">
  <img src="https://www.herokucdn.com/deploy/button.svg" alt="Deploy" height="26"></a>
<a href="https://render.com/deploy">
  <img src="https://render.com/images/deploy-to-render-button.svg" alt="Deploy to Koyeb" height="26"></a>
<a href="https://zeabur.com/templates/5FQIGX?referralCode=reycn">
  <img src="https://zeabur.com/button.svg" alt="Deploy on Zeabur" height="26"></a>
<a href="https://template.sealos.io/deploy?templateName=pdf2zh">
  <img src="https://sealos.io/Deploy-on-Sealos.svg" alt="Deploy on Sealos" height="26"></a>
<a href="https://app.koyeb.com/deploy?type=git&builder=buildpack&repository=github.com/Byaidu/PDFMathTranslate&branch=main&name=pdf-math-translate">
  <img src="https://www.koyeb.com/static/images/deploy/button.svg" alt="Deploy to Koyeb" height="26"></a>
</div>

> [!TIP]
>
> - If you cannot access Docker Hub, please try the image on [GitHub Container Registry](https://github.com/Byaidu/PDFMathTranslate/pkgs/container/pdfmathtranslate).
> ```bash
> docker pull ghcr.io/byaidu/pdfmathtranslate
> docker run -d -p 7860:7860 ghcr.io/byaidu/pdfmathtranslate
> ```
</details>

<details>
  <summary>3.2.* Solutions for network issues in installation</summary>

  Users in specific regions may encounter network difficulties when loading the AI model. The current program relies on the AI model (`wybxc/DocLayout-YOLO-DocStructBench-onnx`), and some users are unable to download it due to these network issues.

  To address issues with downloading this model, use the following environment variable as a workaround:

  ```shell
  set HF_ENDPOINT=https://hf-mirror.com
  ```

  For PowerShell user:

  ```shell
  $env:HF_ENDPOINT = https://hf-mirror.com
  ```

  If the solution does not work to you / you encountered other issues, please refer to [Frequently Asked Questions](https://github.com/Byaidu/PDFMathTranslate/wiki#-faq--%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98).
</details>


<h2 id="usage">4. Technical Details</h2>

### 4.1 Advanced options

Execute the translation command in the command line to generate the translated document `example-mono.pdf` and the bilingual document `example-dual.pdf` in the current working directory. Use Google as the default translation service. More support translation services can find [HERE](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#services).

<img src="./docs/images/cmd.explained.png" width="580px"  alt="cmd"/>

In the following table, we list all advanced options for reference:

| Option                | Function                                                                                                      | Example                                        |
| --------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| files                 | Local files                                                                                                   | `pdf2zh ~/local.pdf`                           |
| links                 | Online files                                                                                                  | `pdf2zh http://arxiv.org/paper.pdf`            |
| `-i`                  | [Enter GUI](#gui)                                                                                             | `pdf2zh -i`                                    |
| `-p`                  | [Partial document translation](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#partial) | `pdf2zh example.pdf -p 1`                      |
| `-li`                 | [Source language](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#languages)            | `pdf2zh example.pdf -li en`                    |
| `-lo`                 | [Target language](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#languages)            | `pdf2zh example.pdf -lo ja`                    |
| `-s`                  | [Translation service](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#services)         | `pdf2zh example.pdf -s deepl`                  |
| `-t`                  | [Multi-threads](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#threads)                | `pdf2zh example.pdf -t 1`                      |
| `-o`                  | Output dir                                                                                                    | `pdf2zh example.pdf -o output`                 |
| `--output-format`     | Choose translated artifact: pdf / md / both                                                                   | `pdf2zh example.pdf --output-format both`      |
| `--markdown-footnotes`| Control Markdown footnotes placement: keep-inline / move-to-end / remove                                      | `pdf2zh example.pdf --output-format md --markdown-footnotes move-to-end` |
| `--no-translate`      | Skip translation and reuse source text (layout debugging)                                                     | `pdf2zh example.pdf --no-translate`            |
| `-f`, `-c`            | [Exceptions](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#exceptions)                | `pdf2zh example.pdf -f "(MS.*)"`               |
| `-cp`                 | Compatibility Mode                                                                                            | `pdf2zh example.pdf --compatible`              |
| `--skip-subset-fonts` | [Skip font subset](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#font-subset)         | `pdf2zh example.pdf --skip-subset-fonts`       |
| `--ignore-cache`      | [Ignore translate cache](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#cache)         | `pdf2zh example.pdf --ignore-cache`            |
| `--share`             | Public link                                                                                                   | `pdf2zh -i --share`                            |
| `--authorized`        | [Authorization](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#auth)                   | `pdf2zh -i --authorized users.txt [auth.html]` |
| `--prompt`            | [Custom Prompt](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#prompt)                 | `pdf2zh --prompt [prompt.txt]`                 |
| `--onnx`              | [Use Custom DocLayout-YOLO ONNX model]                                                                        | `pdf2zh --onnx [onnx/model/path]`              |
| `--serverport`        | [Use Custom WebUI port]                                                                                       | `pdf2zh --serverport 7860`                     |
| `--dir`               | [batch translate]                                                                                             | `pdf2zh --dir /path/to/translate/`             |
| `--config`            | [configuration file](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#cofig)             | `pdf2zh --config /path/to/config/config.json`  |
| `--serverport`        | [custom gradio server port]                                                                                   | `pdf2zh --serverport 7860`                     |
| `--babeldoc`          | Use Experimental backend [BabelDOC](https://funstory-ai.github.io/BabelDOC/) to translate                     | `pdf2zh --babeldoc` -s openai example.pdf      |
| `--mcp`               | Enable MCP STDIO mode                                                                                         | `pdf2zh --mcp`                                 |
| `--sse`               | Enable MCP SSE mode                                                                                           | `pdf2zh --mcp --sse`                           |

For detailed explanations, please refer to our document about [Advanced Usage](./docs/ADVANCED.md) for a full list of each option.

#### Markdown Footnote Handling

When converting PDFs to Markdown, page-level footnotes and copyright notices often interrupt paragraphs. Control their placement via:

```bash
pdf2zh document.pdf --output-format md --markdown-footnotes move-to-end
```

Options:

- `move-to-end` (default): Move footnotes/page footers to a `### Footnotes` section at the end.
- `keep-inline`: Keep them at the original positions (may split ABSTRACT text).
- `remove`: Remove them completely.

Academic papers (ACM/IEEE etc.) usually look cleaner with `move-to-end` or `remove`, because the permission notice on every page no longer breaks the narrative flow.

#### Self-hosted NVIDIA Riva (beta)

- Install optional dependencies: `pip install "pdf2zh[local]"` (or `uv pip install --editable ".[local]"` inside this repo).
- Start an [NVIDIA Riva Quick Start](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide/nmt.html) server with the desired RMIR (e.g., `riva_nmt_en_ja_24.10`), then set:
  ```bash
  export RIVA_ENDPOINT=lab-gpu-01:50051
  export RIVA_MODEL=riva_nmt_en_ja_24.10
  pdf2zh example.pdf -li en -lo ja -s riva
  ```
- TLS-protected deployments can also set `RIVA_USE_SSL`, `RIVA_SSL_ROOT_CERT`, etc. See [docs/ADVANCED.md#riva-hosting](docs/ADVANCED.md#riva-hosting) for the full walkthrough.

<h3 id="downstream">4.2 Downstream Development</h3>
For downstream applications, please refer to our document about [API Details](./docs/APIS.md) for further information about:

- [Python API](./docs/APIS.md#api-python), how to use the program in other Python programs
- [HTTP API](./docs/APIS.md#api-http), how to communicate with a server with the program installed

<h3 id="downstream">4.3 Differences between two major forks</h3>

- [Byaidu/PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate): The present and the original project for stable release.

- [PDFMathTranslate/PDFMathTranslate-next](https://github.com/PDFMathTranslate/PDFMathTranslate-next): A fork with web-ui and additional features. This fork handles a large number of marginal cases, improves PDF compatibility, and optimizes cross-column and cross-page semantic consistency, dynamic scaling, and dynamic scaling consistency, among many other translation quality improvements. However, this fork is intended solely for development and does not address compatibility issues and is not designed for community-contributions.

<h2 id="information">5. Project Information</h2>
<h3 id="citation">5.1 Citation</h3>

This work has been accepted by the *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations* (EMNLP 2025). 

- Pre-print version: [PDFMathTranslate: Scientific Document Translation Preserving Layouts](https://arxiv.org/abs/2507.03009)

  ```
  @online{ouyang2025pdfmathtranslate,
    title = {{{PDFMathTranslate}}: {{Scientific Document Translation Preserving Layouts}}},
    shorttitle = {{{PDFMathTranslate}}},
    author = {Ouyang, Rongxin and Chu, Chang and Xin, Zhikuang and Ma, Xiangyao},
    date = {2025-07-08},
    eprint = {2507.03009},
    eprinttype = {arXiv},
    eprintclass = {cs},
    doi = {10.48550/arXiv.2507.03009},
    url = {http://arxiv.org/abs/2507.03009},
    urldate = {2025-08-27},
    pubstate = {prepublished}
  }
  ```

- The citation for the EMNLP proceedings will be provided upon release.
<!-- ```
@inproceedings{zheng-etal-2024-openresearcher,
    title = "{O}pen{R}esearcher: Unleashing {AI} for Accelerated Scientific Research",
    author = "Ouyang, Rongxin  and
      Chu, Chang and
      Xin, Zhikuang and
      Ma, Xiangyao",
    editor = "TBD",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2025",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/TBD/",
    doi = "TBD",
    pages = "TBD",
    abstract = "Language barriers in scientific documents hinder the diffusion and development of science and technologies. However, prior efforts in translating such documents largely overlooked the information in layouts. To bridge the gap, we introduce PDFMathTranslate, the world’s first open-source software for translating scientific documents while preserving layouts. Leveraging the most recent advances in large language models and precise layout detection, we contribute to the community with key improvements in precision, flexibility, and efficiency. The work is open-sourced at https://github.com/byaidu/pdfmathtranslate with more than 222k downloads."
}
``` -->

<h3 id="acknowledgement">5.2 Acknowledgement</h3>

- [Immersive Translation](https://immersivetranslate.com) sponsors monthly Pro membership redemption codes for active contributors to this project, see details at: [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md)

- New backend: [BabelDOC](https://github.com/funstory-ai/BabelDOC)

- Document merging: [PyMuPDF](https://github.com/pymupdf/PyMuPDF)

- Document parsing: [Pdfminer.six](https://github.com/pdfminer/pdfminer.six)

- Document extraction: [MinerU](https://github.com/opendatalab/MinerU)

- Document Preview: [Gradio PDF](https://github.com/freddyaboulton/gradio-pdf)

- Multi-threaded translation: [MathTranslate](https://github.com/SUSYUSTC/MathTranslate)

- Layout parsing: [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)

- Document standard: [PDF Explained](https://zxyle.github.io/PDF-Explained/), [PDF Cheat Sheets](https://pdfa.org/resource/pdf-cheat-sheets/)

- Multilingual Font: [Go Noto Universal](https://github.com/satbyy/go-noto-universal)

<h3 id="license">5.3 License & Notices</h3>

- This fork is distributed under the [GNU Affero General Public License v3.0](./LICENSE).
- Dual-licensed dependencies such as PyMuPDF / pymupdf4llm, along with other bundled libraries, are listed in [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) for easy attribution when redistributing binaries or Docker images.

<h3 id="contrib">5.4 Contributors</h3>

<a href="https://github.com/Byaidu/PDFMathTranslate/graphs/contributors">
  <img src="https://opencollective.com/PDFMathTranslate/contributors.svg?width=890&button=false" />
</a>

![Alt](https://repobeats.axiom.co/api/embed/dfa7583da5332a11468d686fbd29b92320a6a869.svg "Repobeats analytics image")

For details on how to contribute, please consult the [Contribution Guide](https://github.com/Byaidu/PDFMathTranslate/wiki/Contribution-Guide---%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97).


<h3 id="star_hist">5.5 Star History</h3>

<a href="https://star-history.com/#Byaidu/PDFMathTranslate&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Byaidu/PDFMathTranslate&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Byaidu/PDFMathTranslate&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Byaidu/PDFMathTranslate&type=Date"/>
 </picture>
</a>
