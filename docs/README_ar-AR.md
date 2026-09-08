<div align="center">

[English](../README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md) | [日本語](README_ja-JP.md) | [한국어](README_ko-KR.md) | العربية

<img src="./images/banner.png" width="320px"  alt="PDF2ZH"/>

<h2 id="title">PDFMathTranslate</h2>

<p>
  <!-- PyPI -->
  <a href="https://pypi.org/project/pdf2zh/">
    <img src="https://img.shields.io/pypi/v/pdf2zh"></a>
  <a href="https://pepy.tech/projects/pdf2zh">
    <img src="https://static.pepy.tech/badge/pdf2zh"></a>
  <a href="https://hub.docker.com/r/byaidu/pdf2zh">
    <img src="https://img.shields.io/docker/pulls/byaidu/pdf2zh"></a>
  <a href="https://huggingface.co/spaces/reycn/PDFMathTranslate-Docker">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97-Online%20Demo-FF9E0D"></a>
  <a href="https://www.modelscope.cn/studios/AI-ModelScope/PDFMathTranslate">
    <img src="https://img.shields.io/badge/ModelScope-Demo-blue"></a>
  <a href="https://github.com/Byaidu/PDFMathTranslate/pulls">
    <img src="https://img.shields.io/badge/contributions-welcome-green"></a>
  <a href="https://t.me/+Z9_SgnxmsmA5NzBl">
    <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=flat-squeare&logo=telegram&logoColor=white"></a>
  <!-- License -->
  <a href="../LICENSE">
    <img src="https://img.shields.io/github/license/Byaidu/PDFMathTranslate"></a>
</p>

<a href="https://trendshift.io/repositories/19816" target="_blank"><img src="https://trendshift.io/api/badge/repositories/19816" alt="PDFMathTranslate%2FPDFMathTranslate | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

</div>

<h2 id="what">١. ما الذي يفعله هذا المشروع؟</h2>

ترجمة مستندات PDF العلمية مع الحفاظ على التنسيق الأصلي.

- 📊 الحفاظ على المعادلات والرسوم البيانية وجداول المحتويات والحواشي.
- 🌐 دعم [لغات متعددة](#usage) و[خدمات ترجمة متنوعة](#usage).
- 🤖 يوفّر [أداة سطر أوامر](#usage) و[واجهة رسومية تفاعلية](#install) و[Docker](#install).
- ↔️ دعم كامل للغات التي تُكتب من اليمين إلى اليسار، ومنها العربية والعبرية والفارسية والأردية.

<div align="center">
<img src="./images/preview.gif" width="80%"/>
</div>

<h2 id="rtl">٢. دعم العربية واللغات ذات الاتجاه من اليمين إلى اليسار</h2>

يعتمد إخراج النصوص العربية على مسار تنضيد مستقل يراعي خصائص الكتابة العربية:

- **تشكيل الحروف (Shaping):** تُمرَّر النصوص إلى محرك HarfBuzz لاختيار الأشكال
  السياقية الصحيحة (أولية ووسطية ونهائية ومنفصلة)، مع دعم الرباطات مثل «لا».
- **الخوارزمية ثنائية الاتجاه (UAX #9):** تنفيذ مطابق للمواصفة، يجتاز مجموعة
  اختبارات المطابقة الرسمية `BidiCharacterTest` من قاعدة بيانات يونيكود بالكامل.
- **الأرقام والنصوص المختلطة:** تبقى الأرقام والكلمات اللاتينية بترتيبها الصحيح
  من اليسار إلى اليمين داخل الفقرات العربية.
- **علامات الترقيم والأقواس:** تُطبَّق قاعدة الانعكاس (L4) فتظهر الأقواس
  والعلامات المتناظرة بالشكل الصحيح في السياق العربي.
- **التشكيل (الحركات):** تُوضع الفتحة والضمة والكسرة والشدة وغيرها في مواضعها
  الصحيحة فوق الحروف وتحتها دون أن تشغل عرضًا أفقيًا.
- **ضبط الأسطر والمسافات:** تباعد أسطر مناسب للعربية، وكسر الأسطر عند حدود
  الكلمات، ومحاذاة إلى اليمين، ونقل علامات القوائم النقطية إلى الجهة الصحيحة.

### الاستخدام

```bash
# ترجمة إلى العربية باستخدام خدمة Google
pdf2zh document.pdf -lo ar

# لغات أخرى تُكتب من اليمين إلى اليسار
pdf2zh document.pdf -lo he      # العبرية
pdf2zh document.pdf -lo fa      # الفارسية
pdf2zh document.pdf -lo ur      # الأردية
```

كما يمكن اختيار «Arabic» مباشرةً من قائمة اللغات في الواجهة الرسومية.

### خيارات إضافية

| الخيار | الوظيفة | مثال |
| --- | --- | --- |
| `--rtl` | التحكم في اتجاه التنضيد: `auto` (افتراضي، يُشتق من لغة الهدف) أو `on` أو `off` | `pdf2zh example.pdf -lo ar --rtl on` |
| `--digit-form` | شكل الأرقام: `auto` (كما تُعيده خدمة الترجمة) أو `western` أو `arabic` | `pdf2zh example.pdf -lo ar --digit-form western` |
| `--min-font-scale` | الحد الأدنى لتصغير الخط عند تعذّر احتواء الفقرة | `pdf2zh example.pdf -lo ar --min-font-scale 0.7` |

> [!NOTE]
>
> عند اختيار لغة هدف تُكتب من اليمين إلى اليسار، يتخطّى البرنامج تقليص مجموعة
> المحارف في الخط (font subsetting) تلقائيًا للحفاظ على الأشكال السياقية للحروف.

للاطلاع على التفاصيل التقنية الكاملة، انظر
[وثيقة دعم الكتابة من اليمين إلى اليسار](./RTL_ARABIC_SUPPORT.md).

<h2 id="updates">٣. آخر التحديثات</h2>

- [٢٣ مارس ٢٠٢٦] دعم تجريبي لنواة الترجمة v2.0 في بيئة معزولة (`--mode precise`). (بواسطة [@reycn](https://github.com/reycn))
- [٢٢ مارس ٢٠٢٦] دعم MiniMax (بواسطة [@octo-patch](https://github.com/octo-patch))
- [٢٢ مارس ٢٠٢٦] إصلاح مشكلات متعلقة بـ OpenAI (بواسطة [@samqin123](https://github.com/samqin123))
- [٢٢ مارس ٢٠٢٦] إصلاح مشكلات متعلقة بـ HTTP (بواسطة [@soukouki](https://github.com/soukouki))
- [٢٢ مارس ٢٠٢٦] تسريع تحميل النماذج على منصات mac وONNX، وتحسين بدء الواجهة الرسومية والتكامل المستمر. (بواسطة [@reycn](https://github.com/reycn))
- [٩ مايو ٢٠٢٥] نسخة معاينة pdf2zh 2.0 [#586](https://github.com/Byaidu/PDFMathTranslate/issues/586): ملف ZIP لنظام Windows وصورة Docker متاحان الآن.

  > [!NOTE]
  >
  > انتقل الإصدار 2.0 إلى مستودع جديد ضمن المنظمة: [PDFMathTranslate/PDFMathTranslate-next](https://github.com/PDFMathTranslate/PDFMathTranslate-next)
  >
  > وقد صدر الإصدار 2.0 رسميًا.

<h2 id="use-section">٤. الاستخدام 🌟</h2>
<h3 id="demo">٤.١ الخدمة عبر الإنترنت 🌟</h3>

يمكنك تجربة التطبيق عبر أحد العروض التالية:

- [خدمة عامة مجانية](https://pdf2zh.com/) عبر الإنترنت دون تثبيت _(موصى بها)_.
- [Immersive Translate - BabelDOC](https://app.immersivetranslate.com/babel-doc/) يتوفر حد استخدام مجاني؛ راجع قسم الأسئلة الشائعة في الصفحة للتفاصيل. _(موصى بها)_
- [عرض تجريبي على HuggingFace](https://huggingface.co/spaces/reycn/PDFMathTranslate-Docker)
- [عرض تجريبي على ModelScope](https://www.modelscope.cn/studios/AI-ModelScope/PDFMathTranslate) دون تثبيت.

يُرجى الانتباه إلى أن الموارد الحاسوبية للعروض التجريبية محدودة، فتجنّب الإفراط في استخدامها.

<h3 id="install">٤.٢ التثبيت المحلي</h3>

نوفّر طرقًا مختلفة تناسب حالات الاستخدام المتعددة:

<details open>
  <summary>٤.٢.١ Python: التثبيت باستخدام uv</summary>

1. تثبيت Python (الإصدار بين 3.11 و3.12)

2. تثبيت الحزمة:

   ```bash
   pip install uv
   uv tool install --python 3.12 pdf2zh
   ```

3. تنفيذ الترجمة، وتُنشأ الملفات في مجلد العمل الحالي:

   ```bash
   pdf2zh document.pdf
   ```

</details>
<details>
  <summary>٤.٢.٢ Python: التثبيت باستخدام pip</summary>

1. تثبيت Python (الإصدار بين 3.11 و3.12)
2. تثبيت الحزمة:

   ```bash
   pip install pdf2zh
   ```

3. تنفيذ الترجمة، وتُنشأ الملفات في مجلد العمل الحالي:

   ```bash
   pdf2zh document.pdf
   ```

</details>
<details>
  <summary>٤.٢.٣ Python: الواجهة الرسومية</summary>

1. تثبيت Python (الإصدار بين 3.11 و3.12)

2. تثبيت الحزمة:

   ```bash
   pip install pdf2zh
   ```

3. البدء من المتصفح:

   ```bash
   pdf2zh -i
   ```

4. إذا لم يفتح المتصفح تلقائيًا، انتقل إلى:

   ```bash
   http://localhost:7860/
   ```

   <img src="./images/gui.gif" width="500"/>

راجع [وثيقة الواجهة الرسومية](./README_GUI.md) للمزيد من التفاصيل.

</details>

<details>
  <summary>٤.٢.٤ التطبيق: على نظام Windows</summary>

1. نزّل الملف `pdf2zh-version-win64.zip` من [صفحة الإصدارات](https://github.com/Byaidu/PDFMathTranslate/releases)

2. فُك الضغط وانقر نقرًا مزدوجًا على `pdf2zh.exe` للتشغيل.

  > [!TIP]
  >
  > - إذا تعذّر فتح الملف بعد التنزيل على نظام Windows، فثبّت [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) ثم أعد المحاولة.
  >
</details>

<details>
  <summary>٤.٢.٥ مدير المراجع: إضافة Zotero</summary>

راجع [Zotero PDF2zh](https://github.com/guaguastandup/zotero-pdf2zh) للمزيد من التفاصيل.

</details>

<details>
  <summary>٤.٢.٦ Docker: النشر عبر الحاويات</summary>

1. السحب والتشغيل:

   ```bash
   docker pull byaidu/pdf2zh
   docker run -d -p 7860:7860 byaidu/pdf2zh
   ```

2. الفتح من المتصفح:

   ```
   http://localhost:7860/
   ```

للنشر عبر Docker على الخدمات السحابية:

<div>
<a href="https://www.heroku.com/deploy?template=https://github.com/Byaidu/PDFMathTranslate">
  <img src="https://www.herokucdn.com/deploy/button.svg" alt="Deploy" height="26"></a>
<a href="https://render.com/deploy">
  <img src="https://render.com/images/deploy-to-render-button.svg" alt="Deploy to Render" height="26"></a>
<a href="https://zeabur.com/templates/5FQIGX?referralCode=reycn">
  <img src="https://zeabur.com/button.svg" alt="Deploy on Zeabur" height="26"></a>
<a href="https://template.sealos.io/deploy?templateName=pdf2zh">
  <img src="https://sealos.io/Deploy-on-Sealos.svg" alt="Deploy on Sealos" height="26"></a>
<a href="https://app.koyeb.com/deploy?type=git&builder=buildpack&repository=github.com/Byaidu/PDFMathTranslate&branch=main&name=pdf-math-translate">
  <img src="https://www.koyeb.com/static/images/deploy/button.svg" alt="Deploy to Koyeb" height="26"></a>
</div>

> [!TIP]
>
> - إذا تعذّر الوصول إلى Docker Hub، فجرّب الصورة على [GitHub Container Registry](https://github.com/Byaidu/PDFMathTranslate/pkgs/container/pdfmathtranslate).
> ```bash
> docker pull ghcr.io/byaidu/pdfmathtranslate
> docker run -d -p 7860:7860 ghcr.io/byaidu/pdfmathtranslate
> ```
</details>

<details>
  <summary>٤.٢.* حلول لمشكلات الشبكة أثناء التثبيت</summary>

  قد يواجه المستخدمون في بعض المناطق صعوبات في الشبكة عند تحميل نموذج الذكاء الاصطناعي. يعتمد البرنامج حاليًا على النموذج (`wybxc/DocLayout-YOLO-DocStructBench-onnx`)، ويتعذّر على بعض المستخدمين تنزيله بسبب هذه المشكلات.

  لمعالجة ذلك، استخدم متغيّر البيئة التالي كحل بديل:

  ```shell
  set HF_ENDPOINT=https://hf-mirror.com
  ```

  لمستخدمي PowerShell:

  ```shell
  $env:HF_ENDPOINT = https://hf-mirror.com
  ```

  إذا لم ينجح الحل أو واجهتك مشكلات أخرى، فراجع [الأسئلة الشائعة](https://github.com/Byaidu/PDFMathTranslate/wiki#-faq--%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98).
</details>

<h2 id="usage">٥. التفاصيل التقنية</h2>

### ٥.١ الخيارات المتقدمة

نفّذ أمر الترجمة في سطر الأوامر لإنشاء المستند المترجم `example-mono.pdf` والمستند ثنائي اللغة `example-dual.pdf` في مجلد العمل الحالي. تُستخدم خدمة Google افتراضيًا. يمكن الاطلاع على مزيد من خدمات الترجمة المدعومة [هنا](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#services).

<img src="./images/cmd.explained.png" width="580px"  alt="cmd"/>

في الجدول التالي نسرد الخيارات المتقدمة للرجوع إليها:

| الخيار | الوظيفة | مثال |
| --- | --- | --- |
| files | ملفات محلية | `pdf2zh ~/local.pdf` |
| links | ملفات عبر الإنترنت | `pdf2zh http://arxiv.org/paper.pdf` |
| `-i` | [فتح الواجهة الرسومية](#install) | `pdf2zh -i` |
| `-p` | [ترجمة جزء من المستند](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#partial) | `pdf2zh example.pdf -p 1` |
| `-li` | [لغة المصدر](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#languages) | `pdf2zh example.pdf -li en` |
| `-lo` | [لغة الهدف](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#languages) | `pdf2zh example.pdf -lo ar` |
| `-s` | [خدمة الترجمة](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#services) | `pdf2zh example.pdf -s deepl` |
| `-t` | [تعدد الخيوط](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#threads) | `pdf2zh example.pdf -t 1` |
| `-o` | مجلد الإخراج | `pdf2zh example.pdf -o output` |
| `-f`, `-c` | [الاستثناءات](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#exceptions) | `pdf2zh example.pdf -f "(MS.*)"` |
| `-cp` | وضع التوافق | `pdf2zh example.pdf --compatible` |
| `--rtl` | اتجاه التنضيد: `auto` أو `on` أو `off` | `pdf2zh example.pdf -lo ar --rtl on` |
| `--digit-form` | شكل الأرقام: `auto` أو `western` أو `arabic` | `pdf2zh example.pdf -lo ar --digit-form western` |
| `--min-font-scale` | الحد الأدنى لتصغير الخط عند تعذّر الاحتواء | `pdf2zh example.pdf --min-font-scale 0.7` |
| `--skip-subset-fonts` | [تخطي تقليص الخطوط](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#font-subset) | `pdf2zh example.pdf --skip-subset-fonts` |
| `--ignore-cache` | [تجاهل ذاكرة الترجمة](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#cache) | `pdf2zh example.pdf --ignore-cache` |
| `--share` | رابط عام | `pdf2zh -i --share` |
| `--authorized` | [التفويض](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#auth) | `pdf2zh -i --authorized users.txt [auth.html]` |
| `--prompt` | [موجّه مخصص](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#prompt) | `pdf2zh --prompt [prompt.txt]` |
| `--onnx` | استخدام نموذج DocLayout-YOLO ONNX مخصص | `pdf2zh --onnx [onnx/model/path]` |
| `--serverport` | منفذ مخصص لواجهة الويب | `pdf2zh --serverport 7860` |
| `--dir` | الترجمة الدفعية | `pdf2zh --dir /path/to/translate/` |
| `--config` | [ملف الإعدادات](https://github.com/Byaidu/PDFMathTranslate/blob/main/docs/ADVANCED.md#cofig) | `pdf2zh --config /path/to/config/config.json` |
| `--mode` | وضع الترجمة: `fast` (افتراضي، v1) أو `precise` (v2، تجريبي) | `pdf2zh --mode precise example.pdf` |
| `--babeldoc` | استخدام الواجهة الخلفية التجريبية [BabelDOC](https://funstory-ai.github.io/BabelDOC/) | `pdf2zh --babeldoc -s openai example.pdf` |
| `--mcp` | تفعيل وضع MCP STDIO | `pdf2zh --mcp` |
| `--sse` | تفعيل وضع MCP SSE | `pdf2zh --mcp --sse` |

للشرح التفصيلي، راجع وثيقة [الاستخدام المتقدم](./ADVANCED.md) للاطلاع على القائمة الكاملة لكل خيار.

<h3 id="downstream">٥.٢ التطوير اللاحق</h3>

للتطبيقات اللاحقة، راجع وثيقة [تفاصيل واجهات البرمجة](./APIS.md) لمزيد من المعلومات حول:

- [واجهة Python](./APIS.md#api-python)، كيفية استخدام البرنامج داخل برامج Python أخرى
- [واجهة HTTP](./APIS.md#api-http)، كيفية التواصل مع خادم مثبَّت عليه البرنامج

<h3 id="forks">٥.٣ الفروق بين النسختين الرئيسيتين</h3>

- [Byaidu/PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate): المشروع الأصلي والحالي للإصدارات المستقرة.

- [PDFMathTranslate/PDFMathTranslate-next](https://github.com/PDFMathTranslate/PDFMathTranslate-next): نسخة متفرعة تتضمن واجهة ويب وميزات إضافية. تعالج هذه النسخة عددًا كبيرًا من الحالات الحدّية، وتحسّن توافق ملفات PDF، وتُحسّن الاتساق الدلالي عبر الأعمدة والصفحات والتحجيم الديناميكي، إضافةً إلى تحسينات أخرى في جودة الترجمة. غير أن هذه النسخة مخصصة للتطوير فقط ولا تعالج مشكلات التوافق وليست مصمّمة لمساهمات المجتمع.

<h2 id="information">٦. معلومات المشروع</h2>
<h3 id="citation">٦.١ الاقتباس</h3>

قُبل هذا العمل في [*وقائع مؤتمر ٢٠٢٥ للأساليب التجريبية في معالجة اللغات الطبيعية: عروض الأنظمة*](https://aclanthology.org/2025.emnlp-demos.71/) (EMNLP 2025).

الاقتباس:

```
@inproceedings{ouyang-etal-2025-pdfmathtranslate,
	    title = "{PDFM}ath{T}ranslate: Scientific Document Translation Preserving Layouts",
	    author = "Ouyang, Rongxin  and
	      Chu, Chang  and
	      Xin, Zhikuang  and
	      Ma, Xiangyao",
	    editor = {Habernal, Ivan  and
	      Schulam, Peter  and
	      Tiedemann, J{\"o}rg},
	    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
	    month = nov,
	    year = "2025",
	    address = "Suzhou, China",
	    publisher = "Association for Computational Linguistics",
	    url = "https://aclanthology.org/2025.emnlp-demos.71/",
	    pages = "918--924",
	    ISBN = "979-8-89176-334-0",
	}
```

<h3 id="acknowledgement">٦.٢ شكر وتقدير</h3>

- [Immersive Translation](https://immersivetranslate.com) ترعى أكواد اشتراك Pro شهريًا للمساهمين النشطين في هذا المشروع، انظر التفاصيل في: [CONTRIBUTOR_REWARD.md](https://github.com/funstory-ai/BabelDOC/blob/main/docs/CONTRIBUTOR_REWARD.md)

- الواجهة الخلفية الجديدة: [BabelDOC](https://github.com/funstory-ai/BabelDOC)

- دمج المستندات: [PyMuPDF](https://github.com/pymupdf/PyMuPDF)

- تحليل المستندات: [Pdfminer.six](https://github.com/pdfminer/pdfminer.six)

- استخراج المستندات: [MinerU](https://github.com/opendatalab/MinerU)

- معاينة المستندات: [Gradio PDF](https://github.com/freddyaboulton/gradio-pdf)

- الترجمة متعددة الخيوط: [MathTranslate](https://github.com/SUSYUSTC/MathTranslate)

- تحليل التنسيق: [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)

- معيار المستندات: [PDF Explained](https://zxyle.github.io/PDF-Explained/)، [PDF Cheat Sheets](https://pdfa.org/resource/pdf-cheat-sheets/)

- الخط متعدد اللغات: [Go Noto Universal](https://github.com/satbyy/go-noto-universal)

- تشكيل النصوص: [HarfBuzz](https://harfbuzz.github.io/) عبر [uharfbuzz](https://github.com/harfbuzz/uharfbuzz)

- الخوارزمية ثنائية الاتجاه: [UAX #9](https://www.unicode.org/reports/tr9/)

<h3 id="contrib">٦.٣ المساهمون</h3>

<a href="https://github.com/Byaidu/PDFMathTranslate/graphs/contributors">
  <img src="https://opencollective.com/PDFMathTranslate/contributors.svg?width=890&button=false" />
</a>

![Alt](https://repobeats.axiom.co/api/embed/dfa7583da5332a11468d686fbd29b92320a6a869.svg "Repobeats analytics image")

للاطلاع على تفاصيل المساهمة، راجع [دليل المساهمة](https://github.com/Byaidu/PDFMathTranslate/wiki/Contribution-Guide---%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97).

<h3 id="star_hist">٦.٤ سجل النجوم</h3>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/star-history/star-history-dark.svg">
  <img alt="Star history" src="../assets/star-history/star-history-light.svg">
</picture>
