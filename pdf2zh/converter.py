import concurrent.futures
import logging
import math
import re
import unicodedata
from enum import Enum
from string import Template
from typing import Dict

import numpy as np
from pdfminer.converter import PDFConverter
from pdfminer.layout import LTChar, LTFigure, LTLine, LTPage
from pdfminer.pdffont import PDFCIDFont, PDFUnicodeNotDefined
from pdfminer.pdfinterp import PDFGraphicState, PDFResourceManager
from pdfminer.utils import apply_matrix_pt, mult_matrix
from pymupdf import Font
from tenacity import retry, wait_fixed

from pdf2zh import bidi_shape
from pdf2zh.translator import (
    AnythingLLMTranslator,
    ArgosTranslator,
    AzureOpenAITranslator,
    AzureTranslator,
    BaseTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    DeepseekTranslator,
    DifyTranslator,
    GeminiTranslator,
    GoogleTranslator,
    GrokTranslator,
    GroqTranslator,
    MiniMaxTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAIlikedTranslator,
    OpenAITranslator,
    QwenMtTranslator,
    SiliconTranslator,
    TencentTranslator,
    XinferenceTranslator,
    ZhipuTranslator,
    X302AITranslator,
)

log = logging.getLogger(__name__)

# 正文字体族：这些字体的斜体是强调，不是数学公式。用于把 vflag 的 ".*Ital"
# 规则限制回 LaTeX 数学字体，避免把 Word/InDesign 文档里的斜体正文整段跳过翻译。
TEXT_ITALIC_FONTS = (
    r"(Times|Arial|Helvetica|Georgia|Garamond|Minion|Calibri|Cambria|Baskerville"
    r"|Caslon|Palatino|Charter|Bookman|BookAntiqua|Century|Verdana|Tahoma|Segoe"
    r"|Lato|Merriweather|SourceSerif|SourceSans|Noto|Roboto|OpenSans|Liberation"
    r"|Nimbus|FreeSerif|FreeSans|DejaVu|Charis|Gentium|Alegreya|Lora|Spectral)"
)


class PDFConverterEx(PDFConverter):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
    ) -> None:
        PDFConverter.__init__(self, rsrcmgr, None, "utf-8", 1, None)

    def begin_page(self, page, ctm) -> None:
        # 重载替换 cropbox
        x0, y0, x1, y1 = page.cropbox
        x0, y0 = apply_matrix_pt(ctm, (x0, y0))
        x1, y1 = apply_matrix_pt(ctm, (x1, y1))
        mediabox = (0, 0, abs(x0 - x1), abs(y0 - y1))
        self.cur_item = LTPage(page.pageno, mediabox)

    def end_page(self, page):
        # 重载返回指令流
        return self.receive_layout(self.cur_item)

    def begin_figure(self, name, bbox, matrix) -> None:
        # 重载设置 pageid
        self._stack.append(self.cur_item)
        self.cur_item = LTFigure(name, bbox, mult_matrix(matrix, self.ctm))
        self.cur_item.pageid = self._stack[-1].pageid

    def end_figure(self, _: str) -> None:
        # 重载返回指令流
        fig = self.cur_item
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        self.cur_item = self._stack.pop()
        self.cur_item.add(fig)
        return self.receive_layout(fig)

    def render_char(
        self,
        matrix,
        font,
        fontsize: float,
        scaling: float,
        rise: float,
        cid: int,
        ncs,
        graphicstate: PDFGraphicState,
    ) -> float:
        # 重载设置 cid 和 font
        try:
            text = font.to_unichr(cid)
            assert isinstance(text, str), str(type(text))
        except PDFUnicodeNotDefined:
            text = self.handle_undefined_char(font, cid)
        textwidth = font.char_width(cid)
        textdisp = font.char_disp(cid)
        item = LTChar(
            matrix,
            font,
            fontsize,
            scaling,
            rise,
            text,
            textwidth,
            textdisp,
            ncs,
            graphicstate,
        )
        self.cur_item.add(item)
        item.cid = cid  # hack 插入原字符编码
        item.font = font  # hack 插入原字符字体
        return item.adv


class Paragraph:
    def __init__(self, y, x, x0, x1, y0, y1, size, brk, cls=-1):
        self.y: float = y  # 初始纵坐标
        self.x: float = x  # 初始横坐标
        self.x0: float = x0  # 左边界
        self.x1: float = x1  # 右边界
        self.y0: float = y0  # 上边界
        self.y1: float = y1  # 下边界
        self.size: float = size  # 字体大小
        self.brk: bool = brk  # 换行标记
        self.cls: int = cls  # 所属版面区域（用于在溢出时恢复真实栏宽）


# fmt: off
class TranslateConverter(PDFConverterEx):
    def __init__(
        self,
        rsrcmgr,
        vfont: str = None,
        vchar: str = None,
        thread: int = 0,
        layout={},
        lang_in: str = "",
        lang_out: str = "",
        service: str = "",
        noto_name: str = "",
        noto: Font = None,
        envs: Dict = None,
        prompt: Template = None,
        ignore_cache: bool = False,
        font_path: str = "",
        rtl: str = "auto",
        digit_form: str = "auto",
        min_font_scale: float = 0.6,
    ) -> None:
        super().__init__(rsrcmgr)
        self.vfont = vfont
        self.vchar = vchar
        self.thread = thread
        self.layout = layout
        self.noto_name = noto_name
        self.noto = noto
        # RTL 排版需要字体文件本身（HarfBuzz 整形），见 docs/RTL_ARABIC_SUPPORT.md
        self.font_path = font_path
        self.lang_in_raw = lang_in
        self.lang_out_raw = lang_out
        self.digit_form = digit_form
        self.min_font_scale = min_font_scale
        if rtl == "on":
            self.rtl_out = True
        elif rtl == "off":
            self.rtl_out = False
        else:
            self.rtl_out = bidi_shape.is_rtl_lang(lang_out)
        if self.rtl_out and not font_path:
            log.warning(
                "RTL target language %r but no font path was provided; "
                "falling back to the left-to-right renderer",
                lang_out,
            )
            self.rtl_out = False
        # 源文档是 RTL 时，字符间距/换行的判定方向需要反转（E02）
        self.rtl_in = bidi_shape.is_rtl_lang(lang_in)
        self.translator: BaseTranslator = None
        # e.g. "ollama:gemma2:9b" -> ["ollama", "gemma2:9b"]
        param = service.split(":", 1)
        service_name = param[0]
        service_model = param[1] if len(param) > 1 else None
        if not envs:
            envs = {}
        for translator in [GoogleTranslator, BingTranslator, DeepLTranslator, DeepLXTranslator, OllamaTranslator, XinferenceTranslator, AzureOpenAITranslator,
                           OpenAITranslator, ZhipuTranslator, ModelScopeTranslator, SiliconTranslator, GeminiTranslator, AzureTranslator, TencentTranslator, DifyTranslator, AnythingLLMTranslator, ArgosTranslator, GrokTranslator, GroqTranslator, DeepseekTranslator, MiniMaxTranslator, OpenAIlikedTranslator, QwenMtTranslator, X302AITranslator]:
            if service_name == translator.name:
                self.translator = translator(lang_in, lang_out, service_model, envs=envs, prompt=prompt, ignore_cache=ignore_cache)
        if not self.translator:
            raise ValueError("Unsupported translation service")

    def receive_layout(self, ltpage: LTPage):
        # 段落
        sstk: list[str] = []            # 段落文字栈
        pstk: list[Paragraph] = []      # 段落属性栈
        vbkt: int = 0                   # 段落公式括号计数
        # 公式组
        vstk: list[LTChar] = []         # 公式符号组
        vlstk: list[LTLine] = []        # 公式线条组
        vfix: float = 0                 # 公式纵向偏移
        # 公式组栈
        var: list[list[LTChar]] = []    # 公式符号组栈
        varl: list[list[LTLine]] = []   # 公式线条组栈
        varf: list[float] = []          # 公式纵向偏移栈
        vlen: list[float] = []          # 公式宽度栈
        # 全局
        lstk: list[LTLine] = []         # 全局线条栈
        xt: LTChar = None               # 上一个字符
        xt_cls: int = -1                # 上一个字符所属段落，保证无论第一个字符属于哪个类别都可以触发新段落
        vmax: float = ltpage.width / 4  # 行内公式最大宽度
        ops: str = ""                   # 渲染结果

        def vflag(font: str, char: str):    # 匹配公式（和角标）字体
            if isinstance(font, bytes):     # 不一定能 decode，直接转 str
                try:
                    font = font.decode('utf-8')  # 尝试使用 UTF-8 解码
                except UnicodeDecodeError:
                    font = ""
            font = font.split("+")[-1]      # 字体名截断
            if re.match(r"\(cid:", char):
                return True
            # 基于字体名规则的判定
            if self.vfont:
                if re.match(self.vfont, font):
                    return True
            else:
                # 常见正文字体的斜体只是强调，不是公式。原来的 ".*Ital" 会把
                # Word/InDesign 导出的 TimesNewRomanPS-ItalicMT 之类整段判为公式，
                # 于是所有斜体正文都不会被翻译（见 docs/RTL_FIELD_REPORT_002.md）。
                # LaTeX 的数学斜体是 CMMI/CMTI 等，仍由前面的 CM[^R] 规则命中。
                if re.match(TEXT_ITALIC_FONTS, font, re.IGNORECASE):
                    pass
                elif re.match(                                          # latex 字体
                    r"(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Ital|.*Sym|.*Math)",
                    font,
                ):
                    return True
            # 基于字符集规则的判定
            if self.vchar:
                if re.match(self.vchar, char):
                    return True
            else:
                # 文字组合符号不是公式：阿拉伯语元音符号（Mn）和延长符（Lm）
                # 会被下面的类别规则误判成公式，希伯来语的元音点同理（E01）
                if char and (
                    bidi_shape.is_text_mark(ord(char[0]))
                    or ord(char[0]) == bidi_shape.ARABIC_TATWEEL
                ):
                    return False
                if (
                    char
                    and char != " "                                     # 非空格
                    and (
                        unicodedata.category(char[0])
                        in ["Lm", "Mn", "Sk", "Sm", "Zl", "Zp", "Zs"]   # 文字修饰符、数学符号、分隔符号
                        or ord(char[0]) in range(0x370, 0x400)          # 希腊字母
                    )
                ):
                    return True
            return False

        ############################################################
        # A. 原文档解析
        for child in ltpage:
            if isinstance(child, LTChar):
                cur_v = False
                layout = self.layout[ltpage.pageid]
                # ltpage.height 可能是 fig 里面的高度，这里统一用 layout.shape
                h, w = layout.shape
                # 读取当前字符在 layout 中的类别
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                # 锚定文档中 bullet 的位置
                if child.get_text() == "•":
                    cls = 0
                # 判定当前字符是否属于公式
                if (                                                                                        # 判定当前字符是否属于公式
                    cls == 0                                                                                # 1. 类别为保留区域
                    or (cls == xt_cls and len(sstk[-1].strip()) > 1 and child.size < pstk[-1].size * 0.79)  # 2. 角标字体，有 0.76 的角标和 0.799 的大写，这里用 0.79 取中，同时考虑首字母放大的情况
                    or vflag(child.fontname, child.get_text())                                              # 3. 公式字体
                    or (child.matrix[0] == 0 and child.matrix[3] == 0)                                      # 4. 垂直字体
                ):
                    cur_v = True
                # 判定括号组是否属于公式
                if not cur_v:
                    if vstk and child.get_text() == "(":
                        cur_v = True
                        vbkt += 1
                    if vbkt and child.get_text() == ")":
                        cur_v = True
                        vbkt -= 1
                if (                                                        # 判定当前公式是否结束
                    not cur_v                                               # 1. 当前字符不属于公式
                    or cls != xt_cls                                        # 2. 当前字符与前一个字符不属于同一段落
                    # or (abs(child.x0 - xt.x0) > vmax and cls != 0)        # 3. 段落内换行，可能是一长串斜体的段落，也可能是段内分式换行，这里设个阈值进行区分
                    # 禁止纯公式（代码）段落换行，直到文字开始再重开文字段落，保证只存在两种情况
                    # A. 纯公式（代码）段落（锚定绝对位置）sstk[-1]=="" -> sstk[-1]=="{v*}"
                    # B. 文字开头段落（排版相对位置）sstk[-1]!=""
                    or (sstk[-1] != "" and abs(child.x0 - xt.x0) > vmax)    # 因为 cls==xt_cls==0 一定有 sstk[-1]==""，所以这里不需要再判定 cls!=0
                ):
                    if vstk:
                        if (                                                # 根据公式右侧的文字修正公式的纵向偏移
                            not cur_v                                       # 1. 当前字符不属于公式
                            and cls == xt_cls                               # 2. 当前字符与前一个字符属于同一段落
                            and child.x0 > max([vch.x0 for vch in vstk])    # 3. 当前字符在公式右侧
                        ):
                            vfix = vstk[0].y0 - child.y0
                        if sstk[-1] == "":
                            xt_cls = -1 # 禁止纯公式段落（sstk[-1]=="{v*}"）的后续连接，但是要考虑新字符和后续字符的连接，所以这里修改的是上个字符的类别
                        sstk[-1] += f"{{v{len(var)}}}"
                        var.append(vstk)
                        varl.append(vlstk)
                        varf.append(vfix)
                        vstk = []
                        vlstk = []
                        vfix = 0
                # 当前字符不属于公式或当前字符是公式的第一个字符
                if not vstk:
                    if cls == xt_cls:               # 当前字符与前一个字符属于同一段落
                        # RTL 原文的笔触向左推进，行内空格和换行的判定需要互换，
                        # 否则每个词间空格都会被误判成换行（E02）
                        if self.rtl_in:
                            inline_gap = xt.x0 - child.x1
                            wrapped = child.x0 > xt.x1 + 1
                        else:
                            inline_gap = child.x0 - xt.x1
                            wrapped = child.x1 < xt.x0
                        if wrapped:                 # 添加换行空格并标记原文段落存在换行
                            sstk[-1] += " "
                            pstk[-1].brk = True
                        elif inline_gap > 1:        # 添加行内空格
                            sstk[-1] += " "
                    else:                           # 根据当前字符构建一个新的段落
                        sstk.append("")
                        pstk.append(Paragraph(child.y0, child.x0, child.x0, child.x0, child.y0, child.y1, child.size, False, cls))
                if not cur_v:                                               # 文字入栈
                    if (                                                    # 根据当前字符修正段落属性
                        child.size > pstk[-1].size                          # 1. 当前字符比段落字体大
                        or len(sstk[-1].strip()) == 1                       # 2. 当前字符为段落第二个文字（考虑首字母放大的情况）
                    ) and child.get_text() != " ":                          # 3. 当前字符不是空格
                        pstk[-1].y -= child.size - pstk[-1].size            # 修正段落初始纵坐标，假设两个不同大小字符的上边界对齐
                        pstk[-1].size = child.size
                    sstk[-1] += child.get_text()
                else:                                                       # 公式入栈
                    if (                                                    # 根据公式左侧的文字修正公式的纵向偏移
                        not vstk                                            # 1. 当前字符是公式的第一个字符
                        and cls == xt_cls                                   # 2. 当前字符与前一个字符属于同一段落
                        and child.x0 > xt.x0                                # 3. 前一个字符在公式左侧
                    ):
                        vfix = child.y0 - xt.y0
                    vstk.append(child)
                # 更新段落边界，因为段落内换行之后可能是公式开头，所以要在外边处理
                pstk[-1].x0 = min(pstk[-1].x0, child.x0)
                pstk[-1].x1 = max(pstk[-1].x1, child.x1)
                pstk[-1].y0 = min(pstk[-1].y0, child.y0)
                pstk[-1].y1 = max(pstk[-1].y1, child.y1)
                # 更新上一个字符
                xt = child
                xt_cls = cls
            elif isinstance(child, LTFigure):   # 图表
                pass
            elif isinstance(child, LTLine):     # 线条
                layout = self.layout[ltpage.pageid]
                # ltpage.height 可能是 fig 里面的高度，这里统一用 layout.shape
                h, w = layout.shape
                # 读取当前线条在 layout 中的类别
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                if vstk and cls == xt_cls:      # 公式线条
                    vlstk.append(child)
                else:                           # 全局线条
                    lstk.append(child)
            else:
                pass
        # 处理结尾
        if vstk:    # 公式出栈
            sstk[-1] += f"{{v{len(var)}}}"
            var.append(vstk)
            varl.append(vlstk)
            varf.append(vfix)
        log.debug("\n==========[VSTACK]==========\n")
        for id, v in enumerate(var):  # 计算公式宽度
            l = max([vch.x1 for vch in v]) - v[0].x0
            log.debug(f'< {l:.1f} {v[0].x0:.1f} {v[0].y0:.1f} {v[0].cid} {v[0].fontname} {len(varl[id])} > v{id} = {"".join([ch.get_text() for ch in v])}')
            vlen.append(l)

        ############################################################
        # B. 段落翻译
        log.debug("\n==========[SSTACK]==========\n")

        @retry(wait=wait_fixed(1))
        def worker(s: str):  # 多线程翻译
            if not s.strip() or re.match(r"^\{v\d+\}$", s):  # 空白和公式不翻译
                return s
            try:
                new = self.translator.translate(s)
                return new
            except BaseException as e:
                if log.isEnabledFor(logging.DEBUG):
                    log.exception(e)
                else:
                    log.exception(e, exc_info=False)
                raise e
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.thread
        ) as executor:
            news = list(executor.map(worker, sstk))

        ############################################################
        # C. 新文档排版
        def raw_string(fcur: str, cstk: str):  # 编码字符串
            if fcur == self.noto_name:
                return "".join(["%04x" % self.noto.has_glyph(ord(c)) for c in cstk])
            elif isinstance(self.fontmap[fcur], PDFCIDFont):  # 判断编码长度
                return "".join(["%04x" % ord(c) for c in cstk])
            else:
                return "".join(["%02x" % ord(c) for c in cstk])

        def raw_string_glyphs(glyphs):  # 编码字形（RTL 走整形器，直接写 GID）
            return "".join("%04x" % g.gid for g in glyphs)

        # 根据目标语言获取默认行距
        # RTL 语言的行距见 docs/RTL_ARABIC_SUPPORT.md：阿拉伯语带元音符号时
        # 实测行高需 >= 1.28，原来的 1.0 会导致行间重叠
        LANG_LINEHEIGHT_MAP = {
            "zh-cn": 1.4, "zh-tw": 1.4, "zh-hans": 1.4, "zh-hant": 1.4, "zh": 1.4,
            "ja": 1.1, "ko": 1.2, "en": 1.2, "ru": 0.8, "uk": 0.8, "ta": 0.8,
            "ar": 1.45, "fa": 1.45, "ur": 1.55, "he": 1.35, "iw": 1.35,
        }
        default_line_height = LANG_LINEHEIGHT_MAP.get(self.translator.lang_out.lower(), 1.1) # 小语种默认1.1
        _x, _y = 0, 0
        ops_list = []

        def gen_op_txt(font, size, x, y, rtxt):
            return f"/{font} {size:f} Tf 1 0 0 1 {x:f} {y:f} Tm [<{rtxt}>] TJ "

        def gen_op_line(x, y, xlen, ylen, linewidth):
            return f"ET q 1 0 0 1 {x:f} {y:f} cm [] 0 d 0 J {linewidth:f} w 0 0 m {xlen:f} {ylen:f} l S Q BT "

        def gen_ops_run(font, size, x, y, glyphs):
            # 输出一个整形后的文字段。只有带偏移的字形（如阿拉伯语元音符号，
            # 其 x_advance 为 0）才需要单独定位，其余字形靠自然步进排列。
            ops = []
            buf = []
            pen = x
            run_x = x
            for g in glyphs:
                if g.force_positioning:
                    if buf:
                        ops.append(gen_op_txt(font, size, run_x, y, raw_string_glyphs(buf)))
                        buf = []
                    ops.append(gen_op_txt(
                        font, size, pen + g.x_offset, y + g.y_offset,
                        raw_string_glyphs([g]),
                    ))
                    pen += g.x_advance
                    run_x = pen
                    continue
                if not buf:
                    run_x = pen
                buf.append(g)
                pen += g.x_advance
            if buf:
                ops.append(gen_op_txt(font, size, run_x, y, raw_string_glyphs(buf)))
            return ops

        # 版面区域的真实栏宽。单行段落（标题、题注、表格单元）的 x0/x1 只是
        # 原文墨迹的范围，不是栏宽；译文更长时无处可去，只能不断缩字号。
        # 这里从版面模型的分区图里取回该区域的实际左右边界，溢出时先扩框再缩字。
        region_bounds: dict[int, tuple[float, float]] = {}
        try:
            _lay = self.layout[ltpage.pageid]
            for _c in {p.cls for p in pstk if p.cls > 0}:
                _xs = np.where(_lay == _c)[1]
                if _xs.size:
                    region_bounds[_c] = (float(_xs.min()), float(_xs.max()))
        except Exception:
            region_bounds = {}

        # 每个段落实际可用的纵向空间：原文框高度，加上它与下方最近段落之间的空白。
        # 短句（一行高的框）译成阿拉伯语常常要两三行，只按原框高度算就必然压到
        # 下一段上；这里把真正空着的地方也算进来。
        avail_h: dict[int, float] = {}
        for _i, _p in enumerate(pstk):
            _own = _p.y1 - _p.y0
            _below_top = None
            for _j, _q in enumerate(pstk):
                if _i == _j or _q.y1 > _p.y0 + 0.5:
                    continue                        # 不在下方
                if _q.x1 < _p.x0 or _q.x0 > _p.x1:
                    continue                        # 不同栏，横向不重叠
                _below_top = _q.y1 if _below_top is None else max(_below_top, _q.y1)
            # 下方没有已知段落时保守处理：可能有图片或线条，不擅自扩张
            avail_h[_i] = max(_own, _p.y1 - _below_top) if _below_top is not None else _own

        def plan_rtl(pid: int, text: str):
            # RTL 段落排版：双向算法 + HarfBuzz 整形 + 方向感知的排版
            # 详见 docs/RTL_ARABIC_SUPPORT.md 第 5-8 节
            para = pstk[pid]
            px0, px1 = para.x0, para.x1
            py = para.y
            pheight = avail_h.get(pid, para.y1 - para.y0)
            base_size = para.size
            indent = max(para.x - para.x0, 0.0)
            lang = self.lang_out_raw.lower()
            # 带元音符号的阿拉伯语实测需要 1.28 行高，否则行间重叠
            lh_floor = 1.30 if bidi_shape.has_tashkeel(text) else 1.15
            # 翻译引擎可能改写公式标记本身（Google 会把 {v4} 译成 {الآية ٤}），
            # 这里给出本段实际存在的公式编号，用于容错还原
            src_vids = {
                int(m.replace(" ", ""))
                for m in re.findall(r"\{\s*v([\d\s]+)\}", sstk[pid], re.IGNORECASE)
            }
            valid_vids = {v for v in src_vids if v < len(var)}
            found_vids = {
                v
                for m in re.finditer(r"\{[^{}]{0,32}\}", text)
                if (v := bidi_shape._placeholder_vid(m.group(0), valid_vids)) is not None
            }
            if valid_vids - found_vids:
                log.warning(
                    "translator dropped formula marker(s) %s from an RTL paragraph; "
                    "that content will be missing from the output",
                    sorted(valid_vids - found_vids),
                )
            line_height = default_line_height
            scale = 1.0
            expand_step = 0
            # 原文只有一行的段落（标题、题注、表格单元）默认应保持一行
            want_one_line = not para.brk
            res = None
            # 自然宽度：不换行时整段的总宽（按 base_size 计）。用它可以直接
            # 算出需要的字号，而不是每次缩 5% 盲目试探。
            _wide = bidi_shape.layout_paragraph(
                text, font_path=self.font_path, font_key=self.noto_name,
                size=base_size, x0=0.0, x1=1e7, indent=0.0, lang_out=lang,
                base_rtl=True,
                placeholder_width=lambda v: vlen[v] if v < len(vlen) else 0.0,
                valid_vids=valid_vids, digit_form=self.digit_form,
            )
            natural_w = sum(q.item.width for ln in _wide.lines for q in ln) or 1.0
            while True:
                res = bidi_shape.layout_paragraph(
                    text,
                    font_path=self.font_path,
                    font_key=self.noto_name,
                    size=base_size * scale,
                    x0=px0,
                    x1=px1,
                    indent=indent,
                    lang_out=lang,
                    base_rtl=True,
                    placeholder_width=lambda v: vlen[v] if v < len(vlen) else 0.0,
                    valid_vids=valid_vids,
                    digit_form=self.digit_form,
                )
                nlines = len(res.lines)
                fits_v = nlines <= 1 or nlines * base_size * scale * line_height <= pheight
                fits_h = res.overflow_x <= 0.1 * base_size
                fits_lines = nlines <= 1 or not want_one_line
                if fits_v and fits_h and fits_lines:
                    break
                # 先压行距，再扩框，最后才缩字号：
                # 缩字号对阿拉伯语可读性影响最大（元音符号先糊成一团）
                if not fits_v and line_height - 0.05 >= lh_floor:
                    line_height -= 0.05
                    continue
                bounds = region_bounds.get(para.cls)
                if bounds and expand_step < 2:
                    # 第一步向左扩（RTL 文字靠右对齐，向左生长最不易错位），
                    # 仍不够再向右扩到区域边界
                    if expand_step == 0 and bounds[0] < px0 - 0.5:
                        px0 = bounds[0]
                        expand_step = 1
                        continue
                    if bounds[1] > px1 + 0.5:
                        px1 = bounds[1]
                        expand_step = 2
                        continue
                    expand_step = 2
                    continue
                box_w = max(px1 - px0, 1.0)
                # 原文只有一行的段落（标题、题注、表格单元）应当尽量保持一行。
                # 直接按自然宽度算出所需字号：标题常常要缩 15%~20%，每次 5%
                # 试探会在行数一时减不动时提前放弃，把标题拆成一词一行（G10）。
                if want_one_line and nlines > 1:
                    target = (box_w / natural_w) * 0.99
                    if 0.75 <= target < scale - 1e-3:
                        scale = target
                        continue
                    want_one_line = False   # 缩得太狠反而更差，接受换行
                    continue

                # 纵向放不下时同样直接估算字号。行数约与字号成正比，
                # 占用高度约与字号平方成正比，于是所需比例约为：
                #     sqrt(可用高度 * 栏宽 / (自然宽度 * 字号 * 行距))
                if not fits_v:
                    est = math.sqrt(
                        max(pheight * box_w / (natural_w * base_size * line_height), 0.0)
                    )
                    target = min(est * 0.98, scale - 0.02)
                    if target >= self.min_font_scale:
                        scale = target
                        continue
                    log.debug(
                        "RTL paragraph cannot fit even at the minimum scale; "
                        "keeping %.2f and accepting vertical overflow", scale,
                    )
                    break

                # 只剩横向放不下（单词比栏还宽）时才继续按步缩小
                if scale > 0.6:
                    scale -= 0.05
                elif scale > self.min_font_scale:
                    scale -= 0.10
                else:
                    log.warning(
                        "RTL paragraph does not fit after scaling to %.2f "
                        "(cls=%s box=[%.1f,%.1f] w=%.1f region=%s overflow_x=%.1f "
                        "lines=%d): %r",
                        scale, para.cls, px0, px1, px1 - px0,
                        region_bounds.get(para.cls), res.overflow_x, nlines,
                        text[:60],
                    )
                    break
            size_ = base_size * scale

            # 首行上伸修正：阿拉伯语元音符号比拉丁文高，可能顶到上一段
            overshoot = (py + res.max_ascent) - para.y1
            if overshoot > 0:
                slack = pheight - len(res.lines) * size_ * line_height
                py -= min(overshoot, max(slack, 0.0))

            return {
                "res": res, "size": size_, "lh": line_height, "py": py,
                "x0": px0, "x1": px1, "text": text,
            }

        def emit_rtl(plan):
            res, size_, line_height, py = (
                plan["res"], plan["size"], plan["lh"], plan["py"]
            )
            out = []
            seen_text = False
            for li, line in enumerate(res.lines):
                ly = py - li * size_ * line_height
                for placed in line:
                    item = placed.item
                    if isinstance(item, bidi_shape.PlaceholderRun):
                        vid = item.vid
                        if vid >= len(var):
                            continue
                        # 公式内部始终从左到右绘制，起点是占位框的左边界
                        fix = varf[vid] if seen_text else 0
                        for vch in var[vid]:
                            out.append(gen_op_txt(
                                self.fontid[vch.font], vch.size,
                                placed.x + vch.x0 - var[vid][0].x0,
                                fix + ly + vch.y0 - var[vid][0].y0,
                                raw_string(self.fontid[vch.font], chr(vch.cid)),
                            ))
                        for line_obj in varl[vid]:
                            if line_obj.linewidth < 5:  # hack 有的文档会用粗线条当图片背景
                                out.append(gen_op_line(
                                    line_obj.pts[0][0] + placed.x - var[vid][0].x0,
                                    line_obj.pts[0][1] + fix + ly - var[vid][0].y0,
                                    line_obj.pts[1][0] - line_obj.pts[0][0],
                                    line_obj.pts[1][1] - line_obj.pts[0][1],
                                    line_obj.linewidth,
                                ))
                    else:
                        seen_text = True
                        out.extend(gen_ops_run(
                            item.font_key, size_, placed.x, ly, item.visual_glyphs()
                        ))
            return out

        def mirror_bullet(pid: int, vid: int) -> None:
            # 项目符号在 RTL 里应该在文字的右侧。它被当作保留区域锚定在原始
            # 绝对位置（见 A 部分对 "•" 的处理），所以这里按同一行的正文段落
            # 把它镜像到另一侧，否则整份列表的符号都留在左边（缺陷 B07）。
            para = pstk[pid]
            host = None
            for q in pstk:
                if q is para or q.cls <= 0:
                    continue
                if q.y0 - 1 <= para.y0 and para.y1 <= q.y1 + 1 and q.x0 > para.x1:
                    if host is None or q.x0 < host.x0:
                        host = q
            if host is None:
                return
            gap = host.x0 - para.x1           # 原文中符号与正文之间的间距
            width = para.x1 - para.x0
            new_x0 = host.x1 + gap
            para.x = para.x0 = new_x0
            para.x1 = new_x0 + width

        BULLETS = set("•‣▪▫◦●○·∙*–—-")
        rtl_plans: list[dict] = []

        for id, new in enumerate(news):
            marker = re.match(r"^\{v(\d+)\}$", new.strip())
            if self.rtl_out and marker:
                vid = int(marker.group(1))
                if vid < len(var) and var[vid] and all(
                    (ch.get_text() or "").strip() in BULLETS or not (ch.get_text() or "").strip()
                    for ch in var[vid]
                ):
                    mirror_bullet(id, vid)
            elif self.rtl_out and new.strip():
                rtl_plans.append(plan_rtl(id, new))
                continue
            x: float = pstk[id].x                       # 段落初始横坐标
            y: float = pstk[id].y                       # 段落初始纵坐标
            x0: float = pstk[id].x0                     # 段落左边界
            x1: float = pstk[id].x1                     # 段落右边界
            height: float = pstk[id].y1 - pstk[id].y0   # 段落高度
            size: float = pstk[id].size                 # 段落字体大小
            brk: bool = pstk[id].brk                    # 段落换行标记
            cstk: str = ""                              # 当前文字栈
            fcur: str = None                            # 当前字体 ID
            lidx = 0                                    # 记录换行次数
            tx = x
            fcur_ = fcur
            ptr = 0
            log.debug(f"< {y} {x} {x0} {x1} {size} {brk} > {sstk[id]} | {new}")

            ops_vals: list[dict] = []

            while ptr < len(new):
                vy_regex = re.match(
                    r"\{\s*v([\d\s]+)\}", new[ptr:], re.IGNORECASE
                )  # 匹配 {vn} 公式标记
                mod = 0  # 文字修饰符
                if vy_regex:  # 加载公式
                    ptr += len(vy_regex.group(0))
                    try:
                        vid = int(vy_regex.group(1).replace(" ", ""))
                        adv = vlen[vid]
                    except Exception:
                        continue  # 翻译器可能会自动补个越界的公式标记
                    if var[vid][-1].get_text() and unicodedata.category(var[vid][-1].get_text()[0]) in ["Lm", "Mn", "Sk"]:  # 文字修饰符
                        mod = var[vid][-1].width
                else:  # 加载文字
                    ch = new[ptr]
                    fcur_ = None
                    try:
                        if fcur_ is None and self.fontmap["tiro"].to_unichr(ord(ch)) == ch:
                            fcur_ = "tiro"  # 默认拉丁字体
                    except Exception:
                        pass
                    if fcur_ is None:
                        fcur_ = self.noto_name  # 默认非拉丁字体
                    if fcur_ == self.noto_name: # FIXME: change to CONST
                        adv = self.noto.char_lengths(ch, size)[0]
                    else:
                        adv = self.fontmap[fcur_].char_width(ord(ch)) * size
                    ptr += 1
                if (                                # 输出文字缓冲区
                    fcur_ != fcur                   # 1. 字体更新
                    or vy_regex                     # 2. 插入公式
                    or x + adv > x1 + 0.1 * size    # 3. 到达右边界（可能一整行都被符号化，这里需要考虑浮点误差）
                ):
                    if cstk:
                        ops_vals.append({
                            "type": OpType.TEXT,
                            "font": fcur,
                            "size": size,
                            "x": tx,
                            "dy": 0,
                            "rtxt": raw_string(fcur, cstk),
                            "lidx": lidx
                        })
                        cstk = ""
                if brk and x + adv > x1 + 0.1 * size:  # 到达右边界且原文段落存在换行
                    x = x0
                    lidx += 1
                if vy_regex:  # 插入公式
                    fix = 0
                    if fcur is not None:  # 段落内公式修正纵向偏移
                        fix = varf[vid]
                    for vch in var[vid]:  # 排版公式字符
                        vc = chr(vch.cid)
                        ops_vals.append({
                            "type": OpType.TEXT,
                            "font": self.fontid[vch.font],
                            "size": vch.size,
                            "x": x + vch.x0 - var[vid][0].x0,
                            "dy": fix + vch.y0 - var[vid][0].y0,
                            "rtxt": raw_string(self.fontid[vch.font], vc),
                            "lidx": lidx
                        })
                        if log.isEnabledFor(logging.DEBUG):
                            lstk.append(LTLine(0.1, (_x, _y), (x + vch.x0 - var[vid][0].x0, fix + y + vch.y0 - var[vid][0].y0)))
                            _x, _y = x + vch.x0 - var[vid][0].x0, fix + y + vch.y0 - var[vid][0].y0
                    for l in varl[vid]:  # 排版公式线条
                        if l.linewidth < 5:  # hack 有的文档会用粗线条当图片背景
                            ops_vals.append({
                                "type": OpType.LINE,
                                "x": l.pts[0][0] + x - var[vid][0].x0,
                                "dy": l.pts[0][1] + fix - var[vid][0].y0,
                                "linewidth": l.linewidth,
                                "xlen": l.pts[1][0] - l.pts[0][0],
                                "ylen": l.pts[1][1] - l.pts[0][1],
                                "lidx": lidx
                            })
                else:  # 插入文字缓冲区
                    if not cstk:  # 单行开头
                        tx = x
                        if x == x0 and ch == " ":  # 消除段落换行空格
                            adv = 0
                        else:
                            cstk += ch
                    else:
                        cstk += ch
                adv -= mod # 文字修饰符
                fcur = fcur_
                x += adv
                if log.isEnabledFor(logging.DEBUG):
                    lstk.append(LTLine(0.1, (_x, _y), (x, y)))
                    _x, _y = x, y
            # 处理结尾
            if cstk:
                ops_vals.append({
                    "type": OpType.TEXT,
                    "font": fcur,
                    "size": size,
                    "x": tx,
                    "dy": 0,
                    "rtxt": raw_string(fcur, cstk),
                    "lidx": lidx
                })

            line_height = default_line_height

            while (lidx + 1) * size * line_height > height and line_height >= 1:
                line_height -= 0.05

            for vals in ops_vals:
                if vals["type"] == OpType.TEXT:
                    ops_list.append(gen_op_txt(vals["font"], vals["size"], vals["x"], vals["dy"] + y - vals["lidx"] * size * line_height, vals["rtxt"]))
                elif vals["type"] == OpType.LINE:
                    ops_list.append(gen_op_line(vals["x"], vals["dy"] + y - vals["lidx"] * size * line_height, vals["xlen"], vals["ylen"], vals["linewidth"]))

        # 译文比原文行数多时，段落会压到下一段上（缺陷 G07）。这里在真正绘制
        # 之前做一次自上而下的避让：只有确实会重叠时才把段落往下推，推到页面
        # 底部为止。不重叠的页面完全不受影响，版面保持原样。
        if rtl_plans:
            def span(plan):
                nl = max(len(plan["res"].lines), 1)
                top = plan["py"] + plan["res"].max_ascent
                bottom = plan["py"] - (nl - 1) * plan["size"] * plan["lh"] \
                    - 0.25 * plan["size"]
                return top, bottom

            placed: list[tuple[float, float, float, float]] = []  # top,bottom,x0,x1
            for plan in sorted(rtl_plans, key=lambda p: -span(p)[0]):
                top, bottom = span(plan)
                shift = 0.0
                for _ptop, pbottom, qx0, qx1 in placed:
                    if plan["x1"] < qx0 or plan["x0"] > qx1:
                        continue                      # 不同栏
                    if top - shift > pbottom:
                        shift = max(shift, top - pbottom)
                if shift > 0:
                    # 不要把最后一行推出页面
                    shift = min(shift, max(bottom - 0.25 * plan["size"], 0.0))
                    plan["py"] -= shift
                    top, bottom = span(plan)
                    log.debug("RTL paragraph shifted down %.1fpt to avoid overlap", shift)
                placed.append((top, bottom, plan["x0"], plan["x1"]))
            for plan in rtl_plans:
                ops_list.extend(emit_rtl(plan))

        for l in lstk:  # 排版全局线条
            if l.linewidth < 5:  # hack 有的文档会用粗线条当图片背景
                ops_list.append(gen_op_line(l.pts[0][0], l.pts[0][1], l.pts[1][0] - l.pts[0][0], l.pts[1][1] - l.pts[0][1], l.linewidth))

        ops = f"BT {''.join(ops_list)}ET "
        return ops


class OpType(Enum):
    TEXT = "text"
    LINE = "line"
