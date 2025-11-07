import concurrent.futures
import logging
import re
import statistics
import unicodedata
from enum import Enum
from string import Template
from typing import Dict, List, Optional

import numpy as np
from pdfminer.converter import PDFConverter
from pdfminer.layout import LTChar, LTFigure, LTLine, LTPage
from pdfminer.pdffont import PDFCIDFont, PDFUnicodeNotDefined
from pdfminer.pdfinterp import PDFGraphicState, PDFResourceManager
from pdfminer.utils import apply_matrix_pt, mult_matrix
from pymupdf import Font
from tenacity import retry, wait_fixed

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
    ModelScopeTranslator,
    NoopTranslator,
    OllamaTranslator,
    OpenAIlikedTranslator,
    OpenAITranslator,
    QwenMtTranslator,
    RivaTranslator,
    SiliconTranslator,
    TencentTranslator,
    XinferenceTranslator,
    ZhipuTranslator,
    X302AITranslator,
)

log = logging.getLogger(__name__)


class PDFConverterEx(PDFConverter):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
    ) -> None:
        PDFConverter.__init__(self, rsrcmgr, None, "utf-8", 1, None)

    def begin_page(self, page, ctm) -> None:
        # cropbox を再計算して差し替える
        (x0, y0, x1, y1) = page.cropbox
        (x0, y0) = apply_matrix_pt(ctm, (x0, y0))
        (x1, y1) = apply_matrix_pt(ctm, (x1, y1))
        mediabox = (0, 0, abs(x0 - x1), abs(y0 - y1))
        self.cur_item = LTPage(page.pageno, mediabox)

    def end_page(self, page):
        # レイアウト結果を呼び出し元へ返す
        return self.receive_layout(self.cur_item)

    def begin_figure(self, name, bbox, matrix) -> None:
        # figure にも pageid を引き継ぐ
        self._stack.append(self.cur_item)
        self.cur_item = LTFigure(name, bbox, mult_matrix(matrix, self.ctm))
        self.cur_item.pageid = self._stack[-1].pageid

    def end_figure(self, _: str) -> None:
        # figure のレイアウトを親に戻す
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
        # cid と font 情報を保持できるよう拡張
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
        item.cid = cid  # 元の文字コードを保持するためのハック
        item.font = font  # 元のフォントを保持するためのハック
        return item.adv


class Paragraph:
    def __init__(
        self,
        y,
        x,
        x0,
        x1,
        y0,
        y1,
        size,
        brk,
        *,
        vertical: bool = False,
        vertical_direction: int = 0,
        vertical_positions: Optional[List[tuple[float, float]]] = None,
        vertical_spacing: float = 0.0,
    ):
        self.y: float = y  # 初期の縦座標
        self.x: float = x  # 初期の横座標
        self.x0: float = x0  # 左境界
        self.x1: float = x1  # 右境界
        self.y0: float = y0  # 上境界
        self.y1: float = y1  # 下境界
        self.size: float = size  # フォントサイズ
        self.brk: bool = brk  # 改行フラグ
        self.vertical: bool = vertical
        self.vertical_direction: int = vertical_direction
        self.vertical_positions: List[tuple[float, float]] = (
            vertical_positions or []
        )
        self.vertical_spacing: float = vertical_spacing


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
    ) -> None:
        super().__init__(rsrcmgr)
        self.vfont = vfont
        self.vchar = vchar
        self.thread = thread
        self.layout = layout
        self.noto_name = noto_name
        self.noto = noto
        self.translator: BaseTranslator = None
        # e.g. "ollama:gemma2:9b" -> ["ollama", "gemma2:9b"]
        param = service.split(":", 1)
        service_name = param[0]
        service_model = param[1] if len(param) > 1 else None
        if not envs:
            envs = {}
        for translator in [
            GoogleTranslator,
            BingTranslator,
            DeepLTranslator,
            DeepLXTranslator,
            OllamaTranslator,
            XinferenceTranslator,
            RivaTranslator,
            AzureOpenAITranslator,
            OpenAITranslator,
            ZhipuTranslator,
            ModelScopeTranslator,
            SiliconTranslator,
            GeminiTranslator,
            AzureTranslator,
            TencentTranslator,
            DifyTranslator,
            AnythingLLMTranslator,
            ArgosTranslator,
            GrokTranslator,
            GroqTranslator,
            DeepseekTranslator,
            OpenAIlikedTranslator,
            QwenMtTranslator,
            NoopTranslator,
            X302AITranslator,
        ]:
            if service_name == translator.name:
                self.translator = translator(lang_in, lang_out, service_model, envs=envs, prompt=prompt, ignore_cache=ignore_cache)
        if not self.translator:
            raise ValueError("Unsupported translation service")

    def receive_layout(self, ltpage: LTPage):
        # 段落関連のスタック
        sstk: list[str] = []            # 段落テキストのスタック
        pstk: list[Paragraph] = []      # 段落属性のスタック
        vbkt: int = 0                   # フォーミュラ括弧のネスト深度
        # 数式グループ
        vstk: list[LTChar] = []         # 数式用の文字スタック
        vlstk: list[LTLine] = []        # 数式用の線スタック
        vfix: float = 0                 # 数式の縦方向補正
        # 数式グループの記録
        var: list[list[LTChar]] = []    # 数式文字グループ
        varl: list[list[LTLine]] = []   # 数式線分グループ
        varf: list[float] = []          # 数式ごとの縦方向補正
        vlen: list[float] = []          # 数式の幅
        # ページ全体で使う一時領域
        lstk: list[LTLine] = []         # 全体線分のスタック
        xt: LTChar = None               # 直前の文字
        xt_cls: int = -1                # 直前の文字が属する段落（初回もトリガーできるよう -1）
        vmax: float = ltpage.width / 4  # 行内数式の最大許容幅
        ops: str = ""                   # 描画命令の結果
        vertical_chars: list[LTChar] = []  # 縦書き文字の一時バッファ
        VERTICAL_X_THRESHOLD = 2.0

        def flush_vertical_chars():
            nonlocal vertical_chars
            if not vertical_chars:
                return
            layout_chars = sorted(vertical_chars, key=lambda ch: (-ch.y0, ch.x0))
            matrix_dir = 0.0
            if vertical_chars:
                matrix_dir = vertical_chars[0].matrix[1]
                if abs(matrix_dir) < 1e-6:
                    matrix_dir = vertical_chars[0].matrix[2]
            text_chars = layout_chars
            direction_sign = -1  # default: move downward (y decreases)
            if matrix_dir > 0:
                text_chars = list(reversed(layout_chars))
                direction_sign = 1  # move upward for consecutive chars
            text = "".join(ch.get_text() for ch in text_chars).strip()
            if text:
                x0 = min(ch.x0 for ch in vertical_chars)
                x1 = max(ch.x1 for ch in vertical_chars)
                y0 = min(ch.y0 for ch in vertical_chars)
                y1 = max(ch.y1 for ch in vertical_chars)
                size = max(ch.size for ch in vertical_chars)
                positions = [(ch.x0, ch.y0) for ch in text_chars]
                if len(text_chars) > 1:
                    spacing_candidates = [
                        abs(text_chars[idx + 1].y0 - text_chars[idx].y0)
                        for idx in range(len(text_chars) - 1)
                    ]
                    spacing = statistics.median(spacing_candidates)
                else:
                    spacing = text_chars[0].height if text_chars else size
                sstk.append(text)
                pstk.append(
                    Paragraph(
                        text_chars[0].y0,
                        text_chars[0].x0,
                        x0,
                        x1,
                        y0,
                        y1,
                        size,
                        False,
                        vertical=True,
                        vertical_direction=direction_sign,
                        vertical_positions=positions,
                        vertical_spacing=spacing,
                    )
                )
            vertical_chars = []

        def is_vertical_glyph(char: LTChar) -> bool:
            return abs(char.matrix[0]) < 1e-6 and abs(char.matrix[3]) < 1e-6

        def vflag(font: str, char: str):
            # 数式（および上付き記号）とみなすフォントを判定する
            if isinstance(font, bytes):     # decode できない場合があるので str へ変換
                try:
                    font = font.decode('utf-8')  # UTF-8 で読める場合はそのまま使う
                except UnicodeDecodeError:
                    font = ""
            font = font.split("+")[-1]      # フォント名の末尾だけを使う
            if re.match(r"\(cid:", char):
                return True
            # フォント名による判定
            if self.vfont:
                if re.match(self.vfont, font):
                    return True
            else:
                if re.match(                                            # LaTeX 系フォント
                    r"(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Ital|.*Sym|.*Math)",
                    font,
                ):
                    return True
            # 文字種による判定
            if self.vchar:
                if re.match(self.vchar, char):
                    return True
            else:
                if (
                    char
                    and char != " "                                     # 空白以外
                    and (
                        unicodedata.category(char[0])
                        in ["Lm", "Mn", "Sk", "Sm", "Zl", "Zp", "Zs"]   # 修飾記号・数式記号・区切り記号
                        or ord(char[0]) in range(0x370, 0x400)          # ギリシャ文字
                    )
                ):
                    return True
            return False

        ############################################################
        # A. 元 PDF の解析
        for child in ltpage:
            if not isinstance(child, LTChar):
                flush_vertical_chars()
            if isinstance(child, LTChar):
                if is_vertical_glyph(child):
                    if vertical_chars:
                        last_vertical = vertical_chars[-1]
                        if abs(child.x0 - last_vertical.x0) > VERTICAL_X_THRESHOLD:
                            flush_vertical_chars()
                    vertical_chars.append(child)
                    continue
                else:
                    flush_vertical_chars()
                cur_v = False
                layout = self.layout[ltpage.pageid]
                # ltpage.height が figure 内の高さになることがあるので layout.shape を基準にする
                h, w = layout.shape
                # 現在文字のレイアウトクラスを取得
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                # リストの bullet は常に本文扱いにする
                if child.get_text() == "•":
                    cls = 0
                # この文字が数式扱いかどうかを判定
                if (
                    cls == 0                                                                                # 1. レイアウトクラスが予約領域
                    or (cls == xt_cls and len(sstk[-1].strip()) > 1 and child.size < pstk[-1].size * 0.79)  # 2. 現在段落より小さい文字サイズ（上付き/下付き）
                    or vflag(child.fontname, child.get_text())                                              # 3. 数式フォント
                ):
                    cur_v = True
                # 括弧の開始・終了で数式かどうかを判定
                if not cur_v:
                    if vstk and child.get_text() == "(":
                        cur_v = True
                        vbkt += 1
                    if vbkt and child.get_text() == ")":
                        cur_v = True
                        vbkt -= 1
                if (
                    not cur_v                                               # 1. 今回の文字が数式ではない
                    or cls != xt_cls                                        # 2. 直前と異なる段落に入った
                    # or (abs(child.x0 - xt.x0) > vmax and cls != 0)        # 3. 段落内改行だが長い斜体列や分数改行を区別したい場合の閾値（無効化中）
                    # ここで段落を閉じることで以下の2パターンを維持する
                    # A. 純粋な数式・コード段落（絶対座標で固定）sstk[-1]=="" -> sstk[-1]=="{v*}"
                    # B. テキスト主体の段落（相対定位）sstk[-1]!=""
                    or (sstk[-1] != "" and abs(child.x0 - xt.x0) > vmax)    # テキスト段落かつ横方向のズレが大きい
                ):
                    if vstk:
                        if (
                            not cur_v                                       # 1. 現在は数式外
                            and cls == xt_cls                               # 2. 同じ段落内
                            and child.x0 > max([vch.x0 for vch in vstk])    # 3. 文字が数式の右側にある
                        ):
                            vfix = vstk[0].y0 - child.y0  # 右側テキスト基準で縦位置を合わせる
                        if sstk[-1] == "":
                            xt_cls = -1  # 純数式段落の後続結合を止める（次の文字判定用に変更）
                        sstk[-1] += f"{{v{len(var)}}}"
                        var.append(vstk)
                        varl.append(vlstk)
                        varf.append(vfix)
                        vstk = []
                        vlstk = []
                        vfix = 0
                # 非数式、または数式の先頭文字の場合
                if not vstk:
                    if cls == xt_cls:               # 同じ段落
                        if child.x0 > xt.x1 + 1:    # 行内スペースを挿入
                            sstk[-1] += " "
                        elif child.x1 < xt.x0:      # 折り返しが生じた場合は空白を入れて改行フラグを立てる
                            sstk[-1] += " "
                            pstk[-1].brk = True
                    else:                           # 新しい段落を作成
                        sstk.append("")
                        pstk.append(Paragraph(child.y0, child.x0, child.x0, child.x0, child.y0, child.y1, child.size, False))
                if not cur_v:                                               # テキストを積む
                    if (
                        child.size > pstk[-1].size                          # 1. フォントサイズが段落基準より大きい
                        or len(sstk[-1].strip()) == 1                       # 2. 段落2文字目（ドロップキャップ対策）
                    ) and child.get_text() != " ":
                        pstk[-1].y -= child.size - pstk[-1].size            # 異なるサイズでも上端が揃うよう補正
                        pstk[-1].size = child.size
                    sstk[-1] += child.get_text()
                else:                                                       # 数式スタックに積む
                    if (
                        not vstk                                            # 1. 数式の先頭
                        and cls == xt_cls                                   # 2. 同じ段落内
                        and child.x0 > xt.x0                                # 3. 左隣が本文
                    ):
                        vfix = child.y0 - xt.y0  # 左側テキスト基準で縦位置を合わせる
                    vstk.append(child)
                # 段落境界を更新（折り返し直後に数式が始まる場合もあるため外側で処理）
                pstk[-1].x0 = min(pstk[-1].x0, child.x0)
                pstk[-1].x1 = max(pstk[-1].x1, child.x1)
                pstk[-1].y0 = min(pstk[-1].y0, child.y0)
                pstk[-1].y1 = max(pstk[-1].y1, child.y1)
                # 直前文字情報を更新
                xt = child
                xt_cls = cls
            elif isinstance(child, LTFigure):   # 図表
                pass
            elif isinstance(child, LTLine):     # 線分
                layout = self.layout[ltpage.pageid]
                # figure の高さ対策で layout.shape を基準にする
                h, w = layout.shape
                # 線分が属するレイアウトクラスを取得
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                if vstk and cls == xt_cls:      # 数式内の線
                    vlstk.append(child)
                else:                           # ページ全体の線
                    lstk.append(child)
            else:
                pass
        flush_vertical_chars()
        # 最後に未処理の数式を吐き出す
        if vstk:
            sstk[-1] += f"{{v{len(var)}}}"
            var.append(vstk)
            varl.append(vlstk)
            varf.append(vfix)
        log.debug("\n==========[VSTACK]==========\n")
        for id, v in enumerate(var):  # 数式の幅を計算
            l = max([vch.x1 for vch in v]) - v[0].x0
            log.debug(f'< {l:.1f} {v[0].x0:.1f} {v[0].y0:.1f} {v[0].cid} {v[0].fontname} {len(varl[id])} > v{id} = {"".join([ch.get_text() for ch in v])}')
            vlen.append(l)

        ############################################################
        # B. 段落の翻訳
        log.debug("\n==========[SSTACK]==========\n")

        @retry(wait=wait_fixed(1))
        def worker(s: str):  # 並列に翻訳する
            if not s.strip() or re.match(r"^\{v\d+\}$", s):  # 空白と数式はそのまま
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
        # C. 出力 PDF の再レイアウト
        def raw_string(fcur: str, cstk: str):  # 文字列を font に合わせてエンコード
            if fcur == self.noto_name:
                return "".join(["%04x" % self.noto.has_glyph(ord(c)) for c in cstk])
            elif isinstance(self.fontmap[fcur], PDFCIDFont):  # CID の場合は 2byte
                return "".join(["%04x" % ord(c) for c in cstk])
            else:
                return "".join(["%02x" % ord(c) for c in cstk])

        # 出力言語ごとの既定行送り
        LANG_LINEHEIGHT_MAP = {
            "zh-cn": 1.4, "zh-tw": 1.4, "zh-hans": 1.4, "zh-hant": 1.4, "zh": 1.4,
            "ja": 1.1, "ko": 1.2, "en": 1.2, "ar": 1.0, "ru": 0.8, "uk": 0.8, "ta": 0.8
        }
        default_line_height = LANG_LINEHEIGHT_MAP.get(self.translator.lang_out.lower(), 1.1) # その他は 1.1
        _x, _y = 0, 0
        ops_list = []

        def gen_op_txt(font, size, x, y, rtxt):
            return f"/{font} {size:f} Tf 1 0 0 1 {x:f} {y:f} Tm [<{rtxt}>] TJ "

        def gen_op_txt_vertical(font, size, x, y, rtxt, direction: int):
            if direction >= 0:
                return f"/{font} {size:f} Tf 0 1 -1 0 {x:f} {y:f} Tm [<{rtxt}>] TJ "
            return f"/{font} {size:f} Tf 0 -1 1 0 {x:f} {y:f} Tm [<{rtxt}>] TJ "

        def gen_op_line(x, y, xlen, ylen, linewidth):
            return f"ET q 1 0 0 1 {x:f} {y:f} cm [] 0 d 0 J {linewidth:f} w 0 0 m {xlen:f} {ylen:f} l S Q BT "

        for id, new in enumerate(news):
            x: float = pstk[id].x                       # 段落の初期 X
            y: float = pstk[id].y                       # 段落の初期 Y
            x0: float = pstk[id].x0                     # 左境界
            x1: float = pstk[id].x1                     # 右境界
            height: float = pstk[id].y1 - pstk[id].y0   # 高さ
            size: float = pstk[id].size                 # フォントサイズ
            brk: bool = pstk[id].brk                    # 改行フラグ

            if getattr(pstk[id], "vertical", False):
                direction = pstk[id].vertical_direction or -1
                positions_queue = list(pstk[id].vertical_positions)
                spacing = pstk[id].vertical_spacing or size
                if positions_queue:
                    fallback_x, fallback_y = positions_queue[0]
                else:
                    fallback_x, fallback_y = x, y

                def next_vertical_position():
                    nonlocal fallback_x, fallback_y
                    if positions_queue:
                        fallback_x, fallback_y = positions_queue.pop(0)
                        return fallback_x, fallback_y
                    fallback_y += direction * spacing
                    return fallback_x, fallback_y

                ptr = 0
                while ptr < len(new):
                    if new[ptr] == "\n":
                        ptr += 1
                        continue
                    vy_regex = re.match(r"\{\s*v([\d\s]+)\}", new[ptr:], re.IGNORECASE)
                    if vy_regex:
                        ptr += len(vy_regex.group(0))
                        continue
                    ch = new[ptr]
                    fcur_ = None
                    try:
                        if fcur_ is None and self.fontmap["tiro"].to_unichr(ord(ch)) == ch:
                            fcur_ = "tiro"  # 默认拉丁字体
                    except Exception:
                        pass
                    if fcur_ is None:
                        fcur_ = self.noto_name  # 默认非拉丁字体
                    rtxt = raw_string(fcur_, ch)
                    px, py = next_vertical_position()
                    ops_list.append(gen_op_txt_vertical(fcur_, size, px, py, rtxt, direction))
                    ptr += 1
                continue

            cstk: str = ""                              # 現在の文字バッファ
            fcur: str = None                            # 現在のフォント ID
            lidx = 0                                    # 改行行数
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
                if (                                # テキストバッファを吐き出す条件
                    fcur_ != fcur                   # 1. フォントが変わった
                    or vy_regex                     # 2. 数式を挿入する
                    or x + adv > x1 + 0.1 * size    # 3. 行の右端を超える（記号だけの行もあるので余裕を持つ）
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
                if brk and x + adv > x1 + 0.1 * size:  # 原文段落に改行があり右端へ到達した場合
                    x = x0
                    lidx += 1
                if vy_regex:  # 数式を挿入
                    fix = 0
                    if fcur is not None:  # 段落内での縦位置補正
                        fix = varf[vid]
                    for vch in var[vid]:  # 数式文字を描画
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
                    for l in varl[vid]:  # 数式内の線分
                        if l.linewidth < 5:  # 一部の文書は太線を画像背景に使うのでフィルタ
                            ops_vals.append({
                                "type": OpType.LINE,
                                "x": l.pts[0][0] + x - var[vid][0].x0,
                                "dy": l.pts[0][1] + fix - var[vid][0].y0,
                                "linewidth": l.linewidth,
                                "xlen": l.pts[1][0] - l.pts[0][0],
                                "ylen": l.pts[1][1] - l.pts[0][1],
                                "lidx": lidx
                            })
                else:  # テキストバッファへ追加
                    if not cstk:  # 行頭
                        tx = x
                        if x == x0 and ch == " ":  # 段落頭の空白は落とす
                            adv = 0
                        else:
                            cstk += ch
                    else:
                        cstk += ch
                adv -= mod # 文字修飾ぶんの補正
                fcur = fcur_
                x += adv
                if log.isEnabledFor(logging.DEBUG):
                    lstk.append(LTLine(0.1, (_x, _y), (x, y)))
                    _x, _y = x, y
            # 残りのバッファを吐き出す
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

        for l in lstk:  # ページ全体の線分
            if l.linewidth < 5:  # 太線を画像背景に使う文書もあるので制限
                ops_list.append(gen_op_line(l.pts[0][0], l.pts[0][1], l.pts[1][0] - l.pts[0][0], l.pts[1][1] - l.pts[0][1], l.linewidth))

        ops = f"BT {''.join(ops_list)}ET "
        return ops


class OpType(Enum):
    TEXT = "text"
    LINE = "line"
