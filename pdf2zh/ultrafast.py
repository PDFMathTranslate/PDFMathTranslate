"""Native text layout for the existing translation mask; no rendering or OCR."""

from collections import defaultdict
import logging
import math
from time import perf_counter

import numpy as np

logger = logging.getLogger(__name__)


class PerfTimer:
    """Wall-clock stages; page detail is nested inside page_processing."""

    def __init__(self):
        self.started = self.last = perf_counter()
        self.timings = defaultdict(float)

    def add(self, stage, seconds):
        self.timings[stage] += seconds

    def step(self, stage):
        now = perf_counter()
        self.add(stage, now - self.last)
        logger.info("[ultrafast perf] %s: %.3fs", stage, now - self.last)
        self.last = now

    def report(self):
        logger.info(
            "[ultrafast perf] total: %.3fs; stages: %s",
            perf_counter() - self.started,
            ", ".join(
                f"{name}={seconds:.3f}s" for name, seconds in self.timings.items()
            ),
        )


def extract_layout(data, pages=None):
    try:
        from pdf_inspector import extract_text_with_positions_bytes
    except ImportError as exc:
        raise RuntimeError(
            "--ultrafast requires pip install 'pdf2zh[ultrafast]'"
        ) from exc
    items = defaultdict(list)
    for item in extract_text_with_positions_bytes(
        data, pages=[p + 1 for p in pages] if pages else None
    ):
        if item.item_type == "text" and item.text.strip():
            items[item.page - 1].append(item)
    return items


def layout_mask(page, items):
    x0, y0, x1, y1 = page.cropbox
    width, height = x1 - x0, y1 - y0
    shape = (math.ceil(height), math.ceil(width))
    mask = np.zeros(shape, dtype=np.int32)
    regions = []
    # ponytail: proximity groups native text; use normal mode for complex layouts.
    for item in sorted(items, key=lambda i: (-i.y, i.x)):
        if item.rotation != 0:
            continue  # Preserve text whose extraction frame is rotated.
        size = max(item.font_size, 1)
        rect = [item.x, item.y - size * 0.3, item.x + item.width, item.y + item.height]
        for group in reversed(regions):
            region, previous_size, baseline = group
            horizontal_gap = max(region[0] - rect[2], rect[0] - region[2], 0)
            vertical_gap = max(region[1] - rect[3], rect[1] - region[3], 0)
            same_line = abs(item.y - baseline) < size * 0.5
            if 0.75 < size / previous_size < 1.35 and (
                (same_line and horizontal_gap < size * 1.5)
                or (
                    abs(rect[0] - region[0]) < size * 2
                    and horizontal_gap == 0
                    and vertical_gap < size * 0.7
                )
            ):
                region[:] = [
                    min(region[0], rect[0]),
                    min(region[1], rect[1]),
                    max(region[2], rect[2]),
                    max(region[3], rect[3]),
                ]
                group[2] = item.y
                break
        else:
            regions.append([rect, size, item.y])
    for label, (rect, _, _) in enumerate(regions, 2):
        left, bottom, right, top = rect
        h, w = mask.shape
        left, right = np.clip([math.floor(left - 1), math.ceil(right + 1)], 0, w)
        bottom, top = np.clip([math.floor(bottom - 1), math.ceil(top + 1)], 0, h)
        mask[bottom:top, left:right] = label
    if not regions:
        logger.warning(
            "--ultrafast: page %s has no usable native text; preserved without OCR",
            page.pageno + 1,
        )
    return mask
