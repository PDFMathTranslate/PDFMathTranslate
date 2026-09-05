"""Unicode Bidirectional Algorithm (UAX #9).

A conformant, self-contained implementation.  It is validated against the
Unicode Character Database's `BidiCharacterTest.txt` conformance suite by
`test/test_uba.py`.

Why this exists rather than a third-party dependency: `python-bidi`'s legacy
pure-Python path does not implement rule N0 (paired brackets) and fails ~16% of
the conformance suite, which shows up as brackets and formula placeholders
landing on the wrong side of Arabic text.  Its Rust path is conformant but
exposes only a reordered string -- no embedding levels and no index mapping,
both of which we need in order to segment text into shaping runs and to track
formula placeholders through reordering.

Implemented: P2-P3, X1-X10, W1-W7, N0-N2, I1-I2, L1-L2, L4.
Out of scope: rule P1 (paragraph splitting) and L3 (combining mark ordering),
which the caller handles or which HarfBuzz handles during shaping.
"""

from __future__ import annotations

import unicodedata

from pdf2zh.bidi_tables import (
    CANONICAL_BRACKETS,
    CLOSING_BRACKETS,
    MIRRORED,
    OPENING_BRACKETS,
)

MAX_DEPTH = 125

# Types removed by rule X9.
_X9_REMOVED = frozenset({"RLE", "LRE", "RLO", "LRO", "PDF", "BN"})
_ISOLATE_INITIATORS = frozenset({"LRI", "RLI", "FSI"})
# "NI" in the spec: neutral or isolate formatting.
_NEUTRAL_OR_ISOLATE = frozenset({"B", "S", "WS", "ON", "FSI", "LRI", "RLI", "PDI"})
_STRONG = frozenset({"L", "R", "AL"})


def _bidi_class(ch: str) -> str:
    cls = unicodedata.bidirectional(ch)
    # Unassigned codepoints report "" but default to L/R/AL by block; treat the
    # empty string as L, which matches the default for unassigned in practice.
    return cls or "L"


# ---------------------------------------------------------------------------
# BD9 - matching PDI for each isolate initiator
# ---------------------------------------------------------------------------


def _matching_pdi(types: list[str]) -> dict[int, int]:
    """Index of the isolate initiator -> index of its matching PDI, or len(types)."""
    matching: dict[int, int] = {}
    for i, t in enumerate(types):
        if t not in _ISOLATE_INITIATORS:
            continue
        depth = 1
        j = i + 1
        while j < len(types):
            tj = types[j]
            if tj in _ISOLATE_INITIATORS:
                depth += 1
            elif tj == "PDI":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        matching[i] = j
    return matching


# ---------------------------------------------------------------------------
# P2 / P3 - paragraph embedding level
# ---------------------------------------------------------------------------


def _p2_p3(types: list[str], start: int, end: int, matching: dict[int, int]) -> int:
    """First strong type in [start, end), skipping isolated runs.  Returns 0 or 1."""
    i = start
    while i < end:
        t = types[i]
        if t in _ISOLATE_INITIATORS:
            i = matching.get(i, end)
            i += 1  # skip past the matching PDI
            continue
        if t == "PDI":
            i += 1
            continue
        if t == "L":
            return 0
        if t in ("R", "AL"):
            return 1
        i += 1
    return 0


def base_level(text: str) -> int:
    """Paragraph embedding level per rules P2/P3."""
    types = [_bidi_class(c) for c in text]
    return _p2_p3(types, 0, len(types), _matching_pdi(types))


# ---------------------------------------------------------------------------
# X1-X8 - explicit levels and directions
# ---------------------------------------------------------------------------


def _explicit_levels(
    types: list[str], para_level: int, matching: dict[int, int]
) -> list[int]:
    """Apply X1-X8.  Mutates `types` where a directional override applies."""
    n = len(types)
    levels = [para_level] * n
    # stack entries: (embedding level, override status, isolate status)
    stack: list[tuple[int, str, bool]] = [(para_level, "n", False)]
    overflow_isolate = 0
    overflow_embedding = 0
    valid_isolate = 0

    for i in range(n):
        t = types[i]

        if t in ("RLE", "LRE", "RLO", "LRO"):
            levels[i] = stack[-1][0]
            rtl = t in ("RLE", "RLO")
            new_level = (stack[-1][0] + 1) | 1 if rtl else (stack[-1][0] + 2) & ~1
            if (
                new_level <= MAX_DEPTH
                and overflow_isolate == 0
                and overflow_embedding == 0
            ):
                override = "R" if t == "RLO" else ("L" if t == "LRO" else "n")
                stack.append((new_level, override, False))
            elif overflow_isolate == 0:
                overflow_embedding += 1

        elif t in _ISOLATE_INITIATORS:
            if t == "FSI":
                end = matching.get(i, n)
                rtl = _p2_p3(types, i + 1, end, matching) == 1
            else:
                rtl = t == "RLI"
            levels[i] = stack[-1][0]
            if stack[-1][1] != "n":
                types[i] = stack[-1][1]
            new_level = (stack[-1][0] + 1) | 1 if rtl else (stack[-1][0] + 2) & ~1
            if (
                new_level <= MAX_DEPTH
                and overflow_isolate == 0
                and overflow_embedding == 0
            ):
                valid_isolate += 1
                stack.append((new_level, "n", True))
            else:
                overflow_isolate += 1

        elif t == "PDI":
            if overflow_isolate > 0:
                overflow_isolate -= 1
            elif valid_isolate > 0:
                overflow_embedding = 0
                while not stack[-1][2]:
                    stack.pop()
                stack.pop()
                valid_isolate -= 1
            levels[i] = stack[-1][0]
            if stack[-1][1] != "n":
                types[i] = stack[-1][1]

        elif t == "PDF":
            levels[i] = stack[-1][0]
            if overflow_isolate > 0:
                pass
            elif overflow_embedding > 0:
                overflow_embedding -= 1
            elif not stack[-1][2] and len(stack) > 1:
                stack.pop()

        elif t == "B":
            # X8: paragraph separators reset everything.
            stack = [(para_level, "n", False)]
            overflow_isolate = overflow_embedding = valid_isolate = 0
            levels[i] = para_level

        else:
            levels[i] = stack[-1][0]
            if stack[-1][1] != "n":
                types[i] = stack[-1][1]

    return levels


# ---------------------------------------------------------------------------
# X10 - isolating run sequences
# ---------------------------------------------------------------------------


def _isolating_run_sequences(
    types: list[str], levels: list[int], para_level: int, matching: dict[int, int]
) -> list[tuple[list[int], str, str]]:
    """Return [(indices, sos, eos)] for each isolating run sequence."""
    n = len(types)
    # Indices surviving X9.
    keep = [i for i in range(n) if types[i] not in _X9_REMOVED]
    if not keep:
        return []

    # Level runs over the surviving indices.
    runs: list[list[int]] = []
    for i in keep:
        if runs and levels[runs[-1][-1]] == levels[i]:
            runs[-1].append(i)
        else:
            runs.append([i])

    run_of_index = {run[0]: k for k, run in enumerate(runs)}
    used = [False] * len(runs)
    sequences: list[list[int]] = []

    for k, run in enumerate(runs):
        if used[k]:
            continue
        first = run[0]
        # Start a sequence only at a run that is not a continuation, i.e. whose
        # first character is not a PDI matching an isolate initiator.
        if types[first] == "PDI" and any(matching.get(j) == first for j in matching):
            continue
        seq: list[int] = []
        cur = k
        while True:
            used[cur] = True
            seq.extend(runs[cur])
            last = runs[cur][-1]
            if types[last] in _ISOLATE_INITIATORS and matching.get(last, n) < n:
                nxt = run_of_index.get(matching[last])
                if nxt is not None and not used[nxt]:
                    cur = nxt
                    continue
            break
        sequences.append(seq)

    out: list[tuple[list[int], str, str]] = []
    keep_pos = {idx: p for p, idx in enumerate(keep)}
    for seq in sequences:
        level = levels[seq[0]]

        # sos: compare with the level of the previous surviving character.
        p = keep_pos[seq[0]]
        prev_level = levels[keep[p - 1]] if p > 0 else para_level
        sos = "R" if max(level, prev_level) % 2 else "L"

        # eos: if the sequence ends with an unmatched isolate initiator, the
        # following level is the paragraph level.
        last = seq[-1]
        if types[last] in _ISOLATE_INITIATORS and matching.get(last, n) >= n:
            next_level = para_level
        else:
            q = keep_pos[last]
            next_level = levels[keep[q + 1]] if q + 1 < len(keep) else para_level
        eos = "R" if max(levels[last], next_level) % 2 else "L"

        out.append((seq, sos, eos))
    return out


# ---------------------------------------------------------------------------
# W1-W7 - weak types
# ---------------------------------------------------------------------------


def _resolve_weak(types: list[str], seq: list[int], sos: str) -> None:
    # W1: NSM takes the type of the previous character.
    prev = sos
    for i in seq:
        if types[i] == "NSM":
            types[i] = "ON" if prev in _ISOLATE_INITIATORS or prev == "PDI" else prev
        prev = types[i]

    # W2: EN becomes AN when the last strong type is AL.
    strong = sos
    for i in seq:
        t = types[i]
        if t in _STRONG:
            strong = t
        elif t == "EN" and strong == "AL":
            types[i] = "AN"

    # W3: AL becomes R.
    for i in seq:
        if types[i] == "AL":
            types[i] = "R"

    # W4: a single ES between two EN becomes EN; a single CS between two
    # numbers of the same type becomes that type.
    for k in range(1, len(seq) - 1):
        i, before, after = seq[k], types[seq[k - 1]], types[seq[k + 1]]
        t = types[i]
        if t == "ES" and before == "EN" and after == "EN":
            types[i] = "EN"
        elif t == "CS" and before == after and before in ("EN", "AN"):
            types[i] = before

    # W5: a sequence of ET adjacent to EN becomes EN.
    k = 0
    while k < len(seq):
        if types[seq[k]] == "ET":
            j = k
            while j < len(seq) and types[seq[j]] == "ET":
                j += 1
            before = types[seq[k - 1]] if k > 0 else sos
            after = types[seq[j]] if j < len(seq) else None
            if before == "EN" or after == "EN":
                for m in range(k, j):
                    types[seq[m]] = "EN"
            k = j
        else:
            k += 1

    # W6: remaining separators and terminators become ON.
    for i in seq:
        if types[i] in ("ET", "ES", "CS"):
            types[i] = "ON"

    # W7: EN becomes L when the last strong type is L.
    strong = sos
    for i in seq:
        t = types[i]
        if t in ("L", "R"):
            strong = t
        elif t == "EN" and strong == "L":
            types[i] = "L"


# ---------------------------------------------------------------------------
# N0 - paired brackets (BD16)
# ---------------------------------------------------------------------------


def _canonical(cp: int) -> int:
    return CANONICAL_BRACKETS.get(cp, cp)


def _bracket_pairs(
    text: str, types: list[str], seq: list[int]
) -> list[tuple[int, int]]:
    """BD16: identify bracket pairs within one isolating run sequence."""
    stack: list[tuple[int, int]] = []  # (canonical closing codepoint, position in seq)
    pairs: list[tuple[int, int]] = []
    for pos, i in enumerate(seq):
        if types[i] != "ON":
            continue
        cp = ord(text[i])
        if cp in OPENING_BRACKETS:
            if len(stack) >= 63:
                return sorted(pairs)
            stack.append((_canonical(OPENING_BRACKETS[cp]), pos))
        elif cp in CLOSING_BRACKETS:
            target = _canonical(cp)
            for depth in range(len(stack) - 1, -1, -1):
                if stack[depth][0] == target:
                    pairs.append((stack[depth][1], pos))
                    del stack[depth:]
                    break
    return sorted(pairs)


def _strong_class(t: str) -> str | None:
    """Strong direction for N0/N1 purposes; EN and AN count as R."""
    if t == "L":
        return "L"
    if t in ("R", "EN", "AN"):
        return "R"
    return None


def _resolve_brackets(
    text: str, types: list[str], seq: list[int], sos: str, level: int
) -> None:
    e = "R" if level % 2 else "L"
    o = "L" if e == "R" else "R"
    for open_pos, close_pos in _bracket_pairs(text, types, seq):
        found_e = False
        found_o = False
        for pos in range(open_pos + 1, close_pos):
            d = _strong_class(types[seq[pos]])
            if d == e:
                found_e = True
                break
            if d == o:
                found_o = True

        if found_e:
            new = e
        elif found_o:
            # Check the preceding context back to sos.
            prior = sos
            for pos in range(open_pos - 1, -1, -1):
                d = _strong_class(types[seq[pos]])
                if d is not None:
                    prior = d
                    break
            new = o if prior == o else e
        else:
            continue  # no strong type inside: leave the brackets as they are

        types[seq[open_pos]] = new
        types[seq[close_pos]] = new
        # Any NSM that originally followed a paired bracket takes its new type.
        for pos in (open_pos, close_pos):
            for after in range(pos + 1, len(seq)):
                if _bidi_class(text[seq[after]]) == "NSM":
                    types[seq[after]] = new
                else:
                    break


# ---------------------------------------------------------------------------
# N1-N2 - neutral types
# ---------------------------------------------------------------------------


def _resolve_neutrals(
    types: list[str], seq: list[int], sos: str, eos: str, level: int
) -> None:
    e = "R" if level % 2 else "L"
    k = 0
    while k < len(seq):
        if types[seq[k]] in _NEUTRAL_OR_ISOLATE:
            j = k
            while j < len(seq) and types[seq[j]] in _NEUTRAL_OR_ISOLATE:
                j += 1
            before = _strong_class(types[seq[k - 1]]) if k > 0 else sos
            after = _strong_class(types[seq[j]]) if j < len(seq) else eos
            new = before if before == after and before is not None else e
            for m in range(k, j):
                types[seq[m]] = new
            k = j
        else:
            k += 1


# ---------------------------------------------------------------------------
# I1-I2 - implicit levels
# ---------------------------------------------------------------------------


def _resolve_implicit(types: list[str], levels: list[int], seq: list[int]) -> None:
    for i in seq:
        t = types[i]
        if levels[i] % 2 == 0:  # I1
            if t == "R":
                levels[i] += 1
            elif t in ("AN", "EN"):
                levels[i] += 2
        else:  # I2
            if t in ("L", "AN", "EN"):
                levels[i] += 1


# ---------------------------------------------------------------------------
# L1 / L2 / L4
# ---------------------------------------------------------------------------


def _reset_levels(original: list[str], levels: list[int], para_level: int) -> None:
    """Rule L1, using the ORIGINAL bidi classes."""
    n = len(levels)
    reset_from: int | None = None
    for i in range(n):
        t = original[i]
        if t in ("B", "S"):
            levels[i] = para_level
            if reset_from is not None:
                for j in range(reset_from, i):
                    levels[j] = para_level
            reset_from = None
        elif t in ("WS", "FSI", "LRI", "RLI", "PDI") or t in _X9_REMOVED:
            if reset_from is None:
                reset_from = i
        else:
            reset_from = None
    if reset_from is not None:
        for j in range(reset_from, n):
            levels[j] = para_level


def reorder_indices(levels: list[int], indices: list[int] | None = None) -> list[int]:
    """Rule L2.  Returns the visual order of `indices` (default: 0..len-1)."""
    order = list(range(len(levels))) if indices is None else list(indices)
    if not order:
        return order
    present = [levels[i] for i in order]
    highest = max(present)
    lowest_odd = min((lv for lv in present if lv % 2), default=highest + 1)
    for level in range(highest, lowest_odd - 1, -1):
        i = 0
        while i < len(order):
            if levels[order[i]] >= level:
                j = i
                while j < len(order) and levels[order[j]] >= level:
                    j += 1
                order[i:j] = order[i:j][::-1]
                i = j
            else:
                i += 1
    return order


def mirror_char(ch: str) -> str:
    """Rule L4."""
    cp = MIRRORED.get(ord(ch))
    return chr(cp) if cp is not None else ch


def is_removed(ch: str) -> bool:
    """True for characters removed by rule X9 (explicit formatting codes).

    These carry no glyph and must not reach the shaper.
    """
    return _bidi_class(ch) in _X9_REMOVED


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve(text: str, para_level: int | None = None) -> tuple[list[int], int]:
    """Resolve embedding levels for `text`.

    Returns (levels, paragraph_level).  `levels[i]` is the embedding level of
    `text[i]`; even levels are left-to-right, odd levels right-to-left.
    """
    original = [_bidi_class(c) for c in text]
    matching = _matching_pdi(original)
    if para_level is None:
        para_level = _p2_p3(original, 0, len(original), matching)

    types = list(original)
    levels = _explicit_levels(types, para_level, matching)

    for seq, sos, eos in _isolating_run_sequences(types, levels, para_level, matching):
        level = levels[seq[0]]
        _resolve_weak(types, seq, sos)
        _resolve_brackets(text, types, seq, sos, level)
        _resolve_neutrals(types, seq, sos, eos, level)
        _resolve_implicit(types, levels, seq)

    _reset_levels(original, levels, para_level)
    return levels, para_level


def display_order(text: str, para_level: int | None = None) -> list[int]:
    """Indices of `text` in visual (left-to-right) order, X9 characters removed."""
    levels, _ = resolve(text, para_level)
    keep = [i for i, c in enumerate(text) if _bidi_class(c) not in _X9_REMOVED]
    return reorder_indices(levels, keep)


def get_display(text: str, para_level: int | None = None) -> str:
    """Visual-order string with rule L4 mirroring applied.  Mainly for tests."""
    levels, _ = resolve(text, para_level)
    out = []
    for i in display_order(text, para_level):
        ch = text[i]
        out.append(mirror_char(ch) if levels[i] % 2 else ch)
    return "".join(out)
