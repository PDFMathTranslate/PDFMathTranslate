import ast
from pathlib import Path


def _load_gui_lang_map() -> dict[str, str]:
    gui_path = Path(__file__).resolve().parents[1] / "pdf2zh" / "gui.py"
    module = ast.parse(gui_path.read_text(encoding="utf-8"))

    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "lang_map":
                    return ast.literal_eval(node.value)

    raise AssertionError("lang_map not found in pdf2zh/gui.py")


def test_gui_language_map_includes_thai():
    assert _load_gui_lang_map()["Thai"] == "th"


def test_qwen_mt_maps_thai_language_code():
    from pdf2zh.translator import QwenMtTranslator

    assert QwenMtTranslator.lang_mapping("th") == "Thai"
