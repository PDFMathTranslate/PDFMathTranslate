import ast
from pathlib import Path
import unittest


def load_setup_gui():
    source = Path(__file__).parents[1].joinpath("pdf2zh", "gui.py").read_text()
    tree = ast.parse(source)
    setup_gui = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "setup_gui"
    )
    namespace = {
        "flag_demo": False,
        "parse_user_passwd": lambda _: ([], ""),
        "_has_ipv6": lambda: True,
    }
    exec(compile(ast.Module([setup_gui], type_ignores=[]), "gui.py", "exec"), namespace)
    return namespace


class FakeDemo:
    def __init__(self, failures=0):
        self.failures = failures
        self.calls = []

    def launch(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            raise RuntimeError("launch failed")


class TestGuiBindingOrder(unittest.TestCase):
    def test_legacy_gui_prefers_ipv4_wildcard_address(self):
        namespace = load_setup_gui()
        demo = FakeDemo()
        namespace["demo"] = demo

        namespace["setup_gui"]()

        self.assertEqual(demo.calls[0]["server_name"], "0.0.0.0")
        self.assertEqual(len(demo.calls), 1)

    def test_legacy_gui_uses_loopback_after_wildcard_launch_fails(self):
        namespace = load_setup_gui()
        demo = FakeDemo(failures=1)
        namespace["demo"] = demo

        namespace["setup_gui"]()

        self.assertEqual(demo.calls[1]["server_name"], "127.0.0.1")

    def test_ipv6_is_only_used_after_ipv4_launches_fail(self):
        namespace = load_setup_gui()
        demo = FakeDemo(failures=2)
        namespace["demo"] = demo

        namespace["setup_gui"]()

        self.assertEqual(
            [call["server_name"] for call in demo.calls],
            ["0.0.0.0", "127.0.0.1", "[::]"],
        )

    def test_ipv6_is_skipped_when_unavailable(self):
        namespace = load_setup_gui()
        namespace["_has_ipv6"] = lambda: False
        demo = FakeDemo(failures=2)
        namespace["demo"] = demo

        namespace["setup_gui"]()

        self.assertEqual(
            [call.get("server_name") for call in demo.calls],
            ["0.0.0.0", "127.0.0.1", None],
        )
