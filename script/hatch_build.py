"""Hatchling build hook — builds the Vue submodule and copies dist to pdf2zh/static/."""

import os
import shutil
import subprocess

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class VueBuildHook(BuildHookInterface):
    PLUGIN_NAME = "vue-build"

    def initialize(self, version, build_data):
        root = self.root
        web_dir = os.path.join(root, "pdf2zh", "web")
        dist_dir = os.path.join(web_dir, "dist")
        static_dir = os.path.join(root, "pdf2zh", "static")

        # Only build if the submodule has a package.json
        if not os.path.isfile(os.path.join(web_dir, "package.json")):
            return

        # Check if rebuild is needed: no static dir, or submodule is newer
        needs_build = not os.path.isdir(static_dir)
        if not needs_build:
            # Compare mtime of submodule src vs static index.html
            static_index = os.path.join(static_dir, "index.html")
            if os.path.isfile(static_index):
                src_dir = os.path.join(web_dir, "src")
                if os.path.isdir(src_dir):
                    static_mtime = os.path.getmtime(static_index)
                    for dirpath, _, filenames in os.walk(src_dir):
                        for f in filenames:
                            if (
                                os.path.getmtime(os.path.join(dirpath, f))
                                > static_mtime
                            ):
                                needs_build = True
                                break
                        if needs_build:
                            break
            else:
                needs_build = True

        if not needs_build:
            return

        print("Building Vue frontend from submodule...")
        subprocess.check_call(["bun", "install"], cwd=web_dir)
        subprocess.check_call(["bun", "run", "build"], cwd=web_dir)

        if not os.path.isdir(dist_dir):
            raise RuntimeError(f"Vue build did not produce {dist_dir}")

        if os.path.exists(static_dir):
            shutil.rmtree(static_dir)
        shutil.copytree(dist_dir, static_dir)
        print(f"Copied Vue dist -> {static_dir}")

        # Include static files in the wheel
        build_data["shared_data"] = {}
        build_data["force_include"] = {
            static_dir: "pdf2zh/static",
        }
