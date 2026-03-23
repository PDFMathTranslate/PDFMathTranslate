"""Vue GUI launcher — serves pre-built Vue SPA via Flask."""

import logging
import os
import threading
import webbrowser

logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def launch_vue_gui(host="127.0.0.1", port=8787, model=None):
    """Start Flask server serving the Vue SPA and API endpoints."""
    try:
        from flask import send_from_directory
    except ImportError:
        raise ImportError(
            "Flask is required for the Vue GUI. "
            "Install with: pip install pdf2zh[vue]"
        )

    from pdf2zh.api import create_api_app

    if not os.path.isfile(os.path.join(STATIC_DIR, "index.html")):
        raise FileNotFoundError(
            "Vue GUI not found at %s. "
            "Install with: pip install pdf2zh[vue]" % STATIC_DIR
        )

    app, _jobs = create_api_app(token=None, model=model)

    @app.route("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.route("/<path:path>")
    def static_files(path):
        file_path = os.path.join(STATIC_DIR, path)
        if os.path.isfile(file_path):
            return send_from_directory(STATIC_DIR, path)
        # SPA fallback for client-side routing
        return send_from_directory(STATIC_DIR, "index.html")

    url = f"http://{host}:{port}"
    threading.Timer(1.0, webbrowser.open, args=[url]).start()

    print(f"Vue GUI running at {url}")
    app.run(host=host, port=port, threaded=True)
