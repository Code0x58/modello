"""Mount the ordinary explorer once per example, with shared gallery navigation."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from examples.gallery.catalog import EXAMPLES
from modello.gui import create_app

if TYPE_CHECKING:
    from _typeshed.wsgi import WSGIApplication
    from flask import Flask


def create_gallery() -> Flask:
    """Create one local app; each example retains its own equations and inputs."""
    from flask import Flask, render_template, request
    from werkzeug.middleware.dispatcher import DispatcherMiddleware

    app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/assets")
    app.config["TRUSTED_HOSTS"] = ["localhost", "127.0.0.1", "[::1]"]

    @app.get("/")
    def index() -> str:
        return render_template("gallery.html", examples=EXAMPLES)

    def navigation() -> dict[str, Any]:
        prefix, _, selected = request.script_root.rpartition("/")
        return {
            "gallery_home": prefix + "/",
            "gallery_css": prefix + "/assets/gallery.css",
            "gallery_links": [
                {"title": example.info.title, "url": f"{prefix}/{slug}/", "selected": slug == selected}
                for slug, example in EXAMPLES.items()
            ],
        }

    mounts: dict[str, WSGIApplication] = {}
    for slug, example in EXAMPLES.items():
        explorer = create_app(example.model, inputs=example.inputs)
        explorer.context_processor(navigation)
        mounts["/" + slug] = explorer.wsgi_app
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, mounts)  # type: ignore[method-assign]  # Flask's middleware hook.
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore the Modello example gallery in one local app.")
    parser.add_argument("--port", type=int, default=5070)
    args = parser.parse_args()
    create_gallery().run(host="127.0.0.1", port=args.port, debug=False)
